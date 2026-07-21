"""CLIP / RemoteCLIP: preprocessing, prompt strategies, zero-shot heads.

HOW ZERO-SHOT CLASSIFICATION WORKS HERE. CLIP is trained contrastively to place
an image and its caption close together in one shared embedding space. So a
classifier can be built with no labels and no gradient steps: encode one text
prompt per class, encode the image, take the class whose text embedding has the
highest cosine similarity. The classifier "head" is literally a matrix of
L2-normalised text embeddings.

THE PREPROCESSING MISMATCH, STATED PLAINLY. CLIP consumes 3-channel 8-bit RGB at
224x224 with its own normalisation. A EuroSAT chip is 13-channel 16-bit at
64x64. Bridging that gap means: select B04/B03/B02, percentile-stretch, quantise
to 8-bit, bicubic-upsample 64 -> 224, apply CLIP's normalisation. TEN OF THE
THIRTEEN BANDS ARE DISCARDED — every band that carries the red-edge, NIR and
SWIR information that makes land cover separable in the first place. The
vision-language model only ever sees something that looks like a photograph.
This is a structural handicap of RGB-pretrained VLMs on multispectral data, not
a tuning problem, and any comparison against a 13-band supervised model must say
so. NB03 therefore also compares against an RGB-only supervised arm, which is
the apples-to-apples comparison.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg
from .bands import to_uint8_rgb

# --------------------------------------------------------------------------
# Prompt strategies (the NB03 experiment)
# --------------------------------------------------------------------------
# v4 templates. Averaging several phrasings in embedding space cancels the
# idiosyncratic direction each individual wording contributes, leaving the part
# that is actually about the class. This is the ensembling trick from the
# original CLIP paper; it reliably buys a point or two for free.
PROMPT_TEMPLATES: tuple[str, ...] = (
    "a satellite photo of {}.",
    "an aerial view of {}.",
    "a high-resolution overhead image of {}.",
    "a Sentinel-2 satellite image showing {}.",
    "a top-down aerial photograph of {}.",
    "a remote sensing image of {}.",
    "satellite imagery of {}.",
    "an overhead photo of {} taken from space.",
    "a land cover map region of {}.",
    "a low-resolution satellite image of {}.",
)


def prompts_v1(class_names=cfg.CLASS_NAMES) -> dict[str, list[str]]:
    """Raw dataset labels. 'HerbaceousVegetation' is not English and CLIP's
    tokeniser will shred it; expect this to be the worst strategy."""
    return {c: [c] for c in class_names}


def prompts_v2(class_names=cfg.CLASS_NAMES) -> dict[str, list[str]]:
    """Standard single template over the raw labels: isolates the effect of the
    template alone, with the class wording held fixed."""
    return {c: [f"a satellite photo of {c}."] for c in class_names}


def prompts_v3(class_names=cfg.CLASS_NAMES) -> dict[str, list[str]]:
    """Natural-language class names in the same single template: isolates the
    effect of the wording, with the template held fixed. v2 vs v3 is the
    controlled comparison that shows the class NAME is a design decision."""
    return {c: [f"a satellite photo of {cfg.NATURAL_CLASS_NAMES[c]}."] for c in class_names}


def prompts_v4(class_names=cfg.CLASS_NAMES) -> dict[str, list[str]]:
    """Natural names x the full template ensemble."""
    return {
        c: [t.format(cfg.NATURAL_CLASS_NAMES[c]) for t in PROMPT_TEMPLATES] for c in class_names
    }


PROMPT_STRATEGIES = {
    "v1_raw_labels": prompts_v1,
    "v2_simple_template": prompts_v2,
    "v3_natural_names": prompts_v3,
    "v4_prompt_ensemble": prompts_v4,
}


# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def ms_chip_to_pil(chip: np.ndarray, low: float = 2.0, high: float = 98.0):
    """(13, H, W) reflectance -> PIL RGB image, via percentile stretch to 8-bit.

    The stretch is not cosmetic: feeding raw reflectance would give CLIP a
    near-black image, which is far outside the distribution of anything it saw
    in training. We are already asking it to generalise from photographs to
    overhead imagery; handing it an unrendered array as well would make the
    result uninterpretable.
    """
    from PIL import Image

    return Image.fromarray(to_uint8_rgb(np.asarray(chip), low, high))


def preprocess_chips(chips: np.ndarray, preprocess, batch: int = 256):
    """Apply an OpenCLIP preprocess transform to a batch of (N, 13, H, W) chips.

    Uses the model's OWN preprocess callable rather than a reimplementation, so
    the resize interpolation and normalisation constants always match the
    checkpoint even if a different backbone is swapped in.
    """
    import torch

    out = []
    for start in range(0, len(chips), batch):
        block = np.asarray(chips[start : start + batch])
        out.append(torch.stack([preprocess(ms_chip_to_pil(c)) for c in block]))
    tensor = torch.cat(out)
    assert tensor.ndim == 4 and tensor.shape[1] == 3, f"expected (N,3,H,W), got {tuple(tensor.shape)}"
    return tensor


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_openclip(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str | None = None):
    """Load a generic OpenCLIP checkpoint. Returns (model, preprocess, tokenizer)."""
    import open_clip

    device = device or cfg.get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    # Freeze THIS model's parameters only. Emphatically do not call
    # torch.set_grad_enabled(False) here: that flips a global, session-wide
    # switch, so every later backward() in the notebook — the fine-tuning arm in
    # NB04, for instance — dies with "element 0 of tensors does not require grad
    # and does not have a grad_fn", thousands of lines away from this function
    # and with nothing in the traceback pointing back to CLIP. Local no_grad()
    # blocks around the encode calls give the same speed with none of the reach.
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, tokenizer


# RemoteCLIP publishes OpenCLIP-format state_dicts on the Hub (repo id below,
# from the authors' README at github.com/ChenDelong1999/RemoteCLIP). It is NOT a
# HuggingFace CLIPModel checkpoint, which is why it is loaded by hand into an
# open_clip architecture rather than with transformers' from_pretrained.
REMOTECLIP_REPO = "chendelong/RemoteCLIP"
REMOTECLIP_FILES = {
    "ViT-B-32": "RemoteCLIP-ViT-B-32.pt",
    "ViT-L-14": "RemoteCLIP-ViT-L-14.pt",
    "RN50": "RemoteCLIP-RN50.pt",
}


def load_remoteclip(model_name: str = "ViT-B-32", device: str | None = None):
    """Load RemoteCLIP weights into an OpenCLIP architecture.

    Raises on failure rather than silently falling back — a notebook that
    quietly reports generic CLIP numbers under the heading "RemoteCLIP" would be
    worse than one that says the download failed. NB03 catches this and
    documents the fallback it used.
    """
    import torch
    from huggingface_hub import hf_hub_download

    if model_name not in REMOTECLIP_FILES:
        raise KeyError(f"no RemoteCLIP checkpoint known for {model_name!r}")
    device = device or cfg.get_device()
    path = hf_hub_download(REMOTECLIP_REPO, REMOTECLIP_FILES[model_name], repo_type="model")
    model, preprocess, tokenizer = load_openclip(model_name, pretrained="openai", device=device)
    state = torch.load(path, map_location="cpu")
    state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 10:  # a couple of buffers may legitimately differ
        raise RuntimeError(f"RemoteCLIP load looks wrong: {len(missing)} missing keys, e.g. {missing[:5]}")
    return model.to(device).eval(), preprocess, tokenizer, {"missing": missing, "unexpected": unexpected}


# --------------------------------------------------------------------------
# Zero-shot head
# --------------------------------------------------------------------------
def build_zeroshot_classifier(model, tokenizer, prompts: dict[str, list[str]], device: str | None = None):
    """Text-embedding classifier matrix, shape (num_classes, embed_dim).

    Per class: encode every prompt, L2-normalise each, average, then
    L2-RENORMALISE the average. The final renormalisation matters — the mean of
    unit vectors is not a unit vector, and skipping it makes classes whose
    prompts disagree (a shorter mean vector) systematically less likely to win
    the argmax, silently biasing the classifier by prompt consistency rather
    than by image content.
    """
    import torch

    device = device or cfg.get_device()
    rows = []
    with torch.no_grad():
        for class_name in prompts:
            tokens = tokenizer(prompts[class_name]).to(device)
            emb = model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(dim=0)
            rows.append(mean / mean.norm())
    weights = torch.stack(rows)
    assert weights.shape[0] == len(prompts)
    return weights


def encode_images(model, images, device: str | None = None, batch: int = 256):
    """L2-normalised image embeddings for a preprocessed (N, 3, H, W) tensor."""
    import torch

    device = device or cfg.get_device()
    out = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch):
            block = images[start : start + batch].to(device)
            emb = model.encode_image(block).float()
            out.append((emb / emb.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


def zeroshot_predict(image_features, class_weights, logit_scale: float = 100.0):
    """Cosine similarity logits -> (logits, predictions) as numpy.

    The 100x scale is CLIP's learned temperature; it does not change the argmax
    but it is what makes the softmax over these logits meaningfully peaked, so
    the confidences are comparable to a trained classifier's in NB06.
    """
    import torch

    logits = logit_scale * image_features.float() @ class_weights.float().cpu().T
    return logits.numpy(), torch.argmax(logits, dim=1).numpy()
