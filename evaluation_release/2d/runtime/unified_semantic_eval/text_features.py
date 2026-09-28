from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Sequence


TEXT_PROMPT_TEMPLATES = {
    "single": ("{label}",),
    "scaffold": (
        "{label}",
        "a photo of {label}",
        "a rendering of {label}",
    ),
    "maskclustering": (
        "{label}",
        "a photo of a {label}.",
        "an indoor photo of a {label}.",
        "a photo of the {label} in a room.",
    ),
}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_prompts(class_names: Sequence[str], mode: str) -> tuple[list[str], list[int]]:
    mode = str(mode or "maskclustering").strip().lower()
    if mode not in TEXT_PROMPT_TEMPLATES:
        raise RuntimeError(f"Unsupported text prompt mode: {mode}")
    prompts = []
    owners = []
    for class_index, class_name in enumerate(class_names):
        label = " ".join(str(class_name).strip().split())
        for template in TEXT_PROMPT_TEMPLATES[mode]:
            prompts.append(template.format(label=label))
            owners.append(class_index)
    return prompts, owners


def _group_features(features, owners: Sequence[int], num_classes: int):
    import torch
    import torch.nn.functional as functional

    grouped = []
    owner_tensor = torch.as_tensor(owners, dtype=torch.long, device=features.device)
    for class_index in range(num_classes):
        class_features = features[owner_tensor == class_index]
        grouped.append(functional.normalize(class_features.mean(dim=0), dim=0, eps=1e-6))
    return torch.stack(grouped, dim=0).contiguous()


def encode_openclip_text(
    class_names: Sequence[str],
    model_name: str,
    pretrained: str,
    device: str,
    prompt_mode: str = "maskclustering",
    batch_size: int = 64,
):
    import torch
    import torch.nn.functional as functional

    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("open_clip is required; install open-clip-torch") from exc

    pretrained_path = Path(pretrained).expanduser()
    precision = "fp16" if str(device).startswith("cuda") else "fp32"
    if pretrained_path.is_file():
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None,
            precision=precision,
            device=device,
        )
        try:
            from open_clip.factory import load_checkpoint
        except Exception:
            load_checkpoint = getattr(open_clip, "load_checkpoint", None)
        if load_checkpoint is None:
            raise RuntimeError("The installed open_clip does not expose load_checkpoint")
        load_checkpoint(model, str(pretrained_path))
        weight_source = str(pretrained_path.resolve())
        weight_sha256 = _file_sha256(pretrained_path)
    else:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=device,
        )
        weight_source = str(pretrained)
        weight_sha256 = None
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    prompts, owners = expand_prompts(class_names, prompt_mode)
    encoded = []
    with torch.no_grad():
        for start in range(0, len(prompts), max(1, int(batch_size))):
            tokens = tokenizer(prompts[start : start + int(batch_size)]).to(device)
            encoded.append(functional.normalize(model.encode_text(tokens).float(), dim=-1, eps=1e-6))
    features = _group_features(torch.cat(encoded, dim=0), owners, len(class_names))
    return features, {
        "family": "openclip",
        "model": model_name,
        "pretrained": weight_source,
        "weightSha256": weight_sha256,
        "promptMode": prompt_mode,
        "featureDim": int(features.shape[1]),
    }


def encode_alphaclip_text(
    class_names: Sequence[str],
    model_name: str,
    alpha_checkpoint: str | Path,
    package_path: str | Path,
    download_root: str | Path,
    device: str,
    prompt_mode: str = "maskclustering",
    batch_size: int = 64,
):
    import torch
    import torch.nn.functional as functional

    package_path = Path(package_path).expanduser().resolve()
    alpha_checkpoint = Path(alpha_checkpoint).expanduser().resolve()
    download_root = Path(download_root).expanduser().resolve()
    base_checkpoint = download_root / "ViT-B-16.pt"
    for required in (package_path, alpha_checkpoint, base_checkpoint):
        if not required.exists():
            raise FileNotFoundError(f"Alpha-CLIP dependency is missing: {required}")
    if str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))
    os.environ["ALPHACLIP_PACKAGE_PATH"] = str(package_path)
    os.environ["ALPHACLIP_DOWNLOAD_ROOT"] = str(download_root)
    try:
        import alpha_clip
    except ImportError as exc:
        raise RuntimeError(f"alpha_clip is not importable from {package_path}") from exc

    model, _ = alpha_clip.load(
        model_name,
        alpha_vision_ckpt_pth=str(alpha_checkpoint),
        device=device,
        download_root=str(download_root),
    )
    model.eval()
    prompts, owners = expand_prompts(class_names, prompt_mode)
    encoded = []
    with torch.no_grad():
        for start in range(0, len(prompts), max(1, int(batch_size))):
            tokens = alpha_clip.tokenize(prompts[start : start + int(batch_size)]).to(device)
            encoded.append(functional.normalize(model.encode_text(tokens).float(), dim=-1, eps=1e-6))
    features = _group_features(torch.cat(encoded, dim=0), owners, len(class_names))
    return features, {
        "family": "alphaclip",
        "model": model_name,
        "baseCheckpoint": str(base_checkpoint),
        "baseCheckpointSha256": _file_sha256(base_checkpoint),
        "alphaCheckpoint": str(alpha_checkpoint),
        "alphaCheckpointSha256": _file_sha256(alpha_checkpoint),
        "packagePath": str(package_path),
        "promptMode": prompt_mode,
        "featureDim": int(features.shape[1]),
    }
