import json
import os
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

from gaussian_renderer import prefilter_voxel, render
from scene import Scene, GaussianModel
from utils.clip_text_utils import (
    _load_text_model,
    _resolve_clip_package_dir,
    _resolve_model_path,
    _tokenize_queries,
)


_SEMANTIC_2D_EXTENSIONS = (".png", ".npy", ".npz", ".pt")
_MAX_3D_BRUTEFORCE_PAIRS = 50_000_000
_MAX_3D_QUERY_CHUNK = 4096
_MAX_3D_KDTREE_QUERY_CHUNK = 200_000


def _log(logger, message):
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def _to_abs_path(path_value, source_root):
    if not path_value:
        return None
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return Path(source_root) / path_obj


def semantic_benchmark_enabled(dataset) -> bool:
    prompt_path = getattr(dataset, "semantic_eval_prompt_path", "")
    return bool(
        getattr(dataset, "use_semantic_features", False)
        and prompt_path
        and (getattr(dataset, "semantic_gt_2d", "") or getattr(dataset, "semantic_gt_3d", ""))
    )


def _parse_prompt_definition_line(line, next_auto_id):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if "," in stripped:
        parts = [part.strip() for part in stripped.split(",", 2)]
        if parts and parts[0].lstrip("-").isdigit():
            class_id = int(parts[0])
            class_name = parts[1] if len(parts) > 1 else str(class_id)
            prompt = parts[2] if len(parts) > 2 and parts[2] else class_name
            return class_id, class_name, prompt
        class_name = parts[0]
        prompt = parts[1] if len(parts) > 1 and parts[1] else class_name
        return next_auto_id, class_name, prompt

    parts = stripped.split()
    if len(parts) > 1 and parts[0].lstrip("-").isdigit():
        class_id = int(parts[0])
        class_name = " ".join(parts[1:])
        return class_id, class_name, class_name

    return next_auto_id, stripped, stripped


def _normalize_class_name(value):
    return " ".join(str(value).strip().lower().split())


def load_semantic_prompt_bank(prompt_path):
    prompt_file = Path(prompt_path)
    if not prompt_file.exists():
        raise RuntimeError(f"Semantic evaluation prompt file does not exist: {prompt_file}")

    class_ids = []
    class_names = []
    prompts = []
    next_auto_id = 0

    if prompt_file.suffix.lower() == ".json":
        data = json.loads(prompt_file.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "classes" in data:
            entries = data["classes"]
        elif isinstance(data, dict) and "class_names" in data:
            ids = data.get("class_ids", list(range(len(data["class_names"]))))
            entries = [
                {"id": ids[idx], "name": name}
                for idx, name in enumerate(data["class_names"])
            ]
        elif isinstance(data, dict) and "prompts" in data:
            ids = data.get("class_ids", list(range(len(data["prompts"]))))
            entries = [
                {"id": ids[idx], "name": prompt, "prompt": prompt}
                for idx, prompt in enumerate(data["prompts"])
            ]
        elif isinstance(data, dict):
            entries = [{"id": key, "name": value} for key, value in data.items()]
        elif isinstance(data, list):
            entries = data
        else:
            raise RuntimeError(f"Unsupported semantic prompt JSON layout: {prompt_file}")

        for idx, entry in enumerate(entries):
            if isinstance(entry, str):
                class_id, class_name, prompt = next_auto_id, entry, entry
            elif isinstance(entry, dict):
                class_id = entry.get("id", entry.get("class_id", entry.get("label", next_auto_id)))
                class_name = entry.get("name", entry.get("class_name", entry.get("prompt", class_id)))
                prompt = entry.get("prompt", entry.get("text", class_name))
            else:
                raise RuntimeError(f"Unsupported semantic prompt JSON entry #{idx}: {entry!r}")
            class_id = int(class_id)
            class_name = str(class_name).replace("-", " ")
            prompt = str(prompt).replace("-", " ")
            class_ids.append(class_id)
            class_names.append(class_name)
            prompts.append(prompt)
            next_auto_id = max(next_auto_id, class_id + 1)
    else:
        for raw_line in prompt_file.read_text(encoding="utf-8").splitlines():
            parsed = _parse_prompt_definition_line(raw_line, next_auto_id)
            if parsed is None:
                continue
            class_id, class_name, prompt = parsed
            class_ids.append(int(class_id))
            class_names.append(class_name)
            prompts.append(prompt)
            next_auto_id = max(next_auto_id, int(class_id) + 1)

    if not prompts:
        raise RuntimeError(f"Semantic evaluation prompt file is empty: {prompt_file}")

    return {
        "class_ids": class_ids,
        "class_names": class_names,
        "prompts": prompts,
    }


def _load_class_id_to_name(class_path):
    class_file = Path(class_path)
    if not class_file.exists():
        raise RuntimeError(f"Semantic raw class file does not exist: {class_file}")

    id_to_name = {}
    next_auto_id = 0
    for raw_line in class_file.read_text(encoding="utf-8").splitlines():
        parsed = _parse_prompt_definition_line(raw_line, next_auto_id)
        if parsed is None:
            continue
        class_id, class_name, _ = parsed
        id_to_name[int(class_id)] = class_name
        next_auto_id = max(next_auto_id, int(class_id) + 1)
    if not id_to_name:
        raise RuntimeError(f"Semantic raw class file is empty: {class_file}")
    return id_to_name


def _load_benchmark_semantic_map(map_path):
    map_file = Path(map_path)
    if not map_file.exists():
        raise RuntimeError(f"Semantic benchmark map file does not exist: {map_file}")

    mapping = {}
    with map_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Semantic benchmark map file is empty: {map_file}")
        required = {"class", "semantic_map_to"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"Semantic benchmark map file missing columns {sorted(missing)}: {map_file}"
            )
        for row in reader:
            raw_name = _normalize_class_name(row.get("class", ""))
            mapped_name = _normalize_class_name(row.get("semantic_map_to", ""))
            if raw_name and mapped_name:
                mapping[raw_name] = mapped_name
    return mapping


def _build_prompt_name_to_index(prompt_bank):
    name_to_index = {}
    for class_index, (class_name, prompt) in enumerate(zip(prompt_bank["class_names"], prompt_bank["prompts"])):
        for candidate in (class_name, prompt):
            key = _normalize_class_name(candidate)
            if key:
                name_to_index.setdefault(key, class_index)
    return name_to_index


def _parse_semantic_eval_exclude_classes(dataset):
    raw_value = getattr(dataset, "semantic_eval_exclude_classes", "") or ""
    if isinstance(raw_value, (list, tuple, set)):
        candidates = [str(value).strip() for value in raw_value]
    else:
        raw_text = str(raw_value).strip()
        if not raw_text:
            return []
        if "," in raw_text or ";" in raw_text:
            candidates = [part.strip() for part in raw_text.replace(";", ",").split(",")]
        else:
            candidates = raw_text.split()

    names = []
    seen = set()
    for candidate in candidates:
        key = _normalize_class_name(candidate)
        if key and key not in seen:
            names.append(candidate.strip())
            seen.add(key)
    return names


def _resolve_semantic_eval_exclude_indices(dataset, prompt_bank, logger=None):
    exclude_names = _parse_semantic_eval_exclude_classes(dataset)
    if not exclude_names:
        return set(), []

    prompt_name_to_index = _build_prompt_name_to_index(prompt_bank)
    exclude_indices = set()
    unknown_names = []
    for class_name in exclude_names:
        key = _normalize_class_name(class_name)
        if key not in prompt_name_to_index:
            unknown_names.append(class_name)
            continue
        exclude_indices.add(int(prompt_name_to_index[key]))

    if unknown_names:
        raise RuntimeError(
            "--semantic_eval_exclude_classes contains classes that are not present in "
            f"--semantic_eval_prompt_path: {', '.join(unknown_names)}"
        )

    resolved_names = [prompt_bank["class_names"][idx] for idx in sorted(exclude_indices)]
    _log(
        logger,
        "[Semantic-Eval] Excluding GT classes from metrics: "
        + ", ".join(f"{name}[{idx}]" for idx, name in zip(sorted(exclude_indices), resolved_names)),
    )
    return exclude_indices, resolved_names


def _apply_semantic_eval_exclude_indices(gt_indices, exclude_indices):
    if not exclude_indices:
        return gt_indices
    excluded = torch.zeros_like(gt_indices, dtype=torch.bool)
    for class_index in exclude_indices:
        excluded |= gt_indices == int(class_index)
    if not bool(excluded.any().item()):
        return gt_indices
    filtered = gt_indices.clone()
    filtered[excluded] = -1
    return filtered


def _semantic_eval_class_indices(num_classes, exclude_indices):
    if not exclude_indices:
        return None
    excluded = {int(index) for index in exclude_indices}
    return [index for index in range(int(num_classes)) if index not in excluded]


def _semantic_eval_present_class_indices(gt_indices, num_classes):
    gt_indices = torch.as_tensor(gt_indices).detach().cpu().long().view(-1)
    valid = (gt_indices >= 0) & (gt_indices < int(num_classes))
    if not bool(valid.any().item()):
        return []
    return [int(value) for value in torch.unique(gt_indices[valid], sorted=True).tolist()]


def _semantic_eval_prediction_class_indices(dataset, gt_indices, num_classes, exclude_indices, prompt_bank, logger=None):
    if not bool(getattr(dataset, "semantic_eval_mesh_classes_only", False)):
        return _semantic_eval_class_indices(num_classes, exclude_indices), False

    class_indices = _semantic_eval_present_class_indices(gt_indices, num_classes)
    if not class_indices:
        raise RuntimeError("--semantic_eval_mesh_classes_only found no valid classes in semantic GT.")
    class_names = [prompt_bank["class_names"][idx] for idx in class_indices]
    _log(
        logger,
        "[Semantic-Eval] Restricting prediction/evaluation to GT-present mesh classes: "
        + ", ".join(f"{name}[{idx}]" for idx, name in zip(class_indices, class_names)),
    )
    return class_indices, True


def _select_text_features_for_classes(text_features, class_indices):
    if class_indices is None:
        return text_features, None
    class_indices_tensor = torch.as_tensor(class_indices, dtype=torch.long, device=text_features.device)
    if class_indices_tensor.numel() == 0:
        raise RuntimeError("Semantic evaluation class restriction is empty.")
    return text_features[class_indices_tensor], class_indices_tensor


def _map_local_prediction_indices(pred_indices, class_indices_tensor):
    if class_indices_tensor is None:
        return pred_indices
    return class_indices_tensor[pred_indices]


def _build_gt_label_mapper(dataset, prompt_bank, logger=None):
    ignore_values = {int(getattr(dataset, "semantic_eval_ignore_label", 255))}
    raw_class_path = _to_abs_path(getattr(dataset, "semantic_raw_class_path", ""), dataset.source_path)
    benchmark_map_path = _to_abs_path(getattr(dataset, "semantic_benchmark_map_path", ""), dataset.source_path)

    if raw_class_path is None and benchmark_map_path is None:
        gt_to_index, direct_ignore_values = _build_gt_index_map(prompt_bank["class_ids"], next(iter(ignore_values)))
        _log(logger, "[Semantic-Eval] Using direct GT label-id mapping from prompt file class ids.")
        return gt_to_index, direct_ignore_values, {
            "mode": "direct",
            "mapped_raw_classes": len(gt_to_index),
            "ignored_raw_classes": 0,
        }

    if raw_class_path is None or benchmark_map_path is None:
        raise RuntimeError(
            "ScanNet++ benchmark GT mapping requires both --semantic_raw_class_path "
            "and --semantic_benchmark_map_path."
        )

    raw_id_to_name = _load_class_id_to_name(raw_class_path)
    benchmark_map = _load_benchmark_semantic_map(benchmark_map_path)
    prompt_name_to_index = _build_prompt_name_to_index(prompt_bank)

    gt_to_index = {}
    ignored_names = []
    for raw_id, raw_name in raw_id_to_name.items():
        raw_key = _normalize_class_name(raw_name)
        mapped_key = benchmark_map.get(raw_key, raw_key if raw_key in prompt_name_to_index else "")
        if mapped_key in prompt_name_to_index:
            gt_to_index[int(raw_id)] = int(prompt_name_to_index[mapped_key])
        else:
            ignored_names.append(raw_name)

    if not gt_to_index:
        raise RuntimeError(
            "ScanNet++ benchmark GT mapping produced no valid classes. "
            "Check semantic_raw_class_path, semantic_benchmark_map_path and semantic_eval_prompt_path."
        )

    _log(
        logger,
        "[Semantic-Eval] Using ScanNet++ benchmark GT mapping: "
        f"raw_classes={len(raw_id_to_name)}, mapped={len(gt_to_index)}, ignored={len(ignored_names)}.",
    )
    if ignored_names:
        _log(
            logger,
            "[Semantic-Eval] Ignored raw classes sample: "
            + ", ".join(ignored_names[:12])
            + (" ..." if len(ignored_names) > 12 else ""),
        )

    return gt_to_index, ignore_values, {
        "mode": "scannetpp_benchmark",
        "raw_class_path": str(raw_class_path),
        "benchmark_map_path": str(benchmark_map_path),
        "mapped_raw_classes": len(gt_to_index),
        "ignored_raw_classes": len(ignored_names),
    }


_TEXT_PROMPT_TEMPLATES = {
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


def _normalize_text_prompt_mode(prompt_template_mode):
    mode = str(prompt_template_mode or "scaffold").strip().lower()
    aliases = {
        "default": "scaffold",
        "legacy": "scaffold",
        "mc": "maskclustering",
        "mask_clustering": "maskclustering",
        "teacher": "maskclustering",
    }
    mode = aliases.get(mode, mode)
    if mode not in _TEXT_PROMPT_TEMPLATES:
        raise RuntimeError(
            "Unsupported semantic text prompt mode "
            f"{prompt_template_mode!r}; expected one of {sorted(_TEXT_PROMPT_TEMPLATES)}."
        )
    return mode


def _expand_text_prompts(prompts, prompt_template_mode):
    mode = _normalize_text_prompt_mode(prompt_template_mode)
    templates = _TEXT_PROMPT_TEMPLATES[mode]
    expanded_prompts = []
    prompt_owner = []
    for prompt_index, prompt in enumerate(prompts):
        normalized = " ".join(str(prompt).strip().split())
        for template in templates:
            expanded_prompts.append(template.format(label=normalized))
            prompt_owner.append(prompt_index)
    return expanded_prompts, prompt_owner, mode


def _group_text_features(text_features, prompt_owner, num_prompts):
    grouped = []
    for prompt_index in range(int(num_prompts)):
        owner_indices = [idx for idx, owner in enumerate(prompt_owner) if owner == prompt_index]
        prompt_feats = text_features[owner_indices]
        grouped_feat = F.normalize(prompt_feats.mean(dim=0), dim=0, eps=1e-6)
        grouped.append(grouped_feat)
    return torch.stack(grouped, dim=0).contiguous()


def _encode_open_vocab_text_features(prompts, clip_model_name, expected_dim, device, prompt_template_mode="scaffold"):
    if str(clip_model_name).lower().startswith(("alphaclip:", "alpha_clip:", "alpha-clip:")):
        return _encode_alphaclip_text_features(
            prompts,
            clip_model_name,
            expected_dim,
            device,
            prompt_template_mode=prompt_template_mode,
        )
    if str(clip_model_name).lower().startswith(("openclip:", "open_clip:")):
        return _encode_openclip_text_features(
            prompts,
            clip_model_name,
            expected_dim,
            device,
            prompt_template_mode=prompt_template_mode,
        )

    clip_pkg_dir = _resolve_clip_package_dir()
    model_path = _resolve_model_path(clip_model_name)
    clip_model = _load_text_model(clip_pkg_dir, model_path, device=device)

    expanded_prompts, prompt_owner, _ = _expand_text_prompts(prompts, prompt_template_mode)
    tokens = _tokenize_queries(clip_pkg_dir, expanded_prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens).float()
    text_features = F.normalize(text_features, dim=-1, eps=1e-6)
    prompt_bank = _group_text_features(text_features, prompt_owner, len(prompts))

    if prompt_bank.shape[1] != expected_dim:
        raise RuntimeError(
            f"Encoded text dim {prompt_bank.shape[1]} does not match semantic feature dim {expected_dim}."
        )
    return prompt_bank


def _parse_openclip_spec(clip_model_name):
    spec = str(clip_model_name).strip()
    for prefix in ("openclip:", "open_clip:"):
        if spec.lower().startswith(prefix):
            spec = spec[len(prefix):]
            break
    if "/" in spec and ":" not in spec:
        model_name, pretrained = spec.split("/", 1)
    elif ":" in spec:
        model_name, pretrained = spec.split(":", 1)
    else:
        model_name, pretrained = spec, "laion2b_s34b_b88k"
    model_name = model_name.strip() or "ViT-B-16"
    pretrained = pretrained.strip() or "laion2b_s34b_b88k"
    return model_name, pretrained


def _parse_alphaclip_spec(clip_model_name):
    spec = str(clip_model_name).strip()
    for prefix in ("alphaclip:", "alpha_clip:", "alpha-clip:"):
        if spec.lower().startswith(prefix):
            spec = spec[len(prefix):]
            break
    parts = spec.split(":", 2)
    model_name = parts[0].strip() if parts and parts[0].strip() else "ViT-B/16"
    checkpoint = parts[1].strip() if len(parts) > 1 else ""
    package_path = parts[2].strip() if len(parts) > 2 else ""
    if not checkpoint:
        raise RuntimeError(
            "Alpha-CLIP text encoding needs a model spec like "
            "`alphaclip:ViT-B/16:/path/to/clip_b16_grit+mim_fultune_6xe.pth`."
        )
    return model_name, checkpoint, package_path


def _encode_openclip_text_features(prompts, clip_model_name, expected_dim, device, prompt_template_mode="scaffold"):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "OpenCLIP text encoding requested, but open_clip is not installed. "
            "Install it with `pip install open-clip-torch`."
        ) from exc

    model_name, pretrained = _parse_openclip_spec(clip_model_name)
    precision = "fp16" if str(device).startswith("cuda") else "fp32"
    pretrained_path = Path(str(pretrained)).expanduser()
    if pretrained_path.exists():
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
            raise RuntimeError("This open_clip installation does not expose load_checkpoint for local weights.")
        load_checkpoint(model, str(pretrained_path))
    else:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=device,
        )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    expanded_prompts, prompt_owner, _ = _expand_text_prompts(prompts, prompt_template_mode)
    tokens = tokenizer(expanded_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
    text_features = F.normalize(text_features, dim=-1, eps=1e-6)
    prompt_bank = _group_text_features(text_features, prompt_owner, len(prompts))

    if prompt_bank.shape[1] != expected_dim:
        raise RuntimeError(
            f"Encoded OpenCLIP text dim {prompt_bank.shape[1]} does not match semantic feature dim {expected_dim}."
        )
    return prompt_bank


def _encode_alphaclip_text_features(prompts, clip_model_name, expected_dim, device, prompt_template_mode="scaffold"):
    model_name, checkpoint, package_path = _parse_alphaclip_spec(clip_model_name)
    env_package_path = os.environ.get("ALPHACLIP_PACKAGE_PATH", "")
    package_path = package_path or env_package_path
    if package_path:
        import sys

        resolved = str(Path(package_path).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    try:
        import alpha_clip
    except ImportError as exc:
        raise RuntimeError(
            "Alpha-CLIP text encoding requested, but alpha_clip is not importable. "
            "Set ALPHACLIP_PACKAGE_PATH or append the package path as the fourth field in "
            "`alphaclip:MODEL:CHECKPOINT:PACKAGE_PATH`."
        ) from exc

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise RuntimeError(f"Alpha-CLIP checkpoint does not exist: {checkpoint_path}")
    download_root = os.environ.get("ALPHACLIP_DOWNLOAD_ROOT", "") or None
    try:
        model, _ = alpha_clip.load(
            model_name,
            alpha_vision_ckpt_pth=str(checkpoint_path),
            device=device,
            download_root=download_root,
        )
    except TypeError:
        model, _ = alpha_clip.load(
            model_name,
            alpha_vision_ckpt_pth=str(checkpoint_path),
            device=device,
        )
    model.eval()

    expanded_prompts, prompt_owner, _ = _expand_text_prompts(prompts, prompt_template_mode)
    tokens = alpha_clip.tokenize(expanded_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
    text_features = F.normalize(text_features, dim=-1, eps=1e-6)
    prompt_bank = _group_text_features(text_features, prompt_owner, len(prompts))

    if prompt_bank.shape[1] != expected_dim:
        raise RuntimeError(
            f"Encoded Alpha-CLIP text dim {prompt_bank.shape[1]} does not match semantic feature dim {expected_dim}."
        )
    return prompt_bank


def _find_semantic_2d_label_path(label_dir, image_stem):
    label_dir = Path(label_dir)
    for ext in _SEMANTIC_2D_EXTENSIONS:
        candidate = label_dir / f"{image_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_semantic_label_2d(label_path):
    suffix = label_path.suffix.lower()
    if suffix == ".png":
        with Image.open(label_path) as label_image:
            label_array = np.array(label_image)
    elif suffix == ".npy":
        label_array = np.load(label_path)
    elif suffix == ".npz":
        payload = np.load(label_path)
        if "labels" in payload:
            label_array = payload["labels"]
        else:
            first_key = payload.files[0]
            label_array = payload[first_key]
    elif suffix == ".pt":
        payload = torch.load(label_path, map_location="cpu")
        if isinstance(payload, dict):
            if "labels" in payload:
                label_array = payload["labels"]
            elif "label_map" in payload:
                label_array = payload["label_map"]
            else:
                raise RuntimeError(f"Unsupported 2D semantic .pt payload keys for {label_path}")
        else:
            label_array = payload
        label_array = torch.as_tensor(label_array).cpu().numpy()
    else:
        raise RuntimeError(f"Unsupported 2D semantic label extension: {label_path}")

    if label_array.ndim == 3 and label_array.shape[-1] == 1:
        label_array = label_array[..., 0]
    if label_array.ndim != 2:
        raise RuntimeError(f"Expected a 2D semantic label map, got shape {tuple(label_array.shape)} for {label_path}")
    return torch.as_tensor(label_array, dtype=torch.long)


def _resize_label_map(label_map, target_hw):
    if tuple(label_map.shape[-2:]) == tuple(target_hw):
        return label_map
    resized = F.interpolate(
        label_map.unsqueeze(0).unsqueeze(0).float(),
        size=target_hw,
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).long()


def _update_confusion_matrix(confusion, gt_indices, pred_indices, num_classes):
    if gt_indices.numel() == 0:
        return
    encoded = gt_indices * num_classes + pred_indices
    bincount = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion += bincount.view(num_classes, num_classes)


def _compute_metrics_from_confusion(confusion, class_names, eval_class_indices=None):
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    gt_count = confusion.sum(dim=1)
    pred_count = confusion.sum(dim=0)
    union = gt_count + pred_count - true_positive

    eval_class_mask = torch.ones_like(true_positive, dtype=torch.bool)
    if eval_class_indices is not None:
        eval_class_mask = torch.zeros_like(true_positive, dtype=torch.bool)
        if len(eval_class_indices) > 0:
            eval_class_mask[torch.as_tensor(eval_class_indices, dtype=torch.long)] = True

    valid_iou = (union > 0) & eval_class_mask
    valid_acc = (gt_count > 0) & eval_class_mask

    iou = torch.zeros_like(true_positive)
    acc = torch.zeros_like(true_positive)
    nonzero_iou = union > 0
    nonzero_acc = gt_count > 0
    iou[nonzero_iou] = true_positive[nonzero_iou] / union[nonzero_iou]
    acc[nonzero_acc] = true_positive[nonzero_acc] / gt_count[nonzero_acc]

    total = confusion.sum().item()
    overall_acc = float(true_positive.sum().item() / total) if total > 0 else 0.0
    miou = float(iou[valid_iou].mean().item()) if bool(valid_iou.any().item()) else 0.0
    macc = float(acc[valid_acc].mean().item()) if bool(valid_acc.any().item()) else 0.0

    per_class = {}
    for class_index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "IoU": float(iou[class_index].item()) if bool(valid_iou[class_index].item()) else None,
            "Acc": float(acc[class_index].item()) if bool(valid_acc[class_index].item()) else None,
            "GT_COUNT": int(gt_count[class_index].item()),
            "PRED_COUNT": int(pred_count[class_index].item()),
            "EXCLUDED": not bool(eval_class_mask[class_index].item()),
        }

    return {
        "mIoU": miou,
        "mAcc": macc,
        "overallAcc": overall_acc,
        "numClasses": len(class_names),
        "numLabeled": int(total),
        "perClass": per_class,
    }


def _compute_topk_metrics(gt_indices, pred_topk_indices, num_classes, class_names, eval_class_indices=None):
    gt_indices = torch.as_tensor(gt_indices).detach().cpu().long().view(-1)
    pred_topk_indices = torch.as_tensor(pred_topk_indices).detach().cpu().long()
    if pred_topk_indices.ndim == 1:
        pred_topk_indices = pred_topk_indices.unsqueeze(1)
    if pred_topk_indices.shape[0] != gt_indices.numel():
        raise RuntimeError("Top-k prediction count does not match GT count.")

    true_positive = torch.zeros((num_classes,), dtype=torch.long)
    gt_count = torch.zeros((num_classes,), dtype=torch.long)
    pred_count = torch.zeros((num_classes,), dtype=torch.long)
    for class_index in range(num_classes):
        gt_mask = gt_indices == class_index
        pred_mask = (pred_topk_indices == class_index).any(dim=1)
        true_positive[class_index] = (gt_mask & pred_mask).sum()
        gt_count[class_index] = gt_mask.sum()
        pred_count[class_index] = pred_mask.sum()

    union = gt_count + pred_count - true_positive
    eval_class_mask = torch.ones((num_classes,), dtype=torch.bool)
    if eval_class_indices is not None:
        eval_class_mask = torch.zeros((num_classes,), dtype=torch.bool)
        if len(eval_class_indices) > 0:
            eval_class_mask[torch.as_tensor(eval_class_indices, dtype=torch.long)] = True
    valid_iou = (union > 0) & eval_class_mask
    valid_acc = (gt_count > 0) & eval_class_mask

    iou = torch.zeros((num_classes,), dtype=torch.float64)
    acc = torch.zeros((num_classes,), dtype=torch.float64)
    nonzero_iou = union > 0
    nonzero_acc = gt_count > 0
    iou[nonzero_iou] = true_positive[nonzero_iou].to(torch.float64) / union[nonzero_iou].to(torch.float64)
    acc[nonzero_acc] = true_positive[nonzero_acc].to(torch.float64) / gt_count[nonzero_acc].to(torch.float64)

    total = int(gt_indices.numel())
    matched = int((pred_topk_indices == gt_indices.unsqueeze(1)).any(dim=1).sum().item())
    overall_acc = float(matched / total) if total > 0 else 0.0
    miou = float(iou[valid_iou].mean().item()) if bool(valid_iou.any().item()) else 0.0
    macc = float(acc[valid_acc].mean().item()) if bool(valid_acc.any().item()) else 0.0

    per_class = {}
    for class_index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "IoU": float(iou[class_index].item()) if bool(valid_iou[class_index].item()) else None,
            "Acc": float(acc[class_index].item()) if bool(valid_acc[class_index].item()) else None,
            "GT_COUNT": int(gt_count[class_index].item()),
            "PRED_COUNT": int(pred_count[class_index].item()),
            "EXCLUDED": not bool(eval_class_mask[class_index].item()),
        }

    return {
        "mIoU": miou,
        "mAcc": macc,
        "overallAcc": overall_acc,
        "numClasses": int(num_classes),
        "numLabeled": int(total),
        "matched": int(matched),
        "perClass": per_class,
    }


def _format_top_counts(values, class_names=None, max_items=8):
    values = torch.as_tensor(values).detach().cpu().long().view(-1)
    if values.numel() == 0:
        return "<empty>"
    unique, counts = torch.unique(values, return_counts=True)
    order = torch.argsort(counts, descending=True)[:max_items]
    entries = []
    for item_index in order.tolist():
        value = int(unique[item_index].item())
        count = int(counts[item_index].item())
        if class_names is not None and 0 <= value < len(class_names):
            entries.append(f"{class_names[value]}[{value}]={count}")
        else:
            entries.append(f"{value}={count}")
    return ", ".join(entries)


def _build_gt_index_map(class_ids, ignore_label):
    gt_to_index = {int(class_id): idx for idx, class_id in enumerate(class_ids)}
    ignore_values = {int(ignore_label)}
    return gt_to_index, ignore_values


def _map_gt_labels_to_indices(gt_labels, gt_to_index, ignore_values):
    mapped = torch.full_like(gt_labels, -1, dtype=torch.long)
    for class_id, class_index in gt_to_index.items():
        mapped[gt_labels == int(class_id)] = int(class_index)
    for ignore_value in ignore_values:
        mapped[gt_labels == int(ignore_value)] = -1
    return mapped


def _predict_open_vocab_labels(decoded_feature_map, text_features, class_indices=None):
    text_features, class_indices_tensor = _select_text_features_for_classes(text_features, class_indices)
    logits = torch.einsum("chw,kc->khw", decoded_feature_map, text_features)
    pred_indices = logits.argmax(dim=0).long()
    return _map_local_prediction_indices(pred_indices, class_indices_tensor).long()


def _render_semantic_feature_map_for_view(view, gaussians, pipeline, background):
    voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
    render_pkg = render(
        view,
        gaussians,
        pipeline,
        background,
        visible_mask=voxel_visible_mask,
        render_semantic_features=True,
        semantic_only=True,
        render_alpha=True,
    )
    semantic_latent = render_pkg.get("semantic", None)
    if semantic_latent is None or semantic_latent.numel() == 0:
        raise RuntimeError(f"Semantic render returned an empty latent map for view {view.image_name}.")
    semantic_weight = render_pkg.get("semantic_weight", None)
    if semantic_weight is None or semantic_weight.numel() == 0:
        semantic_weight = render_pkg.get("alpha", None)
    if semantic_weight is not None and semantic_weight.numel() > 0:
        semantic_latent = semantic_latent / semantic_weight.clamp_min(1e-6)
    semantic_feature_map = gaussians.decode_semantic_feature_map(semantic_latent)
    alpha = render_pkg.get("alpha", None)
    return semantic_feature_map, alpha


def _evaluate_open_vocab_2d(
    scene,
    gaussians,
    pipeline,
    dataset,
    text_features,
    prompt_bank,
    gt_to_index=None,
    ignore_values=None,
    logger=None,
):
    gt_dir = _to_abs_path(getattr(dataset, "semantic_gt_2d", ""), dataset.source_path)
    if not gt_dir:
        return None, {}
    if not gt_dir.exists():
        raise RuntimeError(f"2D semantic GT directory does not exist: {gt_dir}")

    split = getattr(dataset, "semantic_eval_split", "test").lower()
    if split == "train":
        cameras = scene.getTrainCameras()
    else:
        cameras = scene.getTestCameras()
    if not cameras:
        raise RuntimeError(f"No cameras found for semantic 2D evaluation split '{split}'.")

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=gaussians.get_anchor.device,
    )
    text_features = text_features.to(background.device)
    if gt_to_index is None or ignore_values is None:
        gt_to_index, ignore_values, _ = _build_gt_label_mapper(dataset, prompt_bank, logger=logger)
    exclude_indices, exclude_names = _resolve_semantic_eval_exclude_indices(dataset, prompt_bank, logger=logger)
    eval_class_indices = _semantic_eval_class_indices(len(prompt_bank["class_ids"]), exclude_indices)

    confusion = torch.zeros((len(prompt_bank["class_ids"]), len(prompt_bank["class_ids"])), dtype=torch.long)
    per_view_metrics = {}
    available_views = 0

    for view in cameras:
        label_path = _find_semantic_2d_label_path(gt_dir, view.image_name)
        if label_path is None:
            continue

        label_map = _load_semantic_label_2d(label_path)
        decoded_feature_map, _ = _render_semantic_feature_map_for_view(view, gaussians, pipeline, background)
        label_map = _resize_label_map(label_map, decoded_feature_map.shape[-2:])
        gt_indices = _map_gt_labels_to_indices(label_map, gt_to_index, ignore_values).to(decoded_feature_map.device)
        gt_indices = _apply_semantic_eval_exclude_indices(gt_indices, exclude_indices)
        valid_mask = gt_indices >= 0
        if not bool(valid_mask.any().item()):
            continue

        pred_indices = _predict_open_vocab_labels(decoded_feature_map, text_features)
        view_confusion = torch.zeros_like(confusion, device=decoded_feature_map.device)
        _update_confusion_matrix(
            view_confusion,
            gt_indices[valid_mask].view(-1),
            pred_indices[valid_mask].view(-1),
            confusion.shape[0],
        )
        confusion += view_confusion.cpu()
        per_view_metrics[view.image_name] = _compute_metrics_from_confusion(
            view_confusion.cpu(),
            prompt_bank["class_names"],
            eval_class_indices=eval_class_indices,
        )
        available_views += 1

    if available_views == 0:
        raise RuntimeError(f"No matching 2D semantic GT files were found in {gt_dir} for split '{split}'.")

    summary = _compute_metrics_from_confusion(
        confusion,
        prompt_bank["class_names"],
        eval_class_indices=eval_class_indices,
    )
    summary["split"] = split
    summary["numViews"] = available_views
    summary["excludedClasses"] = exclude_names
    _log(
        logger,
        f"[Semantic-2D] split={split} views={available_views} "
        f"mIoU={summary['mIoU']:.4f} mAcc={summary['mAcc']:.4f} overallAcc={summary['overallAcc']:.4f}",
    )
    return summary, per_view_metrics


def _resolve_semantic_gt_info_path(gt_path, gt_info_path=None):
    if gt_info_path:
        info_path = Path(gt_info_path)
        if not info_path.exists():
            raise RuntimeError(f"Semantic GT info JSON does not exist: {info_path}")
        return info_path

    for name in ("info_semantic.json", "info_preseg_semantic.json"):
        candidate = Path(gt_path).with_name(name)
        if candidate.exists():
            return candidate
    return None


def _load_replica_object_to_class(info_path):
    data = json.loads(Path(info_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Replica semantic info JSON must be an object: {info_path}")

    mapping = {}
    objects = data.get("objects", None)
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict) or "id" not in obj or "class_id" not in obj:
                continue
            mapping[int(obj["id"])] = int(obj["class_id"])

    if not mapping and isinstance(data.get("id_to_label", None), list):
        for object_id, class_id in enumerate(data["id_to_label"]):
            mapping[int(object_id)] = int(class_id)

    if not mapping:
        raise RuntimeError(
            f"Replica semantic info JSON must contain objects[].id/class_id or id_to_label: {info_path}"
        )
    return mapping


def _face_object_ids_to_vertex_class_labels(face_data, vertex_count, info_path):
    object_to_class = _load_replica_object_to_class(info_path)
    face_names = set(face_data.dtype.names or [])
    if "object_id" not in face_names:
        raise RuntimeError("Replica semantic PLY face element has no object_id property.")

    face_object_ids = np.asarray(face_data["object_id"], dtype=np.int64)
    face_class_ids = np.asarray(
        [object_to_class.get(int(object_id), -1) for object_id in face_object_ids],
        dtype=np.int64,
    )
    valid_faces = face_class_ids >= 0
    max_class_id = int(face_class_ids[valid_faces].max()) if np.any(valid_faces) else 0
    counts = np.zeros((int(vertex_count), max_class_id + 1), dtype=np.uint16)

    try:
        face_vertices = np.vstack(face_data["vertex_indices"]).astype(np.int64)
        repeated_vertex_ids = face_vertices[valid_faces].reshape(-1)
        repeated_class_ids = np.repeat(face_class_ids[valid_faces], face_vertices.shape[1])
        np.add.at(counts, (repeated_vertex_ids, repeated_class_ids), 1)
    except Exception:
        for vertex_ids, class_id in zip(face_data["vertex_indices"], face_class_ids):
            if class_id < 0:
                continue
            for vertex_id in vertex_ids:
                counts[int(vertex_id), int(class_id)] += 1

    labels = counts.argmax(axis=1).astype(np.int64)
    labels[counts.max(axis=1) == 0] = -1
    return labels


def _load_semantic_gt_3d(gt_path, gt_info_path=None):
    gt_path = Path(gt_path)
    suffix = gt_path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(gt_path)
        points = payload["points"] if "points" in payload else None
        labels = payload["labels"] if "labels" in payload else payload[payload.files[-1]]
    elif suffix == ".pt":
        payload = torch.load(gt_path, map_location="cpu")
        if isinstance(payload, dict):
            points = payload.get("points", None)
            labels = payload.get("labels", None)
            if labels is None:
                raise RuntimeError(f"3D semantic .pt payload missing 'labels': {gt_path}")
        else:
            points = None
            labels = payload
    elif suffix == ".ply":
        plydata = PlyData.read(str(gt_path))
        vertices = plydata["vertex"]
        points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
        for label_key in ("label", "semantic_gt", "class_id", "category_id"):
            if label_key in vertices.data.dtype.names:
                labels = np.asarray(vertices[label_key]).astype(np.int64)
                break
        else:
            if "face" not in plydata:
                raise RuntimeError(f"3D semantic PLY file does not contain a supported label field: {gt_path}")
            face_data = plydata["face"].data
            face_names = set(face_data.dtype.names or [])
            if "object_id" not in face_names:
                raise RuntimeError(f"3D semantic PLY file does not contain a supported label field: {gt_path}")
            info_path = _resolve_semantic_gt_info_path(gt_path, gt_info_path)
            if info_path is None:
                raise RuntimeError(
                    "3D semantic PLY has face.object_id but no vertex labels. "
                    "Pass --semantic_gt_info or place info_semantic.json next to the mesh."
                )
            labels = _face_object_ids_to_vertex_class_labels(face_data, len(vertices), info_path)
    elif suffix == ".npy":
        payload = np.load(gt_path)
        if payload.ndim == 1:
            points = None
            labels = payload
        elif payload.ndim == 2 and payload.shape[1] >= 4:
            points = payload[:, :3]
            labels = payload[:, 3]
        else:
            raise RuntimeError(f"Unsupported 3D semantic .npy layout for {gt_path}")
    else:
        raise RuntimeError(f"Unsupported 3D semantic GT extension: {gt_path}")

    points_tensor = None if points is None else torch.as_tensor(points, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long).view(-1)
    return points_tensor, labels_tensor


def _predict_anchor_class_indices(gaussians, text_features, topk=1, class_indices=None):
    anchor_features = gaussians.decode_semantic_features()
    if anchor_features is None or anchor_features.numel() == 0:
        raise RuntimeError("Semantic decoder is unavailable or produced empty anchor features.")
    text_features, class_indices_tensor = _select_text_features_for_classes(text_features, class_indices)
    logits = anchor_features @ text_features.T
    return _topk_from_logits(logits, topk, class_indices_tensor=class_indices_tensor)


def _topk_from_logits(logits, topk, class_indices_tensor=None):
    if int(topk) <= 1:
        pred_indices = logits.argmax(dim=1).long()
        return _map_local_prediction_indices(pred_indices, class_indices_tensor).long()
    k = min(int(topk), int(logits.shape[1]))
    topk_indices = logits.topk(k=k, dim=1).indices.long()
    if k < int(topk):
        pad = topk_indices[:, -1:].expand(-1, int(topk) - k)
        topk_indices = torch.cat([topk_indices, pad], dim=1)
    return _map_local_prediction_indices(topk_indices, class_indices_tensor).long()


def _build_child_xyz_and_latents(gaussians):
    anchor = gaussians.get_anchor
    offsets = gaussians._offset
    scaling = gaussians.get_scaling
    if anchor is None or anchor.numel() == 0:
        raise RuntimeError("3D semantic child evaluation has no anchors.")
    if offsets is None or offsets.numel() == 0:
        raise RuntimeError("3D semantic child evaluation has no child offsets.")
    if offsets.ndim != 3 or offsets.shape[0] != anchor.shape[0]:
        raise RuntimeError(
            "Child offsets are not aligned with anchors: "
            f"anchor={tuple(anchor.shape)}, offset={tuple(offsets.shape)}."
        )

    num_anchors, num_offsets, _ = offsets.shape
    anchor_repeat = anchor[:, None, :].expand(num_anchors, num_offsets, 3).reshape(-1, 3)
    scaling_repeat = scaling[:, None, :].expand(num_anchors, num_offsets, scaling.shape[-1]).reshape(-1, scaling.shape[-1])
    child_xyz = anchor_repeat + offsets.reshape(-1, 3) * scaling_repeat[:, :3]

    child_latents = gaussians.get_semantic_child_latents()
    if child_latents is None or child_latents.numel() == 0:
        raise RuntimeError("Semantic child latents are unavailable for child-level 3D evaluation.")
    expected_shape = (num_anchors, num_offsets, gaussians.semantic_latent_dim)
    if tuple(child_latents.shape) != expected_shape:
        raise RuntimeError(
            "Semantic child latent shape is not aligned with child xyz order: "
            f"expected {expected_shape}, got {tuple(child_latents.shape)}."
        )
    child_latents = child_latents.reshape(-1, gaussians.semantic_latent_dim)
    if child_latents.shape[0] != child_xyz.shape[0]:
        raise RuntimeError(
            "Child xyz and semantic latents are misaligned: "
            f"xyz={tuple(child_xyz.shape)}, latent={tuple(child_latents.shape)}."
        )
    return child_xyz, child_latents


def _predict_child_class_indices(
    gaussians,
    text_features,
    topk=3,
    batch_size=65536,
    child_xyz=None,
    child_latents=None,
    class_indices=None,
):
    if child_xyz is None or child_latents is None:
        child_xyz, child_latents = _build_child_xyz_and_latents(gaussians)
    text_features = F.normalize(text_features.to(child_latents.device), dim=-1, eps=1e-6)
    text_features, class_indices_tensor = _select_text_features_for_classes(text_features, class_indices)
    predictions = []
    for start in range(0, child_latents.shape[0], int(batch_size)):
        end = min(start + int(batch_size), child_latents.shape[0])
        child_features = gaussians.decode_semantic_features(child_latents[start:end])
        if child_features is None or child_features.numel() == 0:
            raise RuntimeError("Semantic decoder produced empty child features.")
        child_features = F.normalize(child_features, dim=-1, eps=1e-6)
        logits = child_features @ text_features.T
        predictions.append(_topk_from_logits(logits, topk, class_indices_tensor=class_indices_tensor).cpu())
    return child_xyz.detach().cpu(), torch.cat(predictions, dim=0)


def _assign_nearest_labels(reference_xyz, reference_pred, query_points, logger=None, reference_name="anchors"):
    num_references = reference_xyz.shape[0]
    num_points = query_points.shape[0]
    if num_references == 0:
        raise RuntimeError(f"3D semantic benchmark has no {reference_name} for nearest assignment.")
    if num_points == 0:
        return torch.empty((0,), dtype=reference_pred.dtype)

    reference_xyz = reference_xyz.detach().cpu().float().contiguous()
    reference_pred = reference_pred.detach().cpu().long().contiguous()
    query_points = query_points.detach().cpu().float().contiguous()

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        _log(
            logger,
            "[Semantic-3D] scipy.spatial.cKDTree is unavailable "
            f"({exc}); falling back to chunked torch.cdist.",
        )
    else:
        _log(
            logger,
            f"[Semantic-3D] Assigning {num_points} GT points to {num_references} {reference_name} with cKDTree.",
        )
        tree = cKDTree(reference_xyz.numpy())
        query_np = query_points.numpy()
        assigned = []
        for start in range(0, num_points, _MAX_3D_KDTREE_QUERY_CHUNK):
            end = min(start + _MAX_3D_KDTREE_QUERY_CHUNK, num_points)
            try:
                _, nearest_np = tree.query(query_np[start:end], k=1, workers=-1)
            except TypeError:
                _, nearest_np = tree.query(query_np[start:end], k=1)
            nearest = torch.as_tensor(nearest_np, dtype=torch.long)
            assigned.append(reference_pred[nearest])
        return torch.cat(assigned, dim=0)

    query_chunk_size = min(_MAX_3D_QUERY_CHUNK, max(1, _MAX_3D_BRUTEFORCE_PAIRS // max(num_references, 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(
        logger,
        f"[Semantic-3D] Assigning {num_points} GT points to {num_references} {reference_name} with "
        f"chunked torch.cdist on {device}; query_chunk_size={query_chunk_size}.",
    )
    reference_xyz = reference_xyz.to(device, non_blocking=True)
    reference_pred_device = reference_pred.to(device, non_blocking=True)
    assigned = []
    for start in range(0, num_points, query_chunk_size):
        query_chunk = query_points[start:start + query_chunk_size].to(device, non_blocking=True)
        distances = torch.cdist(query_chunk, reference_xyz)
        nearest = distances.argmin(dim=1)
        assigned.append(reference_pred_device[nearest].cpu())
    return torch.cat(assigned, dim=0)


def _assign_nearest_anchor_labels(anchor_xyz, anchor_pred, query_points, logger=None):
    return _assign_nearest_labels(anchor_xyz, anchor_pred, query_points, logger=logger, reference_name="anchors")


def _assign_child_knn_labels(
    child_xyz,
    child_latents,
    gaussians,
    text_features,
    query_points,
    k=8,
    weight_mode="inverse_distance",
    logger=None,
    class_indices=None,
):
    num_children = int(child_xyz.shape[0])
    num_points = int(query_points.shape[0])
    if num_children == 0:
        raise RuntimeError("3D semantic child-kNN evaluation has no child Gaussians.")
    if num_points == 0:
        return torch.empty((0, 3), dtype=torch.long)
    k = max(1, min(int(k), num_children))
    weight_mode = str(weight_mode or "inverse_distance").lower()
    if weight_mode not in {"inverse_distance", "gaussian_rbf"}:
        raise RuntimeError(
            f"Unsupported --semantic_3d_child_knn_weight '{weight_mode}'. "
            "Expected inverse_distance or gaussian_rbf."
        )

    child_xyz_cpu = child_xyz.detach().cpu().float().contiguous()
    query_points_cpu = query_points.detach().cpu().float().contiguous()
    text_features = F.normalize(text_features.to(child_latents.device), dim=-1, eps=1e-6)
    text_features, class_indices_tensor = _select_text_features_for_classes(text_features, class_indices)
    assigned = []

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        _log(
            logger,
            "[Semantic-3D] scipy.spatial.cKDTree is unavailable "
            f"({exc}); falling back to chunked child-kNN torch.cdist.",
        )
        query_chunk_size = min(_MAX_3D_QUERY_CHUNK, max(1, _MAX_3D_BRUTEFORCE_PAIRS // max(num_children, 1)))
        child_xyz_device = child_xyz.to(child_latents.device, non_blocking=True)
        for start in range(0, num_points, query_chunk_size):
            end = min(start + query_chunk_size, num_points)
            query_chunk = query_points_cpu[start:end].to(child_latents.device, non_blocking=True)
            distances, indices = torch.cdist(query_chunk, child_xyz_device).topk(k=k, dim=1, largest=False)
            assigned.append(
                _predict_child_knn_chunk(
                    gaussians,
                    child_latents,
                    text_features,
                    indices,
                    distances,
                    weight_mode,
                    class_indices_tensor=class_indices_tensor,
                ).cpu()
            )
        return torch.cat(assigned, dim=0)

    _log(
        logger,
        f"[Semantic-3D] Assigning {num_points} GT points to {num_children} child Gaussians "
        f"with cKDTree k={k}, weight={weight_mode}.",
    )
    tree = cKDTree(child_xyz_cpu.numpy())
    query_np = query_points_cpu.numpy()
    for start in range(0, num_points, _MAX_3D_KDTREE_QUERY_CHUNK):
        end = min(start + _MAX_3D_KDTREE_QUERY_CHUNK, num_points)
        try:
            distances_np, indices_np = tree.query(query_np[start:end], k=k, workers=-1)
        except TypeError:
            distances_np, indices_np = tree.query(query_np[start:end], k=k)
        distances = torch.as_tensor(distances_np, dtype=torch.float32)
        indices = torch.as_tensor(indices_np, dtype=torch.long)
        if indices.ndim == 1:
            indices = indices[:, None]
            distances = distances[:, None]
        for sub_start in range(0, indices.shape[0], 8192):
            sub_end = min(sub_start + 8192, indices.shape[0])
            assigned.append(
                _predict_child_knn_chunk(
                    gaussians,
                    child_latents,
                    text_features,
                    indices[sub_start:sub_end].to(child_latents.device, non_blocking=True),
                    distances[sub_start:sub_end].to(child_latents.device, non_blocking=True),
                    weight_mode,
                    class_indices_tensor=class_indices_tensor,
                ).cpu()
            )
    return torch.cat(assigned, dim=0)


def _predict_child_knn_chunk(gaussians, child_latents, text_features, indices, distances, weight_mode, class_indices_tensor=None):
    flat_indices = indices.reshape(-1)
    neighbor_latents = child_latents[flat_indices].reshape(indices.shape[0], indices.shape[1], -1)
    neighbor_features = gaussians.decode_semantic_features(neighbor_latents.reshape(-1, neighbor_latents.shape[-1]))
    if neighbor_features is None or neighbor_features.numel() == 0:
        raise RuntimeError("Semantic decoder produced empty child features for child-kNN evaluation.")
    neighbor_features = F.normalize(neighbor_features, dim=-1, eps=1e-6).reshape(
        indices.shape[0],
        indices.shape[1],
        -1,
    )
    if weight_mode == "inverse_distance":
        weights = 1.0 / distances.clamp_min(1e-6)
    else:
        sigma = distances[:, -1:].clamp_min(1e-6)
        weights = torch.exp(-(distances * distances) / (2.0 * sigma * sigma))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    vertex_features = F.normalize((neighbor_features * weights[..., None]).sum(dim=1), dim=-1, eps=1e-6)
    logits = vertex_features @ text_features.T
    return _topk_from_logits(logits, 3, class_indices_tensor=class_indices_tensor)


def _evaluate_open_vocab_3d(
    gaussians,
    dataset,
    text_features,
    prompt_bank,
    gt_to_index=None,
    ignore_values=None,
    gt_mapping_info=None,
    logger=None,
):
    gt_path = _to_abs_path(getattr(dataset, "semantic_gt_3d", ""), dataset.source_path)
    if not gt_path:
        return None
    if not gt_path.exists():
        raise RuntimeError(f"3D semantic GT path does not exist: {gt_path}")

    gt_info_path = _to_abs_path(getattr(dataset, "semantic_gt_info", ""), dataset.source_path)
    points, labels = _load_semantic_gt_3d(gt_path, gt_info_path)
    eval_mode = str(getattr(dataset, "semantic_3d_eval_mode", "anchor") or "anchor").lower()
    if eval_mode not in {"anchor", "child_nearest", "child_knn"}:
        raise RuntimeError(
            f"Unsupported --semantic_3d_eval_mode '{eval_mode}'. "
            "Expected anchor, child_nearest, or child_knn."
        )
    if gt_to_index is None or ignore_values is None:
        gt_to_index, ignore_values, gt_mapping_info = _build_gt_label_mapper(dataset, prompt_bank, logger=logger)
    exclude_indices, exclude_names = _resolve_semantic_eval_exclude_indices(dataset, prompt_bank, logger=logger)
    gt_indices = _map_gt_labels_to_indices(labels, gt_to_index, ignore_values)
    gt_indices = _apply_semantic_eval_exclude_indices(gt_indices, exclude_indices)
    valid_mask = gt_indices >= 0
    if not bool(valid_mask.any().item()):
        raise RuntimeError(f"3D semantic GT file contains no valid labels for the configured prompt bank: {gt_path}")
    eval_class_indices, mesh_classes_only = _semantic_eval_prediction_class_indices(
        dataset,
        gt_indices,
        len(prompt_bank["class_ids"]),
        exclude_indices,
        prompt_bank,
        logger=logger,
    )

    text_features_device = text_features.to(gaussians.get_anchor.device)
    if eval_mode == "anchor":
        anchor_top3 = _predict_anchor_class_indices(
            gaussians,
            text_features_device,
            topk=3,
            class_indices=eval_class_indices if mesh_classes_only else None,
        )
        anchor_top3 = anchor_top3.cpu()
        anchor_pred = anchor_top3[:, 0].cpu()
        assignment_name = "nearest-anchor"
        if points is None:
            if labels.numel() != anchor_pred.numel():
                raise RuntimeError(
                    "3D semantic labels are not aligned with anchors. "
                    "Provide a points+labels file or a label array with the same length as the anchor count."
                )
            pred_indices = anchor_pred
            pred_top3_indices = anchor_top3
        else:
            pred_indices = _assign_nearest_anchor_labels(
                gaussians.get_anchor.detach().cpu(),
                anchor_pred,
                points.cpu(),
                logger=logger,
            )
            pred_top3_indices = _assign_nearest_anchor_labels(
                gaussians.get_anchor.detach().cpu(),
                anchor_top3,
                points.cpu(),
                logger=None,
            )
    else:
        if points is None:
            raise RuntimeError(
                f"--semantic_3d_eval_mode {eval_mode} requires 3D GT points. "
                "Label arrays aligned only to anchors are supported only by anchor mode."
            )
        child_xyz, child_latents = _build_child_xyz_and_latents(gaussians)
        if eval_mode == "child_nearest":
            child_xyz_cpu, child_top3 = _predict_child_class_indices(
                gaussians,
                text_features_device,
                topk=3,
                child_xyz=child_xyz,
                child_latents=child_latents,
                class_indices=eval_class_indices if mesh_classes_only else None,
            )
            child_pred = child_top3[:, 0].cpu()
            pred_indices = _assign_nearest_labels(
                child_xyz_cpu,
                child_pred,
                points.cpu(),
                logger=logger,
                reference_name="child Gaussians",
            )
            pred_top3_indices = _assign_nearest_labels(
                child_xyz_cpu,
                child_top3,
                points.cpu(),
                logger=None,
                reference_name="child Gaussians",
            )
            assignment_name = "nearest-child"
        else:
            pred_top3_indices = _assign_child_knn_labels(
                child_xyz,
                child_latents,
                gaussians,
                text_features_device,
                points.cpu(),
                k=int(getattr(dataset, "semantic_3d_child_knn", 8)),
                weight_mode=getattr(dataset, "semantic_3d_child_knn_weight", "inverse_distance"),
                logger=logger,
                class_indices=eval_class_indices if mesh_classes_only else None,
            )
            pred_indices = pred_top3_indices[:, 0]
            assignment_name = "child-knn"

    valid_gt = gt_indices[valid_mask].view(-1)
    valid_pred = pred_indices[valid_mask].view(-1)
    valid_pred_top3 = pred_top3_indices[valid_mask].view(-1, pred_top3_indices.shape[-1])
    correct = int((valid_gt == valid_pred).sum().item())
    top3_correct = int((valid_pred_top3 == valid_gt.unsqueeze(1)).any(dim=1).sum().item())
    _log(
        logger,
        "[Semantic-3D] GT raw top: "
        f"{_format_top_counts(labels[valid_mask], class_names=None)}",
    )
    _log(
        logger,
        "[Semantic-3D] GT mapped top: "
        f"{_format_top_counts(valid_gt, class_names=prompt_bank['class_names'])}",
    )
    _log(
        logger,
        "[Semantic-3D] Pred mapped top: "
        f"{_format_top_counts(valid_pred, class_names=prompt_bank['class_names'])}",
    )
    _log(
        logger,
        f"[Semantic-3D] Correct {assignment_name} labels: {correct}/{int(valid_gt.numel())}",
    )
    _log(
        logger,
        f"[Semantic-3D] Top-3 {assignment_name} hits: {top3_correct}/{int(valid_gt.numel())}",
    )

    confusion = torch.zeros((len(prompt_bank["class_ids"]), len(prompt_bank["class_ids"])), dtype=torch.long)
    _update_confusion_matrix(
        confusion,
        valid_gt,
        valid_pred,
        confusion.shape[0],
    )
    summary = _compute_metrics_from_confusion(
        confusion,
        prompt_bank["class_names"],
        eval_class_indices=eval_class_indices,
    )
    summary["numPoints"] = int(valid_mask.sum().item())
    summary["top1"] = {
        "mIoU": summary["mIoU"],
        "mAcc": summary["mAcc"],
        "overallAcc": summary["overallAcc"],
        "matched": int(correct),
    }
    summary["top3"] = _compute_topk_metrics(
        valid_gt,
        valid_pred_top3,
        len(prompt_bank["class_ids"]),
        prompt_bank["class_names"],
        eval_class_indices=eval_class_indices,
    )
    summary["top1_mIoU"] = summary["top1"]["mIoU"]
    summary["top1_mAcc"] = summary["top1"]["mAcc"]
    summary["top1_overallAcc"] = summary["top1"]["overallAcc"]
    summary["top3_mIoU"] = summary["top3"]["mIoU"]
    summary["top3_mAcc"] = summary["top3"]["mAcc"]
    summary["top3_overallAcc"] = summary["top3"]["overallAcc"]
    summary["evalMode"] = eval_mode
    summary["excludedClasses"] = exclude_names
    summary["meshClassesOnly"] = bool(mesh_classes_only)
    if mesh_classes_only:
        summary["predictionClasses"] = [prompt_bank["class_names"][idx] for idx in eval_class_indices]
    if gt_mapping_info is not None:
        summary["gtMapping"] = gt_mapping_info
    _log(
        logger,
        f"[Semantic-3D] points={summary['numPoints']} "
        f"mIoU={summary['mIoU']:.4f} mAcc={summary['mAcc']:.4f} overallAcc={summary['overallAcc']:.4f}",
    )
    _log(
        logger,
        f"[Semantic-3D][Top-3] mIoU={summary['top3']['mIoU']:.4f} "
        f"mAcc={summary['top3']['mAcc']:.4f} overallAcc={summary['top3']['overallAcc']:.4f}",
    )
    return summary


def run_semantic_benchmark_from_loaded_scene(dataset, scene, gaussians, pipeline, logger=None, wandb=None):
    if not semantic_benchmark_enabled(dataset):
        return None
    if not getattr(dataset, "use_semantic_features", False):
        _log(logger, "[Semantic-Eval] Skipped because semantic features are disabled.")
        return None

    prompt_path = _to_abs_path(
        getattr(dataset, "semantic_eval_prompt_path", ""),
        dataset.source_path,
    )
    if prompt_path is None:
        raise RuntimeError("Semantic evaluation needs --semantic_eval_prompt_path.")

    prompt_bank = load_semantic_prompt_bank(prompt_path)
    text_device = str(gaussians.get_anchor.device)
    text_features = _encode_open_vocab_text_features(
        prompt_bank["prompts"],
        getattr(dataset, "teacher_clip_model", "ViT-B/32"),
        getattr(dataset, "semantic_feature_dim", 512),
        device=text_device,
    )
    gt_to_index, ignore_values, gt_mapping_info = _build_gt_label_mapper(dataset, prompt_bank, logger=logger)

    results = {}
    per_view = {}

    summary_2d, per_view_2d = _evaluate_open_vocab_2d(
        scene,
        gaussians,
        pipeline,
        dataset,
        text_features,
        prompt_bank,
        gt_to_index=gt_to_index,
        ignore_values=ignore_values,
        logger=logger,
    )
    if summary_2d is not None:
        results["2d"] = summary_2d
        per_view["2d"] = per_view_2d
        if wandb is not None:
            wandb.log(
                {
                    "semantic_2d_mIoU": summary_2d["mIoU"],
                    "semantic_2d_mAcc": summary_2d["mAcc"],
                    "semantic_2d_overallAcc": summary_2d["overallAcc"],
                }
            )

    summary_3d = _evaluate_open_vocab_3d(
        gaussians,
        dataset,
        text_features,
        prompt_bank,
        gt_to_index=gt_to_index,
        ignore_values=ignore_values,
        gt_mapping_info=gt_mapping_info,
        logger=logger,
    )
    if summary_3d is not None:
        results["3d"] = summary_3d
        if wandb is not None:
            wandb.log(
                {
                    "semantic_3d_mIoU": summary_3d["mIoU"],
                    "semantic_3d_mAcc": summary_3d["mAcc"],
                    "semantic_3d_overallAcc": summary_3d["overallAcc"],
                    "semantic_3d_top3_mIoU": summary_3d["top3"]["mIoU"],
                    "semantic_3d_top3_mAcc": summary_3d["top3"]["mAcc"],
                    "semantic_3d_top3_overallAcc": summary_3d["top3"]["overallAcc"],
                }
            )

    output_root = Path(dataset.model_path)
    with open(output_root / "semantic_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    with open(output_root / "semantic_per_view.json", "w", encoding="utf-8") as fp:
        json.dump(per_view, fp, indent=2, ensure_ascii=False)

    return results


def run_semantic_benchmark_for_saved_model(dataset, iteration, pipeline, logger=None, wandb=None):
    with torch.no_grad():
        dataset.depths = ""
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
            dataset.use_semantic_features,
            dataset.semantic_latent_dim,
            dataset.semantic_feature_dim,
            semantic_teacher_dim=getattr(dataset, "semantic_teacher_dim", 0),
            semantic_codec_path=getattr(dataset, "semantic_codec_path", ""),
            sh_degree=dataset.sh_degree,
            use_anchor_offset_semantic=getattr(dataset, "use_anchor_offset_semantic", False),
            semantic_residual_dim=getattr(dataset, "semantic_residual_dim", 4),
            semantic_gate_init=getattr(dataset, "semantic_gate_init", -4.0),
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        return run_semantic_benchmark_from_loaded_scene(dataset, scene, gaussians, pipeline, logger=logger, wandb=wandb)
