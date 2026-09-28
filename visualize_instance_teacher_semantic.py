#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import csv
import hashlib
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils.instance_teacher_adapter import InstanceTeacherStore
from utils.semantic_benchmark_utils import (
    _apply_semantic_eval_exclude_indices,
    _build_gt_label_mapper,
    _load_semantic_gt_3d,
    _map_gt_labels_to_indices,
    _resolve_semantic_eval_exclude_indices,
    _semantic_eval_class_indices,
    _semantic_eval_prediction_class_indices,
    load_semantic_prompt_bank,
)
from utils.semantic_codec import load_semantic_codec


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
_MAP_EXTENSIONS = (".npy", ".npz")
_PROMPT_TEMPLATES = (
    "{label}",
    "a photo of a {label}.",
    "an indoor photo of a {label}.",
    "a photo of the {label} in a room.",
)
_LABEL_FIELD_CANDIDATES = (
    "label",
    "semantic_label",
    "semantic_id",
    "semantic",
    "class",
    "class_id",
    "category_id",
    "nyu40id",
)
_BASE_COLORS = np.asarray(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
    ],
    dtype=np.uint8,
)


def _make_palette(num_classes):
    palette = np.zeros((int(num_classes), 3), dtype=np.uint8)
    for idx in range(int(num_classes)):
        if idx < len(_BASE_COLORS):
            palette[idx] = _BASE_COLORS[idx]
        else:
            palette[idx, 0] = (37 * idx + 23) % 256
            palette[idx, 1] = (17 * idx + 119) % 256
            palette[idx, 2] = (97 * idx + 53) % 256
    return palette


def _stable_color(name):
    digest = hashlib.md5(f"maskclustering:{name}".encode("utf-8")).digest()
    return np.asarray([48 + digest[0] % 176, 48 + digest[1] % 176, 48 + digest[2] % 176], dtype=np.uint8)


def _stable_palette(class_names):
    return np.stack([_stable_color(str(name)) for name in class_names], axis=0).astype(np.uint8)


def _instance_palette(num_instances):
    palette = np.zeros((int(num_instances) + 1, 3), dtype=np.uint8)
    for idx in range(1, int(num_instances) + 1):
        palette[idx, 0] = (53 * idx + 41) % 256
        palette[idx, 1] = (97 * idx + 13) % 256
        palette[idx, 2] = (29 * idx + 179) % 256
    return palette


def _safe_stem(frame_id):
    stem = Path(str(frame_id)).stem
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)


def _as_abs_path(path_value, base=None):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute() or base is None:
        return path
    return Path(base) / path


def _find_image(source_path, images_dir, frame_id):
    image_root = _as_abs_path(images_dir, source_path)
    if image_root is None:
        return None
    frame_path = Path(str(frame_id))
    candidates = []
    if frame_path.suffix:
        candidates.append(image_root / frame_path.name)
    stem = frame_path.stem
    candidates.extend(image_root / f"{stem}{ext}" for ext in _IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_rgb(source_path, images_dir, frame_id, width, height):
    image_path = _find_image(source_path, images_dir, frame_id)
    if image_path is None:
        return np.zeros((height, width, 3), dtype=np.uint8), ""
    rgb = Image.open(image_path).convert("RGB")
    if rgb.size != (width, height):
        rgb = rgb.resize((width, height), Image.BILINEAR)
    return np.asarray(rgb), str(image_path)


def _load_array(path):
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        if "data" in payload:
            return np.asarray(payload["data"])
        keys = list(payload.keys())
        if not keys:
            raise RuntimeError(f"Empty npz file: {path}")
        return np.asarray(payload[keys[0]])
    return np.asarray(np.load(path))


def _squeeze_hw(array, path, field_name):
    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"{field_name} must have shape [H,W] or [1,H,W], got {array.shape}: {path}")
    return array


def _find_frame_array(frame_dir, frame_id):
    frame_dir = Path(frame_dir)
    stem = Path(str(frame_id)).stem
    for extension in _MAP_EXTENSIONS:
        candidate = frame_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _blend_overlay(rgb, color, alpha):
    alpha = alpha[..., None].clip(0.0, 1.0)
    blended = rgb.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(blended + 0.5, 0, 255).astype(np.uint8)


def _mask_boundary(mask):
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary & mask


def _text_color_for_background(rgb):
    r, g, b = [float(value) for value in rgb]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 150.0 else (255, 255, 255)


def _load_font(font_path, font_size):
    font_size = max(8, int(font_size))
    if font_path:
        try:
            return ImageFont.truetype(str(font_path), font_size)
        except OSError:
            pass
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_bbox(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except AttributeError:
        return draw.textsize(text, font=font)


def _rects_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place_label(width, height, anchor_x, anchor_y, text_width, text_height, placed, pad):
    offsets = [
        (0, 0),
        (8, 8),
        (-8, 8),
        (8, -8),
        (-8, -8),
        (16, 0),
        (-16, 0),
        (0, 16),
        (0, -16),
        (24, 12),
        (-24, 12),
    ]
    box_width = text_width + 2 * pad
    box_height = text_height + 2 * pad
    for dx, dy in offsets:
        x0 = int(round(anchor_x + dx - box_width / 2))
        y0 = int(round(anchor_y + dy - box_height / 2))
        x0 = max(0, min(x0, max(0, width - box_width)))
        y0 = max(0, min(y0, max(0, height - box_height)))
        rect = (x0, y0, x0 + box_width, y0 + box_height)
        if not any(_rects_overlap(rect, old) for old in placed):
            return rect
    x0 = int(round(anchor_x - box_width / 2))
    y0 = int(round(anchor_y - box_height / 2))
    x0 = max(0, min(x0, max(0, width - box_width)))
    y0 = max(0, min(y0, max(0, height - box_height)))
    return x0, y0, x0 + box_width, y0 + box_height


def _collect_label_candidates(
    instance_map,
    visible,
    instance_pred_classes,
    instance_confidence,
    class_names,
    min_area,
    include_confidence,
):
    visible_ids, counts = np.unique(instance_map[visible], return_counts=True)
    order = np.argsort(-counts)
    candidates = []
    for order_idx in order:
        instance_index = int(visible_ids[order_idx])
        area = int(counts[order_idx])
        if instance_index < 0 or area < int(min_area):
            continue
        mask = visible & (instance_map == instance_index)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        class_index = int(instance_pred_classes[instance_index])
        class_name = str(class_names[class_index])
        confidence = float(instance_confidence[instance_index])
        text = f"{class_name} {confidence:.2f}" if include_confidence else class_name
        candidates.append(
            {
                "instance_index": instance_index,
                "class_index": class_index,
                "class_name": class_name,
                "confidence": confidence,
                "area": area,
                "x": float(xs.min()),
                "y": float(ys.min()),
            }
            | {"text": text}
        )
    return candidates


def _draw_labels(image, candidates, palette, font, max_labels):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    placed = []
    pad = 4
    drawn = 0
    for candidate in candidates[: max(0, int(max_labels))]:
        text = candidate["text"]
        text_w, text_h = _text_bbox(draw, text, font)
        rect = _place_label(width, height, candidate["x"], candidate["y"], text_w, text_h, placed, pad)
        color = tuple(int(v) for v in palette[int(candidate["class_index"])].tolist())
        text_color = _text_color_for_background(color)
        draw.rectangle(rect, fill=color, outline=(0, 0, 0))
        draw.text((rect[0] + pad, rect[1] + pad), text, fill=text_color, font=font)
        placed.append(rect)
        drawn += 1
    return np.asarray(pil_image), drawn


def _parse_frames(raw_value):
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return None
    raw_path = Path(raw_text)
    if raw_path.is_file():
        return [
            line.strip()
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [part.strip() for part in raw_text.replace(";", ",").split(",") if part.strip()]


def _available_frames(store):
    frames = []
    seen = set()
    for row in store.frame_table_rows:
        frame = str(row.get("frame_id", "") or "").strip()
        if frame and frame not in seen:
            frames.append(frame)
            seen.add(frame)
    if frames:
        return frames
    if store.frame_instance_maps_dir.is_dir():
        return sorted(path.stem for path in store.frame_instance_maps_dir.glob("*.npy"))
    return []


def _load_scene_class_table(class_path):
    class_path = Path(class_path)
    if not class_path.is_file():
        raise FileNotFoundError(f"semantic class file not found: {class_path}")
    class_table = {}
    blank_line_ids = []
    for class_id, line in enumerate(class_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        name = line.strip()
        if not name:
            blank_line_ids.append(int(class_id))
            continue
        class_table[int(class_id)] = name
    if not class_table:
        raise RuntimeError(f"No class names were parsed from {class_path}")
    return class_table, {
        "semantic_classes": str(class_path),
        "class_table_format": "line_index",
        "num_class_lines": int(len(class_table) + len(blank_line_ids)),
        "num_non_empty_classes": int(len(class_table)),
        "blank_line_ids": blank_line_ids,
    }


def _load_gt_mesh_classes_maskclustering(semantic_gt_mesh, semantic_classes, label_field):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("Reading semantic GT mesh requires plyfile. Install it with `pip install plyfile`.") from exc

    mesh_path = Path(semantic_gt_mesh)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Semantic GT mesh not found: {mesh_path}")
    vertices = PlyData.read(str(mesh_path))["vertex"].data
    names = vertices.dtype.names or ()
    if not label_field:
        label_field = next((candidate for candidate in _LABEL_FIELD_CANDIDATES if candidate in names), None)
    if not label_field or label_field not in names:
        raise RuntimeError(
            "Could not find a semantic label field in the GT mesh. "
            f"Available vertex fields: {names}. Pass --label_field explicitly."
        )

    class_table, class_info = _load_scene_class_table(semantic_classes)
    labels = np.asarray(vertices[label_field]).astype(np.int64)
    raw_ids = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)
    prediction_classes = []
    missing_ids = []
    for raw_id in raw_ids:
        class_name = class_table.get(raw_id)
        if class_name is None:
            missing_ids.append(raw_id)
            continue
        prediction_classes.append(
            {
                "prediction_id": int(len(prediction_classes)),
                "raw_id": int(raw_id),
                "name": str(class_name),
                "vertex_count": int((labels == raw_id).sum()),
            }
        )
    if not prediction_classes:
        raise RuntimeError(
            "The GT mesh contains labels, but none matched semantic_classes.txt line indices. "
            f"Raw ids found: {raw_ids[:50]}"
        )
    return prediction_classes, {
        "semantic_gt_mesh": str(mesh_path),
        **class_info,
        "label_field": str(label_field),
        "num_vertices": int(labels.shape[0]),
        "num_raw_gt_labels": int(len(raw_ids)),
        "num_prediction_classes": int(len(prediction_classes)),
        "ignored_negative_labels": sorted(int(value) for value in np.unique(labels) if int(value) < 0),
        "missing_class_ids": missing_ids,
        "classes": prediction_classes,
    }


def _parse_alphaclip_spec(clip_model):
    parts = str(clip_model).split(":", 3)
    if len(parts) >= 3 and parts[0].lower() in {"alphaclip", "alpha_clip", "alpha-clip"}:
        model_name = parts[1]
        checkpoint = parts[2]
        package_path = parts[3] if len(parts) > 3 else ""
        return model_name, checkpoint, package_path
    return str(clip_model), "", ""


def _encode_maskclustering_text_features(labels, clip_model, expected_dim, device, batch_size=64):
    model_name, checkpoint, package_path = _parse_alphaclip_spec(clip_model)
    package_path = package_path or os.environ.get("ALPHACLIP_PACKAGE_PATH", "")
    if package_path:
        resolved = str(Path(package_path).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    try:
        import alpha_clip
    except ImportError as exc:
        raise RuntimeError(
            "MaskClustering-style teacher visualization currently needs AlphaCLIP. "
            "Pass --clip_model alphaclip:MODEL:CHECKPOINT:PACKAGE_PATH."
        ) from exc
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AlphaCLIP checkpoint not found: {checkpoint_path}")

    model, _ = alpha_clip.load(str(model_name), alpha_vision_ckpt_pth=str(checkpoint_path), device=device)
    model.eval()
    prompts = [template.format(label=str(label)) for template in _PROMPT_TEMPLATES for label in labels]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(prompts), max(1, int(batch_size))):
            tokens = alpha_clip.tokenize(prompts[start : start + max(1, int(batch_size))]).to(device)
            features = model.encode_text(tokens).float()
            features = F.normalize(features, dim=-1, eps=1e-8)
            chunks.append(features.detach().cpu())
    encoded = torch.cat(chunks, dim=0).view(len(_PROMPT_TEMPLATES), len(labels), -1).mean(dim=0)
    encoded = F.normalize(encoded, dim=-1, eps=1e-8).to(device)
    if int(encoded.shape[1]) != int(expected_dim):
        raise RuntimeError(
            f"Encoded AlphaCLIP text dim {encoded.shape[1]} does not match teacher feature dim {expected_dim}."
        )
    return encoded


def _resolve_prediction_classes(args, prompt_bank):
    mesh_classes_only = not bool(args.all_prompt_classes)
    dataset = SimpleNamespace(
        source_path=str(args.source_path),
        semantic_eval_ignore_label=int(args.semantic_eval_ignore_label),
        semantic_raw_class_path=str(args.semantic_raw_class_path or ""),
        semantic_benchmark_map_path=str(args.semantic_benchmark_map_path or ""),
        semantic_eval_mesh_classes_only=mesh_classes_only,
        semantic_eval_exclude_classes=str(args.semantic_eval_exclude_classes or ""),
    )
    exclude_indices, excluded_names = _resolve_semantic_eval_exclude_indices(dataset, prompt_bank, logger=None)
    if not mesh_classes_only:
        return _semantic_eval_class_indices(len(prompt_bank["class_ids"]), exclude_indices), False, excluded_names

    gt_path = _as_abs_path(args.semantic_gt_3d, args.source_path)
    if gt_path is None:
        raise RuntimeError("GT-mesh class visualization needs --semantic_gt_3d unless --all_prompt_classes is set.")
    _, gt_labels = _load_semantic_gt_3d(gt_path)
    gt_to_index, ignore_values, _ = _build_gt_label_mapper(dataset, prompt_bank, logger=None)
    gt_indices = _map_gt_labels_to_indices(gt_labels, gt_to_index, ignore_values)
    gt_indices = _apply_semantic_eval_exclude_indices(gt_indices, exclude_indices)
    class_indices, _ = _semantic_eval_prediction_class_indices(
        dataset,
        gt_indices,
        len(prompt_bank["class_ids"]),
        exclude_indices,
        prompt_bank,
        logger=None,
    )
    return class_indices, True, excluded_names


def _classify_instances(decoded_features, text_features, active_class_indices, temperature, topk):
    if active_class_indices is None:
        class_index_tensor = torch.arange(text_features.shape[0], device=text_features.device, dtype=torch.long)
        active_text = text_features
    else:
        class_index_tensor = torch.as_tensor(active_class_indices, dtype=torch.long, device=text_features.device)
        if class_index_tensor.numel() == 0:
            raise RuntimeError("No active semantic classes are available for teacher visualization.")
        active_text = text_features[class_index_tensor]

    similarities = decoded_features @ active_text.T
    k = max(1, min(int(topk), int(active_text.shape[0])))
    top_values, top_local = torch.topk(similarities, k=k, dim=1)
    top_global = class_index_tensor[top_local]
    pred_global = top_global[:, 0]
    confidence = top_values[:, 0]
    return pred_global.cpu().numpy(), confidence.cpu().numpy(), top_global.cpu().numpy(), top_values.cpu().numpy()


def _load_teacher_features(args, store, codec_path, device):
    source = str(args.feature_source or "auto").lower()
    candidates = []
    if source == "auto":
        candidates = [
            ("decoded_for_visualization", args.teacher_path / "decoded_for_visualization" / "instance_features.npy"),
            ("decoded_instance_features", args.teacher_path / "decoded_instance_features.npy"),
        ]
        for feature_source, feature_path in candidates:
            if feature_path.is_file():
                features = torch.as_tensor(np.load(feature_path), dtype=torch.float32, device=device)
                if features.ndim != 2:
                    raise RuntimeError(f"{feature_path} must have shape [N,C], got {tuple(features.shape)}")
                if int(features.shape[0]) != store.num_instances:
                    raise RuntimeError(
                        f"{feature_path} row count {features.shape[0]} does not match instances {store.num_instances}."
                    )
                return F.normalize(features, dim=-1, eps=1e-8), feature_source, {"metadata": {"input_dim": int(features.shape[1])}}
        source = "codec_decode"

    if source == "decoded_for_visualization":
        feature_path = args.teacher_path / "decoded_for_visualization" / "instance_features.npy"
        features = torch.as_tensor(np.load(feature_path), dtype=torch.float32, device=device)
        return F.normalize(features, dim=-1, eps=1e-8), str(source), {"metadata": {"input_dim": int(features.shape[1])}}
    if source == "decoded_instance_features":
        feature_path = args.teacher_path / "decoded_instance_features.npy"
        features = torch.as_tensor(np.load(feature_path), dtype=torch.float32, device=device)
        return F.normalize(features, dim=-1, eps=1e-8), str(source), {"metadata": {"input_dim": int(features.shape[1])}}
    if source == "stored":
        features = store.instance_features.to(device).float()
        return F.normalize(features, dim=-1, eps=1e-8), str(source), {"metadata": {"input_dim": int(features.shape[1])}}
    if source != "codec_decode":
        raise RuntimeError(f"Unsupported --feature_source: {args.feature_source}")

    codec, codec_payload = load_semantic_codec(codec_path, map_location="cpu")
    codec = codec.to(device).eval()
    with torch.no_grad():
        features = codec.decode(store.instance_features.to(device)).float()
        features = F.normalize(features, dim=-1, eps=1e-8)
    return features, "codec_decode", codec_payload


def _write_palette_files(output_path, prompt_bank, palette, active_class_indices):
    rows = []
    active_set = None if active_class_indices is None else {int(idx) for idx in active_class_indices}
    for index, (class_id, class_name, prompt) in enumerate(
        zip(prompt_bank["class_ids"], prompt_bank["class_names"], prompt_bank["prompts"])
    ):
        color = palette[index]
        rows.append(
            {
                "index": int(index),
                "class_id": int(class_id),
                "class_name": class_name,
                "prompt": prompt,
                "active": active_set is None or int(index) in active_set,
                "r": int(color[0]),
                "g": int(color[1]),
                "b": int(color[2]),
            }
        )

    with (output_path / "palette.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "class_id", "class_name", "prompt", "active", "r", "g", "b"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (output_path / "palette.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_legend(output_path, prompt_bank, palette, active_class_indices, font):
    indices = list(range(len(prompt_bank["class_names"]))) if active_class_indices is None else list(active_class_indices)
    if not indices:
        return
    row_height = 28
    swatch = 18
    margin = 12
    width = 520
    height = margin * 2 + row_height * len(indices)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for row, class_index in enumerate(indices):
        y = margin + row * row_height
        color = tuple(int(v) for v in palette[int(class_index)].tolist())
        draw.rectangle((margin, y + 4, margin + swatch, y + 4 + swatch), fill=color, outline=(0, 0, 0))
        text = f"{prompt_bank['class_names'][int(class_index)]} [{int(class_index)}]"
        draw.text((margin + swatch + 10, y + 3), text, fill=(0, 0, 0), font=font)
    image.save(output_path / "legend.png")


def _render_frame(
    store,
    frame,
    args,
    output_dirs,
    class_palette,
    instance_palette,
    prompt_bank,
    instance_pred_classes,
    instance_confidence,
    font,
):
    if str(args.map_value_mode).lower() == "remap":
        frame_payload = store.load_frame(frame, expected_dim=store.feature_dim)
        instance_map = frame_payload["region_id_map"].cpu().numpy().astype(np.int64, copy=False)
        pixel_quality = frame_payload["pixel_quality"].cpu().numpy().astype(np.float32, copy=False)
    else:
        map_path = _find_frame_array(store.frame_instance_maps_dir, frame)
        if map_path is None:
            raise FileNotFoundError(f"Frame instance map not found for {frame}: {store.frame_instance_maps_dir}")
        instance_map = _squeeze_hw(_load_array(map_path), map_path, "frame instance map").astype(np.int64, copy=False)
        weight_path = _find_frame_array(store.frame_weights_dir, frame)
        if weight_path is None:
            pixel_quality = np.ones(instance_map.shape, dtype=np.float32)
        else:
            pixel_quality = _squeeze_hw(_load_array(weight_path), weight_path, "frame weight map").astype(
                np.float32,
                copy=False,
            )
            if tuple(pixel_quality.shape) != tuple(instance_map.shape):
                raise RuntimeError(
                    f"frame weight map shape {tuple(pixel_quality.shape)} does not match "
                    f"frame map shape {tuple(instance_map.shape)}: {weight_path}"
                )
    height, width = instance_map.shape
    valid = (instance_map >= 0) & (instance_map < store.num_instances)
    if float(args.min_weight) > 0.0:
        valid &= pixel_quality >= float(args.min_weight)

    rgb, image_path = _load_rgb(args.source_path, args.images, frame, width, height)
    void_color = np.asarray(args.void_color, dtype=np.uint8)
    class_color = np.zeros((height, width, 3), dtype=np.uint8)
    class_color[:, :] = void_color
    instance_color = np.zeros((height, width, 3), dtype=np.uint8)
    instance_color[:, :] = void_color

    label_map = np.full((height, width), -1, dtype=np.int32)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    if np.any(valid):
        visible_instances = instance_map[valid]
        visible_classes = instance_pred_classes[visible_instances]
        label_map[valid] = visible_classes.astype(np.int32)
        confidence_map[valid] = instance_confidence[visible_instances].astype(np.float32)
        class_color[valid] = class_palette[visible_classes]
        instance_color[valid] = instance_palette[(visible_instances + 1) % len(instance_palette)]

    alpha = float(args.overlay_alpha) * valid.astype(np.float32) * np.clip(pixel_quality, 0.0, 1.0)
    class_overlay = _blend_overlay(rgb, class_color, alpha)
    instance_overlay = _blend_overlay(rgb, instance_color, alpha)
    if np.any(valid):
        for instance_index in sorted(int(value) for value in np.unique(instance_map[valid])):
            mask = valid & (instance_map == instance_index)
            boundary = _mask_boundary(mask)
            if np.any(boundary):
                class_overlay[boundary] = class_palette[int(instance_pred_classes[instance_index])]
                instance_overlay[boundary] = instance_palette[(instance_index + 1) % len(instance_palette)]
    candidates = _collect_label_candidates(
        instance_map,
        valid,
        instance_pred_classes,
        instance_confidence,
        prompt_bank["class_names"],
        args.label_min_area,
        args.label_confidence,
    )
    labeled_overlay, labels_drawn = _draw_labels(
        class_overlay,
        candidates,
        class_palette,
        font,
        args.max_labels_per_image,
    )

    stem = _safe_stem(frame)
    Image.fromarray(class_color).save(output_dirs["class_color"] / f"{stem}.png")
    Image.fromarray(class_overlay).save(output_dirs["class_overlay"] / f"{stem}.png")
    Image.fromarray(labeled_overlay).save(output_dirs["class_overlay_labeled"] / f"{stem}.png")
    Image.fromarray(instance_color).save(output_dirs["instance_color"] / f"{stem}.png")
    Image.fromarray(instance_overlay).save(output_dirs["instance_overlay"] / f"{stem}.png")
    Image.fromarray(np.clip(confidence_map * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(
        output_dirs["confidence"] / f"{stem}.png"
    )
    if args.save_label_npy:
        np.save(output_dirs["label_npy"] / f"{stem}.npy", label_map)
    if args.save_rgb:
        Image.fromarray(rgb).save(output_dirs["rgb"] / f"{stem}.png")

    visible_pixels = int(valid.sum())
    visible_instance_indices = np.unique(instance_map[valid]).astype(np.int64).tolist() if visible_pixels else []
    present_class_indices = sorted({int(instance_pred_classes[idx]) for idx in visible_instance_indices})
    return {
        "frame": str(frame),
        "image_path": image_path,
        "visible_pixels": visible_pixels,
        "visible_instances": len(visible_instance_indices),
        "present_classes": ";".join(prompt_bank["class_names"][idx] for idx in present_class_indices),
        "labels_drawn": int(labels_drawn),
    }


def _make_output_dirs(output_path, save_label_npy, save_rgb):
    names = [
        "class_color",
        "class_overlay",
        "class_overlay_labeled",
        "instance_color",
        "instance_overlay",
        "confidence",
    ]
    if save_label_npy:
        names.append("label_npy")
    if save_rgb:
        names.append("rgb")
    dirs = {}
    for name in names:
        path = output_path / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def main():
    parser = ArgumentParser(description="Visualize an instance-level compressed semantic teacher with class labels.")
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--teacher_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--prompt_path", required=True, type=Path)
    parser.add_argument("--semantic_gt_3d", default="", type=str)
    parser.add_argument("--semantic_raw_class_path", default="", type=str)
    parser.add_argument("--semantic_benchmark_map_path", default="", type=str)
    parser.add_argument("--semantic_eval_ignore_label", default=-100, type=int)
    parser.add_argument("--semantic_eval_exclude_classes", default="", type=str)
    parser.add_argument("--all_prompt_classes", action="store_true", help="Classify against all prompts instead of GT-mesh classes only.")
    parser.add_argument("--clip_model", required=True, type=str)
    parser.add_argument("--semantic_codec_path", default="", type=str)
    parser.add_argument(
        "--feature_source",
        default="auto",
        choices=("auto", "decoded_for_visualization", "decoded_instance_features", "codec_decode", "stored"),
        help="Teacher feature source for text classification. auto prefers decoded_for_visualization for compressed teachers.",
    )
    parser.add_argument(
        "--map_value_mode",
        default="direct",
        choices=("direct", "remap"),
        help="direct matches MaskClustering visualization; remap follows the Scaffold training adapter.",
    )
    parser.add_argument("--label_field", default="", type=str)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--temperature", default=15.0, type=float)
    parser.add_argument("--topk", default=3, type=int)
    parser.add_argument("--text_batch_size", default=64, type=int)
    parser.add_argument("--frames", default="", type=str, help="Comma-separated frame ids or a text file with one frame per line.")
    parser.add_argument("--max_frames", default=0, type=int)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--overlay_alpha", default=0.55, type=float)
    parser.add_argument("--min_weight", default=0.0, type=float)
    parser.add_argument("--label_min_area", default=2500, type=int)
    parser.add_argument("--max_labels_per_image", default=80, type=int)
    parser.add_argument("--label_confidence", action="store_true", default=True)
    parser.add_argument("--font_path", default="", type=str)
    parser.add_argument("--font_size", default=44, type=int)
    parser.add_argument("--void_color", nargs=3, default=(0, 0, 0), type=int)
    parser.add_argument("--save_label_npy", action="store_true")
    parser.add_argument("--save_rgb", action="store_true")
    args = parser.parse_args()

    args.source_path = args.source_path.expanduser().resolve()
    args.teacher_path = args.teacher_path.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()
    args.prompt_path = _as_abs_path(args.prompt_path, args.source_path)
    codec_path = _as_abs_path(args.semantic_codec_path, args.teacher_path) if args.semantic_codec_path else args.teacher_path / "semantic_codec.pth"
    args.output_path.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    store = InstanceTeacherStore(args.teacher_path)
    teacher_features, feature_source, codec_payload = _load_teacher_features(args, store, codec_path, device)
    if bool(args.all_prompt_classes):
        prompt_bank = load_semantic_prompt_bank(args.prompt_path)
        gt_info = {
            "mode": "all_prompt_classes",
            "classes": [
                {"prediction_id": int(index), "raw_id": int(class_id), "name": str(class_name)}
                for index, (class_id, class_name) in enumerate(
                    zip(prompt_bank["class_ids"], prompt_bank["class_names"])
                )
            ],
        }
        mesh_classes_only = False
        excluded_class_names = []
    else:
        prediction_classes, gt_info = _load_gt_mesh_classes_maskclustering(
            _as_abs_path(args.semantic_gt_3d, args.source_path),
            args.prompt_path,
            args.label_field or None,
        )
        prompt_bank = {
            "class_ids": [int(row["raw_id"]) for row in prediction_classes],
            "class_names": [str(row["name"]) for row in prediction_classes],
            "prompts": [str(row["name"]) for row in prediction_classes],
        }
        mesh_classes_only = True
        excluded_class_names = []

    text_features = _encode_maskclustering_text_features(
        prompt_bank["class_names"],
        args.clip_model,
        int(teacher_features.shape[1]),
        device=device,
        batch_size=int(args.text_batch_size),
    )
    with torch.no_grad():
        instance_pred_classes, instance_confidence, instance_topk, instance_topk_scores = _classify_instances(
            teacher_features,
            text_features,
            None,
            args.temperature,
            args.topk,
        )

    class_palette = _stable_palette(prompt_bank["class_names"])
    inst_palette = _instance_palette(store.num_instances)
    output_dirs = _make_output_dirs(args.output_path, args.save_label_npy, args.save_rgb)
    font = _load_font(args.font_path, args.font_size)
    _write_palette_files(args.output_path, prompt_bank, class_palette, None)
    _write_legend(args.output_path, prompt_bank, class_palette, None, font)

    with (args.output_path / "instance_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instance_index",
                "instance_id",
                "pred_class_index",
                "class_id",
                "class_name",
                "confidence",
                "topk",
            ]
        )
        for instance_index in range(store.num_instances):
            pred_index = int(instance_pred_classes[instance_index])
            topk_entries = []
            for class_index, score in zip(instance_topk[instance_index], instance_topk_scores[instance_index]):
                topk_entries.append(f"{prompt_bank['class_names'][int(class_index)]}:{float(score):.4f}")
            instance_id = int(store.instance_ids[instance_index]) if instance_index < len(store.instance_ids) else instance_index
            writer.writerow(
                [
                    int(instance_index),
                    instance_id,
                    pred_index,
                    int(prompt_bank["class_ids"][pred_index]),
                    prompt_bank["class_names"][pred_index],
                    float(instance_confidence[instance_index]),
                    ";".join(topk_entries),
                ]
            )

    frames = _parse_frames(args.frames) or _available_frames(store)
    stride = max(1, int(args.frame_stride))
    frames = frames[::stride]
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    if not frames:
        raise RuntimeError(f"No teacher frames were found under {args.teacher_path}.")

    frame_rows = []
    for frame in tqdm(frames, desc="Visualizing teacher frames", dynamic_ncols=True):
        frame_rows.append(
            _render_frame(
                store,
                frame,
                args,
                output_dirs,
                class_palette,
                inst_palette,
                prompt_bank,
                instance_pred_classes,
                instance_confidence,
                font,
            )
        )

    with (args.output_path / "frame_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "image_path",
                "visible_pixels",
                "visible_instances",
                "present_classes",
                "labels_drawn",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_rows)

    summary = {
        "source_path": str(args.source_path),
        "teacher_path": str(args.teacher_path),
        "output_path": str(args.output_path),
        "images": str(args.images),
        "prompt_path": str(args.prompt_path),
        "semantic_gt_3d": str(args.semantic_gt_3d),
        "clip_model": str(args.clip_model),
        "semantic_codec_path": str(codec_path),
        "feature_source": str(feature_source),
        "map_value_mode": str(args.map_value_mode),
        "gt_mesh_classes": gt_info,
        "codec_metadata": codec_payload["metadata"],
        "num_instances": int(store.num_instances),
        "stored_teacher_feature_dim": int(store.feature_dim),
        "classification_feature_dim": int(teacher_features.shape[1]),
        "num_prompt_classes": int(len(prompt_bank["class_names"])),
        "mesh_classes_only": bool(mesh_classes_only),
        "active_class_indices": list(range(len(prompt_bank["class_names"]))),
        "active_class_names": prompt_bank["class_names"],
        "excluded_class_names": excluded_class_names,
        "num_frames_rendered": int(len(frame_rows)),
        "outputs": {name: str(path) for name, path in output_dirs.items()},
    }
    (args.output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
