# Copyright (C) 2026
#
# Diagnostic utility for visualizing cached SAM+CLIP region teachers before
# semantic 3DGS training. It does not modify teacher caches or model files.

import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils.semantic_benchmark_utils import load_semantic_prompt_bank, _encode_open_vocab_text_features
from utils.semantic_feature_utils import IMAGE_EXTENSIONS, find_region_teacher_path


def _as_abs_path(path, root):
    if not path:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def _list_images(image_dir):
    image_dir = Path(image_dir)
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))
    return sorted({str(path): path for path in images}.values(), key=lambda path: path.name)


def _safe_stem(path_or_name):
    stem = Path(str(path_or_name)).stem
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)


def _make_palette(num_classes):
    palette = np.zeros((max(1, num_classes), 3), dtype=np.uint8)
    for idx in range(max(1, num_classes)):
        palette[idx, 0] = (37 * idx + 23) % 256
        palette[idx, 1] = (17 * idx + 119) % 256
        palette[idx, 2] = (97 * idx + 53) % 256
    if num_classes > 0:
        palette[0] = np.array([230, 25, 75], dtype=np.uint8)
    return palette


def _region_palette(num_regions):
    palette = np.zeros((max(1, num_regions), 3), dtype=np.uint8)
    for idx in range(max(1, num_regions)):
        palette[idx, 0] = (53 * idx + 41) % 256
        palette[idx, 1] = (97 * idx + 83) % 256
        palette[idx, 2] = (193 * idx + 17) % 256
    return palette


def _blend_overlay(rgb, color, valid_mask, overlay_alpha):
    alpha = valid_mask.astype(np.float32)[..., None] * float(overlay_alpha)
    overlay = rgb.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(overlay + 0.5, 0, 255).astype(np.uint8)


def _load_torch_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_region_teacher(path, expected_dim):
    payload = _load_torch_payload(path)
    region_id_map = torch.as_tensor(payload.get("region_id_map"), dtype=torch.long)
    if region_id_map.ndim != 2:
        raise RuntimeError(f"region_id_map must be [H,W], got {tuple(region_id_map.shape)} from {path}")

    region_features = torch.as_tensor(payload.get("region_features", torch.empty((0, expected_dim))), dtype=torch.float32)
    if region_features.ndim == 1:
        region_features = region_features.unsqueeze(0)
    if region_features.numel() > 0 and int(region_features.shape[1]) != int(expected_dim):
        raise RuntimeError(
            f"Teacher feature dim {region_features.shape[1]} does not match expected dim {expected_dim}: {path}"
        )

    region_area = torch.as_tensor(payload.get("region_area", torch.empty((0,))), dtype=torch.float32).view(-1)
    if region_area.numel() != region_features.shape[0]:
        valid_ids = region_id_map[region_id_map >= 0]
        if valid_ids.numel() > 0 and region_features.shape[0] > 0:
            region_area = torch.bincount(valid_ids, minlength=region_features.shape[0]).float()
        else:
            region_area = torch.zeros((region_features.shape[0],), dtype=torch.float32)
    raw_region_area = torch.as_tensor(payload.get("raw_region_area", region_area), dtype=torch.float32).view(-1)
    if raw_region_area.numel() != region_features.shape[0]:
        raw_region_area = region_area.clone()

    region_quality = torch.as_tensor(payload.get("region_quality", torch.empty((0,))), dtype=torch.float32).view(-1)
    if region_quality.numel() != region_features.shape[0]:
        region_quality = torch.zeros((region_features.shape[0],), dtype=torch.float32)

    clip_confidence = torch.as_tensor(payload.get("clip_confidence", torch.empty((0,))), dtype=torch.float32).view(-1)
    if clip_confidence.numel() != region_features.shape[0]:
        clip_confidence = torch.zeros((region_features.shape[0],), dtype=torch.float32)
    crop_weight = torch.as_tensor(payload.get("crop_weight", torch.empty((0,))), dtype=torch.float32).view(-1)
    if crop_weight.numel() != region_features.shape[0]:
        crop_weight = torch.zeros((region_features.shape[0],), dtype=torch.float32)
    crop_names = payload.get("crop_names", [])
    if not isinstance(crop_names, list) or len(crop_names) != region_features.shape[0]:
        crop_names = [""] * int(region_features.shape[0])

    return {
        "image_name": payload.get("image_name", Path(path).stem),
        "config": payload.get("config", {}),
        "region_id_map": region_id_map,
        "region_features": F.normalize(region_features, dim=-1, eps=1e-6) if region_features.numel() else region_features,
        "region_area": region_area,
        "raw_region_area": raw_region_area,
        "region_quality": region_quality,
        "clip_confidence": clip_confidence,
        "crop_weight": crop_weight,
        "crop_names": crop_names,
        "has_clip_confidence": bool(payload.get("has_clip_confidence", False)),
    }


def _resize_rgb_to_map(rgb, height, width):
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb
    image = Image.fromarray(rgb)
    return np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.uint8)


def _teacher_predictions(region_features, text_features, temperature):
    if region_features.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.float32),
        )
    logits = region_features @ text_features.T
    probs = torch.softmax(logits * float(temperature), dim=-1)
    confidence, labels = probs.max(dim=-1)
    return labels.cpu().long(), confidence.cpu().float(), probs.cpu().float()


def _map_regions_to_pixels(region_id_map, region_values, fill_value):
    height, width = region_id_map.shape
    output = torch.full((height, width), fill_value, dtype=region_values.dtype)
    valid = region_id_map >= 0
    if valid.any() and region_values.numel() > 0:
        ids = region_id_map[valid].clamp(0, region_values.shape[0] - 1)
        output[valid] = region_values[ids]
    return output


def _save_palette_files(output_root, prompt_bank, palette):
    rows = []
    for idx, (class_id, class_name, prompt) in enumerate(
        zip(prompt_bank["class_ids"], prompt_bank["class_names"], prompt_bank["prompts"])
    ):
        rgb = palette[idx].tolist()
        rows.append(
            {
                "index": int(idx),
                "class_id": int(class_id),
                "class_name": class_name,
                "prompt": prompt,
                "r": int(rgb[0]),
                "g": int(rgb[1]),
                "b": int(rgb[2]),
            }
        )
    with (output_root / "palette.json").open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2, ensure_ascii=False)
    with (output_root / "palette.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["index", "class_id", "class_name", "prompt", "r", "g", "b"])
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _process_one(image_path, teacher_path, text_features, prompt_bank, class_palette, output_dirs, args):
    teacher = _load_region_teacher(teacher_path, args.semantic_feature_dim)
    region_id_map = teacher["region_id_map"]
    region_features = teacher["region_features"]
    height, width = region_id_map.shape
    valid = (region_id_map >= 0).numpy()

    with Image.open(image_path) as image_file:
        rgb = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
    rgb = _resize_rgb_to_map(rgb, height, width)

    region_labels, region_conf, region_probs = _teacher_predictions(region_features, text_features, args.temperature)
    pixel_labels = _map_regions_to_pixels(region_id_map, region_labels, -1)
    pixel_conf = _map_regions_to_pixels(region_id_map, region_conf, 0.0)

    labels_np = pixel_labels.numpy().astype(np.int32)
    conf_np = pixel_conf.numpy().astype(np.float32)
    color = np.zeros((height, width, 3), dtype=np.uint8)
    if class_palette.shape[0] > 0:
        safe_labels = np.clip(labels_np, 0, class_palette.shape[0] - 1)
        color = class_palette[safe_labels]
    color[~valid] = np.array(args.void_color, dtype=np.uint8)

    region_color = np.zeros((height, width, 3), dtype=np.uint8)
    if region_features.shape[0] > 0:
        r_palette = _region_palette(region_features.shape[0])
        safe_region_ids = np.clip(region_id_map.numpy(), 0, region_features.shape[0] - 1)
        region_color = r_palette[safe_region_ids]
    region_color[~valid] = np.array(args.void_color, dtype=np.uint8)

    stem = _safe_stem(image_path)
    pred_color_path = output_dirs["pred_color"] / f"{stem}.png"
    overlay_path = output_dirs["overlay"] / f"{stem}.png"
    confidence_path = output_dirs["confidence"] / f"{stem}.png"
    region_id_path = output_dirs["region_ids"] / f"{stem}.png"
    region_overlay_path = output_dirs["region_overlay"] / f"{stem}.png"

    Image.fromarray(color).save(pred_color_path)
    Image.fromarray(_blend_overlay(rgb, color, valid, args.overlay_alpha)).save(overlay_path)
    Image.fromarray(np.clip(conf_np * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(confidence_path)
    Image.fromarray(region_color).save(region_id_path)
    Image.fromarray(_blend_overlay(rgb, region_color, valid, args.overlay_alpha)).save(region_overlay_path)

    label_npy_path = None
    confidence_npy_path = None
    if args.save_npy:
        label_npy_path = output_dirs["labels"] / f"{stem}.npy"
        confidence_npy_path = output_dirs["confidence"] / f"{stem}.npy"
        np.save(label_npy_path, labels_np)
        np.save(confidence_npy_path, conf_np)

    pixel_rows = []
    region_rows = []
    num_classes = len(prompt_bank["class_names"])
    if num_classes > 0 and valid.any():
        flat_labels = labels_np[valid]
        flat_conf = conf_np[valid]
        pixel_counts = np.bincount(flat_labels[flat_labels >= 0], minlength=num_classes)
        for class_index in np.argsort(pixel_counts)[::-1]:
            count = int(pixel_counts[class_index])
            if count <= 0:
                continue
            class_mask = flat_labels == class_index
            pixel_rows.append(
                {
                    "image_name": stem,
                    "class_index": int(class_index),
                    "class_id": int(prompt_bank["class_ids"][class_index]),
                    "class_name": prompt_bank["class_names"][class_index],
                    "pixels": count,
                    "pixel_ratio": float(count / max(1, int(valid.sum()))),
                    "mean_confidence": float(flat_conf[class_mask].mean()) if class_mask.any() else 0.0,
                }
            )

    if region_features.shape[0] > 0:
        region_area = teacher["region_area"].cpu().numpy()
        raw_region_area = teacher["raw_region_area"].cpu().numpy()
        region_quality = teacher["region_quality"].cpu().numpy()
        clip_conf = teacher["clip_confidence"].cpu().numpy()
        crop_weight = teacher["crop_weight"].cpu().numpy()
        crop_names = teacher["crop_names"]
        for region_id in range(region_features.shape[0]):
            class_index = int(region_labels[region_id].item())
            topk = min(int(args.region_topk), int(region_probs.shape[1])) if region_probs.numel() else 0
            top_entries = []
            if topk > 0:
                scores, indices = torch.topk(region_probs[region_id], k=topk)
                top_entries = [
                    f"{prompt_bank['class_names'][int(idx)]}:{float(score):.4f}"
                    for score, idx in zip(scores.tolist(), indices.tolist())
                ]
            region_rows.append(
                {
                    "image_name": stem,
                    "region_id": int(region_id),
                    "class_index": class_index,
                    "class_id": int(prompt_bank["class_ids"][class_index]),
                    "class_name": prompt_bank["class_names"][class_index],
                    "area": float(region_area[region_id]) if region_id < len(region_area) else 0.0,
                    "raw_area": float(raw_region_area[region_id]) if region_id < len(raw_region_area) else 0.0,
                    "effective_area_ratio": (
                        float(region_area[region_id] / max(raw_region_area[region_id], 1.0))
                        if region_id < len(region_area) and region_id < len(raw_region_area)
                        else 1.0
                    ),
                    "teacher_confidence": float(region_conf[region_id].item()),
                    "sam_quality": float(region_quality[region_id]) if region_id < len(region_quality) else 0.0,
                    "cached_clip_confidence": float(clip_conf[region_id]) if region_id < len(clip_conf) else 0.0,
                    "crop_weight": float(crop_weight[region_id]) if region_id < len(crop_weight) else 0.0,
                    "crop_names": crop_names[region_id] if region_id < len(crop_names) else "",
                    "topk": " | ".join(top_entries),
                }
            )

    if args.save_region_csv:
        _write_csv(
            output_dirs["region_predictions"] / f"{stem}.csv",
            region_rows,
            [
                "image_name",
                "region_id",
                "class_index",
                "class_id",
                "class_name",
                "area",
                "raw_area",
                "effective_area_ratio",
                "teacher_confidence",
                "sam_quality",
                "cached_clip_confidence",
                "crop_weight",
                "crop_names",
                "topk",
            ],
        )

    return {
        "image_name": stem,
        "image_path": str(image_path),
        "teacher_path": str(teacher_path),
        "height": int(height),
        "width": int(width),
        "num_regions": int(region_features.shape[0]),
        "valid_pixel_ratio": float(valid.mean()) if valid.size else 0.0,
        "pred_color": str(pred_color_path),
        "overlay": str(overlay_path),
        "confidence": str(confidence_path),
        "region_ids": str(region_id_path),
        "region_overlay": str(region_overlay_path),
        "label_npy": str(label_npy_path) if label_npy_path is not None else "",
        "confidence_npy": str(confidence_npy_path) if confidence_npy_path is not None else "",
        "pixel_rows": pixel_rows,
        "region_rows": region_rows,
        "config": teacher["config"],
    }


def inspect_region_teachers(args):
    source_path = Path(args.source_path)
    image_dir = _as_abs_path(args.images, source_path)
    region_dir = _as_abs_path(args.region_teachers, source_path)
    if image_dir is None or not image_dir.exists():
        raise RuntimeError(f"Image directory does not exist: {image_dir}")
    if region_dir is None or not region_dir.exists():
        raise RuntimeError(f"Region-teacher directory does not exist: {region_dir}")

    prompt_bank = load_semantic_prompt_bank(args.prompt_path)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    text_features = _encode_open_vocab_text_features(
        prompt_bank["prompts"],
        args.clip_model,
        int(args.semantic_feature_dim),
        device=device,
    ).cpu()
    class_palette = _make_palette(len(prompt_bank["class_names"]))

    output_root = Path(args.output_path) if args.output_path else source_path / "teacher_visualizations" / Path(region_dir).name
    output_dirs = {
        "pred_color": output_root / "pred_color",
        "overlay": output_root / "overlay",
        "confidence": output_root / "confidence",
        "region_ids": output_root / "region_ids",
        "region_overlay": output_root / "region_overlay",
        "labels": output_root / "labels",
        "region_predictions": output_root / "region_predictions",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    _save_palette_files(output_root, prompt_bank, class_palette)

    images = _list_images(image_dir)
    if args.max_images > 0:
        images = images[: int(args.max_images)]
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    manifest = []
    all_pixel_rows = []
    all_region_rows = []
    missing = []
    first_config = None
    for image_path in tqdm(images, desc="Inspecting region teachers"):
        teacher_path = find_region_teacher_path(region_dir, image_path.stem)
        if teacher_path is None:
            missing.append(str(image_path.name))
            if args.strict:
                raise RuntimeError(f"Missing region teacher for {image_path.name} in {region_dir}")
            continue
        result = _process_one(Path(image_path), Path(teacher_path), text_features, prompt_bank, class_palette, output_dirs, args)
        first_config = first_config or result["config"]
        all_pixel_rows.extend(result.pop("pixel_rows"))
        all_region_rows.extend(result.pop("region_rows"))
        manifest.append(result)

    _write_csv(
        output_root / "per_image_top_classes.csv",
        all_pixel_rows,
        ["image_name", "class_index", "class_id", "class_name", "pixels", "pixel_ratio", "mean_confidence"],
    )
    _write_csv(
        output_root / "all_region_predictions.csv",
        all_region_rows,
        [
            "image_name",
            "region_id",
            "class_index",
            "class_id",
            "class_name",
            "area",
            "raw_area",
            "effective_area_ratio",
            "teacher_confidence",
            "sam_quality",
            "cached_clip_confidence",
            "crop_weight",
            "crop_names",
            "topk",
        ],
    )

    overall = {}
    for row in all_pixel_rows:
        class_index = int(row["class_index"])
        entry = overall.setdefault(
            class_index,
            {
                "class_index": class_index,
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "pixels": 0,
                "weighted_confidence_sum": 0.0,
            },
        )
        pixels = int(row["pixels"])
        entry["pixels"] += pixels
        entry["weighted_confidence_sum"] += float(row["mean_confidence"]) * pixels

    total_pixels = sum(item["pixels"] for item in overall.values())
    overall_rows = []
    for item in sorted(overall.values(), key=lambda row: row["pixels"], reverse=True):
        pixels = int(item["pixels"])
        overall_rows.append(
            {
                "class_index": int(item["class_index"]),
                "class_id": int(item["class_id"]),
                "class_name": item["class_name"],
                "pixels": pixels,
                "pixel_ratio": float(pixels / max(1, total_pixels)),
                "mean_confidence": float(item["weighted_confidence_sum"] / max(1, pixels)),
            }
        )
    _write_csv(
        output_root / "overall_top_classes.csv",
        overall_rows,
        ["class_index", "class_id", "class_name", "pixels", "pixel_ratio", "mean_confidence"],
    )

    config = {
        "source_path": str(source_path),
        "images": str(image_dir),
        "region_teachers": str(region_dir),
        "prompt_path": str(args.prompt_path),
        "clip_model": args.clip_model,
        "semantic_feature_dim": int(args.semantic_feature_dim),
        "temperature": float(args.temperature),
        "overlay_alpha": float(args.overlay_alpha),
        "processed_images": len(manifest),
        "missing_teachers": missing,
        "teacher_config_sample": first_config or {},
    }
    with (output_root / "inspect_config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2, ensure_ascii=False)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    print(f"[Teacher Inspect] Processed {len(manifest)} images; missing={len(missing)}")
    print(f"[Teacher Inspect] Saved outputs to {output_root}")
    if overall_rows:
        top_preview = ", ".join(
            f"{row['class_name']}={row['pixels']}" for row in overall_rows[: min(8, len(overall_rows))]
        )
        print(f"[Teacher Inspect] Overall top classes: {top_preview}")


def build_parser():
    parser = ArgumentParser(description="Visualize cached SAM+CLIP region teachers before semantic 3DGS training.")
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--region_teachers", default="region_teachers", type=str)
    parser.add_argument("--prompt_path", required=True, type=str)
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--clip_model", default="ViT-B/32", type=str)
    parser.add_argument("--semantic_feature_dim", default=512, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--temperature", default=15.0, type=float)
    parser.add_argument("--overlay_alpha", default=0.45, type=float)
    parser.add_argument("--max_images", default=0, type=int)
    parser.add_argument("--region_topk", default=5, type=int)
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--save_region_csv", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.set_defaults(void_color=(0, 0, 0))
    return parser


if __name__ == "__main__":
    inspect_region_teachers(build_parser().parse_args())
