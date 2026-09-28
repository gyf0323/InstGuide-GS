from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_3dgs_teacher.visualization_utils import (  # noqa: E402
    UNKNOWN_COLOR,
    encode_text_features,
    find_image,
    format_prediction,
    infer_semantic_classes_path,
    load_gt_mesh_classes,
    normalize_rows,
    resolve_repo_path,
    save_json,
    stable_color,
    write_legend,
)


IMAGE_EXTENSIONS = (".npy", ".npz")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        payload = np.load(path)
        if "data" in payload:
            return np.asarray(payload["data"])
        keys = list(payload.keys())
        if not keys:
            raise RuntimeError(f"Empty npz file: {path}")
        return np.asarray(payload[keys[0]])
    return np.asarray(np.load(path))


def load_instance_map(path: Path) -> np.ndarray:
    data = load_array(path)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise RuntimeError(f"Expected [H,W] or [1,H,W] instance map in {path}, got {data.shape}.")
    return data.astype(np.int64, copy=False)


def find_frame_map(map_dir: Path, frame_id: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = map_dir / f"{frame_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def discover_frame_maps(map_dir: Path, max_images: int) -> List[Path]:
    paths: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(map_dir.glob(f"*{ext}"))
    paths = sorted(paths)
    if int(max_images) > 0:
        paths = paths[: int(max_images)]
    if not paths:
        raise FileNotFoundError(f"No frame instance maps found under {map_dir}.")
    return paths


def infer_frame_dirs(teacher_dir: Path, summary: Dict[str, Any]) -> Tuple[Path, Path]:
    frame_data = dict(summary.get("frame_data", {})) if isinstance(summary.get("frame_data", {}), dict) else {}
    map_dir = frame_data.get("frame_instance_maps_dir") or summary.get("frame_instance_maps_dir")
    weight_dir = frame_data.get("frame_weights_dir") or summary.get("frame_weights_dir")
    if map_dir:
        map_path = Path(str(map_dir)).expanduser()
    else:
        map_path = teacher_dir / "frame_instance_maps"
    if weight_dir:
        weight_path = Path(str(weight_dir)).expanduser()
    else:
        weight_path = teacher_dir / "frame_weights"
    return map_path.resolve(), weight_path.resolve()


def decode_instance_features(teacher_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    features = np.asarray(np.load(teacher_dir / "instance_features.npy"), dtype=np.float32)
    codec_info: Dict[str, Any] = {}
    if features.ndim != 2:
        raise RuntimeError(f"Expected [N,C] instance features in {teacher_dir / 'instance_features.npy'}, got {features.shape}.")
    codec_info["feature_source"] = "instance_features.npy"
    return normalize_rows(features), codec_info


def classify_features(features: np.ndarray, text_features: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    features = normalize_rows(features)
    if features.shape[1] != text_features.shape[1]:
        raise RuntimeError(
            f"Decoded feature dim {features.shape[1]} does not match text feature dim {text_features.shape[1]}."
        )
    similarities = features @ text_features.T
    topk = max(1, min(int(topk), similarities.shape[1]))
    ids = np.argsort(-similarities, axis=1)[:, :topk]
    scores = np.take_along_axis(similarities, ids, axis=1)
    return ids.astype(np.int64), scores.astype(np.float32)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary & mask


def load_font(font_size: int) -> Any:
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            path = Path(candidate)
            if path.is_file():
                return ImageFont.truetype(str(path), int(font_size))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_label(draw: Any, xy: Tuple[int, int], text: str, color: Tuple[int, int, int], font: Any) -> None:
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4), fill=(0, 0, 0))
    except Exception:
        pass
    draw.text((x, y), text, fill=color, font=font)


def frame_id_from_map_path(path: Path) -> str:
    return path.stem


def stable_instance_color(instance_id: int, class_name: str) -> np.ndarray:
    digest = hashlib.md5(f"maskclustering-instance:{instance_id}:{class_name}".encode("utf-8")).digest()
    return np.asarray([48 + digest[0] % 176, 48 + digest[1] % 176, 48 + digest[2] % 176], dtype=np.uint8)


def visualize(args: argparse.Namespace) -> Dict[str, Any]:
    from PIL import Image, ImageDraw

    teacher_dir = resolve_repo_path(args.teacher_dir)
    rgb_dir = resolve_repo_path(args.rgb_dir)
    summary_path = teacher_dir / "summary.json"
    summary_in = load_json(summary_path) if summary_path.is_file() else {}
    map_dir, _ = infer_frame_dirs(teacher_dir, summary_in)
    semantic_classes = infer_semantic_classes_path(teacher_dir, str(args.semantic_classes or ""))
    output_dir = Path(args.output_dir) if args.output_dir else teacher_dir / "semantic_visualization_gt_vocab_instance"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlay"
    labeled_dir = output_dir / "overlay_labeled"
    color_dir = output_dir / "semantic_color"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = np.asarray(np.load(teacher_dir / "instance_ids.npy"), dtype=np.int64)
    instance_features, codec_info = decode_instance_features(teacher_dir)
    if instance_features.shape[0] != instance_ids.shape[0]:
        raise RuntimeError(
            f"instance feature/id count mismatch: {instance_features.shape[0]} vs {instance_ids.shape[0]}."
        )

    prediction_classes, gt_info = load_gt_mesh_classes(
        resolve_repo_path(args.semantic_gt_mesh),
        semantic_classes,
        args.label_field,
    )
    labels = [str(row["name"]) for row in prediction_classes]
    palette = np.stack([stable_color(label) for label in labels], axis=0)
    text_features = encode_text_features(
        labels=labels,
        alphaclip_checkpoint=resolve_repo_path(args.alphaclip_checkpoint),
        alphaclip_package_path=args.alphaclip_package_path,
        clip_model=args.clip_model,
        device_name=args.device,
        batch_size=int(args.text_batch_size),
    )
    top_ids, top_scores = classify_features(instance_features, text_features, int(args.topk))
    instance_predictions = [
        format_prediction(
            top_ids[index],
            top_scores[index],
            prediction_classes,
            float(args.unknown_score_thresh),
            float(args.unknown_margin_thresh),
        )
        for index in range(instance_features.shape[0])
    ]

    frame_paths = discover_frame_maps(map_dir, int(args.max_images))
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    font = load_font(int(args.label_font_size))

    image_records: List[Dict[str, Any]] = []
    region_records: List[Dict[str, Any]] = []
    class_counts: Counter = Counter()
    instance_observations: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    iterator = frame_paths
    try:
        from tqdm import tqdm

        iterator = tqdm(frame_paths, desc="Render instance teacher semantics")
    except Exception:
        pass

    for map_path in iterator:
        frame_id = frame_id_from_map_path(map_path)
        image_path = find_image(rgb_dir, frame_id)
        if image_path is None:
            continue
        inst_map = load_instance_map(map_path)
        image = Image.open(image_path).convert("RGB")
        if image.size != (int(inst_map.shape[1]), int(inst_map.shape[0])):
            image = image.resize((int(inst_map.shape[1]), int(inst_map.shape[0])), Image.BILINEAR)
        image_rgb = np.asarray(image).astype(np.float32)
        overlay = image_rgb.copy()
        semantic_rgb = np.zeros_like(image_rgb, dtype=np.uint8)
        label_positions: List[Tuple[int, int, str, Tuple[int, int, int]]] = []
        current_regions: List[Dict[str, Any]] = []

        for instance_index in sorted(int(value) for value in np.unique(inst_map) if int(value) >= 0):
            if instance_index >= int(instance_ids.shape[0]):
                continue
            mask = inst_map == int(instance_index)
            area = int(mask.sum())
            if area <= 0:
                continue
            prediction = instance_predictions[instance_index]
            instance_id = int(instance_ids[instance_index])
            if bool(args.color_by_instance):
                color = stable_instance_color(instance_id, str(prediction["class_name"]))
            else:
                color = UNKNOWN_COLOR if prediction["is_unknown"] else palette[int(prediction["prediction_id"])]
            semantic_rgb[mask] = color
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color.astype(np.float32)
            overlay[mask_boundary(mask)] = color.astype(np.float32)
            row = {
                "image_id": frame_id,
                "instance_index": int(instance_index),
                "instance_id": instance_id,
                "mask_area": area,
                "mask_area_ratio": float(area) / float(max(inst_map.size, 1)),
                **prediction,
            }
            current_regions.append(row)
            region_records.append(row)
            instance_observations[instance_id].append(row)
            class_counts[str(row["class_name"])] += 1
            if area >= int(args.min_label_area):
                ys, xs = np.nonzero(mask)
                text = f"{row['class_name']} {row['top1_score']:.2f} I{instance_id}"
                label_positions.append((int(xs.min()), int(ys.min()), text, tuple(int(v) for v in color)))

        clean_overlay = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        clean_overlay.save(overlay_dir / f"{frame_id}_semantic_overlay.jpg", quality=95)
        labeled = clean_overlay.copy()
        draw = ImageDraw.Draw(labeled)
        for x, y, text, color in label_positions:
            draw_label(draw, (x, y), text, color, font)
        labeled.save(labeled_dir / f"{frame_id}_semantic_overlay_labeled.jpg", quality=95)
        Image.fromarray(semantic_rgb).save(color_dir / f"{frame_id}_semantic_color.png")
        image_records.append(
            {
                "image_id": frame_id,
                "image_path": str(image_path),
                "num_instance_regions": int(len(current_regions)),
                "teacher_pixel_coverage_ratio": float((inst_map >= 0).sum()) / float(max(inst_map.size, 1)),
                "regions": current_regions,
            }
        )

    instance_summary: List[Dict[str, Any]] = []
    for index, instance_id_raw in enumerate(instance_ids.tolist()):
        prediction = instance_predictions[index]
        rows = instance_observations.get(int(instance_id_raw), [])
        instance_summary.append(
            {
                "instance_index": int(index),
                "instance_id": int(instance_id_raw),
                "num_frame_observations": int(len(rows)),
                "total_visible_pixels": int(sum(int(row["mask_area"]) for row in rows)),
                **prediction,
            }
        )

    summary = {
        "teacher_dir": str(teacher_dir),
        "rgb_dir": str(rgb_dir),
        "map_dir": str(map_dir),
        "semantic_gt_mesh": str(resolve_repo_path(args.semantic_gt_mesh)),
        "semantic_classes": str(semantic_classes),
        "output_dir": str(output_dir),
        "clip_model": str(args.clip_model),
        "alphaclip_checkpoint": str(resolve_repo_path(args.alphaclip_checkpoint)),
        "codec_info": codec_info,
        "gt_mesh_classes": gt_info,
        "num_images": int(len(image_records)),
        "num_instances": int(instance_ids.shape[0]),
        "num_visible_instance_regions": int(len(region_records)),
        "class_region_counts": dict(class_counts),
        "unknown_score_thresh": float(args.unknown_score_thresh),
        "unknown_margin_thresh": float(args.unknown_margin_thresh),
        "notes": [
            "For compressed teachers, visualize the decoded_for_visualization directory generated by compress_teacher.py.",
            "The text vocabulary is restricted to semantic classes that appear in the GT mesh.",
            "This is a qualitative teacher-feature visualization, not a geometric GT-vs-pred pixel evaluation.",
        ],
    }
    save_json(output_dir / "prediction_classes.json", gt_info)
    save_json(output_dir / "per_image_semantic_regions.json", image_records)
    save_json(output_dir / "instance_semantic_predictions.json", instance_summary)
    save_json(output_dir / "summary.json", summary)
    write_legend(output_dir / "legend.tsv", prediction_classes, palette)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize final instance-level semantic teacher predictions."
    )
    parser.add_argument("--teacher_dir", required=True)
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--semantic_gt_mesh", required=True)
    parser.add_argument("--semantic_classes", default="")
    parser.add_argument("--label_field", default="")
    parser.add_argument("--alphaclip_package_path", default="third_party/AlphaCLIP-main")
    parser.add_argument("--alphaclip_checkpoint", required=True)
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--min_label_area", type=int, default=2500)
    parser.add_argument("--label_font_size", type=int, default=44)
    parser.add_argument("--unknown_score_thresh", type=float, default=-1.0)
    parser.add_argument("--unknown_margin_thresh", type=float, default=-1.0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--color_by_instance", action="store_true")
    return parser


def main() -> None:
    visualize(build_parser().parse_args())


if __name__ == "__main__":
    main()
