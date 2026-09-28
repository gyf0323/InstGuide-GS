from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
LABEL_FIELD_CANDIDATES = (
    "label",
    "semantic_label",
    "semantic_id",
    "semantic",
    "class",
    "class_id",
    "category_id",
    "nyu40id",
)
PROMPT_TEMPLATES = (
    "{label}",
    "a photo of a {label}.",
    "an indoor photo of a {label}.",
    "a photo of the {label} in a room.",
)
UNKNOWN_COLOR = np.asarray([112, 112, 112], dtype=np.uint8)


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.exists() or path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    return repo_path if repo_path.exists() else path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_rows(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def infer_semantic_classes_path(teacher_dir: Path, semantic_classes: str) -> Path:
    if semantic_classes:
        path = resolve_repo_path(semantic_classes)
        if path.exists():
            return path
        raise FileNotFoundError(f"semantic_classes.txt not found: {path}")
    candidates = [teacher_dir / "semantic_classes.txt"]
    candidates.extend(parent / "semantic_classes.txt" for parent in teacher_dir.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not infer semantic_classes.txt. Pass --semantic_classes explicitly, for example "
        "`--semantic_classes /path/to/scene/semantic_classes.txt`."
    )


def load_scene_class_table(semantic_classes: Path) -> Tuple[Dict[int, str], Dict[str, Any]]:
    if not semantic_classes.exists():
        raise FileNotFoundError(f"semantic_classes.txt not found: {semantic_classes}")
    raw_lines = semantic_classes.read_text(encoding="utf-8", errors="replace").splitlines()
    class_table: Dict[int, str] = {}
    blank_line_ids: List[int] = []
    for class_id, line in enumerate(raw_lines):
        name = line.strip()
        if not name:
            blank_line_ids.append(int(class_id))
            continue
        class_table[int(class_id)] = name
    if not class_table:
        raise RuntimeError(f"No class names were parsed from {semantic_classes}")
    info = {
        "semantic_classes": str(semantic_classes),
        "class_table_format": "line_index",
        "num_class_lines": int(len(raw_lines)),
        "num_non_empty_classes": int(len(class_table)),
        "blank_line_ids": blank_line_ids,
        "notes": [
            "GT mesh label values are interpreted as zero-based line indices into semantic_classes.txt.",
            "This is scene-specific mapping, not ScanNet++ global semantic IDs.",
        ],
    }
    return class_table, info


def load_gt_mesh_classes(
    semantic_gt_mesh: Path,
    semantic_classes: Path,
    label_field: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("Reading semantic GT mesh requires plyfile. Install it with `pip install plyfile`.") from exc

    if not semantic_gt_mesh.exists():
        raise FileNotFoundError(f"Semantic GT mesh not found: {semantic_gt_mesh}")
    vertices = PlyData.read(str(semantic_gt_mesh))["vertex"].data
    names = vertices.dtype.names or ()
    if not label_field:
        label_field = next((candidate for candidate in LABEL_FIELD_CANDIDATES if candidate in names), None)
    if not label_field or label_field not in names:
        raise RuntimeError(
            "Could not find a semantic label field in the GT mesh. "
            f"Available vertex fields: {names}. Pass --label_field explicitly."
        )

    class_table, class_info = load_scene_class_table(semantic_classes)
    labels = np.asarray(vertices[label_field]).astype(np.int64)
    raw_ids = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)
    prediction_classes: List[Dict[str, Any]] = []
    missing_ids: List[int] = []
    for raw_id in raw_ids:
        name = class_table.get(raw_id)
        if name is None:
            missing_ids.append(raw_id)
            continue
        prediction_classes.append(
            {
                "prediction_id": int(len(prediction_classes)),
                "raw_id": int(raw_id),
                "name": str(name),
                "vertex_count": int((labels == raw_id).sum()),
            }
        )
    if not prediction_classes:
        raise RuntimeError(
            "The GT mesh contains labels, but none matched semantic_classes.txt line indices. "
            f"Raw ids found: {raw_ids[:50]}, semantic_classes lines: {class_info['num_class_lines']}"
        )
    info = {
        "semantic_gt_mesh": str(semantic_gt_mesh),
        **class_info,
        "label_field": str(label_field),
        "num_vertices": int(labels.shape[0]),
        "num_raw_gt_labels": int(len(raw_ids)),
        "num_prediction_classes": int(len(prediction_classes)),
        "ignored_negative_labels": sorted(int(value) for value in np.unique(labels) if int(value) < 0),
        "missing_class_ids": missing_ids,
        "classes": prediction_classes,
    }
    return prediction_classes, info


def install_alphaclip(package_path: str) -> Any:
    if package_path:
        resolved = str(resolve_repo_path(package_path).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    try:
        import alpha_clip
    except Exception as exc:
        raise RuntimeError("Alpha-CLIP is unavailable. Pass --alphaclip_package_path or install it.") from exc
    return alpha_clip


def encode_text_features(
    labels: Sequence[str],
    alphaclip_checkpoint: Path,
    alphaclip_package_path: str,
    clip_model: str,
    device_name: str,
    batch_size: int,
) -> np.ndarray:
    alpha_clip = install_alphaclip(alphaclip_package_path)
    if not alphaclip_checkpoint.exists():
        raise FileNotFoundError(f"Alpha-CLIP checkpoint not found: {alphaclip_checkpoint}")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = alpha_clip.load(
        str(clip_model),
        alpha_vision_ckpt_pth=str(alphaclip_checkpoint),
        device=device,
    )
    model.eval()
    prompts = [template.format(label=label) for template in PROMPT_TEMPLATES for label in labels]
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(prompts), int(batch_size)):
            tokens = alpha_clip.tokenize(prompts[start : start + int(batch_size)]).to(device)
            features = model.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            chunks.append(features.cpu().numpy().astype(np.float32))
    encoded = normalize_rows(np.concatenate(chunks, axis=0))
    encoded = encoded.reshape(len(PROMPT_TEMPLATES), len(labels), -1).mean(axis=0)
    return normalize_rows(encoded)


def find_image(rgb_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = rgb_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_seg_map(path: Path) -> np.ndarray:
    seg = np.load(path)
    if seg.ndim == 3 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim != 2:
        raise RuntimeError(f"Unsupported segment map shape {seg.shape}: {path}")
    return seg.astype(np.int64, copy=False)


def classify_regions(features: np.ndarray, text_features: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    features = normalize_rows(features)
    if features.shape[1] != text_features.shape[1]:
        raise RuntimeError(
            f"Teacher feature dim {features.shape[1]} does not match text feature dim {text_features.shape[1]}."
        )
    similarities = features @ text_features.T
    topk = max(1, min(int(topk), similarities.shape[1]))
    ids = np.argsort(-similarities, axis=1)[:, :topk]
    scores = np.take_along_axis(similarities, ids, axis=1)
    return ids.astype(np.int64), scores.astype(np.float32)


def stable_color(name: str) -> np.ndarray:
    digest = hashlib.md5(f"maskclustering:{name}".encode("utf-8")).digest()
    return np.asarray([48 + digest[0] % 176, 48 + digest[1] % 176, 48 + digest[2] % 176], dtype=np.uint8)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary & mask


def draw_label(draw: Any, xy: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text)
        draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill=(0, 0, 0))
    except Exception:
        pass
    draw.text((x, y), text, fill=color)


def format_prediction(
    region_top_ids: np.ndarray,
    region_top_scores: np.ndarray,
    prediction_classes: Sequence[Dict[str, Any]],
    unknown_score_thresh: float,
    unknown_margin_thresh: float,
) -> Dict[str, Any]:
    top1_id = int(region_top_ids[0])
    top1_score = float(region_top_scores[0])
    top2_score = float(region_top_scores[1]) if len(region_top_scores) > 1 else -1.0
    margin = top1_score - top2_score
    top_predictions = []
    for class_id, score in zip(region_top_ids.tolist(), region_top_scores.tolist()):
        class_row = prediction_classes[int(class_id)]
        top_predictions.append(
            {
                "prediction_id": int(class_id),
                "raw_id": int(class_row["raw_id"]),
                "class_name": str(class_row["name"]),
                "score": float(score),
            }
        )
    is_unknown = False
    if float(unknown_score_thresh) >= 0.0 and top1_score < float(unknown_score_thresh):
        is_unknown = True
    if float(unknown_margin_thresh) >= 0.0 and margin < float(unknown_margin_thresh):
        is_unknown = True
    if is_unknown:
        return {
            "prediction_id": -1,
            "raw_id": -1,
            "class_name": "unknown",
            "top1_score": top1_score,
            "top1_top2_margin": margin,
            "is_unknown": True,
            "top_predictions": top_predictions,
        }
    class_row = prediction_classes[top1_id]
    return {
        "prediction_id": top1_id,
        "raw_id": int(class_row["raw_id"]),
        "class_name": str(class_row["name"]),
        "top1_score": top1_score,
        "top1_top2_margin": margin,
        "is_unknown": False,
        "top_predictions": top_predictions,
    }


def load_mask_table(teacher_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    path = teacher_dir / "mask_feature_table.json"
    if not path.exists():
        return {}
    rows = json.load(path.open("r", encoding="utf-8"))
    table: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        if int(row.get("region_id", -1)) >= 0:
            table[(str(row.get("frame_id") or row.get("image_id")), int(row["region_id"]))] = row
    return table


def write_legend(path: Path, prediction_classes: Sequence[Dict[str, Any]], palette: np.ndarray) -> None:
    lines = ["prediction_id\traw_gt_id\tclass_name\tvertex_count\tcolor_rgb"]
    for row, color in zip(prediction_classes, palette):
        lines.append(
            f"{int(row['prediction_id'])}\t{int(row['raw_id'])}\t{row['name']}\t{int(row.get('vertex_count', 0))}\t"
            f"{int(color[0])},{int(color[1])},{int(color[2])}"
        )
    lines.append(f"-1\t-1\tunknown\t0\t{int(UNKNOWN_COLOR[0])},{int(UNKNOWN_COLOR[1])},{int(UNKNOWN_COLOR[2])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def visualize(args: argparse.Namespace) -> Dict[str, Any]:
    from PIL import Image, ImageDraw

    teacher_dir = resolve_repo_path(args.teacher_dir)
    rgb_dir = resolve_repo_path(args.rgb_dir)
    semantic_classes = infer_semantic_classes_path(teacher_dir, str(args.semantic_classes or ""))
    output_dir = Path(args.output_dir) if args.output_dir else teacher_dir / "semantic_visualization_gt_vocab"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlay"
    labeled_dir = output_dir / "overlay_labeled"
    color_dir = output_dir / "semantic_color"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

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
    mask_table = load_mask_table(teacher_dir)
    seg_paths = sorted(teacher_dir.glob("*_s.npy"))
    if int(args.max_images) > 0:
        seg_paths = seg_paths[: int(args.max_images)]
    if not seg_paths:
        raise RuntimeError(f"No *_s.npy files found in {teacher_dir}")

    image_records: List[Dict[str, Any]] = []
    region_records: List[Dict[str, Any]] = []
    instance_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    class_counts: Counter = Counter()
    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    iterator = seg_paths
    try:
        from tqdm import tqdm

        iterator = tqdm(seg_paths, desc="Render teacher semantic overlays")
    except Exception:
        pass

    for seg_path in iterator:
        frame_id = seg_path.name[:-6]
        feature_path = teacher_dir / f"{frame_id}_f.npy"
        image_path = find_image(rgb_dir, frame_id)
        if image_path is None or not feature_path.exists():
            continue
        seg = load_seg_map(seg_path)
        features = np.load(feature_path).astype(np.float32)
        top_ids, top_scores = (
            classify_regions(features, text_features, int(args.topk))
            if features.shape[0] > 0
            else (np.zeros((0, int(args.topk)), dtype=np.int64), np.zeros((0, int(args.topk)), dtype=np.float32))
        )

        image = Image.open(image_path).convert("RGB")
        if image.size != (int(seg.shape[1]), int(seg.shape[0])):
            image = image.resize((int(seg.shape[1]), int(seg.shape[0])), Image.BILINEAR)
        image_rgb = np.asarray(image).astype(np.float32)
        overlay = image_rgb.copy()
        semantic_rgb = np.zeros_like(image_rgb, dtype=np.uint8)
        label_positions: List[Tuple[int, int, str, Tuple[int, int, int]]] = []
        current_regions: List[Dict[str, Any]] = []

        for region_id in range(int(features.shape[0])):
            mask = seg == int(region_id)
            area = int(mask.sum())
            if area <= 0:
                continue
            prediction = format_prediction(
                top_ids[region_id],
                top_scores[region_id],
                prediction_classes,
                float(args.unknown_score_thresh),
                float(args.unknown_margin_thresh),
            )
            color = UNKNOWN_COLOR if prediction["is_unknown"] else palette[int(prediction["prediction_id"])]
            semantic_rgb[mask] = color
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color.astype(np.float32)
            overlay[mask_boundary(mask)] = color.astype(np.float32)
            meta = mask_table.get((frame_id, region_id), {})
            row = {
                "image_id": frame_id,
                "region_id": int(region_id),
                "mask_area": area,
                "mask_area_ratio": float(area) / float(max(seg.size, 1)),
                "instance_id": None if "instance_id" not in meta else int(meta["instance_id"]),
                "mask_id": None if "mask_id" not in meta else int(meta["mask_id"]),
                "quality_level": str(meta.get("quality_level", "")),
                "feature_source": str(meta.get("feature_source", "")),
                "teacher_confidence": float(meta.get("teacher_confidence", 0.0)),
                **prediction,
            }
            current_regions.append(row)
            region_records.append(row)
            class_counts[str(row["class_name"])] += 1
            if row["instance_id"] is not None:
                instance_records[int(row["instance_id"])].append(row)
            if area >= int(args.min_label_area):
                ys, xs = np.nonzero(mask)
                text = f"{row['class_name']} {row['top1_score']:.2f}"
                if row["instance_id"] is not None:
                    text += f" I{row['instance_id']}"
                label_positions.append(
                    (int(xs.min()), int(ys.min()), text, tuple(int(value) for value in color))
                )

        clean_overlay = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        clean_overlay.save(overlay_dir / f"{frame_id}_semantic_overlay.jpg", quality=95)
        labeled = clean_overlay.copy()
        draw = ImageDraw.Draw(labeled)
        for x, y, text, color in label_positions:
            draw_label(draw, (x, y), text, color)
        labeled.save(labeled_dir / f"{frame_id}_semantic_overlay_labeled.jpg", quality=95)
        Image.fromarray(semantic_rgb).save(color_dir / f"{frame_id}_semantic_color.png")
        image_records.append(
            {
                "image_id": frame_id,
                "image_path": str(image_path),
                "num_regions": int(len(current_regions)),
                "teacher_pixel_coverage_ratio": float((seg >= 0).sum()) / float(max(seg.size, 1)),
                "regions": current_regions,
            }
        )

    instance_summary: List[Dict[str, Any]] = []
    for instance_id, rows in sorted(instance_records.items()):
        known = [row for row in rows if not bool(row["is_unknown"])]
        counter = Counter(str(row["class_name"]) for row in known)
        majority, count = counter.most_common(1)[0] if counter else ("unknown", 0)
        instance_summary.append(
            {
                "instance_id": int(instance_id),
                "num_region_observations": int(len(rows)),
                "num_known_region_observations": int(len(known)),
                "majority_class_name": str(majority),
                "semantic_consistency_ratio": float(count) / float(max(len(known), 1)),
                "mean_top1_score": float(np.mean([float(row["top1_score"]) for row in rows])) if rows else 0.0,
                "label_counts": dict(counter),
            }
        )

    summary = {
        "teacher_dir": str(teacher_dir),
        "rgb_dir": str(rgb_dir),
        "semantic_gt_mesh": str(resolve_repo_path(args.semantic_gt_mesh)),
        "semantic_classes": str(semantic_classes),
        "output_dir": str(output_dir),
        "clip_model": str(args.clip_model),
        "alphaclip_checkpoint": str(resolve_repo_path(args.alphaclip_checkpoint)),
        "gt_mesh_classes": gt_info,
        "num_images": int(len(image_records)),
        "num_region_observations": int(len(region_records)),
        "num_instances_with_region_observations": int(len(instance_summary)),
        "class_region_counts": dict(class_counts),
        "unknown_score_thresh": float(args.unknown_score_thresh),
        "unknown_margin_thresh": float(args.unknown_margin_thresh),
        "notes": [
            "The text vocabulary is restricted to semantic classes that appear in the GT mesh.",
            "This is a qualitative teacher-feature visualization, not a geometric GT-vs-pred pixel evaluation.",
            "The same class uses the same color across all rendered images.",
        ],
    }
    save_json(output_dir / "prediction_classes.json", gt_info)
    save_json(output_dir / "per_image_semantic_regions.json", image_records)
    save_json(output_dir / "instance_semantic_consistency.json", instance_summary)
    save_json(output_dir / "summary.json", summary)
    write_legend(output_dir / "legend.tsv", prediction_classes, palette)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Visualize MaskClustering Alpha-CLIP teacher semantics.")
    parser.add_argument("--teacher_dir", required=True)
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--semantic_gt_mesh", required=True)
    parser.add_argument("--semantic_classes", default="")
    parser.add_argument("--alphaclip_package_path", default="third_party/AlphaCLIP-main")
    parser.add_argument("--alphaclip_checkpoint", default="third_party/checkpoints/clip_b16_grit20m_fultune_2xe.pth")
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--label_field", default="")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--text_batch_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.52)
    parser.add_argument("--min_label_area", type=int, default=900)
    parser.add_argument("--unknown_score_thresh", type=float, default=-1.0)
    parser.add_argument("--unknown_margin_thresh", type=float, default=-1.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.label_field == "":
        args.label_field = None
    visualize(args)


if __name__ == "__main__":
    main()
