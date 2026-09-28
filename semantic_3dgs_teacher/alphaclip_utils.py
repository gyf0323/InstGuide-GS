from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class MaskRecord:
    instance_id: int
    frame_id: str
    mask_id: int
    coverage: float
    key: str
    quality_level: str = "low"
    quality_score: float = 0.0
    mask_area: int = 0
    mask_area_ratio: float = 0.0
    is_prototype_candidate: bool = False
    is_emergency_candidate: bool = False
    encoded: bool = False
    semantic_outlier: bool = False
    prototype_similarity: float = 0.0
    crop_full_similarity: float = 0.0
    full_context_weight: float = 0.0
    feature_source: str = "missing"
    teacher_confidence: float = 0.0
    region_id: int = -1
    num_suppressed_foreign_masks: int = 0
    suppressed_foreign_crop_ratio: float = 0.0
    candidate_foreign_crop_ratio: float = 0.0
    suppressed_foreign_mask_ids: str = ""


@dataclass
class InstanceSummary:
    instance_id: int
    num_masks: int
    num_high: int
    num_medium: int
    num_low: int
    num_encoded: int
    num_prototype_seed: int
    num_semantic_outliers: int
    prototype_available: bool
    mean_seed_similarity: float
    min_seed_similarity: float


class SceneCache:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self._rgb: Dict[str, np.ndarray] = {}
        self._seg: Dict[str, np.ndarray] = {}

    def rgb(self, frame_id: str) -> np.ndarray:
        if frame_id not in self._rgb:
            import cv2

            rgb_path, _ = self.dataset.get_frame_path(frame_id)
            image = cv2.imread(rgb_path)
            if image is None:
                raise FileNotFoundError(f"RGB image not found: {rgb_path}")
            self._rgb[frame_id] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._rgb[frame_id]

    def segmentation(self, frame_id: str) -> np.ndarray:
        if frame_id not in self._seg:
            import cv2

            _, seg_path = self.dataset.get_frame_path(frame_id)
            seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            if seg is None:
                raise FileNotFoundError(f"Mask label image not found: {seg_path}")
            rgb = self.rgb(frame_id)
            if seg.shape[:2] != rgb.shape[:2]:
                seg = cv2.resize(seg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            self._seg[frame_id] = seg
        return self._seg[frame_id]

    def mask(self, frame_id: str, mask_id: int) -> np.ndarray:
        return self.segmentation(frame_id) == int(mask_id)

    def image_size(self, frame_id: str) -> Tuple[int, int]:
        image = self.rgb(frame_id)
        return int(image.shape[1]), int(image.shape[0])


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.exists() or path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    return repo_path if repo_path.exists() else path


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / max(float(np.linalg.norm(x)), eps)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def mask_bounds(mask: np.ndarray, padding_ratio: float = 0.08) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = int(math.ceil((x1 - x0) * padding_ratio))
    pad_y = int(math.ceil((y1 - y0) * padding_ratio))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y)


def pad_square_rgb(image: np.ndarray, value: int = 127) -> "Image.Image":
    from PIL import Image

    h, w = image.shape[:2]
    size = max(h, w)
    canvas = np.full((size, size, 3), int(value), dtype=np.uint8)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image
    return Image.fromarray(canvas, mode="RGB")


def pad_square_alpha(mask: np.ndarray) -> "Image.Image":
    from PIL import Image

    h, w = mask.shape[:2]
    size = max(h, w)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = (mask.astype(np.uint8) * 255)
    return Image.fromarray(canvas, mode="L")


def parse_object_dict(object_dict: Dict[Any, Any]) -> Tuple[List[MaskRecord], List[Dict[str, Any]]]:
    best_by_mask: Dict[Tuple[str, int], MaskRecord] = {}
    duplicates: List[Dict[str, Any]] = []
    for instance_id_raw, payload in object_dict.items():
        instance_id = int(instance_id_raw)
        for entry in payload.get("mask_list", []):
            if len(entry) < 2:
                continue
            frame_id = str(entry[0])
            mask_id = int(entry[1])
            coverage = float(entry[2]) if len(entry) > 2 else 0.0
            key_tuple = (frame_id, mask_id)
            record = MaskRecord(
                instance_id=instance_id,
                frame_id=frame_id,
                mask_id=mask_id,
                coverage=coverage,
                key=f"{frame_id}:{mask_id}",
            )
            previous = best_by_mask.get(key_tuple)
            if previous is None:
                best_by_mask[key_tuple] = record
                continue
            if record.coverage > previous.coverage:
                duplicates.append(
                    {
                        "frame_id": frame_id,
                        "mask_id": int(mask_id),
                        "kept_instance_id": int(record.instance_id),
                        "dropped_instance_id": int(previous.instance_id),
                        "kept_coverage": float(record.coverage),
                        "dropped_coverage": float(previous.coverage),
                    }
                )
                best_by_mask[key_tuple] = record
            else:
                duplicates.append(
                    {
                        "frame_id": frame_id,
                        "mask_id": int(mask_id),
                        "kept_instance_id": int(previous.instance_id),
                        "dropped_instance_id": int(record.instance_id),
                        "kept_coverage": float(previous.coverage),
                        "dropped_coverage": float(record.coverage),
                    }
                )
    return sorted(best_by_mask.values(), key=lambda r: (r.frame_id, r.mask_id, r.instance_id)), duplicates


def assign_mask_quality(records: Sequence[MaskRecord], cache: SceneCache, args: argparse.Namespace) -> None:
    min_area = int(args.min_mask_area_pixels)
    high_cov = float(args.high_quality_coverage)
    mid_cov = float(args.medium_quality_coverage)
    for record in records:
        mask = cache.mask(record.frame_id, record.mask_id)
        area = int(mask.sum())
        h, w = mask.shape[:2]
        record.mask_area = area
        record.mask_area_ratio = float(area / max(1, h * w))
        area_valid = area >= min_area
        if area_valid and record.coverage >= high_cov:
            record.quality_level = "high"
        elif area_valid and record.coverage >= mid_cov:
            record.quality_level = "medium"
        else:
            record.quality_level = "low"
        area_score = min(1.0, math.sqrt(area / max(1.0, float(min_area))))
        record.quality_score = float(np.clip(record.coverage * area_score, 0.0, 1.0))


def group_by_instance(records: Sequence[MaskRecord]) -> Dict[int, List[MaskRecord]]:
    grouped: DefaultDict[int, List[MaskRecord]] = defaultdict(list)
    for record in records:
        grouped[int(record.instance_id)].append(record)
    return {k: sorted(v, key=lambda r: (-r.coverage, r.frame_id, r.mask_id)) for k, v in grouped.items()}


def select_prototype_candidates(grouped: Dict[int, List[MaskRecord]], args: argparse.Namespace) -> List[MaskRecord]:
    selected: List[MaskRecord] = []
    emergency = int(args.emergency_candidates_per_instance)
    for instance_records in grouped.values():
        candidates = [r for r in instance_records if r.quality_level in {"high", "medium"}]
        if candidates:
            for r in candidates:
                r.is_prototype_candidate = True
            selected.extend(candidates)
            continue
        for r in instance_records[: max(0, emergency)]:
            r.is_prototype_candidate = True
            r.is_emergency_candidate = True
            selected.append(r)
    return sorted(selected, key=lambda r: (r.frame_id, r.mask_id, r.instance_id))


def full_context_weight(mask: np.ndarray, args: argparse.Namespace) -> float:
    mode = str(args.alphaclip_context_mode).lower()
    if mode == "crop":
        return 0.0
    if mode == "full":
        return 1.0
    if mode != "adaptive":
        raise RuntimeError("--alphaclip_context_mode must be adaptive, crop, or full")
    ratio = float(mask.mean())
    area_min = float(args.alphaclip_full_context_min_area_ratio)
    area_max = float(args.alphaclip_full_context_max_area_ratio)
    weight_min = float(args.alphaclip_full_context_min_weight)
    weight_max = float(args.alphaclip_full_context_max_weight)
    if area_max <= area_min:
        return float(np.clip(weight_max, 0.0, 1.0))
    blend = float(np.clip((ratio - area_min) / (area_max - area_min), 0.0, 1.0))
    return float(np.clip(weight_min + blend * (weight_max - weight_min), 0.0, 1.0))


def suppress_background(image: np.ndarray, target_mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mode = str(args.alphaclip_mask_background_mode).lower()
    if mode == "keep":
        return image
    if mode == "gray":
        background = np.full_like(image, int(args.alphaclip_mask_background_gray))
    elif mode == "blur":
        from PIL import Image, ImageFilter

        background = np.asarray(
            Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=float(args.alphaclip_mask_blur_radius)))
        )
    else:
        raise RuntimeError("--alphaclip_mask_background_mode must be keep, gray, or blur")
    return np.where(target_mask[..., None], image, background).astype(np.uint8, copy=False)


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def foreign_suppression_mask(
    seg: np.ndarray,
    target_mask: np.ndarray,
    target_mask_id: int,
    args: argparse.Namespace,
    padding_ratio: float = 0.08,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if not bool(args.alphaclip_suppress_foreign_masks):
        return None, {
            "candidate_foreign_crop_ratio": 0.0,
            "suppressed_foreign_crop_ratio": 0.0,
            "num_suppressed_foreign_masks": 0,
            "suppressed_foreign_mask_ids": "",
        }
    bounds = mask_bounds(target_mask, padding_ratio=padding_ratio)
    if bounds is None:
        return None, {
            "candidate_foreign_crop_ratio": 0.0,
            "suppressed_foreign_crop_ratio": 0.0,
            "num_suppressed_foreign_masks": 0,
            "suppressed_foreign_mask_ids": "",
        }
    x0, y0, x1, y1 = bounds
    seg_crop = seg[y0:y1, x0:x1]
    target_crop = target_mask[y0:y1, x0:x1]
    crop_h, crop_w = seg_crop.shape[:2]
    crop_area = max(1, int(crop_h * crop_w))
    ids, counts = np.unique(seg_crop, return_counts=True)

    candidates: List[Tuple[float, int, float, bool]] = []
    for raw_id, raw_count in zip(ids, counts):
        mask_id = int(raw_id)
        if mask_id == int(target_mask_id) or mask_id == 0:
            continue
        other_crop = (seg_crop == mask_id) & (~target_crop)
        area = int(other_crop.sum())
        if area <= 0:
            continue
        ratio = float(area / crop_area)
        if ratio < float(args.foreign_min_individual_crop_ratio):
            continue
        bbox = bbox_from_mask(other_crop)
        if bbox is None:
            continue
        bx0, by0, bx1, by1 = bbox
        touches = int(bx0 <= 1) + int(by0 <= 1) + int(bx1 >= crop_w - 1) + int(by1 >= crop_h - 1)
        bbox_w = max(1, bx1 - bx0)
        bbox_h = max(1, by1 - by0)
        broad_boundary_context = bool(
            touches >= 2
            and (
                bbox_w / max(1, crop_w) >= float(args.foreign_support_min_span_ratio)
                or bbox_h / max(1, crop_h) >= float(args.foreign_support_min_span_ratio)
            )
        )
        cx0 = int(crop_w * 0.25)
        cx1 = int(crop_w * 0.75)
        cy0 = int(crop_h * 0.25)
        cy1 = int(crop_h * 0.75)
        center_area = int(other_crop[cy0:cy1, cx0:cx1].sum())
        center_score = float(center_area / max(1, area))
        interior_score = 1.0 if touches == 0 else (0.5 if touches == 1 else 0.0)
        bbox_fill = float(area / max(1, bbox_w * bbox_h))
        compact_score = float(np.clip((bbox_fill - 0.15) / 0.45, 0.0, 1.0))
        support_multiplier = 0.25 if broad_boundary_context else 1.0
        risk = ratio * (1.0 + 0.90 * center_score + 0.75 * interior_score + 0.35 * compact_score)
        risk *= support_multiplier
        candidates.append((risk, mask_id, ratio, broad_boundary_context))

    candidate_ratio = float(sum(row[2] for row in candidates))
    if candidate_ratio < float(args.foreign_total_crop_ratio_trigger):
        return None, {
            "candidate_foreign_crop_ratio": candidate_ratio,
            "suppressed_foreign_crop_ratio": 0.0,
            "num_suppressed_foreign_masks": 0,
            "suppressed_foreign_mask_ids": "",
        }

    over_ratio = max(0.0, candidate_ratio - float(args.foreign_total_crop_ratio_trigger))
    suppress_budget = float(args.foreign_suppress_max_crop_ratio)
    if bool(args.foreign_suppress_over_threshold_only):
        suppress_budget = min(suppress_budget, max(over_ratio, float(args.foreign_min_suppress_crop_ratio)))

    selected: List[int] = []
    suppressed_ratio = 0.0
    max_masks = int(args.foreign_max_suppressed_masks)
    for risk, mask_id, ratio, is_support in sorted(candidates, key=lambda row: row[0], reverse=True):
        if max_masks > 0 and len(selected) >= max_masks:
            break
        if is_support and risk < float(args.foreign_support_suppress_risk_threshold):
            continue
        if risk < float(args.foreign_risk_threshold):
            continue
        selected.append(mask_id)
        suppressed_ratio += ratio
        if suppressed_ratio >= suppress_budget:
            break

    if not selected:
        return None, {
            "candidate_foreign_crop_ratio": candidate_ratio,
            "suppressed_foreign_crop_ratio": 0.0,
            "num_suppressed_foreign_masks": 0,
            "suppressed_foreign_mask_ids": "",
        }
    distractor = np.isin(seg, np.asarray(selected, dtype=seg.dtype)) & (~target_mask)
    return distractor, {
        "candidate_foreign_crop_ratio": candidate_ratio,
        "suppressed_foreign_crop_ratio": float(min(1.0, suppressed_ratio)),
        "num_suppressed_foreign_masks": int(len(selected)),
        "suppressed_foreign_mask_ids": ",".join(str(x) for x in selected[:32]),
    }


def apply_foreign_suppression(
    image: np.ndarray,
    target_mask: np.ndarray,
    distractor_mask: Optional[np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    image = suppress_background(image, target_mask, args)
    if distractor_mask is None or not bool(distractor_mask.any()):
        return image
    mode = str(args.foreign_suppress_mode).lower()
    if mode == "none":
        return image
    keep = float(np.clip(args.foreign_soft_keep_ratio, 0.0, 1.0))
    gray_value = int(np.clip(args.alphaclip_mask_background_gray, 0, 255))
    if mode in {"hard_gray", "gray"}:
        replacement = np.full_like(image, gray_value)
        keep = 0.0
    elif mode == "soft_gray":
        replacement = np.full_like(image, gray_value)
    elif mode in {"blur", "soft_blur"}:
        from PIL import Image, ImageFilter

        replacement = np.asarray(
            Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=float(args.foreign_blur_radius)))
        )
        if mode == "blur":
            keep = 0.0
    else:
        raise RuntimeError("--foreign_suppress_mode must be none, hard_gray, soft_gray, blur, or soft_blur")
    blended = (keep * image.astype(np.float32) + (1.0 - keep) * replacement.astype(np.float32)).clip(0, 255)
    result = image.copy()
    result[distractor_mask] = blended.astype(np.uint8)[distractor_mask]
    return result


def prepare_pair(
    record: MaskRecord,
    cache: SceneCache,
    preprocess: Any,
    alpha_transform: Any,
    args: argparse.Namespace,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    image = cache.rgb(record.frame_id)
    seg = cache.segmentation(record.frame_id)
    target_mask = seg == int(record.mask_id)
    bounds = mask_bounds(target_mask, padding_ratio=float(args.crop_padding_ratio))
    if bounds is None:
        return None
    distractor, stats = foreign_suppression_mask(
        seg,
        target_mask,
        int(record.mask_id),
        args,
        padding_ratio=float(args.crop_padding_ratio),
    )
    record.candidate_foreign_crop_ratio = float(stats["candidate_foreign_crop_ratio"])
    record.suppressed_foreign_crop_ratio = float(stats["suppressed_foreign_crop_ratio"])
    record.num_suppressed_foreign_masks = int(stats["num_suppressed_foreign_masks"])
    record.suppressed_foreign_mask_ids = str(stats["suppressed_foreign_mask_ids"])

    processed = apply_foreign_suppression(image, target_mask, distractor, args)
    x0, y0, x1, y1 = bounds
    crop_rgb = processed[y0:y1, x0:x1]
    crop_alpha = target_mask[y0:y1, x0:x1]
    if crop_rgb.size == 0 or not bool(crop_alpha.any()):
        return None

    crop_image = pad_square_rgb(crop_rgb, value=int(args.alphaclip_mask_background_gray))
    crop_mask = pad_square_alpha(crop_alpha)
    full_image = pad_square_rgb(processed, value=int(args.alphaclip_mask_background_gray))
    full_mask = pad_square_alpha(target_mask)
    record.full_context_weight = full_context_weight(target_mask, args)
    return (
        preprocess(crop_image),
        alpha_transform(crop_mask),
        preprocess(full_image),
        alpha_transform(full_mask),
    )


def load_alphaclip(args: argparse.Namespace) -> Tuple[Any, Any, Any, torch.device]:
    package_path = str(args.alphaclip_package_path or "").strip()
    if package_path:
        resolved = str(resolve_repo_path(package_path).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    import alpha_clip
    from torchvision import transforms

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = resolve_repo_path(str(args.alphaclip_checkpoint))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Alpha-CLIP checkpoint not found: {checkpoint}")
    model, preprocess = alpha_clip.load(
        args.clip_model,
        alpha_vision_ckpt_pth=str(checkpoint),
        device=device,
    )
    model.eval()
    if bool(args.fp16) and device.type == "cuda":
        model.half()
    input_resolution = int(getattr(model.visual, "input_resolution", 224))
    alpha_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((input_resolution, input_resolution)),
            transforms.Normalize(0.5, 0.26),
        ]
    )
    return model, preprocess, alpha_transform, device


def encode_batch(
    model: Any,
    images: Sequence[torch.Tensor],
    alphas: Sequence[torch.Tensor],
    device: torch.device,
    fp16: bool,
) -> np.ndarray:
    image_tensor = torch.stack(list(images), dim=0).to(device)
    alpha_tensor = torch.stack(list(alphas), dim=0).to(device)
    if fp16 and device.type == "cuda":
        image_tensor = image_tensor.half()
        alpha_tensor = alpha_tensor.half()
    with torch.no_grad():
        if hasattr(model, "encode_image"):
            features = model.encode_image(image_tensor, alpha_tensor)
        else:
            features = model.visual(image_tensor, alpha_tensor)
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return features.cpu().numpy().astype(np.float32)


def extract_alphaclip_features(
    records: Sequence[MaskRecord],
    cache: SceneCache,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    model, preprocess, alpha_transform, device = load_alphaclip(args)
    feature_by_key: Dict[str, np.ndarray] = {}
    crop_batch: List[torch.Tensor] = []
    crop_alpha_batch: List[torch.Tensor] = []
    full_batch: List[torch.Tensor] = []
    full_alpha_batch: List[torch.Tensor] = []
    batch_records: List[MaskRecord] = []

    def flush() -> None:
        if not batch_records:
            return
        crop_features = encode_batch(model, crop_batch, crop_alpha_batch, device, bool(args.fp16))
        full_features = encode_batch(model, full_batch, full_alpha_batch, device, bool(args.fp16))
        for i, record in enumerate(batch_records):
            crop_feat = crop_features[i]
            full_feat = full_features[i]
            record.crop_full_similarity = float(np.dot(crop_feat, full_feat))
            weight = float(record.full_context_weight)
            feature = l2_normalize((1.0 - weight) * crop_feat + weight * full_feat)
            feature_by_key[record.key] = feature.astype(np.float32)
            record.encoded = True
        crop_batch.clear()
        crop_alpha_batch.clear()
        full_batch.clear()
        full_alpha_batch.clear()
        batch_records.clear()

    iterator = records
    try:
        from tqdm import tqdm

        iterator = tqdm(records, desc="Alpha-CLIP masks")
    except Exception:
        pass

    for record in iterator:
        prepared = prepare_pair(record, cache, preprocess, alpha_transform, args)
        if prepared is None:
            continue
        crop_img, crop_alpha, full_img, full_alpha = prepared
        crop_batch.append(crop_img)
        crop_alpha_batch.append(crop_alpha)
        full_batch.append(full_img)
        full_alpha_batch.append(full_alpha)
        batch_records.append(record)
        if len(batch_records) >= int(args.batch_size):
            flush()
    flush()
    return feature_by_key


def build_instance_prototypes(
    grouped: Dict[int, List[MaskRecord]],
    feature_by_key: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[Dict[int, np.ndarray], List[InstanceSummary]]:
    prototypes: Dict[int, np.ndarray] = {}
    summaries: List[InstanceSummary] = []
    min_outlier_masks = int(args.semantic_outlier_min_masks)
    outlier_threshold = float(args.semantic_outlier_similarity_threshold)
    min_seed_masks = int(args.min_prototype_seed_masks)

    for instance_id, records in sorted(grouped.items()):
        encoded = [r for r in records if r.is_prototype_candidate and r.key in feature_by_key]
        high = sum(1 for r in records if r.quality_level == "high")
        medium = sum(1 for r in records if r.quality_level == "medium")
        low = sum(1 for r in records if r.quality_level == "low")
        if not encoded:
            summaries.append(
                InstanceSummary(
                    instance_id=instance_id,
                    num_masks=len(records),
                    num_high=high,
                    num_medium=medium,
                    num_low=low,
                    num_encoded=0,
                    num_prototype_seed=0,
                    num_semantic_outliers=0,
                    prototype_available=False,
                    mean_seed_similarity=0.0,
                    min_seed_similarity=0.0,
                )
            )
            continue

        features = np.stack([feature_by_key[r.key] for r in encoded], axis=0)
        weights = np.asarray([max(0.05, float(r.quality_score)) for r in encoded], dtype=np.float32)
        initial = l2_normalize((features * weights[:, None]).sum(axis=0))
        sims = features @ initial
        keep = np.ones((len(encoded),), dtype=bool)
        if len(encoded) >= min_outlier_masks:
            keep = sims >= outlier_threshold
            if int(keep.sum()) < min_seed_masks:
                keep = np.zeros((len(encoded),), dtype=bool)
                for idx in np.argsort(-sims)[: max(1, min_seed_masks)]:
                    keep[int(idx)] = True
        seed_features = features[keep]
        seed_weights = weights[keep]
        prototype = l2_normalize((seed_features * seed_weights[:, None]).sum(axis=0))
        prototypes[instance_id] = prototype.astype(np.float32)
        final_sims = features @ prototype
        outlier_count = 0
        for i, record in enumerate(encoded):
            record.prototype_similarity = float(final_sims[i])
            if not bool(keep[i]):
                record.semantic_outlier = True
                outlier_count += 1
        summaries.append(
            InstanceSummary(
                instance_id=instance_id,
                num_masks=len(records),
                num_high=high,
                num_medium=medium,
                num_low=low,
                num_encoded=len(encoded),
                num_prototype_seed=int(keep.sum()),
                num_semantic_outliers=outlier_count,
                prototype_available=True,
                mean_seed_similarity=float(np.mean(final_sims[keep])) if int(keep.sum()) > 0 else 0.0,
                min_seed_similarity=float(np.min(final_sims[keep])) if int(keep.sum()) > 0 else 0.0,
            )
        )
    return prototypes, summaries


def assign_final_features(
    grouped: Dict[int, List[MaskRecord]],
    feature_by_key: Dict[str, np.ndarray],
    prototypes: Dict[int, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    final_features: Dict[str, np.ndarray] = {}
    for instance_id, records in grouped.items():
        prototype = prototypes.get(int(instance_id))
        for record in records:
            own_feature = feature_by_key.get(record.key)
            if own_feature is not None and not record.semantic_outlier and record.quality_level in {"high", "medium"}:
                final_features[record.key] = own_feature.astype(np.float32)
                record.feature_source = "own_alphaclip"
                confidence = record.quality_score
            elif prototype is not None:
                final_features[record.key] = prototype.astype(np.float32)
                if record.semantic_outlier:
                    record.feature_source = "prototype_semantic_outlier"
                    confidence = float(args.semantic_outlier_teacher_confidence)
                elif record.quality_level == "low":
                    record.feature_source = "prototype_low_quality"
                    confidence = float(args.low_quality_teacher_confidence)
                elif own_feature is None:
                    record.feature_source = "prototype_missing_feature"
                    confidence = float(args.low_quality_teacher_confidence)
                else:
                    record.feature_source = "prototype_fallback"
                    confidence = float(args.low_quality_teacher_confidence)
            else:
                record.feature_source = "missing_no_prototype"
                confidence = 0.0
                continue
            if record.quality_level == "high" and record.feature_source == "own_alphaclip":
                confidence = max(confidence, float(args.high_quality_teacher_confidence))
            elif record.quality_level == "medium" and record.feature_source == "own_alphaclip":
                confidence = max(confidence, float(args.medium_quality_teacher_confidence))
            record.teacher_confidence = float(np.clip(confidence, 0.0, 1.0))
    return final_features


def frame_sort_key(frame_id: str) -> Tuple[str, int, str]:
    digits = "".join(ch for ch in frame_id if ch.isdigit())
    return ("".join(ch for ch in frame_id if not ch.isdigit()), int(digits) if digits else -1, frame_id)


def write_teacher_files(
    records: Sequence[MaskRecord],
    final_features: Dict[str, np.ndarray],
    cache: SceneCache,
    output_dir: Path,
    feature_dim: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    region_dir = output_dir / "region_teachers"
    region_dir.mkdir(parents=True, exist_ok=True)

    by_frame: DefaultDict[str, List[MaskRecord]] = defaultdict(list)
    for record in records:
        if record.key in final_features:
            by_frame[record.frame_id].append(record)

    assignments: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for frame_id in sorted(by_frame.keys(), key=frame_sort_key):
        width, height = cache.image_size(frame_id)
        seg = cache.segmentation(frame_id)
        region_id_map = np.full((height, width), -1, dtype=np.int32)
        pixel_weight = np.zeros((height, width), dtype=np.float32)
        features: List[np.ndarray] = []
        crop_names: List[str] = []
        written_records: List[MaskRecord] = []
        frame_records = sorted(by_frame[frame_id], key=lambda r: (-r.teacher_confidence, -r.mask_area, r.mask_id))
        for record in frame_records:
            feature = final_features.get(record.key)
            if feature is None:
                continue
            mask = seg == int(record.mask_id)
            if not bool(mask.any()):
                continue
            rid = len(features)
            record.region_id = rid
            features.append(feature.astype(np.float32))
            crop_names.append(record.key)
            written_records.append(record)
            region_id_map[mask] = rid
            pixel_weight[mask] = float(record.teacher_confidence)
            row = asdict(record)
            row.update({"image_id": frame_id, "feature_dim": int(feature.shape[0])})
            rows.append(row)
        feat_arr = (
            np.stack(features, axis=0).astype(np.float32)
            if features
            else np.zeros((0, feature_dim), dtype=np.float32)
        )
        np.save(output_dir / f"{frame_id}_s.npy", region_id_map[None, ...])
        np.save(output_dir / f"{frame_id}_f.npy", feat_arr)
        np.save(output_dir / f"{frame_id}_w.npy", pixel_weight[None, ...])
        payload = {
            "image_name": frame_id,
            "region_id_map": torch.from_numpy(region_id_map),
            "region_features": torch.from_numpy(feat_arr),
            "fused_feature": torch.from_numpy(feat_arr),
            "region_area": torch.bincount(
                torch.from_numpy(region_id_map[region_id_map >= 0]).long(),
                minlength=feat_arr.shape[0],
            ).float()
            if feat_arr.shape[0] > 0
            else torch.zeros((0,), dtype=torch.float32),
            "region_quality": torch.tensor(
                [float(r.teacher_confidence) for r in written_records],
                dtype=torch.float32,
            ),
            "pixel_quality": torch.from_numpy(pixel_weight),
            "clip_confidence": torch.zeros((feat_arr.shape[0],), dtype=torch.float32),
            "has_clip_confidence": False,
            "crop_names": crop_names,
            "config": {
                "teacher_source": "maskclustering_alphaclip_teacher",
                "feature_backend": "alphaclip",
                "feature_space": "per_mask_with_instance_prototype_fallback",
                "feature_dim": int(feature_dim),
            },
        }
        torch.save(payload, region_dir / f"{frame_id}.pt")
        assignments[frame_id] = {
            "num_regions": int(feat_arr.shape[0]),
            "feature_shape": list(feat_arr.shape),
            "seg_shape": list(region_id_map[None, ...].shape),
            "weight_shape": list(pixel_weight[None, ...].shape),
        }
    save_json(output_dir / "mask_feature_table.json", rows)
    save_json(output_dir / "teacher_assignments.json", assignments)
    return {"num_images": len(assignments), "num_regions": len(rows), "assignments": assignments}


def build_dataset_from_args(args: argparse.Namespace) -> Tuple[Any, argparse.Namespace]:
    from semantic_3dgs_teacher.config import build_dataset_from_cli

    return build_dataset_from_cli(args, config_dir=str(getattr(args, "config_dir", "")))


def default_object_dict_path(dataset: Any, config: str) -> Path:
    return Path(dataset.object_dict_dir) / config / "object_dict.npy"


def resolve_output_dir(dataset: Any, config: str, output_dir: str) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path(dataset.root) / "output" / "semantic_3dgs_teacher" / config


def assert_output_safe(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise RuntimeError(f"Output directory is not empty: {path}. Pass --force or choose --output_dir.")
    if path.exists() and force:
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset, ds_args = build_dataset_from_args(args)
    output_dir = resolve_output_dir(dataset, args.config, args.output_dir)
    assert_output_safe(output_dir, bool(args.force))

    object_path = resolve_repo_path(args.object_dict_path) if args.object_dict_path else default_object_dict_path(dataset, args.config)
    if not object_path.exists():
        raise FileNotFoundError(f"object_dict.npy not found: {object_path}")
    object_dict = np.load(object_path, allow_pickle=True).item()
    records, duplicates = parse_object_dict(object_dict)
    cache = SceneCache(dataset)
    assign_mask_quality(records, cache, args)
    grouped = group_by_instance(records)
    selected = select_prototype_candidates(grouped, args)

    feature_by_key = extract_alphaclip_features(selected, cache, args)
    prototypes, instance_summaries = build_instance_prototypes(grouped, feature_by_key, args)
    final_features = assign_final_features(grouped, feature_by_key, prototypes, args)
    feature_dim = int(args.semantic_feature_dim)
    if final_features:
        feature_dim = int(next(iter(final_features.values())).shape[0])
    write_summary = write_teacher_files(records, final_features, cache, output_dir, feature_dim, args)

    mask_counts = {
        "total": len(records),
        "high": sum(1 for r in records if r.quality_level == "high"),
        "medium": sum(1 for r in records if r.quality_level == "medium"),
        "low": sum(1 for r in records if r.quality_level == "low"),
        "prototype_candidates": sum(1 for r in records if r.is_prototype_candidate),
        "encoded": sum(1 for r in records if r.encoded),
        "semantic_outliers": sum(1 for r in records if r.semantic_outlier),
        "final_features": len(final_features),
    }
    summary = {
        "seq_name": args.seq_name,
        "config": args.config,
        "dataset": getattr(ds_args, "dataset", ""),
        "scene_root": str(getattr(dataset, "root", "")),
        "object_dict_path": str(object_path),
        "output_dir": str(output_dir),
        "feature_backend": "alphaclip",
        "clip_model": str(args.clip_model),
        "alphaclip_checkpoint": str(args.alphaclip_checkpoint),
        "alphaclip_context_mode": str(args.alphaclip_context_mode),
        "alphaclip_mask_background_mode": str(args.alphaclip_mask_background_mode),
        "foreign_suppress_mode": str(args.foreign_suppress_mode),
        "semantic_feature_dim": int(feature_dim),
        "recommended_feature_level": 0,
        "mask_counts": mask_counts,
        "num_instances": len(grouped),
        "num_instances_with_prototype": len(prototypes),
        "num_duplicate_mask_bindings": len(duplicates),
        "teacher_files": {
            "num_images": int(write_summary["num_images"]),
            "num_regions": int(write_summary["num_regions"]),
            "format": "per-frame region files: *_s.npy, *_f.npy, *_w.npy",
        },
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "teacher_config.json", vars(args))
    save_json(output_dir / "instance_semantic_summary.json", [asdict(x) for x in instance_summaries])
    save_json(output_dir / "duplicate_mask_bindings.json", duplicates)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Alpha-CLIP semantic teacher files from MaskClustering object_dict.npy.")
    parser.add_argument("--config", default="scannetpp_dslr")
    parser.add_argument("--seq_name", required=True)
    parser.add_argument("--object_dict_path", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--scannetpp_dslr_root", default="")
    parser.add_argument("--scannetpp_dslr_mesh_name", default="")
    parser.add_argument("--scannetpp_dslr_depth_scale", type=float, default=None)

    parser.add_argument("--alphaclip_package_path", default="third_party/AlphaCLIP-main")
    parser.add_argument("--alphaclip_checkpoint", default="third_party/checkpoints/clip_b16_grit20m_fultune_2xe.pth")
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--semantic_feature_dim", type=int, default=512)

    parser.add_argument("--medium_quality_coverage", type=float, default=0.20)
    parser.add_argument("--high_quality_coverage", type=float, default=0.45)
    parser.add_argument("--min_mask_area_pixels", type=int, default=64)
    parser.add_argument("--emergency_candidates_per_instance", type=int, default=1)

    parser.add_argument("--alphaclip_context_mode", default="adaptive", choices=["adaptive", "crop", "full"])
    parser.add_argument("--alphaclip_full_context_min_area_ratio", type=float, default=0.05)
    parser.add_argument("--alphaclip_full_context_max_area_ratio", type=float, default=0.35)
    parser.add_argument("--alphaclip_full_context_min_weight", type=float, default=0.10)
    parser.add_argument("--alphaclip_full_context_max_weight", type=float, default=0.35)
    parser.add_argument("--alphaclip_mask_background_mode", default="keep", choices=["keep", "gray", "blur"])
    parser.add_argument("--alphaclip_mask_background_gray", type=int, default=127)
    parser.add_argument("--alphaclip_mask_blur_radius", type=float, default=12.0)
    parser.add_argument("--crop_padding_ratio", type=float, default=0.08)

    parser.add_argument("--alphaclip_suppress_foreign_masks", action="store_true", default=True)
    parser.add_argument("--no_alphaclip_suppress_foreign_masks", dest="alphaclip_suppress_foreign_masks", action="store_false")
    parser.add_argument("--foreign_suppress_mode", default="soft_blur", choices=["none", "hard_gray", "gray", "soft_gray", "blur", "soft_blur"])
    parser.add_argument("--foreign_soft_keep_ratio", type=float, default=0.40)
    parser.add_argument("--foreign_blur_radius", type=float, default=12.0)
    parser.add_argument("--foreign_total_crop_ratio_trigger", type=float, default=0.20)
    parser.add_argument("--foreign_min_individual_crop_ratio", type=float, default=0.015)
    parser.add_argument("--foreign_min_suppress_crop_ratio", type=float, default=0.05)
    parser.add_argument("--foreign_suppress_max_crop_ratio", type=float, default=0.40)
    parser.add_argument("--foreign_suppress_over_threshold_only", action="store_true", default=True)
    parser.add_argument("--foreign_risk_threshold", type=float, default=0.04)
    parser.add_argument("--foreign_support_min_span_ratio", type=float, default=0.60)
    parser.add_argument("--foreign_support_suppress_risk_threshold", type=float, default=0.20)
    parser.add_argument("--foreign_max_suppressed_masks", type=int, default=8)

    parser.add_argument("--semantic_outlier_min_masks", type=int, default=3)
    parser.add_argument("--semantic_outlier_similarity_threshold", type=float, default=0.70)
    parser.add_argument("--min_prototype_seed_masks", type=int, default=1)
    parser.add_argument("--low_quality_teacher_confidence", type=float, default=0.35)
    parser.add_argument("--semantic_outlier_teacher_confidence", type=float, default=0.45)
    parser.add_argument("--medium_quality_teacher_confidence", type=float, default=0.70)
    parser.add_argument("--high_quality_teacher_confidence", type=float, default=0.90)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
