from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_3dgs_teacher.alphaclip_utils import (  # noqa: E402
    MaskRecord,
    SceneCache,
    assign_mask_quality,
    build_dataset_from_args,
    default_object_dict_path,
    encode_batch,
    group_by_instance,
    l2_normalize,
    load_alphaclip,
    mask_bounds,
    pad_square_alpha,
    pad_square_rgb,
    parse_object_dict,
    prepare_pair,
    resolve_repo_path,
    save_json,
    write_teacher_files,
)


def output_dir_for(dataset: Any, config: str, output_dir: str) -> Path:
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


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_pair_from_arrays(
    rgb: np.ndarray,
    alpha: np.ndarray,
    preprocess: Any,
    alpha_transform: Any,
    gray_value: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = pad_square_rgb(rgb.astype(np.uint8, copy=False), value=int(gray_value))
    mask = pad_square_alpha(alpha.astype(bool, copy=False))
    return preprocess(image), alpha_transform(mask)


def crop_area_ratio(frame_mask: np.ndarray, padding_ratio: float) -> float:
    bounds = mask_bounds(frame_mask, padding_ratio=padding_ratio)
    if bounds is None:
        return 0.0
    x0, y0, x1, y1 = bounds
    h, w = frame_mask.shape[:2]
    return float(((x1 - x0) * (y1 - y0)) / max(1, h * w))


def large_context_masks(
    seg: np.ndarray,
    target_mask: np.ndarray,
    target_mask_id: int,
    crop_bounds: Tuple[int, int, int, int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    x0, y0, x1, y1 = crop_bounds
    seg_crop = seg[y0:y1, x0:x1]
    target_crop = target_mask[y0:y1, x0:x1]
    crop_h, crop_w = seg_crop.shape[:2]
    crop_area = max(1, int(crop_h * crop_w))
    suppress_crop = np.zeros_like(target_crop, dtype=bool)
    rows: List[Dict[str, Any]] = []

    for raw_id in np.unique(seg_crop):
        mask_id = int(raw_id)
        if mask_id == int(target_mask_id) or mask_id == 0:
            continue
        other = (seg_crop == mask_id) & (~target_crop)
        area = int(other.sum())
        if area <= 0:
            continue
        ratio = float(area / crop_area)
        if ratio < float(args.large_context_min_individual_crop_ratio):
            continue
        bbox = bbox_from_mask(other)
        if bbox is None:
            continue
        bx0, by0, bx1, by1 = bbox
        bbox_w = max(1, bx1 - bx0)
        bbox_h = max(1, by1 - by0)
        touches = int(bx0 <= 1) + int(by0 <= 1) + int(bx1 >= crop_w - 1) + int(by1 >= crop_h - 1)
        span_x = float(bbox_w / max(1, crop_w))
        span_y = float(bbox_h / max(1, crop_h))
        bbox_fill = float(area / max(1, bbox_w * bbox_h))

        cx0 = int(crop_w * 0.25)
        cx1 = int(crop_w * 0.75)
        cy0 = int(crop_h * 0.25)
        cy1 = int(crop_h * 0.75)
        center_area = int(other[cy0:cy1, cx0:cx1].sum())
        center_score = float(center_area / max(1, area))
        interior_score = 1.0 if touches == 0 else (0.55 if touches == 1 else 0.0)
        compact_score = float(np.clip((bbox_fill - 0.12) / 0.50, 0.0, 1.0))
        support_context = bool(
            touches >= int(args.large_support_min_edge_touches)
            and max(span_x, span_y) >= float(args.large_support_min_span_ratio)
        )
        support_multiplier = float(args.large_support_risk_multiplier) if support_context else 1.0
        risk = ratio * (1.0 + 1.25 * center_score + 1.10 * interior_score + 0.45 * compact_score)
        risk *= support_multiplier
        suppress = bool(
            risk >= float(args.large_distractor_risk_threshold)
            or (
                interior_score > 0.0
                and center_score >= float(args.large_center_score_threshold)
                and ratio >= float(args.large_center_min_crop_ratio)
            )
        )
        if support_context and risk < float(args.large_support_suppress_risk_threshold):
            suppress = False
        if suppress:
            suppress_crop |= other
        rows.append(
            {
                "mask_id": mask_id,
                "area": area,
                "crop_ratio": ratio,
                "risk": float(risk),
                "suppress": bool(suppress),
                "support_context": bool(support_context),
                "touches": int(touches),
                "span_x": span_x,
                "span_y": span_y,
                "center_score": center_score,
                "interior_score": interior_score,
                "compact_score": compact_score,
                "bbox_fill": bbox_fill,
            }
        )

    suppress = np.zeros_like(target_mask, dtype=bool)
    suppress[y0:y1, x0:x1] = suppress_crop
    unknown = (seg == 0) & (~target_mask)
    return suppress, unknown, rows


def compact_unknown_mask(
    unknown: np.ndarray,
    crop_bounds: Tuple[int, int, int, int],
    args: argparse.Namespace,
) -> np.ndarray:
    import cv2

    x0, y0, x1, y1 = crop_bounds
    crop = unknown[y0:y1, x0:x1].astype(np.uint8)
    if int(crop.sum()) == 0:
        return np.zeros_like(unknown, dtype=bool)
    crop_h, crop_w = crop.shape[:2]
    crop_area = max(1, int(crop_h * crop_w))
    labels_count, labels = cv2.connectedComponents(crop, connectivity=8)
    suppress_crop = np.zeros_like(crop, dtype=bool)
    for label_id in range(1, int(labels_count)):
        comp = labels == label_id
        area = int(comp.sum())
        ratio = float(area / crop_area)
        if ratio < float(args.large_unknown_min_crop_ratio):
            continue
        bbox = bbox_from_mask(comp)
        if bbox is None:
            continue
        bx0, by0, bx1, by1 = bbox
        touches = int(bx0 <= 1) + int(by0 <= 1) + int(bx1 >= crop_w - 1) + int(by1 >= crop_h - 1)
        span_x = float((bx1 - bx0) / max(1, crop_w))
        span_y = float((by1 - by0) / max(1, crop_h))
        broad_support = bool(
            touches >= int(args.large_unknown_support_min_edge_touches)
            and max(span_x, span_y) >= float(args.large_unknown_support_min_span_ratio)
        )
        if not broad_support:
            suppress_crop |= comp

    suppress = np.zeros_like(unknown, dtype=bool)
    suppress[y0:y1, x0:x1] = suppress_crop
    return suppress


def inpaint_regions(image: np.ndarray, replace_mask: np.ndarray, radius: float) -> np.ndarray:
    import cv2

    if not bool(replace_mask.any()):
        return image.copy()
    mask = replace_mask.astype(np.uint8) * 255
    return cv2.inpaint(image.astype(np.uint8, copy=False), mask, float(radius), cv2.INPAINT_TELEA)


def inpaint_highrisk_unknown_pair(
    image: np.ndarray,
    seg: np.ndarray,
    target_mask: np.ndarray,
    target_mask_id: int,
    args: argparse.Namespace,
    preprocess: Any,
    alpha_transform: Any,
) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    bounds = mask_bounds(target_mask, padding_ratio=float(args.crop_padding_ratio))
    if bounds is None:
        return None, {
            "num_suppressed_context_masks": 0,
            "suppressed_context_crop_ratio": 0.0,
            "unknown_compact_crop_ratio": 0.0,
            "candidate_context_masks": [],
        }
    x0, y0, x1, y1 = bounds
    suppress, unknown, context_rows = large_context_masks(seg, target_mask, int(target_mask_id), bounds, args)
    unknown_compact = compact_unknown_mask(unknown, bounds, args)
    replace_mask = suppress | unknown_compact
    processed = inpaint_regions(image, replace_mask, float(args.large_inpaint_radius))

    crop_area = max(1, int((x1 - x0) * (y1 - y0)))
    meta = {
        "num_suppressed_context_masks": int(sum(1 for row in context_rows if row["suppress"])),
        "suppressed_context_crop_ratio": float(suppress[y0:y1, x0:x1].sum() / crop_area),
        "unknown_compact_crop_ratio": float(unknown_compact[y0:y1, x0:x1].sum() / crop_area),
        "candidate_context_masks": context_rows,
    }
    return (
        pad_pair_from_arrays(
            processed[y0:y1, x0:x1],
            target_mask[y0:y1, x0:x1],
            preprocess,
            alpha_transform,
            int(args.alphaclip_mask_background_gray),
        ),
        meta,
    )


def choose_base_strategy(
    record: MaskRecord,
    cache: SceneCache,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    target_mask = cache.mask(record.frame_id, int(record.mask_id))
    crop_ratio = crop_area_ratio(target_mask, float(args.crop_padding_ratio))
    is_large_crop = bool(
        crop_ratio >= float(args.tiered_large_crop_ratio)
        or float(record.mask_area_ratio) >= float(args.tiered_large_mask_area_ratio)
    )
    strategy = "inpaint_highrisk_unknown" if is_large_crop else "current_teacher"
    return {
        "key": record.key,
        "frame_id": record.frame_id,
        "mask_id": int(record.mask_id),
        "instance_id": int(record.instance_id),
        "base_strategy": strategy,
        "crop_area_ratio": crop_ratio,
        "mask_area_ratio": float(record.mask_area_ratio),
        "is_large_crop": is_large_crop,
        "context": {},
    }


def encode_base_features(
    records: Sequence[MaskRecord],
    cache: SceneCache,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    model, preprocess, alpha_transform, device = load_alphaclip(args)
    base_infos = {record.key: choose_base_strategy(record, cache, args) for record in records}

    current_refs: List[MaskRecord] = []
    current_crop_images: List[torch.Tensor] = []
    current_crop_alphas: List[torch.Tensor] = []
    current_full_images: List[torch.Tensor] = []
    current_full_alphas: List[torch.Tensor] = []
    direct_refs: List[Tuple[str, str]] = []
    direct_images: List[torch.Tensor] = []
    direct_alphas: List[torch.Tensor] = []

    def batch_iterator(count: int, desc: str) -> Any:
        starts = range(0, count, int(args.batch_size))
        try:
            from tqdm import tqdm

            total = (count + int(args.batch_size) - 1) // int(args.batch_size)
            return tqdm(starts, total=total, desc=desc, ascii=True, dynamic_ncols=True)
        except Exception:
            return starts

    iterator = records
    try:
        from tqdm import tqdm

        iterator = tqdm(records, desc="Prepare tiered Alpha-CLIP inputs", ascii=True, dynamic_ncols=True)
    except Exception:
        pass

    for record in iterator:
        info = base_infos[record.key]
        strategy = str(info["base_strategy"])
        if strategy == "current_teacher":
            pair = prepare_pair(record, cache, preprocess, alpha_transform, args)
            if pair is None:
                continue
            crop_image, crop_alpha, full_image, full_alpha = pair
            current_refs.append(record)
            current_crop_images.append(crop_image)
            current_crop_alphas.append(crop_alpha)
            current_full_images.append(full_image)
            current_full_alphas.append(full_alpha)
            continue

        image = cache.rgb(record.frame_id)
        seg = cache.segmentation(record.frame_id)
        target_mask = seg == int(record.mask_id)
        if strategy == "inpaint_highrisk_unknown":
            pair, inpaint_meta = inpaint_highrisk_unknown_pair(
                image,
                seg,
                target_mask,
                int(record.mask_id),
                args,
                preprocess,
                alpha_transform,
            )
            info["context"] = inpaint_meta
        else:
            raise RuntimeError(f"Unsupported base strategy: {strategy}")
        if pair is None:
            continue
        direct_refs.append((record.key, strategy))
        direct_images.append(pair[0])
        direct_alphas.append(pair[1])

    feature_by_key: Dict[str, np.ndarray] = {}
    for start in batch_iterator(len(current_refs), "Encode current-context masks"):
        end = min(len(current_refs), start + int(args.batch_size))
        crop_features = encode_batch(
            model,
            current_crop_images[start:end],
            current_crop_alphas[start:end],
            device,
            bool(args.fp16),
        )
        full_features = encode_batch(
            model,
            current_full_images[start:end],
            current_full_alphas[start:end],
            device,
            bool(args.fp16),
        )
        for i, record in enumerate(current_refs[start:end]):
            weight = float(record.full_context_weight)
            feature = l2_normalize((1.0 - weight) * crop_features[i] + weight * full_features[i])
            feature_by_key[record.key] = feature.astype(np.float32)
            record.encoded = True

    for start in batch_iterator(len(direct_refs), "Encode inpainted large masks"):
        end = min(len(direct_refs), start + int(args.batch_size))
        features = encode_batch(
            model,
            direct_images[start:end],
            direct_alphas[start:end],
            device,
            bool(args.fp16),
        )
        for i, feature in enumerate(features):
            key, _ = direct_refs[start + i]
            feature_by_key[key] = feature.astype(np.float32)

    for record in records:
        if record.key in feature_by_key:
            record.encoded = True
    return feature_by_key, base_infos


def tiered_quality_by_rank(records: Sequence[MaskRecord], args: argparse.Namespace) -> Dict[str, str]:
    ordered = sorted(records, key=lambda item: (-float(item.coverage), -int(item.mask_area), item.frame_id, int(item.mask_id)))
    n = len(ordered)
    if n == 0:
        return {}
    high_end = max(1, int(np.ceil(n / 3.0)))
    mid_end = max(high_end + 1, int(np.ceil(2.0 * n / 3.0))) if n > 1 else high_end
    tiers: Dict[str, str] = {}
    for i, record in enumerate(ordered):
        if i < high_end or float(record.coverage) >= float(args.tiered_high_min_coverage):
            tiers[record.key] = "high"
        elif i < mid_end or float(record.coverage) >= float(args.tiered_mid_min_coverage):
            tiers[record.key] = "medium"
        else:
            tiers[record.key] = "low"
    return tiers


def confidence_for_source(tier: str, source: str, args: argparse.Namespace) -> float:
    if source == "tiered_base_fallback":
        return float(args.fallback_teacher_confidence)
    if tier == "high":
        return float(args.high_quality_teacher_confidence)
    if tier == "medium":
        return float(args.medium_quality_teacher_confidence)
    return float(args.low_quality_teacher_confidence)


def build_tiered_consensus_features(
    grouped: Dict[int, List[MaskRecord]],
    base_features: Dict[str, np.ndarray],
    base_infos: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
    final_features: Dict[str, np.ndarray] = {}
    instance_rows: List[Dict[str, Any]] = []
    mask_rows: List[Dict[str, Any]] = []

    for instance_id, records in sorted(grouped.items(), key=lambda item: int(item[0])):
        tiers = tiered_quality_by_rank(records, args)
        base_rows: List[Dict[str, Any]] = []
        for record in records:
            feature = base_features.get(record.key)
            info = base_infos.get(record.key, {})
            tier = tiers.get(record.key, "low")
            base_rows.append(
                {
                    "record": record,
                    "key": record.key,
                    "tier": tier,
                    "feature": feature,
                    "coverage": float(record.coverage),
                    "base_strategy": str(info.get("base_strategy", "missing")),
                    "crop_area_ratio": float(info.get("crop_area_ratio", 0.0)),
                    "mask_area_ratio": float(info.get("mask_area_ratio", record.mask_area_ratio)),
                    "is_large_crop": bool(info.get("is_large_crop", False)),
                    "context": info.get("context", {}),
                }
            )

        seed_rows = [
            row for row in base_rows
            if row["feature"] is not None and row["tier"] in {"high", "medium"}
        ]
        if len(seed_rows) < int(args.tiered_min_seed_masks):
            seed_rows = [
                row for row in sorted(base_rows, key=lambda item: -float(item["coverage"]))
                if row["feature"] is not None
            ][: max(1, int(args.tiered_min_seed_masks))]

        consensus_feature: Optional[np.ndarray] = None
        consensus_meta: Dict[str, Any] = {
            "instance_id": int(instance_id),
            "num_masks": int(len(records)),
            "num_seed_masks": int(len(seed_rows)),
            "consensus_available": False,
        }
        if seed_rows:
            features = np.stack([row["feature"] for row in seed_rows], axis=0)
            weights = np.asarray(
                [
                    max(float(args.tiered_min_seed_weight), float(row["coverage"]))
                    * (1.0 if row["tier"] == "high" else float(args.tiered_mid_weight))
                    for row in seed_rows
                ],
                dtype=np.float32,
            )
            weights = weights / max(float(weights.sum()), 1e-6)
            sims = features @ features.T
            weighted_sims = sims @ weights
            medoid_idx = int(np.argmax(weighted_sims))
            medoid_feature = features[medoid_idx]
            inlier_mask = sims[medoid_idx] >= float(args.tiered_inlier_similarity)
            if int(inlier_mask.sum()) < min(int(args.tiered_min_inliers), len(seed_rows)):
                top_ids = np.argsort(-sims[medoid_idx])[: min(max(1, int(args.tiered_min_inliers)), len(seed_rows))]
                inlier_mask = np.zeros((len(seed_rows),), dtype=bool)
                inlier_mask[top_ids] = True
            inlier_features = features[inlier_mask]
            inlier_weights = weights[inlier_mask]
            inlier_weights = inlier_weights / max(float(inlier_weights.sum()), 1e-6)
            if str(args.tiered_consensus_mode).lower() == "inlier_mean" and inlier_features.shape[0] > 0:
                consensus_feature = l2_normalize((inlier_features * inlier_weights[:, None]).sum(axis=0)).astype(np.float32)
                consensus_mode = "inlier_mean"
            else:
                consensus_feature = medoid_feature.astype(np.float32)
                consensus_mode = "weighted_medoid"
            consensus_meta = {
                "instance_id": int(instance_id),
                "num_masks": int(len(records)),
                "num_seed_masks": int(len(seed_rows)),
                "consensus_available": True,
                "consensus_mode": consensus_mode,
                "medoid_key": str(seed_rows[medoid_idx]["key"]),
                "medoid_strategy": str(seed_rows[medoid_idx]["base_strategy"]),
                "medoid_weighted_similarity": float(weighted_sims[medoid_idx]),
                "num_inliers": int(inlier_mask.sum()),
                "mean_inlier_similarity_to_medoid": float(sims[medoid_idx][inlier_mask].mean()),
                "seed_rows": [
                    {
                        "key": str(row["key"]),
                        "tier": str(row["tier"]),
                        "base_strategy": str(row["base_strategy"]),
                        "coverage": float(row["coverage"]),
                        "crop_area_ratio": float(row["crop_area_ratio"]),
                        "mask_area_ratio": float(row["mask_area_ratio"]),
                        "is_large_crop": bool(row["is_large_crop"]),
                        "similarity_to_medoid": float(sims[medoid_idx][i]),
                        "is_inlier": bool(inlier_mask[i]),
                    }
                    for i, row in enumerate(seed_rows)
                ],
            }
        instance_rows.append(consensus_meta)

        for row in base_rows:
            record = row["record"]
            own_feature = row["feature"]
            tier = str(row["tier"])
            similarity_to_consensus: Optional[float] = None
            semantic_outlier = False
            if own_feature is not None and consensus_feature is not None:
                similarity_to_consensus = float(np.dot(own_feature, consensus_feature))
                semantic_outlier = bool(
                    tier in {"high", "medium"}
                    and similarity_to_consensus < float(args.tiered_outlier_similarity)
                )

            feature: Optional[np.ndarray]
            if consensus_feature is not None:
                feature = consensus_feature
                source = "tiered_instance_unified"
            elif own_feature is not None:
                feature = own_feature
                source = "tiered_base_fallback"
            else:
                feature = None
                source = "missing_no_feature"

            if feature is None:
                record.feature_source = "missing_no_feature"
                record.teacher_confidence = 0.0
                continue
            final_features[record.key] = feature.astype(np.float32)
            record.semantic_outlier = bool(semantic_outlier)
            record.prototype_similarity = float(similarity_to_consensus or 0.0)
            record.feature_source = source
            record.teacher_confidence = float(np.clip(confidence_for_source(tier, source, args), 0.0, 1.0))
            mask_rows.append(
                {
                    "key": record.key,
                    "frame_id": record.frame_id,
                    "mask_id": int(record.mask_id),
                    "instance_id": int(record.instance_id),
                    "tier": tier,
                    "coverage": float(row["coverage"]),
                    "quality_level": record.quality_level,
                    "base_strategy": str(row["base_strategy"]),
                    "feature_source": source,
                    "teacher_confidence": float(record.teacher_confidence),
                    "semantic_outlier": bool(semantic_outlier),
                    "similarity_to_consensus": similarity_to_consensus,
                    "crop_area_ratio": float(row["crop_area_ratio"]),
                    "mask_area_ratio": float(row["mask_area_ratio"]),
                    "is_large_crop": bool(row["is_large_crop"]),
                    "suppressed_context_count": int(row["context"].get("num_suppressed_context_masks", 0)),
                    "suppressed_context_crop_ratio": float(row["context"].get("suppressed_context_crop_ratio", 0.0)),
                    "unknown_compact_crop_ratio": float(row["context"].get("unknown_compact_crop_ratio", 0.0)),
                    "consensus_available": bool(consensus_meta.get("consensus_available", False)),
                    "consensus_medoid_key": str(consensus_meta.get("medoid_key", "")),
                    "consensus_medoid_strategy": str(consensus_meta.get("medoid_strategy", "")),
                }
            )

    return final_features, instance_rows, mask_rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset, ds_args = build_dataset_from_args(args)
    output_dir = output_dir_for(dataset, args.config, args.output_dir)
    assert_output_safe(output_dir, bool(args.force))

    object_path = resolve_repo_path(args.object_dict_path) if args.object_dict_path else default_object_dict_path(dataset, args.config)
    if not object_path.exists():
        raise FileNotFoundError(f"object_dict.npy not found: {object_path}")
    object_dict = np.load(object_path, allow_pickle=True).item()
    records, duplicates = parse_object_dict(object_dict)
    cache = SceneCache(dataset)
    assign_mask_quality(records, cache, args)
    grouped = group_by_instance(records)

    base_features, base_infos = encode_base_features(records, cache, args)
    final_features, instance_rows, mask_rows = build_tiered_consensus_features(grouped, base_features, base_infos, args)
    feature_dim = int(args.semantic_feature_dim)
    if final_features:
        feature_dim = int(next(iter(final_features.values())).shape[0])
    write_summary = write_teacher_files(records, final_features, cache, output_dir, feature_dim, args)

    mask_counts = {
        "total": len(records),
        "high": sum(1 for r in records if r.quality_level == "high"),
        "medium": sum(1 for r in records if r.quality_level == "medium"),
        "low": sum(1 for r in records if r.quality_level == "low"),
        "encoded_base_features": len(base_features),
        "final_features": len(final_features),
    }
    base_strategy_counts = Counter(str(row.get("base_strategy", "missing")) for row in base_infos.values())
    tier_counts = Counter(str(row.get("tier", "missing")) for row in mask_rows)
    feature_source_counts = Counter(str(row.get("feature_source", "missing")) for row in mask_rows)
    semantic_outlier_count = sum(1 for row in mask_rows if bool(row.get("semantic_outlier", False)))
    unified_count = sum(
        1
        for row in mask_rows
        if str(row.get("feature_source", "")) == "tiered_instance_unified"
    )
    summary = {
        "seq_name": args.seq_name,
        "config": args.config,
        "dataset": getattr(ds_args, "dataset", ""),
        "scene_root": str(getattr(dataset, "root", "")),
        "object_dict_path": str(object_path),
        "output_dir": str(output_dir),
        "feature_backend": "alphaclip",
        "teacher_method": "tiered_instance_unified",
        "instance_feature_mode": "instance_unified",
        "clip_model": str(args.clip_model),
        "alphaclip_checkpoint": str(args.alphaclip_checkpoint),
        "semantic_feature_dim": int(feature_dim),
        "recommended_feature_level": 0,
        "mask_counts": mask_counts,
        "base_strategy_counts": dict(base_strategy_counts),
        "tier_counts": dict(tier_counts),
        "feature_source_counts": dict(feature_source_counts),
        "num_semantic_outliers": int(semantic_outlier_count),
        "num_instance_unified_masks": int(unified_count),
        "num_instances": len(grouped),
        "num_instances_with_consensus": sum(1 for row in instance_rows if row.get("consensus_available")),
        "num_duplicate_mask_bindings": len(duplicates),
        "teacher_files": {
            "num_images": int(write_summary["num_images"]),
            "num_regions": int(write_summary["num_regions"]),
            "format": "per-frame region files: *_s.npy, *_f.npy, *_w.npy",
        },
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "teacher_config.json", vars(args))
    save_json(output_dir / "instance_consensus_summary.json", instance_rows)
    save_json(output_dir / "tiered_mask_table.json", mask_rows)
    save_json(output_dir / "base_feature_table.json", list(base_infos.values()))
    save_json(output_dir / "duplicate_mask_bindings.json", duplicates)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build final MaskClustering instance-unified Alpha-CLIP teacher files."
    )
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

    parser.add_argument("--large_inpaint_radius", type=float, default=7.0)
    parser.add_argument("--large_context_min_individual_crop_ratio", type=float, default=0.002)
    parser.add_argument("--large_distractor_risk_threshold", type=float, default=0.012)
    parser.add_argument("--large_center_score_threshold", type=float, default=0.20)
    parser.add_argument("--large_center_min_crop_ratio", type=float, default=0.004)
    parser.add_argument("--large_support_min_edge_touches", type=int, default=2)
    parser.add_argument("--large_support_min_span_ratio", type=float, default=0.65)
    parser.add_argument("--large_support_risk_multiplier", type=float, default=0.25)
    parser.add_argument("--large_support_suppress_risk_threshold", type=float, default=0.08)
    parser.add_argument("--large_unknown_min_crop_ratio", type=float, default=0.004)
    parser.add_argument("--large_unknown_support_min_edge_touches", type=int, default=2)
    parser.add_argument("--large_unknown_support_min_span_ratio", type=float, default=0.65)

    parser.add_argument("--tiered_large_crop_ratio", type=float, default=0.50)
    parser.add_argument("--tiered_large_mask_area_ratio", type=float, default=0.08)
    parser.add_argument("--tiered_high_min_coverage", type=float, default=0.45)
    parser.add_argument("--tiered_mid_min_coverage", type=float, default=0.20)
    parser.add_argument("--tiered_min_seed_masks", type=int, default=2)
    parser.add_argument("--tiered_min_inliers", type=int, default=2)
    parser.add_argument("--tiered_inlier_similarity", type=float, default=0.70)
    parser.add_argument("--tiered_consensus_mode", default="medoid")
    parser.add_argument("--tiered_mid_weight", type=float, default=0.70)
    parser.add_argument("--tiered_min_seed_weight", type=float, default=0.05)
    parser.add_argument("--tiered_outlier_similarity", type=float, default=0.62)

    parser.add_argument("--low_quality_teacher_confidence", type=float, default=0.50)
    parser.add_argument("--medium_quality_teacher_confidence", type=float, default=0.75)
    parser.add_argument("--high_quality_teacher_confidence", type=float, default=0.90)
    parser.add_argument("--fallback_teacher_confidence", type=float, default=0.35)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
