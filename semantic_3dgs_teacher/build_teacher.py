from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_3dgs_teacher.alphaclip_utils import (  # noqa: E402
    MaskRecord,
    SceneCache,
    assign_mask_quality,
    build_dataset_from_args,
    default_object_dict_path,
    frame_sort_key,
    group_by_instance,
    l2_normalize,
    parse_object_dict,
    resolve_repo_path,
    save_json,
)
from semantic_3dgs_teacher.tiered_semantics import (  # noqa: E402
    build_tiered_consensus_features,
    encode_base_features,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **_kwargs: Any) -> Iterable[Any]:
        return iterable


FORMAT_VERSION = 1
IGNORE_INDEX = -1


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


def normalize_rows(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise RuntimeError(f"Expected [N,C] feature matrix, got {features.shape}.")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, eps)).astype(np.float32, copy=False)


def output_feature_array(features: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name == "float16":
        return features.astype(np.float16)
    if dtype_name == "float32":
        return features.astype(np.float32)
    raise ValueError(f"Unsupported feature dtype: {dtype_name}")


def output_weight_array(weights: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name == "float16":
        return weights.astype(np.float16)
    if dtype_name == "float32":
        return weights.astype(np.float32)
    if dtype_name == "uint8":
        return np.clip(np.round(weights * 255.0), 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported weight dtype: {dtype_name}")


def choose_map_dtype(dtype_name: str, num_instances: int) -> np.dtype:
    if dtype_name == "auto":
        return np.dtype(np.int16 if num_instances <= np.iinfo(np.int16).max else np.int32)
    if dtype_name == "int16":
        if num_instances > np.iinfo(np.int16).max:
            raise RuntimeError(f"int16 maps cannot store {num_instances} instances plus ignore index.")
        return np.dtype(np.int16)
    if dtype_name == "int32":
        return np.dtype(np.int32)
    raise ValueError(f"Unsupported map dtype: {dtype_name}")


def save_array(path: Path, array: np.ndarray, compressed: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        output_path = path.with_suffix(".npz")
        np.savez_compressed(output_path, data=array)
        return str(output_path)
    np.save(path, array)
    return str(path)


def weighted_medoid(features: np.ndarray, weights: np.ndarray) -> int:
    if features.shape[0] == 1:
        return 0
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-8)
    sims = features @ features.T
    return int(np.argmax(sims @ weights))


def choose_instance_feature(
    records: Sequence[MaskRecord],
    final_features: Dict[str, np.ndarray],
    mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    valid_records = [record for record in records if record.key in final_features]
    if not valid_records:
        raise RuntimeError("Cannot choose an instance feature from zero valid records.")
    features = normalize_rows(np.stack([final_features[record.key] for record in valid_records], axis=0))
    weights = np.asarray(
        [
            max(1.0, float(record.mask_area))
            * max(1e-3, float(record.teacher_confidence))
            * max(0.05, float(record.coverage))
            for record in valid_records
        ],
        dtype=np.float32,
    )
    weights = weights / max(float(weights.sum()), 1e-8)
    if mode == "weighted_mean":
        feature = l2_normalize((features * weights[:, None]).sum(axis=0)).astype(np.float32)
        chosen_index = weighted_medoid(features, weights)
    elif mode == "first":
        chosen_index = 0
        feature = features[chosen_index].astype(np.float32)
    elif mode == "weighted_medoid":
        chosen_index = weighted_medoid(features, weights)
        feature = features[chosen_index].astype(np.float32)
    else:
        raise ValueError(f"Unsupported --instance_feature_mode: {mode}")
    sims = features @ feature
    chosen = valid_records[int(chosen_index)]
    return feature, {
        "num_mask_observations": int(len(valid_records)),
        "total_pixel_area": int(sum(int(record.mask_area) for record in valid_records)),
        "feature_mode": mode,
        "chosen_frame_id": str(chosen.frame_id),
        "chosen_mask_id": int(chosen.mask_id),
        "chosen_teacher_confidence": float(chosen.teacher_confidence),
        "chosen_coverage": float(chosen.coverage),
        "chosen_quality_level": str(chosen.quality_level),
        "chosen_feature_source": str(chosen.feature_source),
        "mean_similarity_to_instance_feature": float(np.mean(sims)),
        "min_similarity_to_instance_feature": float(np.min(sims)),
        "num_observations_below_0_98": int((sims < 0.98).sum()),
        "num_observations_below_0_95": int((sims < 0.95).sum()),
    }


def write_codec_training_features(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    final_features: Dict[str, np.ndarray],
    dtype_name: str,
) -> Dict[str, Any]:
    features: List[np.ndarray] = []
    weights: List[float] = []
    keys: List[str] = []
    for row in rows:
        key = str(row.get("key", ""))
        feature = final_features.get(key)
        if feature is None:
            continue
        features.append(np.asarray(feature, dtype=np.float32))
        weights.append(max(1e-3, float(row.get("teacher_confidence", 1.0))))
        keys.append(key)
    if features:
        feature_array = output_feature_array(normalize_rows(np.stack(features, axis=0)), dtype_name)
        weight_array = np.asarray(weights, dtype=np.float32)
    else:
        feature_array = np.zeros((0, 0), dtype=np.float32)
        weight_array = np.zeros((0,), dtype=np.float32)
    path = output_dir / "codec_training_features.npz"
    np.savez_compressed(
        path,
        features=feature_array,
        weights=weight_array,
        keys=np.asarray(keys, dtype=object),
    )
    return {
        "path": str(path),
        "num_features": int(feature_array.shape[0]),
        "feature_shape": [int(value) for value in feature_array.shape],
        "feature_dtype": str(feature_array.dtype),
        "weight_dtype": str(weight_array.dtype),
    }


def write_instance_level_teacher(
    records: Sequence[MaskRecord],
    final_features: Dict[str, np.ndarray],
    cache: SceneCache,
    output_dir: Path,
    feature_dim: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dtype = choose_map_dtype(str(args.map_dtype), len({int(record.instance_id) for record in records}))

    grouped: DefaultDict[int, List[MaskRecord]] = defaultdict(list)
    for record in records:
        if record.key in final_features:
            grouped[int(record.instance_id)].append(record)
    instance_ids = np.asarray(sorted(grouped.keys()), dtype=np.int64)
    instance_id_to_index = {int(instance_id): int(index) for index, instance_id in enumerate(instance_ids.tolist())}

    instance_features: List[np.ndarray] = []
    instance_feature_table: List[Dict[str, Any]] = []
    for instance_id in instance_ids.tolist():
        feature, info = choose_instance_feature(grouped[int(instance_id)], final_features, str(args.instance_feature_mode))
        instance_features.append(feature)
        instance_feature_table.append(
            {
                "instance_id": int(instance_id),
                "instance_index": int(instance_id_to_index[int(instance_id)]),
                **info,
            }
        )
    feature_array = (
        normalize_rows(np.stack(instance_features, axis=0))
        if instance_features
        else np.zeros((0, feature_dim), dtype=np.float32)
    )
    np.save(output_dir / "instance_ids.npy", instance_ids.astype(np.int64))
    np.save(output_dir / "instance_features.npy", output_feature_array(feature_array, str(args.instance_feature_dtype)))
    save_json(output_dir / "instance_id_to_index.json", {str(k): int(v) for k, v in instance_id_to_index.items()})
    save_json(output_dir / "instance_feature_table.json", instance_feature_table)

    by_frame: DefaultDict[str, List[MaskRecord]] = defaultdict(list)
    for record in records:
        if record.key in final_features and int(record.instance_id) in instance_id_to_index:
            by_frame[record.frame_id].append(record)

    mask_feature_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []
    frame_region_to_instance: Dict[str, Dict[str, Any]] = {}
    instance_to_masks: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    mask_to_instance: Dict[str, Dict[str, Any]] = {}

    for frame_id in tqdm(
        sorted(by_frame.keys(), key=frame_sort_key),
        desc="Write instance teacher maps",
        ascii=True,
        dynamic_ncols=True,
    ):
        width, height = cache.image_size(frame_id)
        seg = cache.segmentation(frame_id)
        instance_map = np.full((height, width), IGNORE_INDEX, dtype=map_dtype)
        pixel_weight = np.zeros((height, width), dtype=np.float32)
        region_rows: Dict[str, Dict[str, Any]] = {}
        frame_records = sorted(
            by_frame[frame_id],
            key=lambda item: (-float(item.teacher_confidence), -int(item.mask_area), int(item.mask_id)),
        )
        region_id = 0
        for record in frame_records:
            mask = seg == int(record.mask_id)
            if not bool(mask.any()):
                continue
            instance_index = int(instance_id_to_index[int(record.instance_id)])
            record.region_id = int(region_id)
            instance_map[mask] = instance_index
            pixel_weight[mask] = float(record.teacher_confidence)
            row = asdict(record)
            row.update(
                {
                    "image_id": frame_id,
                    "region_id": int(region_id),
                    "instance_index": instance_index,
                    "feature_dim": int(feature_dim),
                }
            )
            mask_feature_rows.append(row)
            compact_row = {
                "frame_id": str(frame_id),
                "mask_id": int(record.mask_id),
                "region_id": int(region_id),
                "instance_id": int(record.instance_id),
                "instance_index": instance_index,
                "coverage": float(record.coverage),
                "teacher_confidence": float(record.teacher_confidence),
                "pixel_area": int(mask.sum()),
                "quality_level": str(record.quality_level),
                "feature_source": str(record.feature_source),
            }
            instance_to_masks[str(int(record.instance_id))].append(compact_row)
            mask_to_instance[f"{frame_id}:{int(record.mask_id)}"] = compact_row
            region_rows[str(int(region_id))] = compact_row
            region_id += 1

        valid = instance_map >= 0
        pixel_weight[~valid] = 0.0
        map_path = save_array(output_dir / "frame_instance_maps" / f"{frame_id}.npy", instance_map[None, ...], bool(args.compressed_maps))
        weight_path = save_array(
            output_dir / "frame_weights" / f"{frame_id}.npy",
            output_weight_array(pixel_weight[None, ...], str(args.weight_dtype)),
            bool(args.compressed_maps),
        )
        frame_region_to_instance[frame_id] = region_rows
        frame_rows.append(
            {
                "frame_id": str(frame_id),
                "instance_map_path": map_path,
                "weight_path": weight_path,
                "shape": [1, int(height), int(width)],
                "map_dtype": str(map_dtype),
                "weight_dtype": str(args.weight_dtype),
                "num_regions": int(len(region_rows)),
                "num_unique_instances": int(len(set(row["instance_index"] for row in region_rows.values()))),
                "teacher_pixel_coverage_ratio": float(valid.sum()) / float(max(valid.size, 1)),
            }
        )

    save_json(output_dir / "mask_feature_table.json", mask_feature_rows)
    save_json(output_dir / "frame_table.json", frame_rows)
    save_json(output_dir / "frame_region_to_instance.json", frame_region_to_instance)
    save_json(output_dir / "instance_to_masks.json", dict(instance_to_masks))
    save_json(output_dir / "mask_to_instance.json", mask_to_instance)

    codec_training_info: Dict[str, Any] = {}
    if bool(args.write_codec_training_features):
        codec_training_info = write_codec_training_features(
            output_dir,
            mask_feature_rows,
            final_features,
            str(args.codec_training_feature_dtype),
        )

    return {
        "num_images": int(len(frame_rows)),
        "num_instances": int(instance_ids.shape[0]),
        "num_mask_observations": int(len(mask_feature_rows)),
        "feature_shape": [int(value) for value in feature_array.shape],
        "instance_feature_dtype": str(args.instance_feature_dtype),
        "instance_map_dtype": str(map_dtype),
        "weight_dtype": str(args.weight_dtype),
        "compressed_maps": bool(args.compressed_maps),
        "codec_training_features": codec_training_info,
    }


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
    write_summary = write_instance_level_teacher(records, final_features, cache, output_dir, feature_dim, args)
    portable_object_path = output_dir / "object_dict.npy"
    shutil.copy2(object_path, portable_object_path)

    base_strategy_counts = Counter(str(row.get("base_strategy", "missing")) for row in base_infos.values())
    tier_counts = Counter(str(row.get("tier", "missing")) for row in mask_rows)
    feature_source_counts = Counter(str(row.get("feature_source", "missing")) for row in mask_rows)
    summary = {
        "format": "maskclustering_final_instance_level_teacher",
        "format_version": FORMAT_VERSION,
        "seq_name": args.seq_name,
        "config": args.config,
        "dataset": getattr(ds_args, "dataset", ""),
        "scene_root": str(getattr(dataset, "root", "")),
        "object_dict_path": str(portable_object_path),
        "output_dir": str(output_dir),
        "feature_backend": "alphaclip",
        "teacher_method": "tiered_instance_unified",
        "storage_format": "instance_level",
        "clip_model": str(args.clip_model),
        "alphaclip_checkpoint": str(args.alphaclip_checkpoint),
        "semantic_feature_dim": int(feature_dim),
        "mask_counts": {
            "total": len(records),
            "high": sum(1 for item in records if item.quality_level == "high"),
            "medium": sum(1 for item in records if item.quality_level == "medium"),
            "low": sum(1 for item in records if item.quality_level == "low"),
            "encoded_base_features": len(base_features),
            "final_features": len(final_features),
        },
        "base_strategy_counts": dict(base_strategy_counts),
        "tier_counts": dict(tier_counts),
        "feature_source_counts": dict(feature_source_counts),
        "num_instances": int(write_summary["num_instances"]),
        "num_instances_with_consensus": sum(1 for row in instance_rows if row.get("consensus_available")),
        "num_duplicate_mask_bindings": len(duplicates),
        "teacher_files": write_summary,
        "notes": [
            "This final pipeline writes compact instance-level teacher files directly; it does not write per-frame feature tables.",
            "The semantic feature for each 3D instance is shared by all masks bound to that instance.",
            "codec_training_features.npz is only for optional MLP-codec compression.",
        ],
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
    parser = argparse.ArgumentParser(description="Build the final instance-level MaskClustering Alpha-CLIP teacher.")
    parser.add_argument("--config", default="scannetpp_dslr")
    parser.add_argument("--config_dir", default="")
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

    parser.add_argument("--instance_feature_mode", default="weighted_medoid", choices=["weighted_medoid", "weighted_mean", "first"])
    parser.add_argument("--instance_feature_dtype", default="float32", choices=["float16", "float32"])
    parser.add_argument("--map_dtype", default="auto", choices=["auto", "int16", "int32"])
    parser.add_argument("--weight_dtype", default="float16", choices=["float16", "float32", "uint8"])
    parser.add_argument("--compressed_maps", action="store_true")
    parser.add_argument("--write_codec_training_features", dest="write_codec_training_features", action="store_true")
    parser.add_argument("--no_write_codec_training_features", dest="write_codec_training_features", action="store_false")
    parser.set_defaults(write_codec_training_features=True)
    parser.add_argument("--codec_training_feature_dtype", default="float16", choices=["float16", "float32"])
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
