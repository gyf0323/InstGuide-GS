from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **_kwargs: Any) -> Iterable[Any]:
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_3dgs_teacher.codec_utils import (  # noqa: E402
    TeacherFeatureFile,
    discover_teacher_files,
    load_semantic_codec,
    resolve_device,
    save_semantic_codec,
    train_codec,
)


FORMAT_VERSION = 1


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rows(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise RuntimeError(f"Expected [N,C] feature matrix, got {features.shape}.")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, eps)).astype(np.float32, copy=False)


def output_array_dtype(array: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name == "float16":
        return array.astype(np.float16)
    if dtype_name == "float32":
        return array.astype(np.float32)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise RuntimeError(f"Output directory is not empty: {path}. Pass --force or choose another path.")
    if path.exists() and force:
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def load_instance_features(input_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    feature_path = input_dir / "instance_features.npy"
    ids_path = input_dir / "instance_ids.npy"
    summary_path = input_dir / "summary.json"
    if not feature_path.is_file():
        raise FileNotFoundError(f"instance_features.npy not found: {feature_path}")
    if not ids_path.is_file():
        raise FileNotFoundError(f"instance_ids.npy not found: {ids_path}")
    features = normalize_rows(np.asarray(np.load(feature_path), dtype=np.float32))
    instance_ids = np.asarray(np.load(ids_path), dtype=np.int64)
    if features.shape[0] != instance_ids.shape[0]:
        raise RuntimeError(
            f"instance_features/instance_ids length mismatch: {features.shape[0]} vs {instance_ids.shape[0]}."
        )
    summary = load_json(summary_path) if summary_path.is_file() else {}
    return features, instance_ids, summary


def build_instance_training_file(features: np.ndarray, input_dir: Path) -> List[TeacherFeatureFile]:
    dummy_path = input_dir / "instance_features.npy"
    weights = np.ones((features.shape[0],), dtype=np.float32)
    return [
        TeacherFeatureFile(
            feature_path=dummy_path,
            segmentation_path=dummy_path,
            weight_path=None,
            features=normalize_rows(features),
            weights=weights,
        )
    ]


def validate_codec_dims(codec: torch.nn.Module, input_dim: int, compressed_dim: int) -> Dict[str, Any]:
    metadata = dict(codec.metadata())
    codec_input_dim = int(metadata.get("input_dim", -1))
    codec_latent_dim = int(metadata.get("latent_dim", -1))
    if codec_input_dim != int(input_dim):
        raise RuntimeError(f"Codec input_dim mismatch: codec={codec_input_dim}, instance_features={input_dim}.")
    if codec_latent_dim != int(compressed_dim):
        raise RuntimeError(f"Codec latent_dim mismatch: codec={codec_latent_dim}, requested={compressed_dim}.")
    return metadata


@torch.no_grad()
def encode_instance_features(
    codec: torch.nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    codec = codec.to(device).eval()
    tensor = torch.from_numpy(normalize_rows(features)).float()
    latents: List[torch.Tensor] = []
    decoded: List[torch.Tensor] = []
    iterator = tqdm(
        range(0, tensor.shape[0], int(batch_size)),
        total=(int(tensor.shape[0]) + int(batch_size) - 1) // int(batch_size),
        desc="Encode instance features",
        ascii=True,
        dynamic_ncols=True,
    )
    for start in iterator:
        batch = tensor[start : start + int(batch_size)].to(device)
        latent = codec.encode(batch)
        reconstruction = codec.decode(latent)
        latents.append(latent.detach().cpu())
        decoded.append(reconstruction.detach().cpu())
    if latents:
        latent_np = torch.cat(latents, dim=0).numpy().astype(np.float32)
        decoded_np = torch.cat(decoded, dim=0).numpy().astype(np.float32)
    else:
        metadata = codec.metadata()
        latent_np = np.zeros((0, int(metadata["latent_dim"])), dtype=np.float32)
        decoded_np = np.zeros((0, int(metadata["input_dim"])), dtype=np.float32)
    return normalize_rows(latent_np), normalize_rows(decoded_np)


def evaluate_reconstruction(original: np.ndarray, decoded: np.ndarray) -> Dict[str, Any]:
    original = normalize_rows(original)
    decoded = normalize_rows(decoded)
    cosine = np.sum(original * decoded, axis=1)
    sorted_cosine = np.sort(cosine)
    pairwise_original = original @ original.T
    pairwise_decoded = decoded @ decoded.T
    return {
        "mean_cosine": float(np.mean(cosine)),
        "min_cosine": float(np.min(cosine)),
        "p05_cosine": float(sorted_cosine[max(0, int(np.floor(0.05 * (len(sorted_cosine) - 1))))]),
        "p50_cosine": float(np.median(cosine)),
        "p95_cosine": float(sorted_cosine[min(len(sorted_cosine) - 1, int(np.ceil(0.95 * (len(sorted_cosine) - 1))))]),
        "num_below_0_99": int((cosine < 0.99).sum()),
        "num_below_0_98": int((cosine < 0.98).sum()),
        "num_below_0_95": int((cosine < 0.95).sum()),
        "pairwise_similarity_mse": float(np.mean((pairwise_original - pairwise_decoded) ** 2)),
    }


def copy_small_sidecars(input_dir: Path, output_dir: Path, names: Sequence[str]) -> List[str]:
    copied = []
    for name in names:
        source = input_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
            copied.append(name)
    return copied


def copy_or_reference_frame_data(input_dir: Path, output_dir: Path, mode: str) -> Dict[str, Any]:
    map_source = input_dir / "frame_instance_maps"
    weight_source = input_dir / "frame_weights"
    frame_table_source = input_dir / "frame_table.json"
    if not map_source.is_dir():
        raise FileNotFoundError(f"frame_instance_maps not found: {map_source}")
    if not weight_source.is_dir():
        raise FileNotFoundError(f"frame_weights not found: {weight_source}")

    frame_rows = load_json(frame_table_source) if frame_table_source.is_file() else []
    mode = str(mode)
    if mode == "reference":
        if frame_table_source.is_file():
            shutil.copy2(frame_table_source, output_dir / "frame_table.json")
        return {
            "frame_data_mode": "reference",
            "frame_instance_maps_dir": str(map_source),
            "frame_weights_dir": str(weight_source),
            "frame_table_path": str(output_dir / "frame_table.json") if frame_table_source.is_file() else "",
            "num_frame_table_rows": int(len(frame_rows)) if isinstance(frame_rows, list) else 0,
        }

    map_target = output_dir / "frame_instance_maps"
    weight_target = output_dir / "frame_weights"
    if mode == "copy":
        shutil.copytree(map_source, map_target)
        shutil.copytree(weight_source, weight_target)
    elif mode == "symlink":
        map_target.symlink_to(map_source, target_is_directory=True)
        weight_target.symlink_to(weight_source, target_is_directory=True)
    else:
        raise ValueError(f"Unsupported --frame_data_mode: {mode}")

    rewritten_rows = []
    if isinstance(frame_rows, list):
        for row in frame_rows:
            row = dict(row)
            frame_id = str(row.get("frame_id", ""))
            if frame_id:
                map_candidates = list(map_target.glob(f"{frame_id}.*"))
                weight_candidates = list(weight_target.glob(f"{frame_id}.*"))
                if map_candidates:
                    row["instance_map_path"] = str(map_candidates[0])
                if weight_candidates:
                    row["weight_path"] = str(weight_candidates[0])
            rewritten_rows.append(row)
        save_json(output_dir / "frame_table.json", rewritten_rows)

    return {
        "frame_data_mode": mode,
        "frame_instance_maps_dir": str(map_target),
        "frame_weights_dir": str(weight_target),
        "frame_table_path": str(output_dir / "frame_table.json") if rewritten_rows else "",
        "num_frame_table_rows": int(len(rewritten_rows)),
    }


def write_decoded_visualization_teacher(
    input_dir: Path,
    output_dir: Path,
    instance_ids: np.ndarray,
    decoded_features: np.ndarray,
    source_summary: Dict[str, Any],
    reconstruction_metrics: Dict[str, Any],
    frame_data_mode: str,
) -> Dict[str, Any]:
    decoded_dir = output_dir / "decoded_for_visualization"
    decoded_dir.mkdir(parents=True, exist_ok=True)
    np.save(decoded_dir / "instance_ids.npy", instance_ids.astype(np.int64))
    np.save(decoded_dir / "instance_features.npy", decoded_features.astype(np.float32))
    copied_sidecars = copy_small_sidecars(
        input_dir,
        decoded_dir,
        [
            "instance_id_to_index.json",
            "instance_feature_table.json",
            "instance_to_masks.json",
            "mask_to_instance.json",
            "frame_region_to_instance.json",
        ],
    )
    frame_info = copy_or_reference_frame_data(input_dir, decoded_dir, str(frame_data_mode))
    summary = {
        "format": "maskclustering_decoded_instance_level_teacher_for_visualization",
        "format_version": FORMAT_VERSION,
        "source_instance_teacher_dir": str(input_dir),
        "source_compressed_teacher_dir": str(output_dir),
        "output_teacher_dir": str(decoded_dir),
        "source_format": str(source_summary.get("format", "")),
        "num_instances": int(instance_ids.shape[0]),
        "semantic_feature_dim": int(decoded_features.shape[1]),
        "reconstruction_metrics": reconstruction_metrics,
        "frame_data": frame_info,
        "copied_sidecars": copied_sidecars,
        "notes": [
            "This directory stores decoded 512D instance features for qualitative visualization only.",
            "Use the compressed parent directory for low-dimensional training.",
        ],
    }
    save_json(decoded_dir / "summary.json", summary)
    return {
        "decoded_teacher_dir": str(decoded_dir),
        "frame_data": frame_info,
    }


def load_or_train_codec(
    args: argparse.Namespace,
    instance_features: np.ndarray,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    output_codec_path = Path(args.codec_path).resolve() if args.codec_path else output_dir / "semantic_codec.pth"
    source_codec_path = Path(args.source_codec_path).resolve() if args.source_codec_path else None

    if source_codec_path is not None:
        codec, payload = load_semantic_codec(source_codec_path, map_location=device)
        validate_codec_dims(codec, int(instance_features.shape[1]), int(args.compressed_dim))
        output_codec_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_codec_path, output_codec_path)
        source_json = source_codec_path.with_suffix(".json")
        if source_json.is_file():
            shutil.copy2(source_json, output_codec_path.with_suffix(".json"))
        return codec.to(device).eval(), dict(payload.get("training_metadata", {})), "reuse_source_codec"

    if args.source_mask_teacher_dir:
        train_files = discover_teacher_files(Path(args.source_mask_teacher_dir).resolve())
        train_source = "source_mask_teacher_dir"
    else:
        train_files = build_instance_training_file(instance_features, input_dir)
        train_source = "instance_features_only"

    codec, training_metadata = train_codec(args, train_files, device)
    save_semantic_codec(output_codec_path, codec, training_metadata=training_metadata)
    return codec.to(device).eval(), dict(training_metadata), train_source


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_dir = Path(args.input_teacher_dir).expanduser().resolve()
    output_dir = Path(args.output_teacher_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input instance-level teacher directory not found: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output teacher directories must be different.")
    if int(args.compressed_dim) <= 0:
        raise ValueError("--compressed_dim must be positive.")
    if int(args.batch_size) <= 0 or int(args.eval_batch_size) <= 0:
        raise ValueError("--batch_size and --eval_batch_size must be positive.")
    if int(args.epochs) <= 0 and not args.source_codec_path:
        raise ValueError("--epochs must be positive when training a codec.")

    instance_features, instance_ids, source_summary = load_instance_features(input_dir)
    if int(args.compressed_dim) >= int(instance_features.shape[1]):
        raise ValueError(
            f"--compressed_dim must be smaller than source feature dim {instance_features.shape[1]}, got {args.compressed_dim}."
        )
    prepare_output_dir(output_dir, bool(args.force))

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))
    device = resolve_device(str(args.device))

    codec, training_metadata, training_source = load_or_train_codec(args, instance_features, input_dir, output_dir, device)
    codec_metadata = validate_codec_dims(codec, int(instance_features.shape[1]), int(args.compressed_dim))
    latents, decoded = encode_instance_features(codec, instance_features, int(args.eval_batch_size), device)
    reconstruction_metrics = evaluate_reconstruction(instance_features, decoded)

    np.save(output_dir / "instance_ids.npy", instance_ids.astype(np.int64))
    np.save(output_dir / "instance_features.npy", output_array_dtype(latents, str(args.latent_dtype)))
    if bool(args.write_decoded_features):
        np.save(output_dir / "decoded_instance_features.npy", decoded.astype(np.float32))

    copied_sidecars = copy_small_sidecars(
        input_dir,
        output_dir,
        [
            "instance_id_to_index.json",
            "instance_feature_table.json",
            "instance_to_masks.json",
            "mask_to_instance.json",
            "frame_region_to_instance.json",
        ],
    )
    frame_info = copy_or_reference_frame_data(input_dir, output_dir, str(args.frame_data_mode))
    decoded_info: Dict[str, Any] = {}
    if bool(args.write_decoded_visualization_teacher):
        decoded_info = write_decoded_visualization_teacher(
            input_dir=input_dir,
            output_dir=output_dir,
            instance_ids=instance_ids,
            decoded_features=decoded,
            source_summary=source_summary,
            reconstruction_metrics=reconstruction_metrics,
            frame_data_mode=str(args.frame_data_mode),
        )

    output_codec_path = Path(args.codec_path).resolve() if args.codec_path else output_dir / "semantic_codec.pth"
    dense_float32_bytes = int(instance_features.size * np.dtype(np.float32).itemsize)
    latent_bytes = int(latents.size * np.dtype(str(args.latent_dtype)).itemsize)
    summary = {
        "format": "maskclustering_mlp_compressed_instance_level_teacher",
        "format_version": FORMAT_VERSION,
        "source_teacher_dir": str(input_dir),
        "output_teacher_dir": str(output_dir),
        "source_format": str(source_summary.get("format", "")),
        "num_instances": int(instance_ids.shape[0]),
        "original_feature_dim": int(instance_features.shape[1]),
        "compressed_dim": int(args.compressed_dim),
        "compression_method": "mlp_codec",
        "codec_path": str(output_codec_path),
        "codec_metadata": codec_metadata,
        "training_source": training_source,
        "source_mask_teacher_dir": str(Path(args.source_mask_teacher_dir).resolve()) if args.source_mask_teacher_dir else "",
        "source_codec_path": str(Path(args.source_codec_path).resolve()) if args.source_codec_path else "",
        "semantic_feature_storage": {
            "original_dense_float32_bytes": dense_float32_bytes,
            "compressed_latent_bytes": latent_bytes,
            "feature_storage_ratio_vs_dense_float32": float(latent_bytes / max(dense_float32_bytes, 1)),
            "note": "Codec parameters are stored separately in semantic_codec.pth.",
        },
        "reconstruction_metrics": reconstruction_metrics,
        "training": training_metadata,
        "frame_data": frame_info,
        "decoded_visualization_teacher": decoded_info,
        "copied_sidecars": copied_sidecars,
        "files": {
            "instance_features": str(output_dir / "instance_features.npy"),
            "instance_ids": str(output_dir / "instance_ids.npy"),
            "semantic_codec": str(output_codec_path),
            "decoded_instance_features": str(output_dir / "decoded_instance_features.npy") if bool(args.write_decoded_features) else "",
        },
        "notes": [
            "This script keeps the instance-level teacher layout but compresses instance_features.npy with an MLP codec.",
            "For best stability, train or reuse a codec learned from the mask-level teacher region features.",
            "At training/query time, decode compressed instance features with semantic_codec.pth.",
        ],
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "teacher_config.json", vars(args))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compress instance-level teacher features with the packaged MLP codec."
        )
    )
    parser.add_argument("--input_teacher_dir", required=True, help="Instance-level teacher directory.")
    parser.add_argument("--output_teacher_dir", required=True, help="Output compressed instance-level teacher directory.")
    parser.add_argument("--source_mask_teacher_dir", default="", help="Optional 512D mask-level teacher dir used to train the codec.")
    parser.add_argument("--source_codec_path", default="", help="Optional existing semantic_codec.pth to reuse instead of training.")
    parser.add_argument("--codec_path", default="", help="Output codec path. Defaults to output_teacher_dir/semantic_codec.pth.")
    parser.add_argument("--compressed_dim", type=int, default=32)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--codec_arch", choices=("langsplat", "mlp"), default="langsplat")
    parser.add_argument("--loss_type", choices=("langsplat", "reconstruction", "enhanced"), default="langsplat")
    parser.add_argument("--encoder_hidden_dims", default="256,128,64")
    parser.add_argument("--decoder_hidden_dims", default="64,128,256,256")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--langsplat_cosine_weight", type=float, default=0.001)
    parser.add_argument("--similarity_weight", type=float, default=0.0)
    parser.add_argument("--decoded_similarity_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.04)
    parser.add_argument("--pair_sample_count", type=int, default=256)
    parser.add_argument("--deduplicate_features", dest="deduplicate_features", action="store_true")
    parser.add_argument("--no_deduplicate_features", dest="deduplicate_features", action="store_false")
    parser.set_defaults(deduplicate_features=False)
    parser.add_argument("--deduplicate_decimals", type=int, default=6)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--latent_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--frame_data_mode", default="copy", choices=("reference", "copy", "symlink"))
    parser.add_argument("--write_decoded_features", action="store_true")
    parser.add_argument("--write_decoded_visualization_teacher", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
