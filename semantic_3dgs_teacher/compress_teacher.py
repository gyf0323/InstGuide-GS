from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_3dgs_teacher.instance_codec_utils import (  # noqa: E402
    copy_or_reference_frame_data,
    copy_small_sidecars,
    encode_instance_features,
    evaluate_reconstruction,
    load_instance_features,
    output_array_dtype,
    prepare_output_dir,
    validate_codec_dims,
    write_decoded_visualization_teacher,
)
from semantic_3dgs_teacher.codec_utils import (  # noqa: E402
    TeacherFeatureFile,
    discover_teacher_files,
    load_semantic_codec,
    resolve_device,
    save_semantic_codec,
    train_codec,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **_kwargs: Any) -> Iterable[Any]:
        return iterable


FORMAT_VERSION = 1


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_rows(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise RuntimeError(f"Expected [N,C] feature matrix, got {features.shape}.")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, eps)).astype(np.float32, copy=False)


def size_report(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "total_mib": 0.0,
            "total_gib": 0.0,
            "skipped_entries": 0,
        }

    total_bytes = 0
    file_count = 0
    skipped_entries = 0
    paths = [path] if path.is_file() else path.rglob("*")
    for item in paths:
        try:
            if item.is_file():
                total_bytes += int(item.stat().st_size)
                file_count += 1
        except OSError:
            skipped_entries += 1

    return {
        "path": str(path),
        "exists": True,
        "file_count": int(file_count),
        "total_bytes": int(total_bytes),
        "total_mib": round(total_bytes / (1024.0 ** 2), 3),
        "total_gib": round(total_bytes / (1024.0 ** 3), 6),
        "skipped_entries": int(skipped_entries),
    }


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


def load_codec_training_features(path: Path) -> List[TeacherFeatureFile]:
    if not path.is_file():
        raise FileNotFoundError(f"Codec training feature file not found: {path}")
    payload = np.load(path, allow_pickle=True)
    if "features" not in payload:
        raise RuntimeError(f"Expected `features` in {path}.")
    features = normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    if "weights" in payload:
        weights = np.asarray(payload["weights"], dtype=np.float32)
    else:
        weights = np.ones((features.shape[0],), dtype=np.float32)
    if weights.shape[0] != features.shape[0]:
        raise RuntimeError(f"Training feature/weight length mismatch in {path}: {features.shape[0]} vs {weights.shape[0]}.")
    weights = np.clip(weights, 1e-3, None)
    return [
        TeacherFeatureFile(
            feature_path=path,
            segmentation_path=path,
            weight_path=None,
            features=features,
            weights=weights,
        )
    ]


def load_or_train_codec(
    args: argparse.Namespace,
    instance_features: np.ndarray,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    output_codec_path = Path(args.codec_path).resolve() if args.codec_path else output_dir / "semantic_codec.pth"
    if args.source_codec_path:
        source_codec_path = Path(args.source_codec_path).resolve()
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
        training_source = "source_mask_teacher_dir"
    else:
        training_features_path = Path(args.source_training_features).resolve() if args.source_training_features else input_dir / "codec_training_features.npz"
        if training_features_path.is_file():
            train_files = load_codec_training_features(training_features_path)
            training_source = "codec_training_features"
        else:
            train_files = build_instance_training_file(instance_features, input_dir)
            training_source = "instance_features_only"

    codec, training_metadata = train_codec(args, train_files, device)
    save_semantic_codec(output_codec_path, codec, training_metadata=training_metadata)
    return codec.to(device).eval(), dict(training_metadata), training_source


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
            "mask_feature_table.json",
            "object_dict.npy",
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
        "format": "maskclustering_final_compressed_instance_level_teacher",
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
        "source_training_features": str(Path(args.source_training_features).resolve()) if args.source_training_features else str(input_dir / "codec_training_features.npz"),
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
            "This is the final compression path: instance-level teacher storage plus an MLP codec.",
            "For visualization, use decoded_for_visualization when --write_decoded_visualization_teacher is enabled.",
        ],
    }
    save_json(output_dir / "teacher_config.json", vars(args))
    summary["output_storage"] = size_report(output_dir)
    save_json(output_dir / "summary.json", summary)
    summary["output_storage"] = size_report(output_dir)
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress final instance-level teacher features with an MLP codec.")
    parser.add_argument("--input_teacher_dir", required=True)
    parser.add_argument("--output_teacher_dir", required=True)
    parser.add_argument("--source_training_features", default="", help="Optional codec_training_features.npz from build_teacher.py.")
    parser.add_argument("--source_mask_teacher_dir", default="", help="Optional 512D mask-level teacher directory used to train the codec.")
    parser.add_argument("--source_codec_path", default="", help="Optional existing semantic_codec.pth to reuse instead of training.")
    parser.add_argument("--codec_path", default="", help="Output codec path. Defaults to output_teacher_dir/semantic_codec.pth.")
    parser.add_argument("--compressed_dim", type=int, default=32)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--codec_arch", choices=("langsplat", "mlp"), default="langsplat", help=argparse.SUPPRESS)
    parser.add_argument("--loss_type", choices=("langsplat", "reconstruction", "enhanced"), default="langsplat", help=argparse.SUPPRESS)
    parser.add_argument("--encoder_hidden_dims", default="256,128,64")
    parser.add_argument("--decoder_hidden_dims", default="64,128,256,256")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--langsplat_cosine_weight", type=float, default=0.001, help=argparse.SUPPRESS)
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


