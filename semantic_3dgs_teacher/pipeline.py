from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


STAGE_ORDER = ("masks", "instances", "teacher", "compress")


def log(message: str) -> None:
    print(f"[teacher-pipeline] {message}", flush=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def run_timed_stage(stage: str, func: Callable[..., Dict[str, Any]], *args: Any) -> Dict[str, Any]:
    started_at = now_iso()
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    result["started_at"] = started_at
    result["finished_at"] = now_iso()
    result["elapsed_seconds"] = round(elapsed, 3)
    return result


def candidate_repo_roots(start: Path) -> Iterable[Path]:
    current = start.resolve()
    yield current
    yield from current.parents


def resolve_repo_root(value: str) -> Path:
    if value:
        return Path(value).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in candidate_repo_roots(here.parent):
        if (candidate / "third_party").is_dir() or (candidate / "main.py").is_file():
            return candidate
    cwd = Path.cwd().resolve()
    for candidate in candidate_repo_roots(cwd):
        if (candidate / "third_party").is_dir() or (candidate / "main.py").is_file():
            return candidate
    return cwd


def as_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def run_command(command: Sequence[str], cwd: Path, cuda_visible_devices: str = "") -> None:
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    package_parent = Path(__file__).resolve().parents[1]
    pythonpath_parts = [str(package_parent), str(cwd)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    log("run: " + " ".join(str(item) for item in command))
    subprocess.run([str(item) for item in command], cwd=str(cwd), env=env, check=True)


def split_stages(value: str) -> List[str]:
    text = str(value).strip().lower()
    if not text or text == "all":
        return list(STAGE_ORDER)
    stages = [item.strip() for item in text.replace("+", ",").split(",") if item.strip()]
    unknown = [stage for stage in stages if stage not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}. Valid stages: {', '.join(STAGE_ORDER)} or all.")
    return stages


def scene_root(args: argparse.Namespace) -> Path:
    if str(args.dataset) != "scannetpp_dslr":
        raise NotImplementedError(
            "The integrated portable pipeline currently supports scannetpp_dslr. "
            "Other dataset readers can still be called through the original repo scripts."
        )
    if not args.scannetpp_dslr_root:
        raise ValueError("--scannetpp_dslr_root is required for scannetpp_dslr.")
    return Path(args.scannetpp_dslr_root).expanduser().resolve() / args.seq_name


def default_object_dict_path(args: argparse.Namespace) -> Path:
    return scene_root(args) / "output" / "object" / args.config / "object_dict.npy"


def default_teacher_dir(args: argparse.Namespace) -> Path:
    return scene_root(args) / "output" / "semantic_3dgs_teacher" / args.config


def default_compressed_teacher_dir(args: argparse.Namespace) -> Path:
    return scene_root(args) / "output" / f"semantic_3dgs_teacher_{int(args.compressed_dim)}d" / args.config


def existing_mask_count(args: argparse.Namespace) -> int:
    mask_dir = scene_root(args) / "output" / "mask"
    if not mask_dir.is_dir():
        return 0
    return len(list(mask_dir.glob("*.png")))


def validate_masks(args: argparse.Namespace) -> int:
    from PIL import Image

    root = scene_root(args)
    # Match the bundled CropFormer runner's registered-view filter.
    registered = set()
    with (root / "sparse/0/images.txt").open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 10:
                raise ValueError("Malformed COLMAP images.txt pose record")
            registered.add(fields[9])
            handle.readline()
    images = [Path(p) for p in glob.glob(str(root / args.image_path_pattern))
              if Path(p).is_file() and Path(p).name in registered]
    if not images:
        raise ValueError("No registered RGB images matched the mask input pattern")
    names = [p.name.split(".")[0] for p in images]
    if len(set(names)) != len(names):
        raise ValueError("CropFormer output names collide between registered images")
    for image, name in zip(images, names):
        path = root / "output/mask" / (name + ".png")
        with Image.open(image) as rgb, Image.open(path) as mask:
            if mask.size != rgb.size or mask.mode not in {"L", "I", "I;16", "P"}:
                raise ValueError(f"Mask shape/type mismatch: {path}")
            mask.load()
    return len(images)


def validate_teacher_output(output_dir, expected_dim, compressed):
    from semantic_3dgs_teacher.validation import validate_teacher

    try:
        return validate_teacher(output_dir, expected_dim=expected_dim, require_codec=compressed)
    except Exception as exc:
        raise RuntimeError(
            f"Incomplete or incompatible teacher at {output_dir}: {exc}. "
            "No files were removed. Choose a new output directory or explicitly use --force to rebuild."
        ) from exc


def run_mask_segmentation(repo_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    mask_dir = scene_root(args) / "output" / "mask"
    if args.skip_existing and not args.force and mask_dir.is_dir() and any(mask_dir.iterdir()):
        try:
            validate_masks(args)
        except Exception as exc:
            raise RuntimeError(
                f"Existing masks are incomplete: {exc}. Use --force to regenerate the pipeline."
            ) from exc
        log(f"stage masks skipped: existing masks in {mask_dir}")
        return {
            "stage": "masks",
            "status": "skipped_existing",
            "mask_dir": str(mask_dir),
            "num_existing_masks": existing_mask_count(args),
        }

    script = as_repo_path(repo_root, args.mask_predict_script)
    if not script.is_file():
        raise FileNotFoundError(f"CropFormer mask prediction script not found: {script}")
    cropformer_config = as_repo_path(repo_root, args.cropformer_config_file)
    cropformer_checkpoint = as_repo_path(repo_root, args.cropformer_checkpoint)
    if not cropformer_config.is_file():
        raise FileNotFoundError(f"CropFormer config not found: {cropformer_config}")
    if not cropformer_checkpoint.is_file():
        raise FileNotFoundError(f"CropFormer checkpoint not found: {cropformer_checkpoint}")

    command = [
        sys.executable,
        str(script),
        "--config-file",
        str(cropformer_config),
        "--root",
        str(Path(args.scannetpp_dslr_root).expanduser().resolve()),
        "--image_path_pattern",
        str(args.image_path_pattern),
        "--dataset",
        str(args.dataset),
        "--seq_name_list",
        str(args.seq_name),
        "--confidence-threshold",
        str(args.mask_confidence_threshold),
        "--opts",
        "MODEL.WEIGHTS",
        str(cropformer_checkpoint),
    ]
    log(f"stage masks start: {args.seq_name}")
    run_command(command, cwd=repo_root, cuda_visible_devices=args.cuda_visible_devices)
    validate_masks(args)
    log(f"stage masks complete: {existing_mask_count(args)} mask files -> {mask_dir}")
    return {
        "stage": "masks",
        "status": "completed",
        "mask_dir": str(mask_dir),
        "num_masks": existing_mask_count(args),
    }


def run_instance_clustering(repo_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    object_path = Path(args.object_dict_path).expanduser().resolve() if args.object_dict_path else default_object_dict_path(args)
    if args.skip_existing and not args.force and object_path.is_file():
        log(f"stage instances skipped: existing object_dict at {object_path}")
        return {
            "stage": "instances",
            "status": "skipped_existing",
            "object_dict_path": str(object_path),
        }

    from argparse import Namespace
    from semantic_3dgs_teacher.clustering.runner import run_instance_clustering as run_local_instance_clustering

    run_args = Namespace(
        seq_name=args.seq_name,
        seq_name_list=args.seq_name,
        config=args.config,
        dataset=args.dataset,
        debug=bool(args.debug),
        scannetpp_dslr_root=str(Path(args.scannetpp_dslr_root).expanduser().resolve()),
        scannetpp_dslr_mesh_name=args.scannetpp_dslr_mesh_name,
        scannetpp_dslr_depth_scale=args.scannetpp_dslr_depth_scale,
    )
    log(f"stage instances start: {args.seq_name}, config={args.config}")
    run_local_instance_clustering(run_args, config_dir=str(args.config_dir))

    if not object_path.is_file():
        raise FileNotFoundError(f"Instance clustering finished but object_dict.npy was not found: {object_path}")
    log(f"stage instances complete: {object_path}")
    return {
        "stage": "instances",
        "status": "completed",
        "object_dict_path": str(object_path),
    }


def run_teacher_build(repo_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.teacher_output_dir).expanduser().resolve() if args.teacher_output_dir else default_teacher_dir(args)
    if args.skip_existing and not args.force and output_dir.is_dir() and any(output_dir.iterdir()):
        validate_teacher_output(output_dir, 512, compressed=False)
        log(f"stage teacher skipped: existing teacher at {output_dir}")
        return {
            "stage": "teacher",
            "status": "skipped_existing",
            "teacher_dir": str(output_dir),
        }

    object_path = Path(args.object_dict_path).expanduser().resolve() if args.object_dict_path else default_object_dict_path(args)
    command = [
        sys.executable,
        "-m",
        "semantic_3dgs_teacher.build_teacher",
        "--config",
        args.config,
        "--config_dir",
        args.config_dir,
        "--seq_name",
        args.seq_name,
        "--scannetpp_dslr_root",
        str(Path(args.scannetpp_dslr_root).expanduser().resolve()),
        "--object_dict_path",
        str(object_path),
        "--output_dir",
        str(output_dir),
        "--alphaclip_package_path",
        str(as_repo_path(repo_root, args.alphaclip_package_path)),
        "--alphaclip_checkpoint",
        str(as_repo_path(repo_root, args.alphaclip_checkpoint)),
        "--clip_model",
        args.clip_model,
        "--tiered_large_crop_ratio",
        str(args.tiered_large_crop_ratio),
        "--tiered_large_mask_area_ratio",
        str(args.tiered_large_mask_area_ratio),
        "--tiered_high_min_coverage",
        str(args.tiered_high_min_coverage),
        "--tiered_mid_min_coverage",
        str(args.tiered_mid_min_coverage),
        "--tiered_consensus_mode",
        args.tiered_consensus_mode,
        "--tiered_inlier_similarity",
        str(args.tiered_inlier_similarity),
        "--tiered_outlier_similarity",
        str(args.tiered_outlier_similarity),
        "--batch_size",
        str(args.semantic_batch_size),
    ]
    if args.scannetpp_dslr_mesh_name:
        command.extend(["--scannetpp_dslr_mesh_name", args.scannetpp_dslr_mesh_name])
    if args.scannetpp_dslr_depth_scale is not None:
        command.extend(["--scannetpp_dslr_depth_scale", str(args.scannetpp_dslr_depth_scale)])
    if args.compressed_maps:
        command.append("--compressed_maps")
    if args.force:
        command.append("--force")
    log(f"stage teacher start: {args.seq_name}, object_dict={object_path}")
    run_command(command, cwd=repo_root, cuda_visible_devices=args.cuda_visible_devices)
    validate_teacher_output(output_dir, 512, compressed=False)
    log(f"stage teacher complete: {output_dir}")
    return {
        "stage": "teacher",
        "status": "completed",
        "teacher_dir": str(output_dir),
    }


def run_teacher_compression(repo_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    input_dir = Path(args.teacher_output_dir).expanduser().resolve() if args.teacher_output_dir else default_teacher_dir(args)
    output_dir = Path(args.compressed_teacher_output_dir).expanduser().resolve() if args.compressed_teacher_output_dir else default_compressed_teacher_dir(args)
    if args.skip_existing and not args.force and output_dir.is_dir() and any(output_dir.iterdir()):
        validate_teacher_output(output_dir, args.compressed_dim, compressed=True)
        log(f"stage compress skipped: existing compressed teacher at {output_dir}")
        return {
            "stage": "compress",
            "status": "skipped_existing",
            "compressed_teacher_dir": str(output_dir),
        }

    command = [
        sys.executable,
        "-m",
        "semantic_3dgs_teacher.compress_teacher",
        "--input_teacher_dir",
        str(input_dir),
        "--output_teacher_dir",
        str(output_dir),
        "--compressed_dim",
        str(args.compressed_dim),
        "--eval_batch_size",
        str(args.codec_eval_batch_size),
        "--latent_dtype",
        args.latent_dtype,
        "--frame_data_mode",
        args.frame_data_mode,
        "--device",
        args.device,
    ]
    if args.source_codec_path:
        command.extend(["--source_codec_path", str(as_repo_path(repo_root, args.source_codec_path))])
    else:
        command.extend(
            [
                "--codec_arch",
                "langsplat",
                "--loss_type",
                "langsplat",
                "--epochs",
                str(args.codec_epochs),
                "--batch_size",
                str(args.codec_batch_size),
                "--lr",
                str(args.codec_lr),
            ]
        )
    if args.write_decoded_features:
        command.append("--write_decoded_features")
    if args.write_decoded_visualization_teacher:
        command.append("--write_decoded_visualization_teacher")
    if args.force:
        command.append("--force")
    log(f"stage compress start: {input_dir} -> {output_dir}")
    run_command(command, cwd=repo_root, cuda_visible_devices=args.cuda_visible_devices)
    validate_teacher_output(output_dir, args.compressed_dim, compressed=True)
    log(f"stage compress complete: {output_dir}")
    return {
        "stage": "compress",
        "status": "completed",
        "compressed_teacher_dir": str(output_dir),
    }


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    pipeline_started_at = now_iso()
    pipeline_start = time.perf_counter()
    repo_root = resolve_repo_root(args.repo_root)
    stages = split_stages(args.stages)
    results: List[Dict[str, Any]] = []
    log(f"selected stages: {','.join(stages)}")

    if "masks" in stages:
        results.append(run_timed_stage("masks", run_mask_segmentation, repo_root, args))
    if "instances" in stages:
        results.append(run_timed_stage("instances", run_instance_clustering, repo_root, args))
    if "teacher" in stages:
        results.append(run_timed_stage("teacher", run_teacher_build, repo_root, args))
    if "compress" in stages:
        results.append(run_timed_stage("compress", run_teacher_compression, repo_root, args))

    teacher_dir = Path(args.teacher_output_dir).expanduser().resolve() if args.teacher_output_dir else default_teacher_dir(args)
    compressed_dir = (
        Path(args.compressed_teacher_output_dir).expanduser().resolve()
        if args.compressed_teacher_output_dir
        else default_compressed_teacher_dir(args)
    )
    pipeline_elapsed = time.perf_counter() - pipeline_start
    summary = {
        "format": "semantic_3dgs_teacher_pipeline_summary",
        "repo_root": str(repo_root),
        "seq_name": args.seq_name,
        "config": args.config,
        "dataset": args.dataset,
        "scene_root": str(scene_root(args)),
        "stages": stages,
        "results": results,
        "object_dict_path": str(Path(args.object_dict_path).expanduser().resolve() if args.object_dict_path else default_object_dict_path(args)),
        "teacher_dir": str(teacher_dir),
        "compressed_teacher_dir": str(compressed_dir),
        "timing": {
            "started_at": pipeline_started_at,
            "finished_at": now_iso(),
            "total_elapsed_seconds": round(pipeline_elapsed, 3),
            "total_elapsed_minutes": round(pipeline_elapsed / 60.0, 3),
        },
        "storage": {
            "teacher_dir": size_report(teacher_dir),
            "compressed_teacher_dir": size_report(compressed_dir),
        },
    }
    report_path = teacher_dir.parent / f"{teacher_dir.name}_pipeline_summary.json"
    save_json(report_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end semantic 3DGS teacher generation pipeline.")
    parser.add_argument("--repo_root", default="", help="Root used to resolve relative third_party/config/checkpoint paths. The package code itself is self-contained.")
    parser.add_argument("--stages", default="all", help="Comma-separated stages: masks,instances,teacher,compress, or all.")
    parser.add_argument("--seq_name", required=True)
    parser.add_argument("--config", default="scannetpp_dslr")
    parser.add_argument("--config_dir", default="", help="Optional directory containing config JSON files. Defaults to semantic_3dgs_teacher/configs.")
    parser.add_argument("--dataset", default="scannetpp_dslr")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--cuda_visible_devices", default="")

    parser.add_argument("--scannetpp_dslr_root", required=True)
    parser.add_argument("--scannetpp_dslr_mesh_name", default="")
    parser.add_argument("--scannetpp_dslr_depth_scale", type=float, default=None)
    parser.add_argument("--object_dict_path", default="")

    parser.add_argument("--mask_predict_script", default="third_party/Entity/Entityv2/CropFormer/demo_cropformer/mask_predict.py")
    parser.add_argument("--cropformer_config_file", default="third_party/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml")
    parser.add_argument("--cropformer_checkpoint", default="third_party/checkpoints/Mask2Former_hornet_3x_576d0b.pth")
    parser.add_argument("--image_path_pattern", default="images/*")
    parser.add_argument("--mask_confidence_threshold", type=float, default=0.5)

    parser.add_argument("--alphaclip_package_path", default="third_party/AlphaCLIP-main")
    parser.add_argument("--alphaclip_checkpoint", default="third_party/checkpoints/clip_b16_grit20m_fultune_2xe.pth")
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--semantic_batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--teacher_output_dir", default="")
    parser.add_argument("--compressed_teacher_output_dir", default="")
    parser.add_argument("--compressed_maps", action="store_true")
    parser.add_argument("--source_codec_path", default="")
    parser.add_argument("--compressed_dim", type=int, default=32)
    parser.add_argument("--latent_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--frame_data_mode", default="copy", choices=("reference", "copy", "symlink"))
    parser.add_argument("--write_decoded_features", action="store_true", default=True)
    parser.add_argument("--no_write_decoded_features", dest="write_decoded_features", action="store_false")
    parser.add_argument("--write_decoded_visualization_teacher", action="store_true", default=True)
    parser.add_argument("--no_write_decoded_visualization_teacher", dest="write_decoded_visualization_teacher", action="store_false")
    parser.add_argument("--codec_epochs", type=int, default=100)
    parser.add_argument("--codec_batch_size", type=int, default=512)
    parser.add_argument("--codec_eval_batch_size", type=int, default=4096)
    parser.add_argument("--codec_lr", type=float, default=7e-4)

    parser.add_argument("--tiered_large_crop_ratio", type=float, default=0.50)
    parser.add_argument("--tiered_large_mask_area_ratio", type=float, default=0.08)
    parser.add_argument("--tiered_high_min_coverage", type=float, default=0.45)
    parser.add_argument("--tiered_mid_min_coverage", type=float, default=0.20)
    parser.add_argument("--tiered_consensus_mode", default="medoid")
    parser.add_argument("--tiered_inlier_similarity", type=float, default=0.70)
    parser.add_argument("--tiered_outlier_similarity", type=float, default=0.62)
    return parser


def main() -> None:
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
