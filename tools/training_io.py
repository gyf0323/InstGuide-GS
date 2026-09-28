"""CPU preflight and consistent inference-artifact accounting for launch scripts."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_config(root):
    path = Path(root) / "cfg_args"
    expr = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != "Namespace" or expr.args:
        raise ValueError(f"Invalid Namespace configuration: {path}")
    if any(item.arg is None for item in expr.keywords):
        raise ValueError(f"Unsupported configuration unpacking: {path}")
    return {item.arg: ast.literal_eval(item.value) for item in expr.keywords}


def model_artifacts(root, iteration, stage):
    root = Path(root)
    config = read_config(root)
    folder = root / "point_cloud" / f"iteration_{int(iteration)}"
    required = {"point_cloud.ply", "opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"}
    if int(config.get("appearance_dim", 32)) > 0:
        required.add("embedding_appearance.pt")
    if config.get("use_feat_bank", False):
        required.add("feature_bank_mlp.pt")
    if stage == "semantic":
        if config.get("use_anchor_offset_semantic", False) or config.get("semantic_storage_dtype") == "float16":
            required.add("semantic_anchor_offset.pt")
        codec = root / "semantic_codec.pth"
        if config.get("semantic_codec_path") and not codec.is_file():
            raise ValueError(f"Missing portable model codec: {codec}")
        if not codec.is_file() and not (folder / "semantic_decoder.pt").is_file():
            raise ValueError("Semantic model has neither a codec nor a saved decoder")
    missing = [str(folder / name) for name in sorted(required) if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"Missing inference artifacts: {missing}")
    optional = {"embedding_appearance.pt", "feature_bank_mlp.pt"}
    if stage == "semantic":
        optional |= {"semantic_decoder.pt", "semantic_anchor_offset.pt"}
    files = [folder / name for name in sorted(required | optional) if (folder / name).is_file()]
    if stage == "semantic" and (root / "semantic_codec.pth").is_file():
        files.append(root / "semantic_codec.pth")
    files.append(root / "cfg_args")
    for path in files:
        if path.stat().st_size == 0:
            raise ValueError(f"Empty inference artifact: {path}")
    return files


def summarize_model(root, iteration, stage, seconds, start_iteration=0):
    if seconds < 0 or iteration <= start_iteration or start_iteration < 0:
        raise ValueError("Invalid timing or iteration interval")
    root = Path(root)
    files = model_artifacts(root, iteration, stage)
    weights = [path for path in files if path.name != "cfg_args"]
    result = {
        "stage": stage, "iteration": int(iteration),
        "optimization_steps": int(iteration - start_iteration),
        "wall_seconds_including_setup_and_final_save": float(seconds),
        "timing_boundary": "train.py process and console logging; excludes wrapper preflight and separate evaluation; includes process setup and final saves",
        "inference_artifact_bytes": sum(path.stat().st_size for path in files),
        "inference_weight_bytes": sum(path.stat().st_size for path in weights),
        "size_boundary": "saved inference weights, codec, and cfg_args; excludes optimizer checkpoints, logs, GT, teachers, and evaluation outputs",
        "inference_artifact_files": [str(path.relative_to(root)) for path in files],
    }
    if stage == "semantic":
        result["semantic_steps"] = result["optimization_steps"]
    return result


def check_scene(source, images="images"):
    from PIL import Image

    source = Path(source)
    sparse = source / "sparse/0"
    # Use the existing COLMAP parser without importing scene's CUDA-dependent __init__.
    spec = importlib.util.spec_from_file_location("_preflight_colmap", REPO_ROOT / "scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    try:
        extr = loader.read_extrinsics_binary(str(sparse / "images.bin"))
        intr = loader.read_intrinsics_binary(str(sparse / "cameras.bin"))
        fmt = "binary"
    except (FileNotFoundError, OSError):
        extr = loader.read_extrinsics_text(str(sparse / "images.txt"))
        intr = loader.read_intrinsics_text(str(sparse / "cameras.txt"))
        fmt = "text"
    if not any((sparse / ("points3D." + suffix)).is_file() for suffix in ("ply", "bin", "txt")):
        raise ValueError(f"Missing initial point cloud in {sparse}")
    frames = {}
    for record in extr.values():
        camera = intr[record.camera_id]
        if camera.model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError(f"Undistorted cameras required, got {camera.model}")
        path = source / images / Path(record.name).name
        if path.stem in frames:
            raise ValueError(f"Duplicate registered image stem: {path.stem}")
        with Image.open(path) as image:
            if image.size != (camera.width, camera.height):
                raise ValueError(f"RGB/COLMAP size mismatch: {path}")
            frames[path.stem] = (image.height, image.width)
    ordered = sorted(frames)
    train = {name: frames[name] for i, name in enumerate(ordered) if i % 8 != 0}
    test = [name for i, name in enumerate(ordered) if i % 8 == 0]
    if not train:
        raise ValueError("The --eval split has no training views")
    return {"colmap_format": fmt, "registered_views": len(frames),
            "train_views": len(train), "test_views": len(test)}, train


def check_teacher(teacher, dimension, instance_point_cloud, train_frames):
    from semantic_3dgs_teacher.validation import validate_teacher
    from utils.instance_teacher_adapter import InstanceTeacherStore

    if not Path(instance_point_cloud).is_file():
        raise ValueError(f"Missing exact indexed instance cloud: {instance_point_cloud}")
    result = validate_teacher(teacher, expected_dim=dimension, require_codec=True,
                              expected_frames=train_frames, allow_npz=False)
    store = InstanceTeacherStore(teacher, map_mode="direct")
    for frame, shape in train_frames.items():
        issue = store.frame_issue(frame, dimension, shape)
        if issue:
            raise ValueError(issue)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--source", required=True, type=Path)
    preflight.add_argument("--images", default="images")
    preflight.add_argument("--teacher", type=Path)
    preflight.add_argument("--dimension", type=int)
    preflight.add_argument("--geometry", type=Path)
    preflight.add_argument("--start-iteration", type=int, default=30000)
    preflight.add_argument("--final-iteration", type=int, default=32000)
    preflight.add_argument("--instance-point-cloud", type=Path)
    summary = commands.add_parser("summary")
    summary.add_argument("--model", type=Path, required=True)
    summary.add_argument("--iteration", type=int, required=True)
    summary.add_argument("--stage", choices=("geometry", "semantic"), required=True)
    summary.add_argument("--seconds", type=float, required=True)
    summary.add_argument("--start-iteration", type=int, default=0)
    args = parser.parse_args()
    if args.command == "summary":
        result = summarize_model(args.model, args.iteration, args.stage, args.seconds, args.start_iteration)
        path = args.model / "run_summary.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(path)
    else:
        result, train = check_scene(args.source, args.images)
        if args.teacher:
            if not args.geometry or not args.dimension or not args.instance_point_cloud:
                parser.error("Semantic preflight requires geometry, dimension, and instance-point-cloud")
            if args.final_iteration <= args.start_iteration:
                parser.error("final-iteration must exceed start-iteration")
            model_artifacts(args.geometry, args.start_iteration, "geometry")
            result["teacher"] = check_teacher(args.teacher, args.dimension, args.instance_point_cloud, train)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
