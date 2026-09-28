"""One launcher for geometry reconstruction and instance-guided semantic training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.training_io import check_scene, check_teacher, model_artifacts, summarize_model


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("both", "geometry", "semantic"), default="both")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--instance-point-cloud", type=Path)
    parser.add_argument("--images", default="images")
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--geometry-iterations", type=int, default=30000)
    parser.add_argument("--semantic-steps", type=int, default=2000)
    parser.add_argument("--semantic-dim", type=int, default=12)
    parser.add_argument("--residual-dim", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only; no input validation, output creation, or training")
    return parser


def contains(parent, child):
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_args(args):
    for key in ("source", "geometry", "model", "teacher", "instance_point_cloud"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    if min(args.geometry_iterations, args.semantic_steps, args.semantic_dim, args.residual_dim) <= 0:
        raise ValueError("Iterations and dimensions must be positive")
    if args.voxel_size <= 0 or args.residual_dim > args.semantic_dim:
        raise ValueError("Use a positive voxel size and residual_dim <= semantic_dim")
    if args.stage != "geometry" and any(getattr(args, key) is None for key in
                                        ("model", "teacher", "instance_point_cloud")):
        raise ValueError("Semantic training requires --model, --teacher, and --instance-point-cloud")
    if args.model and (contains(args.geometry, args.model) or contains(args.model, args.geometry)):
        raise ValueError("Geometry and semantic output directories must be separate, not nested")
    outputs = [args.geometry] if args.stage == "geometry" else [args.model]
    if args.stage == "both":
        outputs.insert(0, args.geometry)
    for output in outputs:
        # Outputs below the dataset root are normal, but must not contain inputs.
        protected = [args.source, args.source / args.images, args.source / "sparse"]
        protected += [path for path in (args.teacher, args.instance_point_cloud) if path is not None]
        if any(contains(output, path) for path in protected):
            raise ValueError(f"Output overlaps an input: {output}")
        if any(contains(path.resolve(), output) for path in
               (args.source / args.images, args.source / "sparse")):
            raise ValueError(f"Output must not be inside camera/image inputs: {output}")
        if args.teacher and contains(args.teacher, output):
            raise ValueError(f"Output must not be inside the teacher: {output}")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Output must be new or empty; refusing to overwrite: {output}")
    return args


def build_commands(args):
    common = [sys.executable, "-u", str(REPO_ROOT / "train.py"),
              "--source_path", str(args.source), "--images", args.images,
              "--voxel_size", str(args.voxel_size), "--eval"]
    finish = ["--test_iterations", "0", "--skip_final_render", "--skip_final_eval",
              "--skip_final_semantic_eval", "--disable_gui"]
    commands = []
    if args.stage in ("both", "geometry"):
        command = common + ["--model_path", str(args.geometry), "--iterations", str(args.geometry_iterations),
                            "--lambda_semantic", "0", "--lambda_semantic_gate", "0",
                            "--lambda_semantic_residual", "0", "--lambda_instance_geometry", "0",
                            "--save_iterations", str(args.geometry_iterations)] + finish
        commands.append(("geometry", args.geometry, args.geometry_iterations, 0, command))
    if args.stage in ("both", "semantic"):
        final = args.geometry_iterations + args.semantic_steps
        command = common + [
            "--model_path", str(args.model), "--start_model_path", str(args.geometry),
            "--start_point_cloud_iteration", str(args.geometry_iterations), "--iterations", str(final),
            "--region_teachers", str(args.teacher), "--semantic_codec_path", str(args.teacher / "semantic_codec.pth"),
            "--instance_teacher_map_mode", "direct", "--semantic_3dgs", "--semantic_finetune_only",
            "--use_anchor_offset_semantic", "--semantic_feature_dim", "512",
            "--semantic_latent_dim", str(args.semantic_dim), "--semantic_teacher_dim", str(args.semantic_dim),
            "--semantic_residual_dim", str(args.residual_dim), "--semantic_gate_init", "-4.0",
            "--semantic_storage_dtype", "float16", "--use_instance_geometry_semantic",
            "--instance_geometry_residual_mode", "sparse", "--instance_geometry_teacher", str(args.teacher),
            "--instance_geometry_object_dict", str(args.teacher / "object_dict.npy"),
            "--instance_geometry_point_cloud", str(args.instance_point_cloud),
            "--instance_geometry_initialize_interior", "--instance_geometry_unknown_residual",
            "--lambda_semantic", "1.0", "--lambda_semantic_gate", "1e-3", "--lambda_semantic_residual", "1e-4",
            "--lambda_instance_geometry", "0.1", "--semantic_min_alpha", "0.05", "--semantic_debug_interval", "500",
            "--save_iterations", str(final), "--checkpoint_iterations", str(final)] + finish
        commands.append(("semantic", args.model, final, args.geometry_iterations, command))
    return commands


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_process(command, output):
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.perf_counter()
    with (output / "training.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    return status, time.perf_counter() - start


def run_pipeline(args):
    args = validate_args(args)
    commands = build_commands(args)
    if args.dry_run:
        print(json.dumps({"validated_inputs": False, "commands": [item[-1] for item in commands]}, indent=2))
        return []
    scene, train_frames = check_scene(args.source, args.images)
    if args.stage != "geometry":
        scene["teacher"] = check_teacher(args.teacher, args.semantic_dim, args.instance_point_cloud, train_frames)
    if args.stage == "semantic":
        model_artifacts(args.geometry, args.geometry_iterations, "geometry")
    print(json.dumps({"preflight": scene}, indent=2), flush=True)
    results = []
    for stage, output, iteration, start, command in commands:
        if stage == "semantic":
            model_artifacts(args.geometry, args.geometry_iterations, "geometry")
        output.mkdir(parents=True, exist_ok=True)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        write_json(output / "launch_config.json", {"options": config, "stage": stage, "command": command,
                                                   "split": scene, "train_frame_names": sorted(train_frames)})
        status, seconds = run_process(command, output)
        write_json(output / "process_status.json", {"returncode": status, "wall_seconds": seconds})
        if status != 0:
            raise RuntimeError(f"{stage} training failed (exit={status}); see {output / 'training.log'}")
        result = summarize_model(output, iteration, stage, seconds, start)
        write_json(output / "run_summary.json", result)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    write_json(commands[-1][1] / "pipeline_summary.json", {
        "stage_selection": args.stage, "stages": results,
        "training_process_wall_seconds": sum(item["wall_seconds_including_setup_and_final_save"] for item in results),
        "excludes": "preflight, teacher generation, between-stage checks, and separate evaluation"})
    return results


def main():
    try:
        run_pipeline(build_parser().parse_args())
    except KeyboardInterrupt:
        print("Training interrupted; completed outputs and logs were retained.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
