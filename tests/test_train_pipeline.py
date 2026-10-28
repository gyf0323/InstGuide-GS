import argparse
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import train_pipeline as launcher


class TrainPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, stage="both"):
        return launcher.build_parser().parse_args([
            "--stage", stage, "--source", str(self.root / "scene"),
            "--geometry", str(self.root / "geometry"), "--model", str(self.root / "semantic"),
            "--teacher", str(self.root / "teacher"), "--instance-point-cloud", str(self.root / "mesh.ply"),
            "--semantic-dim", "16"])

    def test_main_method_flags_and_iteration_accounting(self):
        commands = launcher.build_commands(self.args())
        self.assertEqual([row[0] for row in commands], ["geometry", "semantic"])
        geometry, semantic = (row[-1] for row in commands)
        self.assertNotIn("--semantic_3dgs", geometry)
        self.assertIn("--eval", geometry)
        self.assertIn("--eval", semantic)
        for key, value in {"--iterations": "32000", "--semantic_latent_dim": "16",
                           "--semantic_residual_dim": "4", "--semantic_gate_init": "-4.0",
                           "--instance_geometry_residual_mode": "sparse",
                           "--lambda_instance_geometry": "0.1", "--semantic_storage_dtype": "float16"}.items():
            self.assertEqual(semantic[semantic.index(key) + 1], value)
        for key in ("--instance_geometry_initialize_interior", "--instance_geometry_unknown_residual",
                    "--use_anchor_offset_semantic", "--semantic_finetune_only"):
            self.assertIn(key, semantic)
        self.assertNotIn("--use_depth_visible_registration", semantic)
        self.assertNotIn("--use_strict_topk_registration", semantic)

    def test_public_defaults_match_reference_semantic_command(self):
        args = launcher.build_parser().parse_args([
            "--source", str(self.root / "scene"), "--geometry", str(self.root / "geometry"),
            "--model", str(self.root / "semantic"), "--teacher", str(self.root / "teacher"),
            "--instance-point-cloud", str(self.root / "mesh.ply")])
        self.assertEqual((args.geometry_iterations, args.semantic_steps), (30000, 2000))
        self.assertEqual((args.semantic_dim, args.residual_dim), (12, 4))
        self.assertEqual(args.voxel_size, 0.003)
        semantic = launcher.build_commands(args)[1][-1]
        self.assertEqual(semantic[semantic.index("--semantic_latent_dim") + 1], "12")
        self.assertEqual(semantic[semantic.index("--semantic_gate_init") + 1], "-4.0")
        self.assertIn("--instance_geometry_initialize_interior", semantic)
        self.assertIn("--instance_geometry_unknown_residual", semantic)

    def test_selected_stage_only(self):
        for stage in ("geometry", "semantic"):
            commands = launcher.build_commands(self.args(stage))
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][0], stage)

    def test_commands_use_existing_training_options(self):
        from arguments import ModelParams, OptimizationParams, PipelineParams
        parser = argparse.ArgumentParser()
        ModelParams(parser)
        OptimizationParams(parser)
        PipelineParams(parser)
        options = set(parser._option_string_actions)
        tree = ast.parse((launcher.REPO_ROOT / "train.py").read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                options.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                               and isinstance(arg.value, str) and arg.value.startswith("--"))
        for stage, _, _, _, command in launcher.build_commands(self.args()):
            for arg in command:
                if arg.startswith("--"):
                    self.assertIn(arg, options, (stage, arg))

    def test_rejects_overwrite_nesting_and_input_overlap(self):
        args = self.args()
        args.geometry.mkdir()
        (args.geometry / "existing.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            launcher.validate_args(args)
        args = self.args()
        args.model = args.geometry / "child"
        with self.assertRaisesRegex(ValueError, "nested"):
            launcher.validate_args(args)
        args = self.args("geometry")
        args.geometry = args.source
        with self.assertRaisesRegex(ValueError, "overlaps"):
            launcher.validate_args(args)

    def test_dry_run_has_no_side_effects(self):
        args = self.args()
        args.dry_run = True
        with patch.object(launcher, "check_scene") as check, patch("builtins.print"):
            self.assertEqual(launcher.run_pipeline(args), [])
            check.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_bad_teacher_stops_before_geometry(self):
        with patch.object(launcher, "check_scene", return_value=({}, {"a": (2, 2)})), \
             patch.object(launcher, "check_teacher", side_effect=ValueError("bad teacher")), \
             patch.object(launcher, "run_process") as run:
            with self.assertRaisesRegex(ValueError, "bad teacher"):
                launcher.run_pipeline(self.args())
            run.assert_not_called()

    def test_geometry_failure_stops_semantic(self):
        with patch.object(launcher, "check_scene", return_value=({}, {})), \
             patch.object(launcher, "check_teacher", return_value={}), \
             patch.object(launcher, "run_process", return_value=(1, 2.0)) as run, \
             patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "geometry training failed"):
                launcher.run_pipeline(self.args())
            self.assertEqual(run.call_count, 1)
        self.assertFalse((self.root / "semantic").exists())
        status = json.loads((self.root / "geometry/process_status.json").read_text())
        self.assertEqual(status["returncode"], 1)

    def test_stage_summaries_and_start_iteration(self):
        def summary(root, iteration, stage, seconds, start):
            return {"stage": stage, "optimization_steps": iteration - start,
                    "wall_seconds_including_setup_and_final_save": seconds}
        with patch.object(launcher, "check_scene", return_value=({"train_views": 7, "test_views": 1}, {})), \
             patch.object(launcher, "check_teacher", return_value={}), \
             patch.object(launcher, "model_artifacts", return_value=[]), \
             patch.object(launcher, "run_process", side_effect=[(0, 10.0), (0, 2.0)]), \
             patch.object(launcher, "summarize_model", side_effect=summary), patch("builtins.print"):
            results = launcher.run_pipeline(self.args())
        self.assertEqual([r["optimization_steps"] for r in results], [30000, 2000])
        summary = json.loads((self.root / "semantic/pipeline_summary.json").read_text())
        self.assertEqual(summary["training_process_wall_seconds"], 12.0)

    def test_console_logging_and_child_exit(self):
        with patch("builtins.print"):
            status, seconds = launcher.run_process([sys.executable, "-c", "print(123); raise SystemExit(3)"], self.root)
        self.assertEqual(status, 3)
        self.assertGreaterEqual(seconds, 0)
        self.assertIn("123", (self.root / "training.log").read_text())


if __name__ == "__main__":
    unittest.main()
