import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from semantic_3dgs_teacher import pipeline
from semantic_3dgs_teacher.codec_utils import SceneFeatureCodec, save_semantic_codec
from semantic_3dgs_teacher.validation import validate_teacher
from utils.instance_teacher_adapter import InstanceTeacherStore


class ReleasePipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def teacher(self, reference=False, npz=False):
        root = self.root / "teacher"
        root.mkdir()
        features = np.zeros((2, 16), dtype=np.float16)
        features[:, 0] = 1
        np.save(root / "instance_features.npy", features)
        np.save(root / "instance_ids.npy", np.array([10, 20]))
        np.save(root / "object_dict.npy", {10: {"point_ids": [0]}, 20: {"point_ids": [1]}})
        frames = self.root / "frames" if reference else root
        for name, array in (("frame_instance_maps", np.array([[0, 1], [-1, 1]])),
                            ("frame_weights", np.array([[0.9, 0.5], [0, 0.5]], dtype=np.float16))):
            folder = frames / name
            folder.mkdir(parents=True)
            if npz:
                np.savez_compressed(folder / "a.npz", data=array)
            else:
                np.save(folder / "a.npy", array)
        summary = {"format": "maskclustering_mlp_compressed_instance_level_teacher",
                   "frame_data": {name + "_dir": str(frames / name)
                                  for name in ("frame_instance_maps", "frame_weights")}}
        (root / "summary.json").write_text(json.dumps(summary))
        (root / "frame_table.json").write_text(json.dumps([{"frame_id": "a", "shape": [1, 2, 2]}]))
        save_semantic_codec(root / "semantic_codec.pth", SceneFeatureCodec(512, 16))
        return root

    def test_local_and_reference_teachers_match_runtime_resolution(self):
        root = self.teacher(reference=True)
        result = validate_teacher(root, expected_dim=16, require_codec=True, expected_frames={"a": (2, 2)})
        store = InstanceTeacherStore(root, map_mode="direct")
        self.assertEqual(Path(result["frame_weights_dir"]), store.frame_weights_dir)
        self.assertEqual(result["checked_frames"], 1)

    def test_missing_map_weight_and_codec_are_rejected(self):
        root = self.teacher()
        for name in ("frame_instance_maps/a.npy", "frame_weights/a.npy", "semantic_codec.pth"):
            file = root / name
            saved = file.read_bytes()
            file.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                validate_teacher(root, require_codec=True)
            file.write_bytes(saved)

    def test_wrong_dim_zero_and_nonfinite_features_are_rejected(self):
        root = self.teacher()
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_teacher(root, expected_dim=32)
        np.save(root / "instance_features.npy", np.zeros((2, 16)))
        with self.assertRaisesRegex(ValueError, "zero"):
            validate_teacher(root)
        np.save(root / "instance_features.npy", np.full((2, 16), np.nan))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_teacher(root)

    def test_bad_indices_weights_shapes_and_frame_coverage_are_rejected(self):
        root = self.teacher()
        with self.assertRaisesRegex(ValueError, "missing selected frames"):
            validate_teacher(root, expected_frames={"b": (2, 2)})
        with self.assertRaisesRegex(ValueError, "RGB/teacher"):
            validate_teacher(root, expected_frames={"a": (3, 2)})
        np.save(root / "frame_instance_maps/a.npy", np.array([[0, 2], [0, 1]]))
        with self.assertRaisesRegex(ValueError, "index out of range"):
            validate_teacher(root)
        np.save(root / "frame_instance_maps/a.npy", np.zeros((2, 2), dtype=int))
        np.save(root / "frame_weights/a.npy", np.full((2, 2), np.nan))
        with self.assertRaisesRegex(ValueError, "weights"):
            validate_teacher(root)

    def test_codec_is_strict_loaded_not_only_dimension_checked(self):
        root = self.teacher()
        codec = SceneFeatureCodec(512, 16)
        payload = {"metadata": codec.metadata(), "state_dict": {}}
        torch.save(payload, root / "semantic_codec.pth")
        with self.assertRaises(RuntimeError):
            validate_teacher(root, require_codec=True)

    def test_geometry_membership_must_cover_feature_ids(self):
        root = self.teacher()
        np.save(root / "object_dict.npy", {10: {"point_ids": [0]}})
        with self.assertRaisesRegex(ValueError, "membership"):
            validate_teacher(root)

    def test_semantic_preflight_cli_accepts_reference_teacher(self):
        root = self.teacher(reference=True)
        scene = self.root / "scene"
        (scene / "images").mkdir(parents=True)
        (scene / "sparse/0").mkdir(parents=True)
        for name in ("0", "a"):
            Image.new("RGB", (2, 2)).save(scene / "images" / (name + ".png"))
        (scene / "sparse/0/cameras.txt").write_text("1 PINHOLE 2 2 1 1 1 1\n")
        (scene / "sparse/0/images.txt").write_text("1 1 0 0 0 0 0 0 1 0.png\n\n2 1 0 0 0 0 0 0 1 a.png\n\n")
        (scene / "sparse/0/points3D.ply").write_bytes(b"ply")
        geometry = self.root / "geometry"
        folder = geometry / "point_cloud/iteration_30000"
        folder.mkdir(parents=True)
        (geometry / "cfg_args").write_text("Namespace(appearance_dim=0, use_feat_bank=False)")
        for name in ("point_cloud.ply", "opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"):
            (folder / name).write_bytes(b"fixture")
        cloud = scene / "mesh.ply"
        cloud.write_bytes(b"ply")
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-B", str(repo / "tools/training_io.py"), "preflight",
                                 "--source", str(scene), "--teacher", str(root), "--dimension", "16",
                                 "--geometry", str(geometry), "--instance-point-cloud", str(cloud)],
                                cwd=repo, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["teacher"]["checked_frames"], 1)

    def test_npz_generator_output_checked_but_training_requires_npy(self):
        root = self.teacher(npz=True)
        self.assertEqual(validate_teacher(root)["checked_frames"], 1)
        with self.assertRaisesRegex(ValueError, "Missing frame array"):
            validate_teacher(root, allow_npz=False)

    def test_complete_compression_skips_but_partial_output_fails_without_overwrite(self):
        root = self.teacher()
        args = pipeline.build_parser().parse_args([
            "--seq_name", "scene", "--scannetpp_dslr_root", str(self.root),
            "--teacher_output_dir", str(self.root / "raw"),
            "--compressed_teacher_output_dir", str(root), "--compressed_dim", "16"])
        with patch.object(pipeline, "run_command") as run:
            self.assertEqual(pipeline.run_teacher_compression(self.root, args)["status"], "skipped_existing")
            (root / "semantic_codec.pth").unlink()
            with self.assertRaisesRegex(RuntimeError, "Incomplete or incompatible"):
                pipeline.run_teacher_compression(self.root, args)
            run.assert_not_called()
        self.assertTrue((root / "instance_features.npy").is_file())
        self.assertIn("semantic_3dgs_teacher_16d", str(pipeline.default_compressed_teacher_dir(args)))

    def test_public_defaults_match_reference_teacher_command(self):
        args = pipeline.build_parser().parse_args([
            "--seq_name", "scene", "--scannetpp_dslr_root", str(self.root)])
        expected = {
            "config": "scannetpp_dslr_small", "dataset": "scannetpp_dslr",
            "compressed_dim": 12, "latent_dtype": "float16", "frame_data_mode": "copy",
            "clip_model": "ViT-B/16", "semantic_batch_size": 32,
            "codec_epochs": 100, "codec_batch_size": 512,
            "codec_eval_batch_size": 4096, "codec_lr": 7e-4,
            "write_decoded_features": False, "write_decoded_visualization_teacher": False,
        }
        for key, value in expected.items():
            self.assertEqual(getattr(args, key), value, key)
        explicit = pipeline.build_parser().parse_args([
            "--seq_name", "scene", "--scannetpp_dslr_root", str(self.root),
            "--config", "scannetpp_dslr", "--compressed_dim", "16",
            "--write_decoded_features", "--write_decoded_visualization_teacher"])
        self.assertEqual(explicit.config, "scannetpp_dslr")
        self.assertEqual(explicit.compressed_dim, 16)
        self.assertTrue(explicit.write_decoded_features)
        self.assertTrue(explicit.write_decoded_visualization_teacher)

    def test_partial_mask_directory_is_not_treated_as_complete(self):
        scene = self.root / "scene"
        for name in ("images", "sparse/0", "output/mask"):
            (scene / name).mkdir(parents=True)
        (scene / "sparse/0/images.txt").write_text("1 1 0 0 0 0 0 0 1 a.png\n\n2 1 0 0 0 0 0 0 1 b.png\n\n")
        for name in ("a", "b"):
            Image.new("RGB", (2, 2)).save(scene / "images" / (name + ".png"))
        Image.new("L", (2, 2)).save(scene / "output/mask/a.png")
        args = pipeline.build_parser().parse_args(["--seq_name", "scene", "--scannetpp_dslr_root", str(self.root)])
        with patch.object(pipeline, "run_command") as run:
            with self.assertRaisesRegex(RuntimeError, "masks are incomplete"):
                pipeline.run_mask_segmentation(self.root, args)
            run.assert_not_called()
        Image.new("L", (2, 2)).save(scene / "output/mask/b.png")
        self.assertEqual(pipeline.run_mask_segmentation(self.root, args)["status"], "skipped_existing")


if __name__ == "__main__":
    unittest.main()
