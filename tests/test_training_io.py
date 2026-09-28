import struct
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image

from tools.training_io import check_scene, model_artifacts, summarize_model


class TrainingIOTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def model(self, stage="semantic", feat_bank=True):
        folder = self.root / "point_cloud/iteration_32000"
        folder.mkdir(parents=True)
        config = Namespace(appearance_dim=32, use_feat_bank=feat_bank, use_anchor_offset_semantic=True,
                           semantic_storage_dtype="float16", semantic_codec_path="old_server/semantic_codec.pth")
        (self.root / "cfg_args").write_text(str(config))
        names = ["point_cloud.ply", "opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt", "embedding_appearance.pt"]
        if feat_bank:
            names.append("feature_bank_mlp.pt")
        if stage == "semantic":
            names += ["semantic_decoder.pt", "semantic_anchor_offset.pt"]
            (self.root / "semantic_codec.pth").write_bytes(b"codec")
        for name in names:
            (folder / name).write_bytes(b"weight")
        (folder / "debug.npy").write_bytes(b"not inference" * 100)
        (self.root / "chkpnt32000.pth").write_bytes(b"optimizer" * 100)
        return folder, names

    def test_size_includes_optional_networks_codec_and_config_not_logs(self):
        folder, names = self.model()
        result = summarize_model(self.root, 32000, "semantic", 20, 30000)
        expected = sum((folder / n).stat().st_size for n in names) + 5 + (self.root / "cfg_args").stat().st_size
        self.assertEqual(result["inference_artifact_bytes"], expected)
        self.assertEqual(result["semantic_steps"], 2000)
        self.assertEqual(len(result["inference_artifact_files"]), len(names) + 2)

    def test_missing_inference_weights_fail_instead_of_undercount(self):
        folder, _ = self.model()
        (folder / "embedding_appearance.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing inference"):
            model_artifacts(self.root, 32000, "semantic")

    def test_geometry_has_same_accounting_boundary(self):
        folder, names = self.model(stage="geometry", feat_bank=False)
        result = summarize_model(self.root, 32000, "geometry", 20)
        self.assertEqual(len(result["inference_artifact_files"]), len(names) + 1)
        self.assertNotIn("semantic_steps", result)

    def test_missing_portable_codec_fails(self):
        self.model()
        (self.root / "semantic_codec.pth").unlink()
        with self.assertRaisesRegex(ValueError, "portable model codec"):
            model_artifacts(self.root, 32000, "semantic")

    def scene(self, binary=False):
        sparse = self.root / "sparse/0"
        sparse.mkdir(parents=True)
        (self.root / "images").mkdir()
        (sparse / "points3D.ply").write_bytes(b"ply")
        for index in range(9):
            Image.new("RGB", (3, 2)).save(self.root / "images" / f"frame{index:03d}.png")
        if binary:
            with (sparse / "cameras.bin").open("wb") as handle:
                handle.write(struct.pack("<QiiQQdddd", 1, 1, 1, 3, 2, 1., 1., 1.5, 1.))
            with (sparse / "images.bin").open("wb") as handle:
                handle.write(struct.pack("<Q", 9))
                for i in reversed(range(9)):
                    handle.write(struct.pack("<idddddddi", i+1, 1., 0., 0., 0., 0., 0., 0., 1))
                    handle.write(f"frame{i:03d}.png".encode() + b"\x00")
                    handle.write(struct.pack("<Q", 0))
        else:
            (sparse / "cameras.txt").write_text("1 PINHOLE 3 2 1 1 1.5 1\n")
            (sparse / "images.txt").write_text("".join(f"{i+1} 1 0 0 0 0 0 0 1 frame{i:03d}.png\n\n" for i in reversed(range(9))))

    def test_text_scene_uses_sorted_every_eighth_split(self):
        self.scene()
        result, train = check_scene(self.root)
        self.assertEqual((result["train_views"], result["test_views"]), (7, 2))
        self.assertNotIn("frame000", train)
        self.assertNotIn("frame008", train)

    def test_binary_colmap_does_not_require_text_files(self):
        self.scene(binary=True)
        result, train = check_scene(self.root)
        self.assertEqual(result["colmap_format"], "binary")
        self.assertEqual(len(train), 7)

    def test_missing_registered_rgb_fails(self):
        self.scene()
        (self.root / "images/frame001.png").unlink()
        with self.assertRaises(FileNotFoundError):
            check_scene(self.root)


if __name__ == "__main__":
    unittest.main()
