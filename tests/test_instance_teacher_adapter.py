import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.instance_teacher_adapter import InstanceTeacherStore


class InstanceTeacherStoreTest(unittest.TestCase):
    def _make_teacher(self, root: Path, *, reference=False):
        root.mkdir(parents=True)
        np.save(root / "instance_features.npy", np.asarray([[3.0, 0.0], [0.0, 4.0]], np.float16))
        np.save(root / "instance_ids.npy", np.asarray([10, 20], np.int64))
        (root / "semantic_codec.pth").write_bytes(b"test-placeholder")

        frame_root = root.parent / "frame_data" if reference else root
        maps = frame_root / "frame_instance_maps"
        weights = frame_root / "frame_weights"
        maps.mkdir(parents=True)
        weights.mkdir(parents=True)
        np.save(maps / "frame000.npy", np.asarray([[0, 1], [-1, 5]], np.int16))
        np.save(weights / "frame000.npy", np.asarray([[0.9, 0.5], [0.8, 1.0]], np.float16))
        summary = {
            "format": "maskclustering_final_compressed_instance_level_teacher",
            "frame_data": {
                "map_value_mode": "instance_index",
                "frame_instance_maps_dir": str(maps) if reference else "frame_instance_maps",
                "frame_weights_dir": str(weights) if reference else "frame_weights",
            },
        }
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (root / "frame_table.json").write_text(
            json.dumps([{"frame_id": "frame000", "shape": [2, 2]}]), encoding="utf-8"
        )

    def test_direct_map_and_pixel_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "teacher"
            self._make_teacher(root)
            store = InstanceTeacherStore(root, map_mode="auto")
            payload = store.load_frame("frame000.png", expected_dim=2)

            self.assertEqual(store.map_mode, "direct")
            torch.testing.assert_close(
                payload["region_features"], torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            )
            torch.testing.assert_close(
                payload["region_id_map"], torch.tensor([[0, 1], [-1, -1]], dtype=torch.int64)
            )
            torch.testing.assert_close(payload["region_area"], torch.tensor([1.0, 1.0]))
            torch.testing.assert_close(
                payload["pixel_quality"], torch.tensor([[0.89990234, 0.5], [0.0, 0.0]])
            )

    def test_dimension_and_image_shape_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "teacher"
            self._make_teacher(root)
            store = InstanceTeacherStore(root)
            self.assertIn("feature dim", store.frame_issue("frame000", expected_dim=3))
            self.assertIn("image shape", store.frame_issue("frame000", expected_dim=2, image_hw=(3, 2)))

    def test_reference_mode_frame_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "teacher"
            self._make_teacher(root, reference=True)
            store = InstanceTeacherStore(root)
            self.assertEqual(store.frame_instance_maps_dir, root.parent / "frame_data" / "frame_instance_maps")
            self.assertEqual(store.frame_weights_dir, root.parent / "frame_data" / "frame_weights")
            self.assertIsNone(store.frame_issue("frame000", expected_dim=2, image_hw=(2, 2)))


if __name__ == "__main__":
    unittest.main()
