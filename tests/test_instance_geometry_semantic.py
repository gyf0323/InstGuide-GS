import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from utils.instance_geometry_semantic import (
    GEOMETRY_STATE_BOUNDARY,
    GEOMETRY_STATE_INTERIOR,
    GEOMETRY_STATE_UNKNOWN,
    InstanceGeometryBinding,
    classify_anchor_geometry,
    export_instance_geometry_binding,
    geometry_binding_is_complete,
    load_instance_geometry_binding,
    prepare_instance_geometry_binding,
    validate_instance_geometry_binding,
)
from tools.export_instance_geometry_binding import build_point_instance_owners


class InstanceGeometrySemanticTest(unittest.TestCase):
    def test_export_excludes_conflicting_membership(self):
        owners, diagnostics = build_point_instance_owners(
            5,
            {
                10: {"point_ids": [0, 1, 2]},
                20: {"point_ids": [2, 3, 99]},
            },
            {10, 20},
        )
        np.testing.assert_array_equal(owners, np.asarray([10, 10, -2, 20, -1], np.int64))
        self.assertEqual(diagnostics["conflicting_points"], 1)
        self.assertEqual(diagnostics["invalid_point_ids"], 1)

    def test_export_builds_complete_atomic_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = root / "teacher"
            teacher.mkdir()
            np.save(teacher / "instance_ids.npy", np.asarray([10, 20], np.int64))
            object_path = root / "object_dict.npy"
            np.save(
                object_path,
                {
                    10: {"point_ids": [0, 1, 2]},
                    20: {"point_ids": [2, 3, 99]},
                },
                allow_pickle=True,
            )
            point_path = root / "mesh.ply"
            point_path.write_bytes(b"placeholder")
            points = np.asarray(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
                np.float32,
            )
            with patch("utils.instance_geometry_semantic._load_ply_xyz", return_value=points):
                metadata = export_instance_geometry_binding(
                    teacher,
                    object_path,
                    point_path,
                )

            binding_dir = teacher / "geometry_binding"
            self.assertTrue(geometry_binding_is_complete(binding_dir))
            self.assertEqual(metadata["format_version"], 2)
            self.assertEqual(metadata["exported_point_count"], 3)
            self.assertEqual(metadata["conflicting_points"], 1)
            self.assertEqual(metadata["invalid_point_ids"], 1)
            binding = validate_instance_geometry_binding(binding_dir, np.asarray([10, 20], np.int64))
            np.testing.assert_array_equal(binding.raw_instance_ids, np.asarray([10, 10, 20], np.int64))
            self.assertFalse(any(path.name.startswith(".geometry_binding.tmp-") for path in teacher.iterdir()))

            with self.assertRaisesRegex(RuntimeError, "different teacher|instance count|absent from"):
                validate_instance_geometry_binding(binding_dir, np.asarray([10, 30], np.int64))

    def test_binding_remaps_raw_instance_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "points_xyz.npy", np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32))
            np.save(root / "point_instance_ids.npy", np.asarray([10, 20, 999], np.int64))
            binding = load_instance_geometry_binding(root, np.asarray([20, 10], np.int64))
        self.assertEqual(binding.points_xyz.shape, (2, 3))
        np.testing.assert_array_equal(binding.instance_indices, np.asarray([1, 0], np.int64))
        self.assertEqual(binding.metadata["dropped_unknown_instance_points"], 1)

    def test_prepare_auto_generates_then_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "scene"
            teacher = root / "teacher" / "scannetpp_dslr_small"
            source.mkdir()
            teacher.mkdir(parents=True)
            np.save(teacher / "instance_ids.npy", np.asarray([10], np.int64))
            np.save(
                teacher / "object_dict.npy",
                {10: {"point_ids": [0, 1]}},
                allow_pickle=True,
            )
            (source / "mesh_aligned_0.05.ply").write_bytes(b"placeholder")
            points = np.asarray([[0, 0, 0], [1, 0, 0]], np.float32)
            messages = []
            with patch("utils.instance_geometry_semantic._load_ply_xyz", return_value=points):
                first = prepare_instance_geometry_binding(
                    teacher,
                    np.asarray([10], np.int64),
                    source,
                    log_callback=messages.append,
                )
            self.assertTrue(geometry_binding_is_complete(first.root))
            self.assertTrue(any("preparing it automatically" in message for message in messages))

            with patch(
                "utils.instance_geometry_semantic._load_ply_xyz",
                side_effect=AssertionError("cache reuse must not reload the source PLY"),
            ):
                second = prepare_instance_geometry_binding(
                    teacher,
                    np.asarray([10], np.int64),
                    source,
                    log_callback=messages.append,
                )
            self.assertEqual(first.root, second.root)
            self.assertTrue(any("Reusing cached" in message for message in messages))

    def test_classifies_interior_boundary_and_unknown(self):
        left = np.asarray(
            [[-0.20, y, z] for y in (-0.10, 0.0, 0.10) for z in (-0.10, 0.0, 0.10)],
            dtype=np.float32,
        )
        right = np.asarray(
            [[0.20, y, z] for y in (-0.10, 0.0, 0.10) for z in (-0.10, 0.0, 0.10)],
            dtype=np.float32,
        )
        binding = InstanceGeometryBinding(
            root=Path("synthetic"),
            points_xyz=np.concatenate([left, right], axis=0),
            instance_indices=np.concatenate(
                [np.zeros(left.shape[0], np.int64), np.ones(right.shape[0], np.int64)]
            ),
            raw_instance_ids=np.concatenate(
                [np.full(left.shape[0], 10, np.int64), np.full(right.shape[0], 20, np.int64)]
            ),
            metadata={},
        )
        anchors = np.asarray([[-0.20, 0, 0], [0.0, 0, 0], [4.0, 0, 0]], np.float32)
        children = np.asarray(
            [
                [[-0.22, -0.05, 0], [-0.18, 0.05, 0]],
                [[-0.18, 0, 0], [0.18, 0, 0]],
                [[4.0, 0.1, 0], [4.0, -0.1, 0]],
            ],
            np.float32,
        )
        assignment = classify_anchor_geometry(
            anchors,
            children,
            binding,
            num_instances=2,
            max_distance=0.24,
            interior_purity=0.80,
            min_support_fraction=0.50,
            min_query_confidence=0.20,
            unknown_uses_residual=True,
        )
        self.assertEqual(int(assignment.states[0]), GEOMETRY_STATE_INTERIOR)
        self.assertEqual(int(assignment.states[1]), GEOMETRY_STATE_BOUNDARY)
        self.assertEqual(int(assignment.states[2]), GEOMETRY_STATE_UNKNOWN)
        np.testing.assert_array_equal(assignment.residual_anchor_indices, np.asarray([1, 2], np.int64))


if __name__ == "__main__":
    unittest.main()
