import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'evaluation_release'
sys.path.insert(0, str(EVAL / '2d/runtime'))
from unified_semantic_eval.protocol import UnifiedSemanticGT, PredictionWriter, score_prediction_directory
from unified_semantic_eval.adapter_utils import select_contract_views


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DIRECT = load_module('release_direct_3d', EVAL / '3d/eval_direct_3d_19.py')
PREPARE = load_module('release_prepare_gt', EVAL / 'common/prepare_fixed_vocabulary_gt.py')
RASTER = load_module('release_raster_gt', EVAL / 'common/precompute_gtmesh_2d_labels.py')


class FixedVocabularyEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.split = self.root / 'source' / 'scene' / 'test'
        (self.split / 'labels').mkdir(parents=True)
        manifest = {
            'protocol': 'rasterized_gt_mesh_2d_gt_only', 'scene': 'scene', 'split': 'test',
            'outputWidth': 2, 'outputHeight': 2, 'numClasses': 3,
            'selectedViews': ['view'], 'numSelectedViews': 1,
            'fixedVocabulary': {'name': 'fixture', 'source': 'classes.txt', 'sourceSha256': 'abc'},
        }
        (self.split / 'manifest.json').write_text(json.dumps(manifest))
        (self.split / 'view_names.txt').write_text('view\n')
        with (self.split / 'semantic_palette.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['class_index', 'class_id', 'class_name', 'r', 'g', 'b'])
            for index, name in enumerate(['wall', 'chair', 'object']):
                writer.writerow([index, index, name, index * 30, 0, 0])
        np.save(self.split / 'labels/view.npy', np.array([[0, 0], [1, -1]], dtype=np.int32))
        self.gt = UnifiedSemanticGT.open(self.root / 'source', scene='scene')

    def test_default_argmax_includes_classes_absent_from_gt(self):
        self.assertEqual(self.gt.present_class_indices, [0, 1])
        self.assertEqual(self.gt.prediction_class_indices(), [0, 1, 2])

    def test_metrics_count_missing_predictions_as_wrong(self):
        output = self.root / 'pred'
        writer = PredictionWriter(output, self.gt, 'ours')
        writer.write('view', np.array([[0, -1], [2, 2]], dtype=np.int32))
        writer.finalize()
        summary = score_prediction_directory(self.gt, output / 'predictions', output / 'metrics')
        self.assertAlmostEqual(summary['mIoU'], 0.25)
        self.assertAlmostEqual(summary['mAcc'], 0.25)
        self.assertAlmostEqual(summary['overallAcc'], 1 / 3)
        self.assertAlmostEqual(summary['predictionCoverage'], 2 / 3)
        self.assertEqual(summary['predictionClassCount'], 3)
        self.assertEqual(summary['predictionVocabulary'], 'fixture')

    def test_exclusions_are_recorded_as_used(self):
        writer = PredictionWriter(self.root / 'pred', self.gt, 'ours', exclude_classes=['object'])
        self.assertEqual(writer.metadata_payload['predictionClassIndices'], [0, 1])
        self.assertEqual(writer.metadata_payload['requestedExcludedClasses'], ['object'])

    def test_stale_predictions_cannot_complete_a_run(self):
        writer = PredictionWriter(self.root / 'pred', self.gt, 'ours')
        np.save(writer.prediction_dir / 'view.npy', np.zeros((2, 2), dtype=np.int32))
        with self.assertRaisesRegex(RuntimeError, 'stale predictions'):
            writer.finalize()

    def test_gt_scene_and_split_are_checked(self):
        with self.assertRaisesRegex(RuntimeError, 'scene/split mismatch'):
            UnifiedSemanticGT.open(self.split, scene='another')
        with self.assertRaisesRegex(RuntimeError, 'scene/split mismatch'):
            UnifiedSemanticGT.open(self.split, scene='scene', split='train')

    def test_test_cameras_must_match_gt(self):
        view = SimpleNamespace(image_name='view', image_height=2, image_width=2)
        scene = SimpleNamespace(getTestCameras=lambda: [view])
        self.assertEqual(select_contract_views(scene, self.gt), [view])
        view.image_width = 3
        with self.assertRaisesRegex(RuntimeError, 'expects'):
            select_contract_views(scene, self.gt)

    def test_gt_conversion_rejects_in_place_overwrite(self):
        before = (self.split / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            PREPARE.prepare_scene(self.root / 'source', self.root / 'source', self.root,
                                  'scene', 'test', True)
        self.assertEqual(before, (self.split / 'manifest.json').read_bytes())

    def test_raster_cache_is_not_silently_reused(self):
        args = SimpleNamespace(output_root=self.root / 'source', split='test', overwrite=True)
        before = (self.split / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(FileExistsError, 'fresh GT output'):
            RASTER.run_scene(Path('scene'), args, torch.device('cpu'))
        self.assertEqual(before, (self.split / 'manifest.json').read_bytes())

    def test_official_ignore_mapping_cannot_fall_back_to_raw_name(self):
        raw = self.root / 'classes.txt'
        raw.write_text('wall\nchair\n')
        mapping = self.root / 'map.csv'
        mapping.write_text('class,semantic_map_to\nwall,None\nchair,\n')
        mesh = self.root / 'mesh.ply'
        mesh.touch()
        vertices = np.zeros(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'i4')])
        vertices['label'] = [0, 1]
        args = SimpleNamespace(gt_mesh=str(mesh), raw_classes=str(raw),
                               benchmark_map=str(mapping), top100_classes=str(raw))
        with patch.object(DIRECT, 'CLASS_PROTOCOL', 'scannetpp100'), \
                patch.object(DIRECT, 'CLASS_NAMES', ('wall', 'chair')), \
                patch.object(DIRECT, 'read_ply_vertex', return_value=vertices):
            result = DIRECT.load_gt(args)
        np.testing.assert_array_equal(result[2], [-1, 1])


class SparseSemanticRestoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ['point_cloud.ply', 'semantic_anchor_offset.pt', 'semantic_decoder.pt']:
            (self.root / name).touch()
        fields = ['x', 'y', 'z'] + ['f_offset_%d' % i for i in range(6)] + \
            ['scale_%d' % i for i in range(6)] + ['f_semantic_feat_%d' % i for i in range(3)]
        self.vertices = np.zeros(2, dtype=[(name, 'f4') for name in fields])
        self.vertices['x'] = [1, 3]
        self.vertices['scale_0'] = np.log(2)
        self.vertices['f_offset_0'] = 0.25
        self.vertices['f_offset_1'] = -0.25
        self.sidecar = {
            'n_offsets': 2, 'semantic_latent_dim': 3, 'semantic_residual_dim': 1,
            'semantic_anchor': torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float16),
            'semantic_offset_residual': torch.tensor([[[1.], [2.]]], dtype=torch.float16),
            'semantic_offset_gate': torch.full((1, 2, 1), -4., dtype=torch.float16),
            'semantic_residual_is_sparse': True,
            'semantic_residual_anchor_indices': torch.tensor([1], dtype=torch.int32),
            'offset_residual_proj': {
                'weight': torch.tensor([[0.], [0.], [1.]], dtype=torch.float16),
                'bias': torch.tensor([0.125, 0., 0.], dtype=torch.float16),
            },
        }
        self.args = SimpleNamespace(point_cloud=str(self.root / 'point_cloud.ply'),
                                    geometry_transfer='knn_latent', classify_chunk_size=2)

    def restore(self):
        with patch.object(DIRECT, 'read_ply_vertex', return_value=self.vertices), \
                patch.object(DIRECT, 'torch_load', return_value=self.sidecar), \
                patch.object(DIRECT, 'encode_alphaclip', return_value=(torch.eye(3), {})), \
                patch.object(DIRECT, 'torchscript_load', return_value=torch.nn.Identity()):
            return DIRECT.load_ours(self.args, torch.device('cpu'))

    def test_fp16_sparse_restore_matches_native_formula(self):
        xyz, output, meta = self.restore()
        base = self.sidecar['semantic_anchor'].float()[:, None, :].expand(-1, 2, -1).clone()
        projected = F.linear(self.sidecar['semantic_offset_residual'].float(),
                             self.sidecar['offset_residual_proj']['weight'].float(),
                             self.sidecar['offset_residual_proj']['bias'].float())
        base[1] += torch.sigmoid(self.sidecar['semantic_offset_gate'].float()[0]) * projected[0]
        expected = F.normalize(base.reshape(-1, 3), dim=1, eps=1e-6).numpy()
        np.testing.assert_allclose(output['latents'], expected)
        np.testing.assert_allclose(xyz[:, 0], [1.5, 0.5, 3.5, 2.5])
        self.assertEqual(output['latents'].dtype, np.float32)
        self.assertEqual(meta['residualAnchorCount'], 1)

    def test_fp32_anchor_in_ply_is_supported(self):
        expected = self.sidecar.pop('semantic_anchor').float().numpy()
        for index in range(3):
            self.vertices['f_semantic_feat_%d' % index] = expected[:, index]
        _, output, _ = self.restore()
        np.testing.assert_allclose(output['latents'][:2], np.tile(expected[0], (2, 1)))

    def test_invalid_sparse_index_fails(self):
        self.sidecar['semantic_residual_anchor_indices'] = torch.tensor([2])
        with self.assertRaisesRegex(RuntimeError, 'indices are invalid'):
            self.restore()

    def test_nonfinite_semantics_fail(self):
        self.sidecar['semantic_anchor'][0, 0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'NaN or Inf'):
            self.restore()

    def test_knn_weights_handle_exact_coincident_points(self):
        actual = DIRECT._knn_weights(np.array([[0., 0., 3.], [1., 2., 2.]]), 'inverse_distance')
        np.testing.assert_allclose(actual, [[0.5, 0.5, 0.], [0.5, 0.25, 0.25]])


if __name__ == '__main__':
    unittest.main()
