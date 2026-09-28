import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arguments import ModelParams


class InferenceInputsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        parser = argparse.ArgumentParser()
        self.params = ModelParams(parser)
        self.args = parser.parse_args([])
        self.args.source_path = str(self.root / 'scene')
        self.args.model_path = str(self.root / 'model')
        self.args.semantic_3dgs = True
        self.args.semantic_codec_path = str(self.root / 'teacher' / 'semantic_codec.pth')
        self.args.region_teachers = str(self.root / 'teacher')
        self.args.instance_geometry_teacher = self.args.region_teachers
        self.metadata = {'metadata': {'input_dim': 512, 'latent_dim': 12}}

    def test_model_codec_wins_and_teachers_are_not_needed(self):
        local = Path(self.args.model_path) / 'semantic_codec.pth'
        local.parent.mkdir()
        local.touch()
        with patch('utils.semantic_codec.load_semantic_codec', return_value=(None, self.metadata)) as load:
            result = self.params.extract(self.args, for_inference=True)
        self.assertEqual(Path(result.semantic_codec_path), local)
        self.assertEqual(result.region_teachers, '')
        self.assertEqual(result.instance_geometry_teacher, '')
        load.assert_called_once_with(str(local), map_location='cpu')

    def test_missing_model_codec_does_not_fall_back_to_teacher(self):
        teacher = Path(self.args.semantic_codec_path)
        teacher.parent.mkdir()
        teacher.touch()
        with self.assertRaisesRegex(RuntimeError, 'model-owned'):
            self.params.extract(self.args, for_inference=True)

    def test_geometry_ignores_stale_semantic_config(self):
        self.args.semantic_3dgs = False
        local = Path(self.args.model_path) / 'semantic_codec.pth'
        local.parent.mkdir()
        local.touch()
        with patch('utils.semantic_codec.load_semantic_codec') as load:
            result = self.params.extract(self.args, for_inference=True)
        load.assert_not_called()
        self.assertEqual(result.semantic_codec_path, '')
        self.assertEqual(result.region_teachers, '')

    def test_training_keeps_teacher_codec(self):
        with patch('utils.semantic_codec.load_semantic_codec', return_value=(None, self.metadata)) as load:
            result = self.params.extract(self.args)
        self.assertEqual(result.region_teachers, self.args.region_teachers)
        load.assert_called_once_with(self.args.semantic_codec_path, map_location='cpu')


if __name__ == '__main__':
    unittest.main()
