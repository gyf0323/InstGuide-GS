import json
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.prepare_release import collect_files, inspect_release, export_release


@unittest.skipUnless(shutil.which('git'), 'Git is required')
class PrepareReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / '.gitignore').write_text(
            '.release/\n11/\n/checkpoints/\nthird_party/checkpoints/*\n'
            '!third_party/checkpoints/README.md\n*.pth\n', encoding='utf-8')
        for name in ['train.py', '11/unused.py', 'checkpoints/old.pt',
                     'third_party/checkpoints/README.md', 'third_party/checkpoints/weights.pth',
                     'third_party/alpha/bpe.txt.gz']:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')

    def test_ignore_semantics_keep_checkpoint_readme_and_tokenizer(self):
        files = collect_files(self.root)
        self.assertIn('third_party/checkpoints/README.md', files)
        self.assertIn('third_party/alpha/bpe.txt.gz', files)
        self.assertNotIn('11/unused.py', files)
        self.assertNotIn('checkpoints/old.pt', files)
        self.assertNotIn('third_party/checkpoints/weights.pth', files)
        self.assertFalse((self.root / '.git').exists())

    def test_missing_runtime_asset_fails(self):
        with self.assertRaisesRegex(ValueError, 'Required files'):
            inspect_release(self.root, ['missing.py'])

    def test_root_data_ignore_keeps_nested_runtime_code(self):
        with (self.root / '.gitignore').open('a') as handle:
            handle.write('/data/\n')
        nested = self.root / 'third_party/mask2former/data/__init__.py'
        nested.parent.mkdir(parents=True)
        nested.touch()
        (self.root / 'data').mkdir()
        (self.root / 'data/image.png').write_bytes(b'fixture')
        files = collect_files(self.root)
        self.assertIn('third_party/mask2former/data/__init__.py', files)
        self.assertNotIn('data/image.png', files)

    def test_unignored_binary_fails(self):
        (self.root / 'accidental.pt').write_bytes(b'weights')
        with self.assertRaisesRegex(ValueError, 'artifacts'):
            inspect_release(self.root, ['train.py'])

    def test_export_is_hashed_and_never_overwrites(self):
        output = self.root / '.release' / 'candidate'
        summary = export_release(self.root, output, ['train.py'])
        manifest = json.loads((output / 'release_manifest.json').read_text())
        self.assertEqual(summary['file_count'], len(manifest['files']))
        self.assertEqual((output / 'train.py').read_bytes(), (self.root / 'train.py').read_bytes())
        self.assertTrue(all(len(row['sha256']) == 64 for row in manifest['files']))
        self.assertFalse((output / '11').exists())
        self.assertFalse(any(name.startswith('.release/') for name in collect_files(self.root)))
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            export_release(self.root, output, ['train.py'])

    def test_export_cannot_write_into_source_module(self):
        with self.assertRaisesRegex(ValueError, 'under .release'):
            export_release(self.root, self.root / 'scene', ['train.py'])

    def test_reexport_excludes_previous_manifest(self):
        first = self.root / '.release' / 'candidate'
        second = self.root / '.release' / 'second'
        export_release(self.root, first, ['train.py'])
        self.assertNotIn('release_manifest.json', collect_files(first))
        summary = export_release(first, second, ['train.py'])
        manifest = json.loads((second / 'release_manifest.json').read_text())
        self.assertNotIn('release_manifest.json', [row['path'] for row in manifest['files']])
        self.assertEqual(summary['file_count'], len(collect_files(first)))


if __name__ == '__main__':
    unittest.main()
