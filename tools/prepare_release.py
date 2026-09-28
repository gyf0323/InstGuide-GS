"""Inspect or export a clean source candidate without changing the working Git index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


REQUIRED_FILES = (
    '.gitignore', '.gitattributes', 'README.md', 'LICENSE.md', 'THIRD_PARTY_NOTICES.md',
    'environment.yml', 'train.py', 'scene/gaussian_model.py',
    'utils/instance_teacher_adapter.py', 'utils/instance_geometry_semantic.py',
    'semantic_3dgs_teacher/pipeline.py', 'semantic_3dgs_teacher/validation.py',
    'scripts/train.sh', 'tools/train_pipeline.py', 'utils/clip_text_utils.py',
    'scripts/evaluate_fixed_vocab_2d.sh', 'scripts/evaluate_fixed_vocab_3d.sh',
    'evaluation_release/README.md',
    'evaluation_release/2d/eval_unified_scaffoldgs_text_encoder.py',
    'evaluation_release/2d/semantic_metrics_2d_gtmesh.py',
    'evaluation_release/2d/runtime/unified_semantic_eval/protocol.py',
    'evaluation_release/3d/eval_direct_3d_19.py',
    'third_party/checkpoints/README.md',
    'third_party/AlphaCLIP-main/LICENSE',
    'third_party/AlphaCLIP-main/alpha_clip/bpe_simple_vocab_16e6.txt.gz',
    'third_party/Entity/Entityv2/CropFormer/LICENSE',
    'third_party/Entity/Entityv2/CropFormer/demo_cropformer/mask_predict.py',
    'third_party/Entity/Entityv2/CropFormer/mask2former/data/__init__.py',
    'third_party/Entity/Entityv2/CropFormer/mask2former/data/datasets/__init__.py',
    'third_party/Entity/Entityv2/CropFormer/mask2former/data/dataset_mappers/crop_augmentations.py',
    'third_party/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml',
    'utils/__init__.py', 'tools/__init__.py',
    'submodules/diff-gaussian-rasterization/setup.py',
    'submodules/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp',
    'submodules/diff-gaussian-rasterization/third_party/glm/copying.txt',
    'submodules/simple-knn/setup.py',
)
FORBIDDEN_SUFFIXES = {'.pth', '.pt', '.ckpt', '.onnx', '.pyd', '.so', '.dll', '.exe', '.pyc', '.obj', '.pem', '.key'}


def collect_files(root):
    """Apply actual Git ignore semantics even before the source repo is initialized."""
    root = Path(root).resolve()
    if not (root / '.gitignore').is_file():
        raise ValueError('Source must contain .gitignore')
    if shutil.which('git') is None:
        raise RuntimeError('Git is required for release file selection')
    with tempfile.TemporaryDirectory(prefix='instguide-release-') as temporary:
        subprocess.run(['git', 'init', '--quiet', temporary], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        command = ['git', '-c', 'core.excludesFile=',
                   '--git-dir=' + str(Path(temporary) / '.git'),
                   '--work-tree=' + str(root), 'ls-files', '--others', '--exclude-standard', '-z']
        raw = subprocess.check_output(command)
    # A generated manifest must not become a source entry when re-exporting.
    files = sorted(os.fsdecode(item) for item in raw.split(b'\0')
                   if item and os.fsdecode(item) != 'release_manifest.json')
    for name in files:
        path = root / name
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f'Release path escapes source: {name}') from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Release entries must be regular files: {name}')
    return files


def inspect_release(root, required_files=REQUIRED_FILES):
    root = Path(root).resolve()
    files = collect_files(root)
    missing = sorted(set(required_files) - set(files))
    if missing:
        raise ValueError(f'Required files are missing or ignored: {missing}')
    rejected = []
    total = 0
    for name in files:
        path = root / name
        size = path.stat().st_size
        total += size
        if (path.suffix.lower() in FORBIDDEN_SUFFIXES or size > 50 * 1024**2
                or path.name == '.env' or path.name in {'id_rsa', 'id_ed25519'}
                or (path.name.startswith('.env.') and path.name != '.env.example')):
            rejected.append(name)
    if rejected:
        raise ValueError(f'Local/binary/large artifacts would enter the release: {rejected}')
    return files, {'file_count': len(files), 'source_bytes': total,
                   'source_mib': round(total / 1024**2, 2),
                   'validation': 'file selection only; not GPU validation or license clearance'}


def export_release(root, output, required_files=REQUIRED_FILES):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output must not exist; refusing to overwrite: {output}')
    try:
        relative = output.relative_to(root)
    except ValueError:
        relative = None
    if relative is not None and (len(relative.parts) < 2 or relative.parts[0] != '.release'):
        raise ValueError('An in-repository output must be under .release/<new-name>')
    files, summary = inspect_release(root, required_files)
    if '.release' in {Path(name).parts[0] for name in files}:
        raise ValueError('.release must be ignored before exporting')
    output.mkdir(parents=True)
    records = []
    for name in files:
        source = root / name
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != hashlib.sha256(source.read_bytes()).hexdigest():
            raise RuntimeError(f'Source changed during export or copy verification failed: {name}')
        records.append({'path': name, 'bytes': destination.stat().st_size, 'sha256': digest})
    # The manifest accounts for source files; it does not count itself.
    manifest = dict(summary, format='instguide-source-release-v1', files=records)
    (output / 'release_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return dict(summary, output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='Optional new directory; omitted means read-only audit')
    args = parser.parse_args()
    result = (export_release(args.source, args.output) if args.output
              else inspect_release(args.source)[1])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
