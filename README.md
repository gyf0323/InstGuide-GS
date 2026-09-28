# InstGuide-GS

InstGuide-GS is a two-stage open-vocabulary semantic Gaussian splatting system for
static indoor scenes. It extends Scaffold-GS with:

1. a geometry-indexed, instance-level semantic teacher that aggregates
   multi-view CropFormer masks with Alpha-CLIP features and stores one prototype
   per 3D instance; and
2. geometry-guided Anchor-Offset semantics, where an Anchor stores a shared base
   latent and only transition or geometrically uncertain Anchors allocate
   low-rank child-Gaussian residuals.

## Repository layout

```text
arguments/                         command-line parameters
scene/                             Scaffold-GS model and semantic carrier
utils/instance_teacher_adapter.py  instance teacher runtime adapter
utils/instance_geometry_semantic.py geometry binding and Anchor state assignment
semantic_3dgs_teacher/             complete teacher-generation package
tools/                             training orchestration and artifact utilities
scripts/                           one training and two evaluation launchers
tests/                             CPU unit tests for teacher and geometry logic
third_party/AlphaCLIP-main/        Alpha-CLIP source
third_party/Entity/.../CropFormer/ CropFormer source
submodules/                        CUDA rasterizer and nearest-neighbor extensions
```

## Environment

The last validated server stack used Python 3.11, a CUDA-enabled PyTorch build,
and locally compiled CUDA extensions. Create the base environment first:

```bash
conda env create -f environment.yml
conda activate instguide-gs
```

This is a base environment, not a complete GPU installation. Dependencies such
as LPIPS/timm may install a default PyTorch build; verify or replace it with a
matching CUDA-enabled **torch and torchvision** pair for the host. Install
**torch-scatter** against that same PyTorch/CUDA combination (a matching binary
wheel, or a source build with the local CUDA toolkit). Then build both extensions
in the active environment:

```bash
python -m pip install torch-scatter --no-build-isolation
python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation -e submodules/simple-knn
```

The first command is a source-build fallback when no matching wheel is available;
it requires a compatible compiler and CUDA toolkit. Do not reuse compiled
extensions from a different PyTorch/CUDA environment. On the GPU server, check:

```bash
export OMP_NUM_THREADS=8
nvidia-smi
python - <<'PY'
import torch, torchvision, torch_scatter, cv2, plyfile, lpips, wandb
import diff_gaussian_rasterization._C
from simple_knn._C import distCUDA2
assert torch.cuda.is_available(), "A GPU allocation and driver are required"
points = torch.rand(32, 3, device="cuda")
assert torch.isfinite(distCUDA2(points)).all().item()
torch.cuda.synchronize()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
python train.py --help
```

These checks do not replace a short render/backward/save/reload smoke run.

For teacher generation, also install Detectron2, PyTorch3D, Open3D, and the
requirements of CropFormer and Alpha-CLIP. These packages are not required when
training from an already generated teacher.

Semantic text evaluation/visualization needs the same Alpha-CLIP package and
checkpoint as the teacher, even when teacher generation was done elsewhere.
After installing the matching PyTorch/torchvision pair, install the inference
package with `python -m pip install -e third_party/AlphaCLIP-main`. Its metadata
declares the tokenizer and LoRA dependencies; no model weights are bundled.

Model checkpoints are not redistributed. Put the two teacher-generation weights
under `third_party/checkpoints/` as documented there.

Use ASCII-only model output paths on Windows; TorchScript export to paths
containing non-ASCII characters is not supported by the current saving code.
The shell launchers below require Bash.

## Data layout

A scene must provide registered images and a COLMAP sparse model:

```text
SCENE/
  images/
  sparse/0/cameras.txt
  sparse/0/images.txt
  sparse/0/points3D.ply
```

Training also accepts `cameras.bin` + `images.bin`, using the same binary-first
selection as the scene loader. Initial points may be PLY, COLMAP BIN, or TXT;
the scene loader converts the latter two to PLY. The bundled teacher-generation
reader/CropFormer runner still uses COLMAP TXT files, so supply TXT exports for
teacher generation even when the geometry trainer uses BIN.

Depth maps are used by the RGB-D teacher construction pipeline. The public
geometry launcher trains RGB reconstruction without depth supervision; the
semantic loss also does not use depth-visible registration. Optional depth
handling in the numerical backend is not the public main configuration.

For geometry-guided semantic capacity allocation, provide the exact PLY whose
vertex indices are referenced by the teacher's `object_dict.npy`. Coordinate
alignment alone is insufficient because `point_ids` are vertex indices.

With `--eval`, registered cameras are sorted and every eighth view is held out.
Only the remaining views are sampled for optimization.

## 1. Generate the instance teacher

The complete teacher pipeline is in `semantic_3dgs_teacher/`. A self-contained
12D teacher can be produced with:

```bash
python -m semantic_3dgs_teacher.pipeline \
  --repo_root "$PWD" \
  --stages all \
  --seq_name SCENE_ID \
  --config scannetpp_dslr_small \
  --dataset scannetpp_dslr \
  --scannetpp_dslr_root /path/to/data \
  --compressed_dim 12 \
  --latent_dtype float16 \
  --frame_data_mode copy
```

`frame_data_mode=copy` is recommended for portability. The compressed directory
contains `instance_features.npy`, `instance_ids.npy`, `semantic_codec.pth`,
`object_dict.npy`, frame instance maps, frame weights, and metadata. See
`semantic_3dgs_teacher/README.md` for stage-specific commands.

Default output names include the requested compressed dimension. Existing mask,
teacher-export and compression stages are skipped after read-only checks; a
partial teacher/mask output stops with an error rather than being accepted or
silently overwritten. Use a new output directory, or explicitly use `--force`
to rebuild. Validation is structural, not proof that an old output was generated
with identical settings: after changing teacher settings, use a new output or
rebuild all affected stages. Avoid `--compressed_maps` for the current trainer,
whose runtime adapter consumes NPY frame arrays (the generator can also emit NPZ).
Instance clustering still reuses an existing `object_dict.npy` by file presence;
its provenance is not established by this check. Rebuild clustering after changes
to its source geometry, masks or configuration.

## 2. Train geometry and semantics

One public launcher runs the two optimization stages in sequence:

```bash
export SRC=/path/to/scene
export GEOM=$SRC/out_geometry_30000
export MODEL=$SRC/out_instguide_32000
export TEACHER=/path/to/compressed_teacher
export INSTANCE_POINT_CLOUD=/path/to/exact_object_dict_indexed_mesh.ply
export SEMANTIC_DIM=12
export RESIDUAL_DIM=4
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
bash scripts/train.sh
```

The example matches the 12D teacher generated above. Teacher, codec, and model
latent dimensions must agree. Defaults are 12D latents, residual rank 4, gate
initialization -4, 30k geometry steps, and 2k semantic steps.

The semantic stage freezes geometry, uses alpha-rendered 2D distillation,
allocates sparse child residuals to boundary/unknown anchors, and enables
reliable-interior teacher initialization and direct 3D supervision. Teacher
geometry binding is generated automatically.

Use `STAGE=geometry` or `STAGE=semantic` with the same launcher for a single stage.
The latter requires an existing geometry model. Nonempty output directories are
rejected. Dataset and teacher preflight runs before optimization; a failed stage
stops the pipeline. See [two-stage training](SEMANTIC_TWO_STAGE.md) for parameters,
checks and the Python interface. `train.py` remains the shared numerical backend.

Both stages use `--eval`; logs record the registered-view split. Each stage saves
its configuration, log, process status and successful timing/model-size summary.
Timing includes process setup and final saves, but excludes teacher generation,
preflight and separate evaluation. Size includes inference weights, codec and
configuration, excluding optimizer checkpoints, logs and evaluation files.
Semantic sidecar floats are stored as FP16; geometry and normal saved MLP/decoder
weights remain FP32, and restored computation is FP32.

## Evaluation and visualization

The latest [fixed-vocabulary evaluation package](evaluation_release/README.md)
is the supported metric implementation:

- `scripts/evaluate_fixed_vocab_2d.sh`: rendered test views against precomputed GT.
- `scripts/evaluate_fixed_vocab_3d.sh`: camera-free native-latent 3D transfer.
- `render.py` and `metrics.py`: RGB reconstruction rendering and metrics.
- `render_semantic.py` and `visualize_semantic_3dgs_model.py`: semantic renders and
  class overlays.
- `semantic_3dgs_teacher.visualize_teacher`: teacher inspection.

Report split, vocabulary, text encoder checkpoint, GT construction,
geometry transfer, model-size boundary and timing boundary with results.

## Tests

CPU-side tests do not require the rasterizer:

```bash
python -m unittest discover -s tests -v
```

A full training smoke test additionally requires a CUDA device and extensions
compiled against the active PyTorch environment.

See [script protocols](scripts/README.md) for launcher options,
[two-stage training](SEMANTIC_TWO_STAGE.md) for stage configuration, and
[code layout](docs/CODE_LAYOUT.md) for the main modules.

For a source-only export to a new directory:

```bash
python tools/prepare_release.py --output .release/InstGuide-GS-source
```

It excludes local weights, datasets, and build products without deleting them.

## Acknowledgements and licenses

We thank the authors of [Scaffold-GS](https://github.com/city-super/Scaffold-GS),
[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[MaskClustering](https://github.com/PKU-EPIC/MaskClustering),
[Alpha-CLIP](https://github.com/SunzeY/AlphaCLIP), and
[CropFormer](https://github.com/qqlu/Entity) for their work and shared code.

See [LICENSE.md](LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md)
for component sources and applicable terms. Original copyright notices and
component licenses are retained; model weights and datasets are obtained
separately.
