# InstGuide-GS

**Instance-Guided Semantic Learning for Open-Vocabulary 3D Gaussian Splatting**

InstGuide-GS learns open-vocabulary 3D semantics through a cross-view instance
teacher and a geometry-guided sparse Anchor-Offset representation.

## Results

**ScanNet++ Scene 2**

![ScanNet++ Scene 2 semantic segmentation comparison](assets/results/scannetpp_scene2.jpg)

**Replica room0**

![Replica room0 semantic segmentation comparison](assets/results/replica_room0.jpg)

## Installation

Server environments and CUDA build instructions are in [environment.yml](environment.yml):
`instguide-teacher` for teacher generation; `instguide-gs` for 3DGS training and evaluation.

```bash
conda env create -f environment.yml
conda activate instguide-gs
# Complete the native-extension installation commands in environment.yml.
```

Download the required weights using the [checkpoint guide](third_party/checkpoints/README.md).
Alpha-CLIP text evaluation also requires the base `ViT-B-16.pt`.

## Data layout

```text
SCENE/
  images/
  depths/                 # registered depth for teacher construction
  sparse/0/cameras.txt
  sparse/0/images.txt
  sparse/0/points3D.ply
  INSTANCE_MESH.ply        # exact geometry indexed by object_dict.npy
```

`INSTANCE_MESH.ply` must preserve the vertex order indexed by `object_dict.npy`.
See the [teacher guide](semantic_3dgs_teacher/README.md) for mesh names and depth scales.

## 1. Generate the instance teacher

In `instguide-teacher`, from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
python -m semantic_3dgs_teacher.pipeline \
  --seq_name SCENE_ID \
  --scannetpp_dslr_root /path/to/data
```

The default 12D teacher is saved to:

```text
SCENE/output/semantic_3dgs_teacher_12d/scannetpp_dslr_small/
```

## 2. Train geometry and semantics

In `instguide-gs`, from the repository root:

```bash
export SRC=/path/to/scene
export GEOM=$SRC/out_geometry_30000
export MODEL=$SRC/out_instguide_32000
export TEACHER=/path/to/compressed_teacher
export INSTANCE_POINT_CLOUD=/path/to/exact_object_dict_indexed_mesh.ply
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
bash scripts/train.sh
```

Defaults: 30,000 geometry steps, 2,000 semantic steps, 12D anchor semantics and
4D sparse residuals. Geometry binding, interior initialization and 3D supervision
are enabled. Logs, training time and model size are saved automatically.

To reuse a geometry model, run `STAGE=semantic bash scripts/train.sh`.
See [training options](SEMANTIC_TWO_STAGE.md) for other dimensions and iteration counts.

Optimization holds out every eighth registered view and uses only training views.
The teacher uses all registered views (a transductive scene protocol).

## Evaluation and visualization

Prepare GT and set evaluation paths as described in the
[evaluation guide](evaluation_release/README.md), then run:

```bash
bash scripts/evaluate_fixed_vocab_2d.sh
bash scripts/evaluate_fixed_vocab_3d.sh
```

Use `visualize_semantic_3dgs_model.py` for model overlays and
`visualize_instance_teacher_semantic.py` for teacher inspection.

## Acknowledgements and licenses

We thank the authors of [Scaffold-GS](https://github.com/city-super/Scaffold-GS),
[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[MaskClustering](https://github.com/PKU-EPIC/MaskClustering),
[Alpha-CLIP](https://github.com/SunzeY/AlphaCLIP), and
[CropFormer](https://github.com/qqlu/Entity) for their work and shared code.

See [LICENSE.md](LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
