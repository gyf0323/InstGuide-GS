# Public code layout

## Teacher (Innovation 1)

- `semantic_3dgs_teacher/pipeline.py`: masks, instance clustering, robust teacher
  extraction, and compression orchestration.
- `semantic_3dgs_teacher/`: dataset readers, MaskClustering-derived clustering,
  Alpha-CLIP region features, instance consensus, codec, export and validation.
- `third_party/`: required Alpha-CLIP and CropFormer source, tokenizer and notices.
  Model weights are obtained separately.

## Representation and training (Innovation 2)

- `scripts/train.sh`: the single public launcher, selecting both stages or one.
- `tools/train_pipeline.py`: preflight, sequential subprocesses, logs and summaries.
- `train.py`: geometry optimization and frozen-geometry semantic optimization.
- `scene/gaussian_model.py`: shared anchor semantics, projected gated residuals,
  sparse residual rows, optimizer and saved-model lifecycle.
- `utils/instance_geometry_semantic.py`: instance geometry preparation, spatial
  support queries and interior/boundary/unknown allocation.
- `utils/instance_teacher_adapter.py`: compressed instance table, maps and weights.
- `utils/semantic_codec.py`: strict codec loading and latent decoding. The
  LangSplat-style codec architecture is a live dependency of compressed teachers.
- `gaussian_renderer/` and `submodules/`: geometry/semantic rasterization and CUDA
  nearest-neighbor support.

## Evaluation and inspection

- `evaluation_release/`: latest fixed-vocabulary 2D evaluation, GT preparation,
  direct 3D evaluation and multi-scene aggregation for this method.
- `scripts/evaluate_fixed_vocab_2d.sh` and `evaluate_fixed_vocab_3d.sh`: documented
  evaluation launchers. No old root evaluation entry points remain.
- `render.py`, `metrics.py`: RGB rendering and reconstruction metrics.
- `render_semantic.py`, `visualize_semantic_3dgs_model.py`: semantic renders and
  overlays; teacher inspection is also retained.
- `utils/clip_text_utils.py`: shared text helpers used by batch evaluation and
  visualization, separated from the removed interactive viewer.

## Maintenance

`tools/` also contains geometry export, semantic storage export and release
inspection. `tests/` contains regression tests, not paper ablation experiments.
Keep these tests: they protect teacher contracts, sparse restoration and metrics.
Environment, documentation, license files and the generated source manifest
complete the package.

## Intentional compatibility boundary

Old per-frame LangSplat teacher loading/generation, SAM encoders, interactive
viewers, old metric entry points, scene experiment scripts, alternate uncompiled
CUDA implementations, and GLM example/test documentation have been removed.

The modified rasterizer's default-off contributor ABI is retained pending a
GPU-tested ABI change. Saved-model restore and pre-binding allocation still need
dense tensor handling internally; those branches are not public dense-residual
experiment modes. Removing them blindly would break initialization or restore.
