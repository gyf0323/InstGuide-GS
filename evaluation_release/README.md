# Fixed-vocabulary semantic evaluation

This package is integrated from the author's `evaluation_release` snapshot. It
is the supported evaluation path; old root metric entry points were removed.
The original snapshot remains outside this public-source candidate.

## Protocols and dependencies

- 2D: render held-out cameras, compare against precomputed mesh-rasterized labels.
  Test cameras are sorted by image stem, with every eighth registered view held
  out. Camera names and HxW must exactly match the GT contract; no silent resize.
- Prediction vocabulary: all ScanNet++ Top-100 or Replica 101 classes compete in
  cosine argmax. Ignored GT pixels and `object` do not enter the metric denominator.
  Macro mIoU/mAcc average valid GT-present classes, not only correctly predicted
  classes; OA pools valid pixels. Invalid predictions remain errors.
- 3D: camera-free transfer of native child-Gaussian latents to labeled vertices,
  followed by normalization, the trained decoder and text classification. It is
  not view voting. The provided main-method wrapper uses inverse-distance kNN;
  set `KNN_K` explicitly to match your experiment. Other transfer modes in the
  underlying evaluator are different protocols, not interchangeable results.
- 3D Top-100 here uses a single vertex label and dataset vocabulary. It is an
  adapted protocol, not a claim of reproducing the official multi-label server.
  The 3D loader expects ScanNet++ `vertex.label`; Replica `face.object_id` is
  supported by the 2D GT preparation, not by this 3D wrapper.
- Use the same vocabulary, templates, encoder weights, GT construction and
  transfer settings across compared methods. GT generation views and teacher
  generation views are separate from the held-out optimization split.

Use the main repository environment, plus a working Alpha-CLIP installation.
Both the visual checkpoint `clip_b16_grit20m_fultune_2xe.pth` and the base
`ViT-B-16.pt` are required. The latter supplies the base model including the text
encoder; a visual checkpoint alone is not the complete text-encoder dependency.
Weights must be downloaded separately. Only load trusted model checkpoints.
Actual rendering/decoding entry points require CUDA; CPU unit tests do not verify
GPU execution. GT preparation also requires `plyfile`, Pillow, NumPy and PyTorch;
3D transfer uses SciPy.

## Prepare 2D GT

```bash
python evaluation_release/common/precompute_gtmesh_2d_labels.py \
  --data_root /path/to/datasets --scenes cs0 room0 \
  --output_root /path/to/new_raw_gt --split test \
  --cs_target_height 1080 --replica_target_height 0

python evaluation_release/common/prepare_fixed_vocabulary_gt.py \
  --source_gt_root /path/to/new_raw_gt \
  --output_gt_root /path/to/new_fixed_gt \
  --scannetpp_metadata_root /path/to/scannetpp/metadata \
  --scenes cs0 room0 --split test
```

Set GT image sizes to the actual rendered model resolution. `1080` is an example,
not a mandatory model resolution. This precompute tool requires COLMAP text
`cameras.txt`/`images.txt`, PINHOLE/SIMPLE_PINHOLE cameras and triangle faces in the
GT PLY. It follows the repository's centered-principal-point camera convention.
BIN-only COLMAP data must first be converted to text; use the same undistorted
cameras as training. The default mixed-label triangle policy is `ignore`.
Do not compare metrics generated using different rasterization conventions.

Use fresh GT directories. Pre-rasterization refuses nonempty scene/split outputs
instead of silently accepting labels cached under different settings. The legacy
`--overwrite` option does not bypass that guard. Fixed-vocabulary conversion
rejects overlapping input/output scene directories. Replica conversion reads the
original `classMappingPath` in the source manifest; that source metadata must
remain accessible. Optional manual Replica object overrides are recorded by the
precompute tool and are not enabled by the examples above.

## Evaluate 2D

From the repository root, with the model's matching dimensions:

```bash
export SRC=/path/to/datasets/room0
export MODEL=/path/to/trained_semantic_model
export GT_ROOT=/path/to/new_fixed_gt
export SCENE=room0
export ITERATION=32000
export SEMANTIC_DIM=16
export RESIDUAL_DIM=4
export ALPHACLIP="$PWD/third_party/AlphaCLIP-main"
export ALPHACLIP_CKPT=/path/to/clip_b16_grit20m_fultune_2xe.pth
export ALPHACLIP_DOWNLOAD_ROOT=/path/to/folder_containing_ViT-B-16.pt
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
bash scripts/evaluate_fixed_vocab_2d.sh
```

`IMAGES` defaults to `images`; set it when trained images have another directory
name. Outputs default to `$MODEL/eval_fixed_vocab_2d_$ITERATION`: predictions,
`prediction_manifest.json`, `metrics/summary.json`, confusion matrix, per-class
and per-view metrics. `--save_visualizations` adds paired GT/prediction images.
The underlying adapter may also be invoked directly and fixes the same
vocabulary protocol itself; the redundant `run_adapter.py` wrapper was removed.
The copied `2d/semantic_metrics_2d_gtmesh.py`
provides the native loader/render helpers and is explicitly loaded by file path.

## Evaluate 3D (ScanNet++)

Reuse the model and Alpha-CLIP environment variables above, then set:

```bash
export GT_MESH=/path/to/cs0/semantic_gt_3d/mesh_aligned_0.05_semantic.ply
export RAW_CLASSES=/path/to/scannetpp/metadata/semantic_classes.txt
export BENCHMARK_MAP=/path/to/scannetpp/metadata/semantic_benchmark/map_benchmark.csv
export TOP100_CLASSES=/path/to/scannetpp/metadata/semantic_benchmark/top100.txt
export KNN_K=1
bash scripts/evaluate_fixed_vocab_3d.sh
```

Use a model from the same scene and coordinate system as `GT_MESH`. `KNN_K=1`
above is a runnable example, not a verified final paper hyperparameter; match the
saved experiment configuration when reproducing numbers. Outputs default to
`$MODEL/eval_fixed_vocab_3d_$ITERATION`, including `metrics.json`, predictions,
distances, class metrics and labeled PLY files. The evaluator refuses an existing
output directory. It uses all stored child centers, without a view-dependent
opacity filter or distance rejection; report that evaluation boundary.

`3d/aggregate_scannetpp_top100.py` optionally aggregates matching scene outputs.
Pass explicit `--scenes`; `--methods` defaults to `ours`. Inputs are arranged as
`OUTPUT_ROOT/SCENE/ours/`. Other-method adapters and historical scene defaults
have been removed. Aggregation keeps the equal-weight scene mean separate from
metrics computed from pooled confusion matrices.
Raw evaluator defaults are historical: explicitly set `--class_protocol
scannetpp100`, `--prompt_mode maskclustering` and the transfer settings, or use the
wrapper. The wrapper does not infer paper settings from old output names.

## Model precision

The semantic stage of `scripts/train.sh` passes `--semantic_storage_dtype float16`.
`semantic_anchor_offset.pt` stores floating semantic anchors, sparse residuals,
gates and projection weights in FP16; sparse row indices remain integers. Native
PLY geometry/RGB data and normal saved decoder/MLP weights remain FP32. Training
and restored semantic computation use FP32. This is semantic FP16 storage, not
full-model FP16 inference or automatic mixed-precision training. The optional
storage-export tool can additionally convert some files; inspect actual tensors
when auditing older exported artifacts.

The 2D path uses the native restore code. The 3D path restores FP16 sidecars to
FP32 and also accepts FP32 semantic anchors stored in PLY. Both retain sparse
residual indexing and projection bias.

Inference does not load frame teachers. Native rendering and 2D evaluation use
the model-owned `semantic_codec.pth`, even when an old configuration still names
an external teacher. A missing model codec is an error for compressed semantic
inference; it is not silently replaced by a teacher codec. The 3D direct evaluator
loads the saved iteration's decoder. Semantic training still reads its teacher
and copies the codec into the output model before writing `cfg_args`.

## Integration corrections and validation boundary

- Removed the private absolute repository default and made imports self-contained.
- Kept fixed-vocabulary selection identical with or without the wrapper; recorded
  the actual exclusion list and vocabulary metadata.
- Added scene/split, current-run prediction and input/output GT safety checks.
- Validated sparse row indices, shapes and finite semantic parameters on restore.
- For 3D Top-100, explicit benchmark exclusions no longer fall back to raw names.
  This aligns with fixed-vocabulary 2D remapping. Old numbers must be rerun if that
  fallback affected a class; do not claim all historical metrics are unchanged.

Run `python -B -m unittest discover -s tests -v` for CPU regression checks. Full
scene rendering and numerical agreement with server results still require a GPU
run with the original data, weights, vocabulary and experiment configuration.
