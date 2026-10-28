# Two stages, one training entry point

Use `scripts/train.sh` for the public method. Geometry reconstruction and semantic
distillation remain distinct optimization stages: semantics loads the saved
geometry, freezes it, and learns the geometry-guided Anchor-Offset representation.
The launcher runs each stage in a fresh process, retaining the original stage
initialization and random-seed lifecycle.

## Complete run

Activate the installed environment, then run from the repository root:

```bash
export SRC=/path/to/scene
export TEACHER=/path/to/compressed_teacher
export INSTANCE_POINT_CLOUD=/path/to/exact_object_dict_indexed_mesh.ply
export GEOM=$SRC/out_geometry_30000
export MODEL=$SRC/out_instguide_32000
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
bash scripts/train.sh
```

The default teacher, model latent and codec dimension is 12. Set `SEMANTIC_DIM`
explicitly only for a teacher/codec with another dimension. Other defaults are
30,000 geometry steps,
2,000 semantic steps, a 4D residual and a -4 gate initialization. Change
`GEOMETRY_ITERS` or `SEMANTIC_STEPS` to adjust the iteration counts.

The semantic stage enables both sparse geometry allocation and reliable-interior
teacher initialization/direct supervision. Boundary and unknown anchors receive
residual rows; interior anchors do not. Ordinary alpha-rendered 2D distillation
and gate/residual regularization remain enabled. No semantic memory, depth-visible
registration, or dense-residual ablation is selected.

## Run a single stage

Use the same launcher, not another training script:

```bash
STAGE=geometry bash scripts/train.sh
STAGE=semantic bash scripts/train.sh
```

`STAGE=geometry` only requires `SRC` and `GEOM`. `STAGE=semantic` requires a complete
geometry model at `GEOMETRY_ITERS` and a fresh semantic output. It does not resume
an interrupted semantic optimizer. Existing nonempty outputs are rejected, not
overwritten. For a preview, append `--dry-run`; this prints commands without
validating data or starting training.

The corresponding Python entry is `python tools/train_pipeline.py --help`.
`train.py` remains the shared numerical backend, not a duplicate implementation.

## Checks, split and records

Before a complete run, the launcher validates the dataset and teacher before
starting expensive geometry training. Checks cover registered images, camera
sizes, teacher dimensions, strict codec loading, frame maps/weights, and training
view coverage. The instance cloud must be the exact vertex ordering referenced
by `object_dict.npy`; file existence alone does not prove provenance. Geometric
binding is generated automatically at the start of semantic training.

Both stages use `--eval`: sorted camera stems at indices 0, 8, 16, ... are held
out; optimization samples the remaining cameras. All-view teacher construction
is a separate transductive protocol choice and must be disclosed.

Each stage writes `training.log`, `launch_config.json`, `process_status.json`,
`cfg_args` and, on success, `run_summary.json`. The last selected stage also writes
`pipeline_summary.json`. A failed stage stops the pipeline. A second-stage failure
does not remove the completed geometry model.

Timing covers the training subprocess and console logging, including setup and
final saves. Preflight, teacher generation and separate evaluation are excluded.
Size accounting lists inference weights, optional appearance/feature-bank weights,
decoder, codec and configuration; optimizer checkpoints, logs and metrics are
excluded. FP16 applies to semantic sidecar storage, not the entire model or training.
