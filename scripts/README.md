# Public launchers

Only three shell launchers are shipped:

| Script | Purpose | Required environment variables |
| --- | --- | --- |
| `train.sh` | Geometry then sparse instance-guided semantics | `SRC`, `GEOM`, `MODEL`, `TEACHER`, `INSTANCE_POINT_CLOUD` |
| `evaluate_fixed_vocab_2d.sh` | Held-out fixed-vocabulary 2D evaluation | `SRC`, `MODEL`, `GT_ROOT`, `SCENE`, `ALPHACLIP_CKPT`, `ALPHACLIP_DOWNLOAD_ROOT` |
| `evaluate_fixed_vocab_3d.sh` | Camera-free ScanNet++ native-latent 3D transfer | `MODEL`, `GT_MESH`, `RAW_CLASSES`, `BENCHMARK_MAP`, `TOP100_CLASSES`, `KNN_K`, `ALPHACLIP_CKPT`, `ALPHACLIP_DOWNLOAD_ROOT` |

Activate the environment first; no launcher assumes a particular Conda or server
path. Set `SEMANTIC_DIM` and `RESIDUAL_DIM` to the matching model configuration.

`train.sh` defaults to `STAGE=both`; `STAGE=geometry` needs only `SRC`/`GEOM`, and
`STAGE=semantic` uses an already saved geometry model. See the complete
[training example](../SEMANTIC_TWO_STAGE.md) for iteration counts, split, logging,
timing, size accounting and output protection.

See [evaluation protocols](../evaluation_release/README.md) for vocabulary,
encoder and GT requirements. 2D test-view metrics and direct 3D metrics are
different measurements. Historical view-voting, baseline adapters, experiment
matrices and separate geometry/semantic shell scripts are not shipped.
