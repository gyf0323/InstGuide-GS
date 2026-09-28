#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SRC:?Set SRC to the dataset root}"
: "${GEOM:?Set GEOM to the geometry model directory}"
STAGE="${STAGE:-both}"
ARGS=(--stage "$STAGE" --source "$SRC" --geometry "$GEOM"
      --images "${IMAGES:-images}" --voxel-size "${VOXEL_SIZE:-0.003}"
      --geometry-iterations "${GEOMETRY_ITERS:-30000}"
      --semantic-steps "${SEMANTIC_STEPS:-2000}"
      --semantic-dim "${SEMANTIC_DIM:-12}" --residual-dim "${RESIDUAL_DIM:-4}")
if [[ "$STAGE" != geometry ]]; then
    : "${MODEL:?Set MODEL to a new semantic model directory}"
    : "${TEACHER:?Set TEACHER to the compressed instance teacher directory}"
    : "${INSTANCE_POINT_CLOUD:?Set INSTANCE_POINT_CLOUD to the exact object_dict-indexed cloud}"
    ARGS+=(--model "$MODEL" --teacher "$TEACHER" --instance-point-cloud "$INSTANCE_POINT_CLOUD")
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
exec python "$ROOT/tools/train_pipeline.py" "${ARGS[@]}" "$@"
