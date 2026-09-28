#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to the trained semantic-model directory}"
: "${GT_MESH:?Set GT_MESH to a ScanNet++ mesh with vertex.label}"
: "${RAW_CLASSES:?Set RAW_CLASSES to semantic_classes.txt}"
: "${BENCHMARK_MAP:?Set BENCHMARK_MAP to the official map_benchmark.csv}"
: "${TOP100_CLASSES:?Set TOP100_CLASSES to the official top100.txt}"
: "${KNN_K:?Set KNN_K explicitly to match the reported 3D transfer protocol}"
: "${ALPHACLIP_CKPT:?Set ALPHACLIP_CKPT to the Alpha-CLIP visual checkpoint}"
: "${ALPHACLIP_DOWNLOAD_ROOT:?Set ALPHACLIP_DOWNLOAD_ROOT to the directory containing ViT-B-16.pt}"
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ALPHACLIP=${ALPHACLIP:-$REPO_ROOT/third_party/AlphaCLIP-main}
ITERATION=${ITERATION:-32000}
OUT=${OUT:-$MODEL/eval_fixed_vocab_3d_${ITERATION}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
cd "$REPO_ROOT"
python evaluation_release/3d/eval_direct_3d_19.py \
  --method ours --repo_root "$REPO_ROOT" \
  --point_cloud "$MODEL/point_cloud/iteration_${ITERATION}/point_cloud.ply" \
  --gt_mesh "$GT_MESH" --raw_classes "$RAW_CLASSES" --benchmark_map "$BENCHMARK_MAP" \
  --class_protocol scannetpp100 --top100_classes "$TOP100_CLASSES" \
  --geometry_transfer knn_latent --knn_k "$KNN_K" --knn_weight inverse_distance \
  --alphaclip_model ViT-B/16 --alphaclip_checkpoint "$ALPHACLIP_CKPT" \
  --alphaclip_package "$ALPHACLIP" --alphaclip_download_root "$ALPHACLIP_DOWNLOAD_ROOT" \
  --prompt_mode maskclustering --output_dir "$OUT" "$@"
