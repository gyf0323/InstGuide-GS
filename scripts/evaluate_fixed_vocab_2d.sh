#!/usr/bin/env bash
set -euo pipefail
: "${SRC:?Set SRC to the scene directory}"
: "${MODEL:?Set MODEL to the trained semantic-model directory}"
: "${GT_ROOT:?Set GT_ROOT to the prepared fixed-vocabulary 2D GT root}"
: "${SCENE:?Set SCENE to the GT manifest scene name}"
: "${ALPHACLIP_CKPT:?Set ALPHACLIP_CKPT to the Alpha-CLIP visual checkpoint}"
: "${ALPHACLIP_DOWNLOAD_ROOT:?Set ALPHACLIP_DOWNLOAD_ROOT to the directory containing ViT-B-16.pt}"
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ALPHACLIP=${ALPHACLIP:-$REPO_ROOT/third_party/AlphaCLIP-main}
ITERATION=${ITERATION:-32000}
SEMANTIC_DIM=${SEMANTIC_DIM:-12}
RESIDUAL_DIM=${RESIDUAL_DIM:-4}
IMAGES=${IMAGES:-images}
OUT=${OUT:-$MODEL/eval_fixed_vocab_2d_${ITERATION}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
cd "$REPO_ROOT"
python evaluation_release/2d/eval_unified_scaffoldgs_text_encoder.py \
  --repo_root "$REPO_ROOT" --source_path "$SRC" --model_path "$MODEL" \
  --iteration "$ITERATION" --images "$IMAGES" --eval \
  --semantic_3dgs --use_anchor_offset_semantic \
  --semantic_feature_dim 512 --semantic_latent_dim "$SEMANTIC_DIM" \
  --semantic_teacher_dim "$SEMANTIC_DIM" --semantic_residual_dim "$RESIDUAL_DIM" \
  --semantic_codec_path "$MODEL/semantic_codec.pth" \
  --gt_root "$GT_ROOT" --scene "$SCENE" --output_dir "$OUT" \
  --eval_text_encoder alphaclip --alphaclip_model ViT-B/16 \
  --alphaclip_checkpoint "$ALPHACLIP_CKPT" --alphaclip_package_path "$ALPHACLIP" \
  --alphaclip_download_root "$ALPHACLIP_DOWNLOAD_ROOT" \
  --prompt_mode maskclustering --exclude_classes object "$@"
