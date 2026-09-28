from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))
os.environ["UNIFIED_PROTOCOL_NAME"] = "unified_precomputed_gt_2d_fixed_vocab_v1"
os.environ["UNIFIED_PREDICTION_CLASS_SCOPE"] = "all"
from unified_semantic_eval import UnifiedSemanticGT
from unified_semantic_eval.adapter_utils import (
    bootstrap_repository,
    ensure_eval_import_shims,
    finalize_predictions,
    load_repo_module,
    runtime_metadata,
    select_contract_views,
    tensor_map_to_numpy,
    writer_for_method,
)
from unified_semantic_eval.text_features import (
    encode_alphaclip_text,
    encode_openclip_text,
)


def _bootstrap_options() -> tuple[str, str]:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--gpu", default="-1")
    known, _ = parser.parse_known_args()
    return known.repo_root, known.gpu


def _add_argument_once(parser: ArgumentParser, option: str, **kwargs) -> None:
    if option not in parser._option_string_actions:
        parser.add_argument(option, **kwargs)


def main() -> None:
    repo_path, requested_gpu = _bootstrap_options()
    if str(requested_gpu) != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)
    repo_root = bootstrap_repository(repo_path)

    from arguments import ModelParams, PipelineParams, get_combined_args

    parser = ArgumentParser(description="Unified precomputed-GT evaluation for Scaffold-GS models with a teacher-matched text encoder")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    _add_argument_once(parser, "--repo_root", default=str(repo_root))
    _add_argument_once(parser, "--gt_root", required=True)
    _add_argument_once(parser, "--scene", required=True)
    _add_argument_once(parser, "--output_dir", required=True)
    _add_argument_once(parser, "--eval_text_encoder", choices=("alphaclip", "openclip"), required=True)
    _add_argument_once(parser, "--alphaclip_checkpoint", default="")
    _add_argument_once(parser, "--alphaclip_package_path", default="")
    _add_argument_once(parser, "--alphaclip_download_root", default="")
    _add_argument_once(parser, "--alphaclip_model", default="ViT-B/16")
    _add_argument_once(parser, "--openclip_weight", default="")
    _add_argument_once(parser, "--openclip_model", default="ViT-B-16")
    _add_argument_once(parser, "--iteration", type=int, default=32000)
    _add_argument_once(parser, "--text_batch_size", type=int, default=64)
    _add_argument_once(parser, "--classify_chunk_size", type=int, default=262144)
    _add_argument_once(parser, "--prompt_mode", choices=("single", "scaffold", "maskclustering"), default="maskclustering")
    _add_argument_once(parser, "--pred_min_alpha", type=float, default=0.0)
    _add_argument_once(parser, "--exclude_classes", nargs="*", default=["object"])
    _add_argument_once(parser, "--gpu", default="-1")
    _add_argument_once(parser, "--quiet", action="store_true")
    _add_argument_once(parser, "--overwrite", action="store_true")
    _add_argument_once(parser, "--save_visualizations", action="store_true")
    args = get_combined_args(parser)

    if args.text_batch_size < 1 or args.classify_chunk_size < 1:
        raise ValueError("Text and classification chunk sizes must be positive")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Scaffold-GS unified evaluation requires CUDA")
    ensure_eval_import_shims()
    from utils.general_utils import safe_state
    native = load_repo_module(
        "instguide_fixed_vocab_native",
        Path(__file__).resolve().with_name("semantic_metrics_2d_gtmesh.py"),
    )
    safe_state(args.quiet)

    required_paths = [args.source_path, args.model_path]
    if args.eval_text_encoder == "alphaclip":
        required_paths.extend((args.alphaclip_checkpoint, args.alphaclip_package_path, Path(args.alphaclip_download_root) / "ViT-B-16.pt"))
    else:
        required_paths.append(args.openclip_weight)
    for required in required_paths:
        if not Path(required).expanduser().exists():
            raise FileNotFoundError(required)

    gt = UnifiedSemanticGT.open(args.gt_root, scene=args.scene, split="test")
    if not gt.manifest.get("fixedVocabulary"):
        raise RuntimeError("Use GT produced by common/prepare_fixed_vocabulary_gt.py")
    dataset = model_params.extract(args, for_inference=True)
    pipeline = pipeline_params.extract(args)
    dataset.depths = ""
    dataset.use_semantic_features = True

    codec_path = Path(str(dataset.semantic_codec_path)).expanduser().resolve()
    if not codec_path.is_file():
        raise FileNotFoundError(f"The trained semantic decoder is required but was not found: {codec_path}")
    if int(dataset.semantic_feature_dim) != 512:
        raise RuntimeError(f"Unified text evaluation requires a 512D decoded feature; resolved semantic_feature_dim={dataset.semantic_feature_dim}")

    gaussians = native._build_gaussian_model(dataset)
    scene = native.Scene(dataset, gaussians, load_iteration=int(args.iteration), shuffle=False)
    gaussians.eval()
    views = select_contract_views(scene, gt)
    device = gaussians.get_anchor.device

    if args.eval_text_encoder == "alphaclip":
        text_features, text_metadata = encode_alphaclip_text(gt.class_names, args.alphaclip_model, args.alphaclip_checkpoint, args.alphaclip_package_path, args.alphaclip_download_root, str(device), prompt_mode=args.prompt_mode, batch_size=args.text_batch_size)
    else:
        text_features, text_metadata = encode_openclip_text(gt.class_names, args.openclip_model, args.openclip_weight, str(device), prompt_mode=args.prompt_mode, batch_size=args.text_batch_size)
    if int(text_features.shape[1]) != int(dataset.semantic_feature_dim):
        raise RuntimeError(f"Text/model feature dimensions differ: text={text_features.shape[1]} model={dataset.semantic_feature_dim}")

    loaded_iteration = int(getattr(scene, "loaded_iter", args.iteration))
    method_name = f"Scaffold-GS-{args.eval_text_encoder}"
    writer = writer_for_method(args.output_dir, gt, method_name, {
        "repoRoot": str(repo_root),
        "modelPath": str(Path(dataset.model_path).resolve()),
        "iteration": loaded_iteration,
        "semanticCodecPath": str(codec_path),
        "semanticFeatureDim": int(dataset.semantic_feature_dim),
        "semanticLatentDim": int(dataset.semantic_latent_dim),
        "textEncoder": text_metadata,
        "semanticDecision": "normalized_cosine_nearest_text",
        "predictionVocabulary": "full_fixed_dataset_vocabulary",
        "metricClassScope": "gt_present",
        "predMinAlpha": float(args.pred_min_alpha),
        "runtime": runtime_metadata(torch),
    }, args.overwrite, exclude_classes=args.exclude_classes)

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    class_indices = gt.prediction_class_indices(args.exclude_classes)
    with torch.no_grad():
        for view in native.tqdm(views, desc=f"Scaffold-GS {args.eval_text_encoder} unified semantic rendering", dynamic_ncols=True):
            feature_map, alpha = native._render_semantic_feature_map_for_view(view, gaussians, pipeline, background)
            prediction = native._predict_semantic_map(feature_map, text_features, args.classify_chunk_size, class_indices=class_indices)
            prediction_np = tensor_map_to_numpy(prediction)
            if float(args.pred_min_alpha) > 0:
                if alpha is None or alpha.numel() == 0:
                    raise RuntimeError("--pred_min_alpha > 0 requires the semantic renderer to return alpha")
                alpha_map = alpha[0] if alpha.ndim == 3 else alpha
                valid = alpha_map.detach().cpu().numpy() > float(args.pred_min_alpha)
                prediction_np = np.where(valid, prediction_np, -1).astype(np.int32)
            writer.write(str(view.image_name), prediction_np)

    finalize_predictions(writer, gt, args.output_dir, save_visualizations=args.save_visualizations, exclude_classes=args.exclude_classes)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Scaffold-GS teacher-matched unified evaluation failed: {exc}")
        raise
