#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import csv
import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, prefilter_voxel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.semantic_benchmark_utils import (
    load_semantic_prompt_bank,
    _apply_semantic_eval_exclude_indices,
    _build_gt_label_mapper,
    _build_prompt_name_to_index,
    _encode_open_vocab_text_features,
    _load_semantic_gt_3d,
    _map_gt_labels_to_indices,
    _resolve_semantic_eval_exclude_indices,
    _semantic_eval_prediction_class_indices,
    _to_abs_path,
)
from utils.system_utils import configure_visible_devices


SELECTED_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES_SOURCE = configure_visible_devices(sys.argv[1:])


def _safe_stem(view, index):
    name = str(getattr(view, "image_name", "") or f"{index:05d}")
    stem = Path(name).stem
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return f"{index:05d}_{safe}"


def _make_palette(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for idx in range(num_classes):
        # Deterministic high-contrast hash palette.
        palette[idx, 0] = (37 * idx + 23) % 256
        palette[idx, 1] = (17 * idx + 119) % 256
        palette[idx, 2] = (97 * idx + 53) % 256
    if num_classes > 0:
        palette[0] = np.array([230, 25, 75], dtype=np.uint8)
    return palette


def _tensor_rgb_to_uint8(image):
    image = image.detach().clamp(0.0, 1.0).cpu()
    if image.ndim != 3 or image.shape[0] < 3:
        raise RuntimeError(f"Expected RGB tensor [3,H,W], got {tuple(image.shape)}")
    array = image[:3].permute(1, 2, 0).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def _alpha_to_numpy(alpha, height, width):
    if alpha is None or not torch.is_tensor(alpha) or alpha.numel() == 0:
        return np.ones((height, width), dtype=np.float32)
    alpha = alpha.detach().float().cpu()
    if alpha.ndim == 3:
        alpha = alpha[0]
    return alpha.clamp(0.0, 1.0).numpy().astype(np.float32)


def _save_palette_files(output_root, prompt_bank, palette):
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (class_id, class_name, prompt) in enumerate(
        zip(prompt_bank["class_ids"], prompt_bank["class_names"], prompt_bank["prompts"])
    ):
        rgb = palette[idx].tolist()
        rows.append(
            {
                "index": int(idx),
                "class_id": int(class_id),
                "class_name": class_name,
                "prompt": prompt,
                "r": int(rgb[0]),
                "g": int(rgb[1]),
                "b": int(rgb[2]),
            }
        )

    with (output_root / "palette.json").open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2, ensure_ascii=False)
    with (output_root / "palette.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["index", "class_id", "class_name", "prompt", "r", "g", "b"])
        writer.writeheader()
        writer.writerows(rows)


def _parse_semantic_render_classes(raw_value):
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return []
    # Class names can contain spaces, e.g. "ceiling lamp"; use comma/semicolon
    # separators for multiple classes and keep a space-only value as one class.
    parts = raw_text.replace(";", ",").split(",")
    classes = []
    seen = set()
    for part in parts:
        name = part.strip()
        key = " ".join(name.lower().split())
        if name and key not in seen:
            classes.append(name)
            seen.add(key)
    return classes


def _resolve_semantic_render_class_indices(raw_value, prompt_bank):
    class_names = _parse_semantic_render_classes(raw_value)
    if not class_names:
        return None, []

    prompt_name_to_index = _build_prompt_name_to_index(prompt_bank)
    indices = []
    missing = []
    for class_name in class_names:
        key = " ".join(class_name.lower().split())
        if key not in prompt_name_to_index:
            missing.append(class_name)
            continue
        indices.append(int(prompt_name_to_index[key]))

    if missing:
        raise RuntimeError(
            "--semantic_render_classes contains classes that are not present in "
            f"the prompt file: {', '.join(missing)}"
        )

    unique_indices = []
    seen = set()
    for index in indices:
        if index not in seen:
            unique_indices.append(index)
            seen.add(index)
    resolved_names = [prompt_bank["class_names"][idx] for idx in unique_indices]
    return unique_indices, resolved_names


def _blend_overlay(rgb, color, alpha_np, overlay_alpha):
    blend = (alpha_np[..., None] * float(overlay_alpha)).clip(0.0, 1.0)
    overlay = (rgb.astype(np.float32) * (1.0 - blend) + color.astype(np.float32) * blend)
    return np.clip(overlay + 0.5, 0, 255).astype(np.uint8)


def _feature_map_to_pca_rgb(semantic_feature, pca_basis, pca_mean, pca_min, pca_max, visible):
    channels, height, width = semantic_feature.shape
    flat = semantic_feature.permute(1, 2, 0).reshape(-1, channels)
    coords = (flat - pca_mean) @ pca_basis
    denom = (pca_max - pca_min).clamp_min(1e-6)
    coords = ((coords - pca_min) / denom).clamp(0.0, 1.0)
    pca_rgb = (coords.view(height, width, 3).detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    pca_rgb[~visible] = 0
    return pca_rgb


def _render_semantic_view(
    view,
    gaussians,
    pipeline,
    background,
    text_features,
    palette,
    pca_state,
    args,
    class_indices=None,
    render_class_indices=None,
):
    voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
    render_pkg = render(
        view,
        gaussians,
        pipeline,
        background,
        visible_mask=voxel_visible_mask,
        render_semantic_features=True,
        semantic_only=False,
        render_alpha=True,
    )

    semantic_latent = render_pkg.get("semantic", None)
    if semantic_latent is None or semantic_latent.numel() == 0:
        raise RuntimeError(f"Semantic render returned an empty latent map for view {view.image_name}.")
    semantic_weight = render_pkg.get("semantic_weight", None)
    if semantic_weight is None or semantic_weight.numel() == 0:
        semantic_weight = render_pkg.get("alpha", None)
    if semantic_weight is not None and semantic_weight.numel() > 0:
        semantic_latent = semantic_latent / semantic_weight.clamp_min(1e-6)

    semantic_feature = gaussians.decode_semantic_feature_map(semantic_latent)
    semantic_feature = F.normalize(semantic_feature, dim=0, eps=1e-6)
    rgb = _tensor_rgb_to_uint8(render_pkg["render"])
    height, width = rgb.shape[:2]
    alpha_np = _alpha_to_numpy(render_pkg.get("alpha", None), height, width)
    visible = alpha_np >= float(args.alpha_threshold)

    result = {
        "rgb": rgb,
        "visible_ratio": float(visible.mean()),
    }
    if text_features is not None and palette is not None:
        class_indices_tensor = None
        if class_indices is not None:
            class_indices_tensor = torch.as_tensor(class_indices, dtype=torch.long, device=text_features.device)
            if class_indices_tensor.numel() == 0:
                raise RuntimeError("Semantic rendering class restriction is empty.")
            text_features = text_features[class_indices_tensor]
        logits = torch.einsum("chw,kc->khw", semantic_feature, text_features)
        logits = logits * float(args.temperature)
        confidence, labels = torch.softmax(logits, dim=0).max(dim=0)
        if class_indices_tensor is not None:
            labels = class_indices_tensor[labels]

        labels_np = labels.detach().cpu().numpy().astype(np.int16)
        confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
        color = palette[labels_np.clip(0, max(0, palette.shape[0] - 1))]

        display_visible = visible
        if render_class_indices is not None:
            render_class_indices = np.asarray(render_class_indices, dtype=np.int64)
            if render_class_indices.size == 0:
                raise RuntimeError("Semantic render class display filter is empty.")
            display_visible = visible & np.isin(labels_np, render_class_indices)

        color[~display_visible] = np.array(args.void_color, dtype=np.uint8)
        confidence_np[~display_visible] = 0.0
        labels_np[~display_visible] = -1
        overlay_alpha_np = alpha_np
        if render_class_indices is not None:
            overlay_alpha_np = alpha_np.copy()
            overlay_alpha_np[~display_visible] = 0.0

        result.update(
            {
                "pred_color": color,
                "overlay": _blend_overlay(rgb, color, overlay_alpha_np, args.overlay_alpha),
                "pred_label": labels_np,
                "confidence": confidence_np,
                "confidence_png": np.clip(confidence_np * 255.0 + 0.5, 0, 255).astype(np.uint8),
            }
        )
    if pca_state is not None:
        pca_rgb = _feature_map_to_pca_rgb(
            semantic_feature,
            pca_state["basis"],
            pca_state["mean"],
            pca_state["min"],
            pca_state["max"],
            visible,
        )
        result.update(
            {
                "feature_pca": pca_rgb,
                "feature_pca_overlay": _blend_overlay(rgb, pca_rgb, alpha_np, args.overlay_alpha),
            }
        )
    return result


def _select_views(scene, split):
    split = split.lower()
    if split == "train":
        return [("train", scene.getTrainCameras())]
    if split == "test":
        return [("test", scene.getTestCameras())]
    if split == "all":
        return [("train", scene.getTrainCameras()), ("test", scene.getTestCameras())]
    raise RuntimeError(f"Unsupported split: {split}")


def _get_semantic_decoder(gaussians):
    decoder = getattr(gaussians, "semantic_decoder", None)
    if decoder is not None:
        return decoder

    getter = getattr(gaussians, "get_semantic_decoder", None)
    if getter is None:
        return None

    # Some repository revisions expose this as a @property returning the module;
    # calling it would accidentally call the decoder forward() with no input.
    if isinstance(getter, torch.nn.Module):
        return getter
    if callable(getter):
        return getter()
    return getter


def _compute_anchor_pca_state(gaussians, max_samples):
    anchor_features = gaussians.decode_semantic_features()
    if anchor_features is None or anchor_features.numel() == 0:
        raise RuntimeError("Semantic decoder produced no anchor features for PCA rendering.")
    features = anchor_features.detach().float().cpu()
    if int(max_samples) > 0 and features.shape[0] > int(max_samples):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        sample_indices = torch.randperm(features.shape[0], generator=generator)[: int(max_samples)]
        features_for_fit = features[sample_indices]
    else:
        features_for_fit = features

    mean = features_for_fit.mean(dim=0, keepdim=True)
    centered = features_for_fit - mean
    _, _, basis = torch.pca_lowrank(centered, q=3, niter=6, center=False)
    projected = centered @ basis
    pca_min = projected.quantile(0.01, dim=0)
    pca_max = projected.quantile(0.99, dim=0)
    device = gaussians.get_anchor.device
    return {
        "mean": mean.to(device),
        "basis": basis.to(device),
        "min": pca_min.to(device),
        "max": pca_max.to(device),
        "num_fit_samples": int(features_for_fit.shape[0]),
        "num_anchors": int(features.shape[0]),
    }


def _render_split(
    output_root,
    split_name,
    views,
    gaussians,
    pipeline,
    background,
    text_features,
    palette,
    pca_state,
    args,
    class_indices=None,
    render_class_indices=None,
):
    split_root = output_root / split_name
    class_enabled = args.render_mode in {"class", "both"}
    pca_enabled = args.render_mode in {"pca", "both"}
    pred_color_dir = split_root / "pred_color"
    overlay_dir = split_root / "overlay"
    label_dir = split_root / "pred_label"
    confidence_dir = split_root / "confidence"
    pca_dir = split_root / "feature_pca"
    pca_overlay_dir = split_root / "feature_pca_overlay"
    rgb_dir = split_root / "rgb"
    if class_enabled:
        for directory in (pred_color_dir, overlay_dir, label_dir, confidence_dir):
            directory.mkdir(parents=True, exist_ok=True)
    if pca_enabled:
        for directory in (pca_dir, pca_overlay_dir):
            directory.mkdir(parents=True, exist_ok=True)
    if args.save_rgb:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    if args.max_views > 0:
        views = views[: int(args.max_views)]

    manifest = []
    timings = []
    for idx, view in enumerate(tqdm(views, desc=f"Semantic rendering {split_name}")):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        result = _render_semantic_view(
            view,
            gaussians,
            pipeline,
            background,
            text_features,
            palette,
            pca_state,
            args,
            class_indices=class_indices,
            render_class_indices=render_class_indices,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.time() - start)

        stem = _safe_stem(view, idx)
        entry = {
            "index": int(idx),
            "image_name": str(getattr(view, "image_name", stem)),
            "visible_ratio": result["visible_ratio"],
        }
        if class_enabled:
            color_path = pred_color_dir / f"{stem}.png"
            overlay_path = overlay_dir / f"{stem}.png"
            label_path = label_dir / f"{stem}.npy"
            confidence_path = confidence_dir / f"{stem}.png"

            Image.fromarray(result["pred_color"]).save(color_path)
            Image.fromarray(result["overlay"]).save(overlay_path)
            np.save(label_path, result["pred_label"])
            Image.fromarray(result["confidence_png"]).save(confidence_path)
            entry.update(
                {
                    "pred_color": str(color_path),
                    "overlay": str(overlay_path),
                    "pred_label": str(label_path),
                    "confidence": str(confidence_path),
                }
            )
            if args.save_confidence_npy:
                confidence_npy_path = confidence_dir / f"{stem}.npy"
                np.save(confidence_npy_path, result["confidence"])
                entry["confidence_npy"] = str(confidence_npy_path)
        if pca_enabled:
            pca_path = pca_dir / f"{stem}.png"
            pca_overlay_path = pca_overlay_dir / f"{stem}.png"
            Image.fromarray(result["feature_pca"]).save(pca_path)
            Image.fromarray(result["feature_pca_overlay"]).save(pca_overlay_path)
            entry.update(
                {
                    "feature_pca": str(pca_path),
                    "feature_pca_overlay": str(pca_overlay_path),
                }
            )
        if args.save_rgb:
            rgb_path = rgb_dir / f"{stem}.png"
            Image.fromarray(result["rgb"]).save(rgb_path)
            entry["rgb"] = str(rgb_path)

        manifest.append(entry)

    with (split_root / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    if timings:
        mean_time = float(np.mean(timings))
        fps = 1.0 / mean_time if mean_time > 0 else 0.0
        print(f"[Semantic Render] {split_name}: views={len(views)}, mean_time={mean_time:.4f}s, fps={fps:.2f}")


def render_semantic_sets(dataset, iteration, pipeline, args):
    with torch.no_grad():
        dataset.depths = ""
        dataset.use_semantic_features = True
        if getattr(dataset, "semantic_3dgs", False):
            dataset.use_semantic_features = True

        class_enabled = args.render_mode in {"class", "both"}
        pca_enabled = args.render_mode in {"pca", "both"}
        prompt_path = args.prompt_path or dataset.semantic_eval_prompt_path
        if class_enabled and not prompt_path:
            raise RuntimeError("Class semantic rendering needs --prompt_path or --semantic_eval_prompt_path.")

        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
            dataset.use_semantic_features,
            dataset.semantic_latent_dim,
            dataset.semantic_feature_dim,
            semantic_teacher_dim=getattr(dataset, "semantic_teacher_dim", 0),
            semantic_codec_path=getattr(dataset, "semantic_codec_path", ""),
            sh_degree=dataset.sh_degree,
            use_anchor_offset_semantic=getattr(dataset, "use_anchor_offset_semantic", False),
            semantic_residual_dim=getattr(dataset, "semantic_residual_dim", 4),
            semantic_gate_init=getattr(dataset, "semantic_gate_init", -4.0),
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        if not gaussians.use_semantic_features or _get_semantic_decoder(gaussians) is None:
            raise RuntimeError("Loaded model does not contain semantic features/decoder.")

        prompt_bank = None
        text_features = None
        palette = None
        class_indices = None
        mesh_classes_only = False
        excluded_class_names = []
        prediction_class_names = None
        render_class_indices = None
        render_class_names = []
        if class_enabled:
            prompt_bank = load_semantic_prompt_bank(prompt_path)
            text_features = _encode_open_vocab_text_features(
                prompt_bank["prompts"],
                args.clip_model or getattr(dataset, "teacher_clip_model", "ViT-B/32"),
                getattr(dataset, "semantic_feature_dim", 512),
                device=str(gaussians.get_anchor.device),
            ).to(gaussians.get_anchor.device)
            palette = _make_palette(len(prompt_bank["class_names"]))
            exclude_indices, excluded_class_names = _resolve_semantic_eval_exclude_indices(dataset, prompt_bank, logger=None)
            gt_indices = None
            if bool(getattr(dataset, "semantic_eval_mesh_classes_only", False)):
                gt_path = _to_abs_path(getattr(dataset, "semantic_gt_3d", ""), dataset.source_path)
                if gt_path is None:
                    raise RuntimeError("--semantic_eval_mesh_classes_only rendering requires --semantic_gt_3d.")
                gt_info_path = _to_abs_path(getattr(dataset, "semantic_gt_info", ""), dataset.source_path)
                _, labels = _load_semantic_gt_3d(gt_path, gt_info_path)
                gt_to_index, ignore_values, _ = _build_gt_label_mapper(dataset, prompt_bank, logger=None)
                gt_indices = _map_gt_labels_to_indices(labels, gt_to_index, ignore_values)
                gt_indices = _apply_semantic_eval_exclude_indices(gt_indices, exclude_indices)
            class_indices, mesh_classes_only = _semantic_eval_prediction_class_indices(
                dataset,
                torch.empty((0,), dtype=torch.long) if gt_indices is None else gt_indices,
                len(prompt_bank["class_ids"]),
                exclude_indices,
                prompt_bank,
                logger=None,
            )
            prediction_class_names = [prompt_bank["class_names"][idx] for idx in class_indices] if class_indices is not None else None
            render_class_indices, render_class_names = _resolve_semantic_render_class_indices(
                getattr(args, "semantic_render_classes", ""),
                prompt_bank,
            )
            if render_class_indices is not None and class_indices is not None:
                allowed = {int(index) for index in class_indices}
                unavailable = [idx for idx in render_class_indices if int(idx) not in allowed]
                if unavailable:
                    unavailable_names = [prompt_bank["class_names"][idx] for idx in unavailable]
                    raise RuntimeError(
                        "--semantic_render_classes includes classes outside the active prediction set "
                        "(after mesh-only/exclude filtering): "
                        + ", ".join(unavailable_names)
                    )

        pca_state = _compute_anchor_pca_state(gaussians, args.pca_max_samples) if pca_enabled else None

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=gaussians.get_anchor.device)

        output_root = Path(args.output_path) if args.output_path else Path(dataset.model_path) / "semantic_renders" / f"ours_{scene.loaded_iter}"
        output_root.mkdir(parents=True, exist_ok=True)
        if class_enabled:
            _save_palette_files(output_root, prompt_bank, palette)

        config = {
            "model_path": dataset.model_path,
            "source_path": dataset.source_path,
            "loaded_iteration": int(scene.loaded_iter),
            "render_mode": args.render_mode,
            "prompt_path": str(prompt_path) if prompt_path else "",
            "split": args.split,
            "overlay_alpha": float(args.overlay_alpha),
            "alpha_threshold": float(args.alpha_threshold),
            "temperature": float(args.temperature),
            "num_classes": len(prompt_bank["class_names"]) if prompt_bank is not None else 0,
            "mesh_classes_only": bool(mesh_classes_only),
            "excluded_classes": excluded_class_names,
            "prediction_classes": prediction_class_names,
            "render_classes": render_class_names,
        }
        if pca_state is not None:
            config["pca"] = {
                "num_anchors": pca_state["num_anchors"],
                "num_fit_samples": pca_state["num_fit_samples"],
                "max_samples": int(args.pca_max_samples),
            }
        with (output_root / "render_config.json").open("w", encoding="utf-8") as fp:
            json.dump(config, fp, indent=2, ensure_ascii=False)

        for split_name, views in _select_views(scene, args.split):
            _render_split(
                output_root,
                split_name,
                views,
                gaussians,
                pipeline,
                background,
                text_features,
                palette,
                pca_state,
                args,
                class_indices=class_indices,
                render_class_indices=render_class_indices,
            )

        print(f"[Semantic Render] Saved outputs to {output_root}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render semantic maps from a trained semantic Scaffold-GS model.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--prompt_path", default="", type=str, help="Class/prompt file. Defaults to semantic_eval_prompt_path.")
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--render_mode", default="both", choices=["class", "pca", "both"])
    parser.add_argument("--overlay_alpha", default=0.45, type=float)
    parser.add_argument("--alpha_threshold", default=0.01, type=float)
    parser.add_argument("--temperature", default=15.0, type=float)
    parser.add_argument("--pca_max_samples", default=200000, type=int)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--save_rgb", action="store_true")
    parser.add_argument("--save_confidence_npy", action="store_true")
    parser.add_argument(
        "--semantic_render_classes",
        default="",
        type=str,
        help=(
            "Comma/semicolon-separated class names to display in class renders. "
            "Classification still uses the active prompt set; non-selected top-1 pixels are hidden."
        ),
    )
    parser.add_argument("--clip_model", default="", type=str, help="Override CLIP model/checkpoint for prompt encoding.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default="-1")
    args = get_combined_args(parser)

    print("Semantic rendering " + args.model_path)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        "Using GPU visibility:",
        visible_devices if visible_devices else "<unset>",
        f"(source={CUDA_VISIBLE_DEVICES_SOURCE})",
    )

    args.void_color = (0, 0, 0)
    safe_state(args.quiet)
    render_semantic_sets(model.extract(args, for_inference=True), args.iteration, pipeline.extract(args), args)
