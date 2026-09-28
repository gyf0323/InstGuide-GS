#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import csv
import hashlib
import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, prefilter_voxel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.semantic_benchmark_utils import (
    _encode_open_vocab_text_features,
    _load_semantic_gt_3d,
    load_semantic_prompt_bank,
)
from utils.system_utils import configure_visible_devices


SELECTED_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES_SOURCE = configure_visible_devices(sys.argv[1:])


def _safe_stem(view, index):
    name = str(getattr(view, "image_name", "") or f"{index:05d}")
    stem = Path(name).stem
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return f"{index:05d}_{safe}"


def _stable_color(name):
    digest = hashlib.md5(f"maskclustering:{name}".encode("utf-8")).digest()
    return np.asarray([48 + digest[0] % 176, 48 + digest[1] % 176, 48 + digest[2] % 176], dtype=np.uint8)


def _stable_palette(class_names):
    if not class_names:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.stack([_stable_color(str(name)) for name in class_names], axis=0).astype(np.uint8)


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


def _blend_overlay(rgb, color, alpha_np, overlay_alpha):
    blend = (alpha_np[..., None] * float(overlay_alpha)).clip(0.0, 1.0)
    overlay = rgb.astype(np.float32) * (1.0 - blend) + color.astype(np.float32) * blend
    return np.clip(overlay + 0.5, 0, 255).astype(np.uint8)


def _load_font(font_path, font_size):
    font_size = max(8, int(font_size))
    if font_path:
        try:
            return ImageFont.truetype(str(font_path), font_size)
        except OSError:
            pass
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_bbox(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except AttributeError:
        return draw.textsize(text, font=font)


def _text_color_for_background(rgb):
    r, g, b = [float(value) for value in rgb]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 150.0 else (255, 255, 255)


def _rects_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place_label(width, height, anchor_x, anchor_y, text_width, text_height, placed, pad):
    offsets = ((0, 0), (10, 10), (-10, 10), (10, -10), (-10, -10), (24, 0), (0, 24))
    box_width = text_width + 2 * pad
    box_height = text_height + 2 * pad
    for dx, dy in offsets:
        x0 = int(round(anchor_x + dx))
        y0 = int(round(anchor_y + dy))
        x0 = max(0, min(x0, max(0, width - box_width)))
        y0 = max(0, min(y0, max(0, height - box_height)))
        rect = (x0, y0, x0 + box_width, y0 + box_height)
        if not any(_rects_overlap(rect, old) for old in placed):
            return rect
    x0 = max(0, min(int(anchor_x), max(0, width - box_width)))
    y0 = max(0, min(int(anchor_y), max(0, height - box_height)))
    return x0, y0, x0 + box_width, y0 + box_height


def _draw_labels(image, candidates, palette, font, max_labels):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    placed = []
    pad = 4
    for candidate in candidates[: max(0, int(max_labels))]:
        text = candidate["text"]
        text_w, text_h = _text_bbox(draw, text, font)
        rect = _place_label(width, height, candidate["x"], candidate["y"], text_w, text_h, placed, pad)
        color = tuple(int(value) for value in palette[int(candidate["class_index"])].tolist())
        draw.rectangle(rect, fill=color, outline=(0, 0, 0))
        draw.text((rect[0] + pad, rect[1] + pad), text, fill=_text_color_for_background(color), font=font)
        placed.append(rect)
    return np.asarray(pil_image), len(placed)


def _load_line_index_classes(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"semantic class file not found: {path}")
    class_table = {}
    blank_line_ids = []
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for class_id, line in enumerate(raw_lines):
        name = line.strip()
        if not name:
            blank_line_ids.append(int(class_id))
            continue
        class_table[int(class_id)] = name
    if not class_table:
        raise RuntimeError(f"No class names were parsed from {path}")
    return class_table, {
        "semantic_classes": str(path),
        "class_table_format": "line_index",
        "num_class_lines": int(len(raw_lines)),
        "num_non_empty_classes": int(len(class_table)),
        "blank_line_ids": blank_line_ids,
    }


def _build_maskclustering_prompt_bank(prompt_path, semantic_gt_3d, all_prompt_classes, semantic_gt_info=""):
    class_table, class_info = _load_line_index_classes(prompt_path)
    if bool(all_prompt_classes):
        raw_ids = sorted(class_table)
        gt_info = {"mode": "all_prompt_classes", **class_info, "classes": []}
    else:
        gt_path = Path(semantic_gt_3d)
        if not gt_path.is_file():
            raise FileNotFoundError(f"semantic GT mesh not found: {gt_path}")
        gt_info_path = Path(semantic_gt_info) if semantic_gt_info else None
        if gt_info_path is not None and not gt_info_path.is_absolute() and not gt_info_path.exists():
            gt_info_path = gt_path.parent / gt_info_path.name
        _, labels = _load_semantic_gt_3d(gt_path, gt_info_path)
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        raw_ids = sorted(int(value) for value in np.unique(labels_np) if int(value) >= 0)
        gt_info = {
            "mode": "gt_mesh_classes_only_line_index",
            "semantic_gt_3d": str(gt_path),
            **class_info,
            "num_raw_gt_labels": int(len(raw_ids)),
            "ignored_negative_labels": sorted(int(value) for value in np.unique(labels_np) if int(value) < 0),
            "classes": [],
        }
    missing_ids = []
    class_ids = []
    class_names = []
    for raw_id in raw_ids:
        class_name = class_table.get(int(raw_id))
        if class_name is None:
            missing_ids.append(int(raw_id))
            continue
        local_index = len(class_ids)
        class_ids.append(int(raw_id))
        class_names.append(str(class_name))
        gt_info["classes"].append(
            {
                "prediction_id": int(local_index),
                "raw_id": int(raw_id),
                "name": str(class_name),
            }
        )
    if not class_ids:
        raise RuntimeError("No visualization classes were resolved from semantic classes and GT mesh.")
    gt_info["missing_class_ids"] = missing_ids
    return {"class_ids": class_ids, "class_names": class_names, "prompts": class_names}, gt_info


def _write_legend(output_root, prompt_bank, palette):
    rows = []
    for index, (raw_id, class_name, prompt) in enumerate(
        zip(prompt_bank["class_ids"], prompt_bank["class_names"], prompt_bank["prompts"])
    ):
        color = palette[index]
        rows.append(
            {
                "prediction_id": int(index),
                "raw_gt_id": int(raw_id),
                "class_name": str(class_name),
                "prompt": str(prompt),
                "r": int(color[0]),
                "g": int(color[1]),
                "b": int(color[2]),
            }
        )
    with (output_root / "legend.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prediction_id", "raw_gt_id", "class_name", "prompt", "r", "g", "b"])
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "legend.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _select_views(scene, split):
    split = str(split or "test").lower()
    selected = []
    if split in {"train", "all"}:
        selected.append(("train", scene.getTrainCameras()))
    if split in {"test", "all"}:
        selected.append(("test", scene.getTestCameras()))
    if split not in {"train", "test", "all"}:
        raise RuntimeError("--split must be one of train, test, all.")
    return selected


def _build_gaussian_model(dataset):
    return GaussianModel(
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


def _render_semantic_prediction(view, gaussians, pipeline, background, text_features, palette, args):
    visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
    render_pkg = render(
        view,
        gaussians,
        pipeline,
        background,
        visible_mask=visible_mask,
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

    feature_map = gaussians.decode_semantic_feature_map(semantic_latent)
    feature_map = F.normalize(feature_map, dim=0, eps=1e-6)
    logits = torch.einsum("chw,kc->khw", feature_map, F.normalize(text_features, dim=-1, eps=1e-6))
    confidence, labels = logits.max(dim=0)

    rgb = _tensor_rgb_to_uint8(render_pkg["render"])
    height, width = rgb.shape[:2]
    alpha_np = _alpha_to_numpy(render_pkg.get("alpha", None), height, width)
    visible = alpha_np >= float(args.alpha_threshold)

    labels_np = labels.detach().cpu().numpy().astype(np.int32)
    confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
    labels_np[~visible] = -1
    confidence_np[~visible] = 0.0

    color = np.zeros((height, width, 3), dtype=np.uint8)
    color[:, :] = np.asarray(args.void_color, dtype=np.uint8)
    valid = labels_np >= 0
    if np.any(valid):
        color[valid] = palette[labels_np[valid]]
    overlay = _blend_overlay(rgb, color, alpha_np * valid.astype(np.float32), args.overlay_alpha)
    confidence_png = np.clip((confidence_np + 1.0) * 0.5 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return {
        "rgb": rgb,
        "pred_color": color,
        "overlay": overlay,
        "pred_label": labels_np,
        "confidence": confidence_np,
        "confidence_png": confidence_png,
        "visible": valid,
        "visible_ratio": float(valid.mean()),
    }


def _label_candidates(labels_np, confidence_np, class_names, label_min_area, label_confidence):
    candidates = []
    valid_classes = [int(value) for value in np.unique(labels_np) if int(value) >= 0]
    for class_index in valid_classes:
        mask = labels_np == int(class_index)
        area = int(mask.sum())
        if area < int(label_min_area):
            continue
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        score = float(np.mean(confidence_np[mask])) if np.any(mask) else 0.0
        text = f"{class_names[class_index]} {score:.2f}" if label_confidence else str(class_names[class_index])
        candidates.append(
            {
                "class_index": int(class_index),
                "class_name": str(class_names[class_index]),
                "text": text,
                "area": area,
                "confidence": score,
                "x": float(xs.min()),
                "y": float(ys.min()),
            }
        )
    candidates.sort(key=lambda row: row["area"], reverse=True)
    return candidates


def _render_split(output_root, split_name, views, gaussians, pipeline, background, text_features, prompt_bank, palette, args):
    split_root = output_root / split_name
    dirs = {
        "class_color": split_root / "class_color",
        "class_overlay": split_root / "class_overlay",
        "class_overlay_labeled": split_root / "class_overlay_labeled",
        "pred_label": split_root / "pred_label",
        "confidence": split_root / "confidence",
    }
    if args.save_rgb:
        dirs["rgb"] = split_root / "rgb"
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    if int(args.max_views) > 0:
        views = views[: int(args.max_views)]
    font = _load_font(args.font_path, args.font_size)
    manifest = []
    class_pixel_counts = np.zeros((len(prompt_bank["class_names"]),), dtype=np.int64)
    timings = []
    for index, view in enumerate(tqdm(views, desc=f"Visualizing semantic 3DGS {split_name}", dynamic_ncols=True)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        result = _render_semantic_prediction(view, gaussians, pipeline, background, text_features, palette, args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.time() - start)

        labels_np = result["pred_label"]
        valid = labels_np >= 0
        if np.any(valid):
            class_pixel_counts += np.bincount(labels_np[valid], minlength=len(prompt_bank["class_names"]))
        candidates = _label_candidates(
            labels_np,
            result["confidence"],
            prompt_bank["class_names"],
            args.label_min_area,
            args.label_confidence,
        )
        labeled_overlay, labels_drawn = _draw_labels(
            result["overlay"],
            candidates,
            palette,
            font,
            args.max_labels_per_image,
        )

        stem = _safe_stem(view, index)
        Image.fromarray(result["pred_color"]).save(dirs["class_color"] / f"{stem}.png")
        Image.fromarray(result["overlay"]).save(dirs["class_overlay"] / f"{stem}.png")
        Image.fromarray(labeled_overlay).save(dirs["class_overlay_labeled"] / f"{stem}.png")
        Image.fromarray(result["confidence_png"]).save(dirs["confidence"] / f"{stem}.png")
        np.save(dirs["pred_label"] / f"{stem}.npy", labels_np)
        if args.save_rgb:
            Image.fromarray(result["rgb"]).save(dirs["rgb"] / f"{stem}.png")
        manifest.append(
            {
                "index": int(index),
                "image_name": str(getattr(view, "image_name", stem)),
                "visible_ratio": result["visible_ratio"],
                "labels_drawn": int(labels_drawn),
                "present_classes": [candidate["class_name"] for candidate in candidates],
            }
        )

    with (split_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    class_rows = []
    for class_index, count in enumerate(class_pixel_counts.tolist()):
        if count <= 0:
            continue
        class_rows.append(
            {
                "prediction_id": int(class_index),
                "raw_gt_id": int(prompt_bank["class_ids"][class_index]),
                "class_name": prompt_bank["class_names"][class_index],
                "pixel_count": int(count),
            }
        )
    with (split_root / "predicted_class_pixel_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prediction_id", "raw_gt_id", "class_name", "pixel_count"])
        writer.writeheader()
        writer.writerows(class_rows)
    if timings:
        mean_time = float(np.mean(timings))
        print(f"[Semantic3DGS-Vis] {split_name}: views={len(views)}, mean_time={mean_time:.4f}s")
    return {
        "num_views": int(len(views)),
        "predicted_class_pixel_counts": class_rows,
    }


def visualize_model(dataset, pipeline, iteration, args):
    with torch.no_grad():
        dataset.depths = ""
        dataset.use_semantic_features = True
        gaussians = _build_gaussian_model(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        if not gaussians.use_semantic_features or gaussians.get_semantic_decoder is None:
            raise RuntimeError("Loaded model does not contain semantic features/decoder.")

        prompt_path = Path(args.prompt_path or dataset.semantic_eval_prompt_path)
        if not prompt_path.is_absolute():
            prompt_path = Path(dataset.source_path) / prompt_path
        if args.class_protocol == "maskclustering":
            prompt_bank, gt_info = _build_maskclustering_prompt_bank(
                prompt_path,
                args.semantic_gt_3d or getattr(dataset, "semantic_gt_3d", ""),
                args.all_prompt_classes,
                args.semantic_gt_info or getattr(dataset, "semantic_gt_info", ""),
            )
        else:
            prompt_bank = load_semantic_prompt_bank(prompt_path)
            gt_info = {"mode": "prompt_bank", "classes": []}
        palette = _stable_palette(prompt_bank["class_names"])
        text_features = _encode_open_vocab_text_features(
            prompt_bank["prompts"],
            args.clip_model or getattr(dataset, "teacher_clip_model", "ViT-B/32"),
            getattr(dataset, "semantic_feature_dim", 512),
            device=str(gaussians.get_anchor.device),
            prompt_template_mode=args.semantic_text_prompt_mode,
        ).to(gaussians.get_anchor.device)

        background = torch.tensor(
            [1, 1, 1] if dataset.white_background else [0, 0, 0],
            dtype=torch.float32,
            device=gaussians.get_anchor.device,
        )
        output_root = Path(args.output_path) if args.output_path else Path(dataset.model_path) / "semantic_3dgs_labeled_renders"
        output_root.mkdir(parents=True, exist_ok=True)
        _write_legend(output_root, prompt_bank, palette)

        split_summaries = {}
        for split_name, views in _select_views(scene, args.split):
            split_summaries[split_name] = _render_split(
                output_root,
                split_name,
                views,
                gaussians,
                pipeline,
                background,
                text_features,
                prompt_bank,
                palette,
                args,
            )
        summary = {
            "model_path": str(dataset.model_path),
            "source_path": str(dataset.source_path),
            "loaded_iteration": int(scene.loaded_iter),
            "output_path": str(output_root),
            "split": str(args.split),
            "class_protocol": str(args.class_protocol),
            "prompt_path": str(prompt_path),
            "semantic_gt_3d": str(args.semantic_gt_3d or getattr(dataset, "semantic_gt_3d", "")),
            "clip_model": str(args.clip_model or getattr(dataset, "teacher_clip_model", "ViT-B/32")),
            "semantic_text_prompt_mode": str(args.semantic_text_prompt_mode),
            "num_classes": int(len(prompt_bank["class_names"])),
            "class_names": prompt_bank["class_names"],
            "gt_info": gt_info,
            "split_summaries": split_summaries,
        }
        (output_root / "render_config.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Semantic3DGS-Vis] Saved outputs to {output_root}")


def main():
    parser = ArgumentParser(description="Visualize a trained semantic 3DGS model with MaskClustering-style labels.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", default="test", choices=("train", "test", "all"))
    parser.add_argument("--prompt_path", default="", type=str)
    parser.add_argument("--class_protocol", default="maskclustering", choices=("maskclustering", "prompt_bank"))
    parser.add_argument("--all_prompt_classes", action="store_true", default=False)
    parser.add_argument("--clip_model", default="", type=str)
    parser.add_argument("--semantic_text_prompt_mode", default="maskclustering", choices=("single", "scaffold", "maskclustering"))
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--overlay_alpha", default=0.55, type=float)
    parser.add_argument("--alpha_threshold", default=0.01, type=float)
    parser.add_argument("--label_min_area", default=2500, type=int)
    parser.add_argument("--max_labels_per_image", default=80, type=int)
    parser.add_argument("--label_confidence", action="store_true", default=True)
    parser.add_argument("--font_path", default="", type=str)
    parser.add_argument("--font_size", default=44, type=int)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--save_rgb", action="store_true")
    parser.add_argument("--void_color", nargs=3, default=(0, 0, 0), type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default="-1")
    args = get_combined_args(parser)

    print("Visualizing semantic 3DGS model", args.model_path)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        "Using GPU visibility:",
        visible_devices if visible_devices else "<unset>",
        f"(source={CUDA_VISIBLE_DEVICES_SOURCE})",
    )
    safe_state(args.quiet)
    dataset = model.extract(args, for_inference=True)
    dataset.use_semantic_features = True
    visualize_model(dataset, pipeline.extract(args), args.iteration, args)


if __name__ == "__main__":
    main()
