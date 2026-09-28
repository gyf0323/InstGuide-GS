#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import (
    prefilter_voxel,
    render,
    network_gui,
    prepare_visible_gaussians,
    rasterize_precomputed_gaussians,
)
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr, warpped_depth, read_image_pairs
from utils.semantic_feature_utils import IMAGE_EXTENSIONS
from utils.instance_teacher_adapter import (
    get_instance_teacher_store,
    is_instance_level_teacher_dir,
    is_instance_teacher_ref,
    load_instance_teacher_from_ref,
)
from utils.instance_geometry_semantic import (
    GEOMETRY_STATE_INTERIOR,
    classify_anchor_geometry,
    prepare_instance_geometry_binding,
)
from utils.semantic_benchmark_utils import semantic_benchmark_enabled, run_semantic_benchmark_for_saved_model
from utils.system_utils import configure_visible_devices
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


SELECTED_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES_SOURCE = configure_visible_devices(sys.argv[1:])

# torch.set_num_threads(32)
LPIPS_FN_CACHE = {}

def get_lpips_fn(device):
    torch_device = torch.device(device)
    cache_key = str(torch_device)
    if cache_key not in LPIPS_FN_CACHE:
        LPIPS_FN_CACHE[cache_key] = lpips.LPIPS(net='vgg').to(torch_device)
    return LPIPS_FN_CACHE[cache_key]


def _compute_fps(timings, warmup=5):
    if not timings:
        return None
    effective_timings = timings[warmup:] if len(timings) > warmup else timings
    if not effective_timings:
        return None
    mean_time = float(torch.tensor(effective_timings, dtype=torch.float32).mean().item())
    if mean_time <= 0.0:
        return None
    return 1.0 / mean_time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


REGION_TEACHER_CACHE = OrderedDict()
MAX_REGION_TEACHER_CACHE_VIEWS = 64
AUTO_DEPTH_LAMBDA = 0.02


def _log_info(logger, message):
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def _clone_namespace(args, **updates):
    values = vars(args).copy()
    values.update(updates)
    return Namespace(**values)


def _unique_iterations(values, final_iteration):
    final_iteration = int(final_iteration)
    items = {
        int(value)
        for value in (values or [])
        if int(value) > 0 and int(value) <= final_iteration
    }
    items.add(final_iteration)
    return sorted(items)


def _resolve_path(path_value, root):
    if not path_value:
        return None
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return Path(root) / path_obj


def _list_source_images(args):
    image_dir = _resolve_path(getattr(args, "images", "images"), getattr(args, "source_path", ""))
    if image_dir is None:
        return []
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))
    return sorted({str(path): path for path in images}.values(), key=lambda path: path.name)


def _list_colmap_camera_images(args):
    source_value = str(getattr(args, "source_path", "") or "")
    if not source_value:
        return []
    source_path = Path(source_value)
    sparse_dir = source_path / "sparse" / "0"
    reading_dir = getattr(args, "images", "images")
    image_dir = _resolve_path(reading_dir, source_path)
    if image_dir is None:
        return []

    image_names = []
    binary_path = sparse_dir / "images.bin"
    text_path = sparse_dir / "images.txt"
    try:
        if binary_path.is_file():
            import struct

            with binary_path.open("rb") as handle:
                num_images = struct.unpack("<Q", handle.read(8))[0]
                for _ in range(num_images):
                    handle.read(4 + 8 * 7 + 4)
                    name_bytes = bytearray()
                    while True:
                        char = handle.read(1)
                        if char == b"\x00" or char == b"":
                            break
                        name_bytes.extend(char)
                    image_names.append(name_bytes.decode("utf-8"))
                    num_points2d = struct.unpack("<Q", handle.read(8))[0]
                    handle.seek(int(num_points2d) * 24, os.SEEK_CUR)
        elif text_path.is_file():
            with text_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 10:
                        image_names.append(parts[9])
        else:
            return []
    except Exception:
        return []

    image_paths = []
    for image_name in image_names:
        image_name = os.path.basename(str(image_name))
        if image_name:
            image_paths.append(Path(image_dir) / image_name)
    return sorted({str(path): path for path in image_paths}.values(), key=lambda path: path.name)


def _list_semantic_teacher_images(args):
    colmap_images = _list_colmap_camera_images(args)
    return colmap_images if colmap_images else _list_source_images(args)


















def _image_hw(image_path):
    with Image.open(image_path) as image:
        width, height = image.size
    return int(height), int(width)




















def _format_teacher_examples(items, limit=5):
    return ", ".join(f"{image_path.name}:{level_name}({reason})" for image_path, level_name, _, reason in items[:limit])


def _ensure_instance_level_teacher(args, region_dir, images, logger=None):
    expected_dim = int(
        getattr(args, "semantic_teacher_dim", 0)
        or getattr(args, "semantic_feature_dim", 512)
    )
    try:
        store = get_instance_teacher_store(
            region_dir,
            map_mode=getattr(args, "instance_teacher_map_mode", "auto"),
        )
    except Exception as exc:
        raise RuntimeError(f"[Semantic Teacher] Failed to open instance-level teacher at {region_dir}: {exc}") from exc

    reusable = []
    missing = []
    stale = []
    corrupt = []
    for image_path in images:
        frame_id = Path(image_path).stem
        image_hw = None
        try:
            image_hw = _image_hw(image_path)
        except Exception as exc:
            corrupt.append((image_path, "instance", region_dir, f"corrupt: failed to read source image ({exc})"))
            continue
        issue = store.frame_issue(frame_id, expected_dim, image_hw=image_hw)
        if issue is None:
            reusable.append((image_path, "instance", region_dir, "ok"))
        elif str(issue).startswith("missing:"):
            missing.append((image_path, "instance", region_dir, issue))
        elif str(issue).startswith("stale:"):
            stale.append((image_path, "instance", region_dir, issue))
        else:
            corrupt.append((image_path, "instance", region_dir, issue))

    total_required = len(reusable) + len(missing) + len(stale) + len(corrupt)
    if not missing and not stale and not corrupt:
        _log_info(
            logger,
            "[Semantic Teacher] Reusing "
            f"{len(reusable)}/{total_required} matched instance-level teachers from {region_dir} "
            f"(instances={store.num_instances}, dim={store.feature_dim}, "
            f"map_mode={store.map_mode}, requested_map_mode={store.requested_map_mode}).",
        )
        return

    examples = _format_teacher_examples(missing + stale + corrupt)
    raise RuntimeError(
        "[Semantic Teacher] Instance-level teacher is missing, stale, or corrupt. "
        "For reference-mode teachers, ensure summary.json frame_data.frame_instance_maps_dir "
        "and frame_data.frame_weights_dir point to valid local directories, or regenerate with "
        "--frame_data_mode copy. "
        f"reusable={len(reusable)}, missing={len(missing)}, stale={len(stale)}, corrupt={len(corrupt)}. "
        f"Examples: {examples}"
    )


def ensure_semantic_teachers(args, logger=None):
    """Validate the instance-level teacher before semantic training."""
    if not getattr(args, "semantic_3dgs", False):
        return
    images = _list_semantic_teacher_images(args)
    if not images:
        image_dir = _resolve_path(getattr(args, "images", "images"), args.source_path)
        raise RuntimeError(f"[Semantic Teacher] No registered RGB images found in {image_dir}.")
    region_dir = _resolve_path(getattr(args, "region_teachers", ""), args.source_path)
    if region_dir is None or not is_instance_level_teacher_dir(region_dir):
        raise RuntimeError(
            "[Semantic Teacher] --region_teachers must point to an instance-level teacher "
            "with instance_features.npy, instance_ids.npy and frame maps/weights. "
            "Historical per-frame LangSplat and region-PT teachers are not supported by this release."
        )
    _ensure_instance_level_teacher(args, region_dir, images, logger=logger)


def saveRuntimeCode(dst: str) -> None:
    log_dir = pathlib.Path(__file__).parent.resolve()
    dst_path = pathlib.Path(dst).resolve()

    if dst_path.exists():
        if dst_path.is_dir():
            print(f"Backup already exists at {dst_path}, skipping runtime code backup.")
            return
        raise RuntimeError(f"Backup destination exists and is not a directory: {dst_path}")

    additional_ignore_patterns = {
        ".git",
        ".gitignore",
        "__pycache__",
        "out",
        "outputs",
        "backup",
        "build",
        ".idea",
        "point_cloud",
        "region_teachers",
        "region_teachers_*",
        "teacher_visualizations",
    }
    ignore_patterns = set()
    gitignore_path = log_dir / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8") as gitignore_file:
            for line in gitignore_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.endswith("/"):
                    line = line[:-1]
                ignore_patterns.add(line)

    try:
        relative_dst = dst_path.relative_to(log_dir)
    except ValueError:
        relative_dst = None

    if relative_dst is not None and len(relative_dst.parts) > 0:
        # Avoid recursively copying the backup destination when it lives inside the repo.
        additional_ignore_patterns.add(relative_dst.parts[0])

    ignore_patterns.update(additional_ignore_patterns)

    shutil.copytree(log_dir, dst_path, ignore=shutil.ignore_patterns(*sorted(ignore_patterns)))
    
    print('Backup Finished!')

def _match_rendered_depth_to_target(viewpoint_cam, rendered_depth):
    if rendered_depth is None:
        return None
    if getattr(viewpoint_cam, "depth_is_warped", False):
        return warpped_depth(rendered_depth)
    return rendered_depth


def compute_depth_edge_mask(gt_depth, gt_depth_mask, threshold, dilation):
    depth = gt_depth[0]
    valid = gt_depth_mask[0]
    edge = torch.zeros_like(valid, dtype=torch.bool)

    horiz_valid = valid[:, 1:] & valid[:, :-1]
    horiz_edge = torch.abs(depth[:, 1:] - depth[:, :-1]) > threshold
    edge[:, 1:] |= horiz_valid & horiz_edge
    edge[:, :-1] |= horiz_valid & horiz_edge

    vert_valid = valid[1:, :] & valid[:-1, :]
    vert_edge = torch.abs(depth[1:, :] - depth[:-1, :]) > threshold
    edge[1:, :] |= vert_valid & vert_edge
    edge[:-1, :] |= vert_valid & vert_edge

    if dilation > 0:
        kernel = 2 * dilation + 1
        edge = F.max_pool2d(edge.float().unsqueeze(0).unsqueeze(0), kernel_size=kernel, stride=1, padding=dilation).squeeze(0).squeeze(0) > 0.5
    return edge


def get_depth_loss_weight(opt, iteration):
    if opt.lambda_depth <= 0.0 or iteration < opt.depth_start_iter:
        return 0.0
    duration_iters = max(0, int(getattr(opt, "depth_duration_iters", 0)))
    if duration_iters > 0 and iteration >= opt.depth_start_iter + duration_iters:
        return 0.0
    ramp_iters = max(0, int(getattr(opt, "depth_ramp_iters", 0)))
    if ramp_iters <= 0:
        return float(opt.lambda_depth)
    ramp = min(1.0, float(iteration - opt.depth_start_iter + 1) / float(ramp_iters))
    return float(opt.lambda_depth) * ramp


def robust_depth_residual_terms(depth_residual, opt=None):
    eps = float(getattr(opt, "depth_charbonnier_eps", 1e-3)) if opt is not None else 1e-3
    huber_delta = float(getattr(opt, "depth_residual_clip", 0.0)) if opt is not None else 0.0
    if huber_delta > 0.0:
        huber_delta = max(huber_delta, eps)
        abs_residual = torch.sqrt(depth_residual * depth_residual + eps * eps)
        quadratic = 0.5 * depth_residual * depth_residual / huber_delta
        linear = abs_residual - 0.5 * huber_delta
        return torch.where(abs_residual <= huber_delta, quadratic, linear)
    return torch.sqrt(depth_residual * depth_residual + eps * eps)


def robust_depth_residual_loss(depth_residual, opt=None):
    return robust_depth_residual_terms(depth_residual, opt).mean()


def weighted_depth_residual_loss(depth_residual, weights, opt=None):
    terms = robust_depth_residual_terms(depth_residual, opt)
    weights = weights.to(terms.dtype).clamp_min(0.0)
    return (terms * weights).sum() / weights.sum().clamp_min(1e-6)


def _resize_chw_tensor(tensor, target_hw, mode="nearest"):
    if tensor is None or tuple(tensor.shape[-2:]) == tuple(target_hw):
        return tensor
    kwargs = {"size": target_hw, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)


def _depth_valid_mask(gt_depth, gt_depth_mask, opt=None):
    valid = gt_depth_mask.clone()
    valid &= torch.isfinite(gt_depth)
    valid &= gt_depth > 0
    grad_threshold = float(getattr(opt, "depth_grad_filter_threshold", 0.0)) if opt is not None else 0.0
    if grad_threshold > 0.0:
        dilation = int(getattr(opt, "depth_grad_filter_dilation", 0)) if opt is not None else 0
        edge_mask = compute_depth_edge_mask(gt_depth, gt_depth_mask, grad_threshold, dilation)
        valid &= ~edge_mask.unsqueeze(0)
    return valid


def _normalize_rendered_depth_to_metric(rendered_depth, rendered_alpha, viewpoint_cam, target_hw, opt=None, alpha_min=None):
    rendered_depth = _resize_chw_tensor(rendered_depth, target_hw, mode="bilinear")
    rendered_alpha_mask = None
    if rendered_alpha is not None:
        rendered_alpha = rendered_alpha.to(rendered_depth.device, non_blocking=True)
        rendered_alpha = _resize_chw_tensor(rendered_alpha, target_hw, mode="bilinear")
        if alpha_min is None:
            alpha_min = float(getattr(opt, "depth_alpha_min", 0.05)) if opt is not None else 0.05
        alpha_min = max(float(alpha_min), 1e-6)
        rendered_alpha_mask = rendered_alpha > alpha_min
        rendered_depth = rendered_depth / rendered_alpha.detach().clamp_min(1e-6)
    rendered_depth = _match_rendered_depth_to_target(viewpoint_cam, rendered_depth)
    return rendered_depth, rendered_alpha, rendered_alpha_mask


def rendered_depth_loss(viewpoint_cam, rendered_depth, rendered_alpha=None, opt=None):
    if rendered_depth is None or viewpoint_cam.depth is None or viewpoint_cam.depth_mask is None:
        return None

    min_valid = int(getattr(opt, "depth_min_valid_pixels", 128)) if opt is not None else 128
    gt_depth = viewpoint_cam.depth.to(rendered_depth.device, non_blocking=True)
    gt_depth_mask = viewpoint_cam.depth_mask.to(rendered_depth.device, non_blocking=True) > 0.5
    rendered_depth, _, rendered_alpha_mask = _normalize_rendered_depth_to_metric(
        rendered_depth,
        rendered_alpha,
        viewpoint_cam,
        gt_depth.shape[-2:],
        opt,
    )

    if gt_depth_mask.sum().item() < min_valid:
        return None

    valid = _depth_valid_mask(gt_depth, gt_depth_mask, opt)
    valid &= torch.isfinite(rendered_depth)
    valid &= rendered_depth > viewpoint_cam.znear
    if rendered_alpha_mask is not None:
        valid &= rendered_alpha_mask

    if valid.sum().item() < min_valid:
        return None

    pred_depth = rendered_depth[valid]
    target_depth = gt_depth[valid]
    depth_residual = pred_depth - target_depth
    return robust_depth_residual_loss(depth_residual, opt)


def front_floater_opacity_loss(viewpoint_cam, rendered_depth, rendered_alpha, opt=None):
    if (
        rendered_depth is None
        or rendered_alpha is None
        or viewpoint_cam.depth is None
        or viewpoint_cam.depth_mask is None
    ):
        return None
    front_weight = float(getattr(opt, "depth_front_weight", 0.0)) if opt is not None else 0.0
    if front_weight <= 0.0:
        return None

    min_valid = int(getattr(opt, "depth_min_valid_pixels", 128)) if opt is not None else 128
    gt_depth = viewpoint_cam.depth.to(rendered_depth.device, non_blocking=True)
    gt_depth_mask = viewpoint_cam.depth_mask.to(rendered_depth.device, non_blocking=True) > 0.5
    alpha_min = float(getattr(opt, "depth_front_alpha_min", 0.02)) if opt is not None else 0.02
    pred_depth, rendered_alpha, alpha_mask = _normalize_rendered_depth_to_metric(
        rendered_depth,
        rendered_alpha,
        viewpoint_cam,
        gt_depth.shape[-2:],
        opt,
        alpha_min=alpha_min,
    )

    margin = float(getattr(opt, "depth_front_margin", 0.05)) if opt is not None else 0.05
    valid = _depth_valid_mask(gt_depth, gt_depth_mask, opt)
    valid &= torch.isfinite(pred_depth)
    valid &= pred_depth > viewpoint_cam.znear
    if alpha_mask is not None:
        valid &= alpha_mask
    front_residual = gt_depth - pred_depth - margin
    valid &= front_residual > 0
    if valid.sum().item() < min_valid:
        return None

    weights = rendered_alpha[valid].clamp_min(0.0)
    return weighted_depth_residual_loss(front_residual[valid], weights, opt)


def back_floater_opacity_loss(viewpoint_cam, rendered_depth, rendered_alpha, opt=None):
    if (
        rendered_depth is None
        or rendered_alpha is None
        or viewpoint_cam.depth is None
        or viewpoint_cam.depth_mask is None
    ):
        return None
    back_weight = float(getattr(opt, "depth_back_weight", 0.0)) if opt is not None else 0.0
    if back_weight <= 0.0:
        return None

    min_valid = int(getattr(opt, "depth_min_valid_pixels", 128)) if opt is not None else 128
    gt_depth = viewpoint_cam.depth.to(rendered_depth.device, non_blocking=True)
    gt_depth_mask = viewpoint_cam.depth_mask.to(rendered_depth.device, non_blocking=True) > 0.5
    alpha_min = float(getattr(opt, "depth_back_alpha_min", 0.02)) if opt is not None else 0.02
    pred_depth, rendered_alpha, alpha_mask = _normalize_rendered_depth_to_metric(
        rendered_depth,
        rendered_alpha,
        viewpoint_cam,
        gt_depth.shape[-2:],
        opt,
        alpha_min=alpha_min,
    )

    margin = float(getattr(opt, "depth_back_margin", 0.05)) if opt is not None else 0.05
    valid = _depth_valid_mask(gt_depth, gt_depth_mask, opt)
    valid &= torch.isfinite(pred_depth)
    valid &= pred_depth > viewpoint_cam.znear
    if alpha_mask is not None:
        valid &= alpha_mask
    back_residual = pred_depth - gt_depth - margin
    valid &= back_residual > 0
    if valid.sum().item() < min_valid:
        return None

    weights = rendered_alpha[valid].clamp_min(0.0)
    return weighted_depth_residual_loss(back_residual[valid], weights, opt)


def _evict_region_teacher_cache(max_views=MAX_REGION_TEACHER_CACHE_VIEWS):
    while len(REGION_TEACHER_CACHE) > max_views:
        REGION_TEACHER_CACHE.popitem(last=False)


def _load_region_teacher_from_disk(region_teacher_path, expected_dim):
    if not is_instance_teacher_ref(region_teacher_path):
        raise RuntimeError("Only instance-level teacher references are supported by this release.")
    return load_instance_teacher_from_ref(region_teacher_path, expected_dim)


def _get_cached_region_teacher(viewpoint_cam, expected_dim):
    region_teacher_path = getattr(viewpoint_cam, "region_teacher_path", None)
    if region_teacher_path is None:
        return None
    cache_key = (str(region_teacher_path), int(expected_dim))
    if cache_key in REGION_TEACHER_CACHE:
        REGION_TEACHER_CACHE.move_to_end(cache_key)
        return REGION_TEACHER_CACHE[cache_key]

    payload = _load_region_teacher_from_disk(region_teacher_path, expected_dim)
    REGION_TEACHER_CACHE[cache_key] = payload
    REGION_TEACHER_CACHE.move_to_end(cache_key)
    _evict_region_teacher_cache()
    return payload


def load_region_teacher(viewpoint_cam, expected_dim, target_hw=None, device=None):
    if device is None:
        device = viewpoint_cam.data_device
    raw_payload = _get_cached_region_teacher(viewpoint_cam, expected_dim)
    if raw_payload is None:
        return None
    if raw_payload["region_id_map"].numel() == 0 or raw_payload["region_features"].numel() == 0:
        return None

    region_id_map = raw_payload["region_id_map"]
    if target_hw is not None and tuple(region_id_map.shape) != tuple(target_hw):
        region_id_map = F.interpolate(
            region_id_map.float().unsqueeze(0).unsqueeze(0),
            size=target_hw,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
    else:
        region_id_map = region_id_map.clone()

    region_id_map = region_id_map.to(device, non_blocking=True)
    region_features = raw_payload["region_features"].to(device, non_blocking=True)
    region_quality = raw_payload["region_quality"].to(device, non_blocking=True)
    clip_confidence = raw_payload["clip_confidence"].to(device, non_blocking=True)
    pixel_quality = raw_payload["pixel_quality"]
    if pixel_quality.numel() > 0:
        if target_hw is not None and tuple(pixel_quality.shape) != tuple(target_hw):
            pixel_quality = F.interpolate(
                pixel_quality.float().unsqueeze(0).unsqueeze(0),
                size=target_hw,
                mode="nearest",
            ).squeeze(0).squeeze(0)
        else:
            pixel_quality = pixel_quality.clone()
        pixel_quality = pixel_quality.to(device, non_blocking=True)

    valid_mask = region_id_map >= 0
    if pixel_quality.numel() > 0:
        valid_mask &= pixel_quality > 0
    if (
        target_hw is None
        or tuple(raw_payload["region_id_map"].shape) == tuple(region_id_map.shape)
    ):
        region_area = raw_payload["region_area"].to(device, non_blocking=True)
    elif region_features.numel() > 0 and bool(valid_mask.any().item()):
        flat_ids = region_id_map.view(-1)
        valid_flat = flat_ids >= 0
        region_area = torch.bincount(flat_ids[valid_flat], minlength=region_features.shape[0]).float().to(device)
    else:
        region_area = raw_payload["region_area"].to(device, non_blocking=True)

    return {
        "image_name": raw_payload["image_name"],
        "region_id_map": region_id_map,
        "valid_mask": valid_mask,
        "region_features": region_features,
        "region_quality": region_quality,
        "clip_confidence": clip_confidence,
        "has_clip_confidence": raw_payload["has_clip_confidence"],
        "region_area": region_area,
        "pixel_quality": pixel_quality,
    }


def project_points_to_image(points, viewpoint_cam, image_hw, use_depth_camera=False):
    width = int(image_hw[1])
    height = int(image_hw[0])
    if points.numel() == 0:
        empty = torch.empty((0,), device=points.device, dtype=torch.long)
        return {
            "pixel_x": empty,
            "pixel_y": empty,
            "pixel_x_float": torch.empty((0,), device=points.device, dtype=points.dtype),
            "pixel_y_float": torch.empty((0,), device=points.device, dtype=points.dtype),
            "depth": torch.empty((0,), device=points.device, dtype=points.dtype),
            "valid": torch.empty((0,), device=points.device, dtype=torch.bool),
        }

    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)
    if use_depth_camera and getattr(viewpoint_cam, "depth_full_proj_transform", None) is not None:
        full_proj_transform = viewpoint_cam.depth_full_proj_transform.to(points.device)
    else:
        full_proj_transform = viewpoint_cam.full_proj_transform.to(points.device)
    world_view_transform = viewpoint_cam.world_view_transform.to(points.device)
    clip = torch.matmul(points_h, full_proj_transform)
    denom = clip[:, 3].clamp_min(1e-8)
    ndc = clip[:, :3] / denom.unsqueeze(1)

    view = torch.matmul(points_h, world_view_transform)
    depth = view[:, 2]

    pixel_x_float = ((ndc[:, 0] + 1.0) * 0.5) * (width - 1)
    pixel_y_float = ((1.0 - ndc[:, 1]) * 0.5) * (height - 1)
    pixel_x = pixel_x_float.round().long()
    pixel_y = pixel_y_float.round().long()

    valid = torch.isfinite(depth) & (depth > viewpoint_cam.znear)
    valid &= torch.isfinite(pixel_x_float) & torch.isfinite(pixel_y_float)
    valid &= ndc[:, 0] >= -1.0
    valid &= ndc[:, 0] <= 1.0
    valid &= ndc[:, 1] >= -1.0
    valid &= ndc[:, 1] <= 1.0
    valid &= pixel_x >= 0
    valid &= pixel_x < width
    valid &= pixel_y >= 0
    valid &= pixel_y < height

    pixel_x = pixel_x.clamp(0, width - 1)
    pixel_y = pixel_y.clamp(0, height - 1)

    return {
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "pixel_x_float": pixel_x_float,
        "pixel_y_float": pixel_y_float,
        "depth": depth,
        "valid": valid,
    }


def sample_image_tensor(image_tensor, pixel_x, pixel_y):
    if image_tensor is None:
        return None
    if image_tensor.ndim == 3:
        return image_tensor[:, pixel_y, pixel_x]
    if image_tensor.ndim == 2:
        return image_tensor[pixel_y, pixel_x]
    raise RuntimeError(f"Unsupported tensor shape for image sampling: {tuple(image_tensor.shape)}")


def _match_projected_depth_to_target(viewpoint_cam, projected_depth):
    if getattr(viewpoint_cam, "depth_is_warped", False):
        return warpped_depth(projected_depth)
    return projected_depth


def _sample_depth_residual_gate(
    viewpoint_cam,
    rendered_depth,
    rendered_alpha,
    pixel_x,
    pixel_y,
    target_depth,
    opt=None,
    mode="back",
    margin=0.05,
):
    if rendered_depth is None:
        return None
    mode = str(mode).lower()
    alpha_attr = "depth_front_alpha_min" if mode == "front" else "depth_back_alpha_min"
    pred_depth, _, alpha_mask = _normalize_rendered_depth_to_metric(
        rendered_depth,
        rendered_alpha,
        viewpoint_cam,
        target_depth.shape[-2:],
        opt,
        alpha_min=float(getattr(opt, alpha_attr, 0.02)) if opt is not None else 0.02,
    )
    sampled_pred_depth = sample_image_tensor(pred_depth[0], pixel_x, pixel_y)
    sampled_target_depth = sample_image_tensor(target_depth[0], pixel_x, pixel_y)
    valid = torch.isfinite(sampled_pred_depth) & torch.isfinite(sampled_target_depth)
    valid &= sampled_pred_depth > viewpoint_cam.znear
    if alpha_mask is not None:
        valid &= sample_image_tensor(alpha_mask[0], pixel_x, pixel_y).bool()

    if mode == "front":
        residual = sampled_target_depth - sampled_pred_depth - margin
    elif mode in {"both", "band"}:
        residual = torch.abs(sampled_pred_depth - sampled_target_depth) - margin
    else:
        residual = sampled_pred_depth - sampled_target_depth - margin
    return valid & (residual > 0)


def gaussian_surface_depth_loss(
    viewpoint_cam,
    prepared,
    visibility_filter,
    opt=None,
    use_depth_camera=False,
    rendered_depth=None,
    rendered_alpha=None,
):
    if viewpoint_cam.depth is None or viewpoint_cam.depth_mask is None:
        return None
    surface_weight = float(getattr(opt, "depth_surface_weight", 0.0)) if opt is not None else 0.0
    if surface_weight <= 0.0:
        return None

    xyz = prepared.get("xyz", None)
    opacity = prepared.get("opacity", None)
    if xyz is None or opacity is None or xyz.numel() == 0:
        return None

    device = xyz.device
    gt_depth = viewpoint_cam.depth.to(device, non_blocking=True)
    gt_depth_mask = viewpoint_cam.depth_mask.to(device, non_blocking=True) > 0.5
    target_hw = gt_depth.shape[-2:]
    projection = project_points_to_image(xyz, viewpoint_cam, target_hw, use_depth_camera=use_depth_camera)
    if projection["depth"].numel() == 0:
        return None

    pixel_x = projection["pixel_x"]
    pixel_y = projection["pixel_y"]
    valid = projection["valid"].clone()
    if visibility_filter is not None:
        valid &= visibility_filter.to(device=device, dtype=torch.bool)

    valid_depth_mask = _depth_valid_mask(gt_depth, gt_depth_mask, opt)[0]
    sampled_valid = sample_image_tensor(valid_depth_mask, pixel_x, pixel_y).bool()
    sampled_depth = sample_image_tensor(gt_depth[0], pixel_x, pixel_y)
    gaussian_depth = _match_projected_depth_to_target(viewpoint_cam, projection["depth"])

    opacity = opacity.view(-1)
    opacity_for_gate = opacity.detach().clamp_min(0.0)
    min_opacity = float(getattr(opt, "depth_surface_min_opacity", 0.0)) if opt is not None else 0.0
    if min_opacity > 0.0:
        valid &= opacity_for_gate > min_opacity
    valid &= sampled_valid
    valid &= torch.isfinite(sampled_depth) & torch.isfinite(gaussian_depth)
    valid &= sampled_depth > 0
    valid &= gaussian_depth > viewpoint_cam.znear

    margin = float(getattr(opt, "depth_surface_margin", 0.05)) if opt is not None else 0.05
    surface_mode = str(getattr(opt, "depth_surface_mode", "back")).lower() if opt is not None else "back"
    if surface_mode == "front":
        surface_residual = sampled_depth - gaussian_depth - margin
    elif surface_mode in {"both", "band"}:
        surface_residual = torch.abs(gaussian_depth - sampled_depth) - margin
    else:
        surface_residual = gaussian_depth - sampled_depth - margin
    valid &= surface_residual > 0
    residual_gate = _sample_depth_residual_gate(
        viewpoint_cam,
        rendered_depth,
        rendered_alpha,
        pixel_x,
        pixel_y,
        gt_depth,
        opt=opt,
        mode=surface_mode,
        margin=margin,
    )
    if residual_gate is not None:
        valid &= residual_gate
    min_valid = int(getattr(opt, "depth_min_valid_pixels", 128)) if opt is not None else 128
    if valid.sum().item() < min_valid:
        return None

    weights = opacity_for_gate[valid]
    return weighted_depth_residual_loss(surface_residual[valid], weights, opt)


def depth_aware_densification_filter(
    viewpoint_cam,
    prepared,
    update_filter,
    opt=None,
    use_depth_camera=False,
    rendered_depth=None,
    rendered_alpha=None,
):
    if (
        update_filter is None
        or viewpoint_cam.depth is None
        or viewpoint_cam.depth_mask is None
        or not bool(getattr(opt, "depth_densify_gate", True))
    ):
        return update_filter

    xyz = prepared.get("xyz", None)
    if xyz is None or xyz.numel() == 0:
        return update_filter

    device = xyz.device
    gt_depth = viewpoint_cam.depth.to(device, non_blocking=True)
    gt_depth_mask = viewpoint_cam.depth_mask.to(device, non_blocking=True) > 0.5
    target_hw = gt_depth.shape[-2:]
    projection = project_points_to_image(xyz, viewpoint_cam, target_hw, use_depth_camera=use_depth_camera)
    if projection["depth"].numel() == 0:
        return update_filter

    pixel_x = projection["pixel_x"]
    pixel_y = projection["pixel_y"]
    valid_depth_mask = _depth_valid_mask(gt_depth, gt_depth_mask, opt)[0]
    sampled_valid = sample_image_tensor(valid_depth_mask, pixel_x, pixel_y).bool()
    sampled_depth = sample_image_tensor(gt_depth[0], pixel_x, pixel_y)
    gaussian_depth = _match_projected_depth_to_target(viewpoint_cam, projection["depth"])

    densify_mode = str(getattr(opt, "depth_densify_mode", "back")).lower() if opt is not None else "back"
    if densify_mode == "front":
        margin = float(getattr(opt, "depth_densify_front_margin", 0.08)) if opt is not None else 0.08
        gate_margin = margin
        outlier_residual = sampled_depth - gaussian_depth
    elif densify_mode in {"both", "band"}:
        front_margin = float(getattr(opt, "depth_densify_front_margin", 0.08)) if opt is not None else 0.08
        back_margin = float(getattr(opt, "depth_densify_back_margin", front_margin)) if opt is not None else front_margin
        outlier_residual = torch.maximum(sampled_depth - gaussian_depth - front_margin, gaussian_depth - sampled_depth - back_margin)
        margin = 0.0
        gate_margin = max(front_margin, back_margin)
    else:
        margin = float(getattr(opt, "depth_densify_back_margin", 0.08)) if opt is not None else 0.08
        gate_margin = margin
        outlier_residual = gaussian_depth - sampled_depth

    depth_outlier = projection["valid"] & sampled_valid
    depth_outlier &= torch.isfinite(sampled_depth) & torch.isfinite(gaussian_depth)
    depth_outlier &= sampled_depth > 0
    depth_outlier &= gaussian_depth > viewpoint_cam.znear
    depth_outlier &= outlier_residual > margin
    residual_gate = _sample_depth_residual_gate(
        viewpoint_cam,
        rendered_depth,
        rendered_alpha,
        pixel_x,
        pixel_y,
        gt_depth,
        opt=opt,
        mode=densify_mode,
        margin=gate_margin,
    )
    if residual_gate is not None:
        depth_outlier &= residual_gate

    filtered = update_filter.clone()
    filtered &= ~depth_outlier.to(filtered.device)
    return filtered


def _semantic_skip(reason, **debug):
    debug["loss"] = None
    debug["skip_reason"] = reason
    return debug


def _teacher_pixel_quality_map(region_teacher, region_id_map):
    pixel_quality = region_teacher.get("pixel_quality", None)
    if torch.is_tensor(pixel_quality) and pixel_quality.numel() > 0:
        if tuple(pixel_quality.shape) != tuple(region_id_map.shape):
            raise RuntimeError(
                f"Teacher pixel_quality shape {tuple(pixel_quality.shape)} does not match "
                f"region_id_map shape {tuple(region_id_map.shape)}."
            )
        return pixel_quality.to(region_id_map.device).float().clamp(0.0, 1.0)

    region_quality = region_teacher.get("region_quality", None)
    if torch.is_tensor(region_quality) and region_quality.numel() > 0:
        safe_ids = region_id_map.clamp(0, max(int(region_quality.numel()) - 1, 0))
        quality = region_quality.to(region_id_map.device).float()[safe_ids]
        return torch.where(region_id_map >= 0, quality, torch.zeros_like(quality)).clamp(0.0, 1.0)
    return (region_id_map >= 0).float()


def compute_alpha_blended_semantic_loss(
    viewpoint_cam,
    gaussians,
    pipe,
    background,
    prepared,
    region_teacher,
    opt,
):
    if region_teacher is None or prepared is None:
        return _semantic_skip("missing_teacher_or_prepared")
    latent_per_gaussian = prepared.get("semantic_feature", None)
    xyz = prepared.get("xyz", None)
    if latent_per_gaussian is None or latent_per_gaussian.numel() == 0 or xyz is None:
        return _semantic_skip("missing_semantic_gaussians")
    if latent_per_gaussian.shape[0] != xyz.shape[0]:
        raise RuntimeError(
            "Semantic latent expansion is misaligned with generated gaussians: "
            f"semantic={tuple(latent_per_gaussian.shape)}, xyz={tuple(xyz.shape)}."
        )

    target_hw = tuple(region_teacher["region_id_map"].shape)
    semantic_pkg = rasterize_precomputed_gaussians(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        prepared,
        retain_grad=False,
        image_height_override=target_hw[0],
        image_width_override=target_hw[1],
        semantic_only=True,
        render_alpha=True,
    )
    rendered_latent = semantic_pkg.get("semantic", None)
    semantic_weight = semantic_pkg.get("semantic_weight", None)
    semantic_weight_source = "semantic_weight"
    if semantic_weight is None or semantic_weight.numel() == 0:
        semantic_weight = semantic_pkg.get("alpha", None)
        semantic_weight_source = "alpha"
    if rendered_latent is None:
        return _semantic_skip("missing_rendered_latent")
    if semantic_weight is None:
        return _semantic_skip("missing_semantic_denominator")
    if rendered_latent.numel() == 0 or semantic_weight.numel() == 0:
        return _semantic_skip(
            "empty_semantic_render",
            rendered_latent_shape=tuple(rendered_latent.shape) if rendered_latent is not None else None,
            semantic_weight_shape=tuple(semantic_weight.shape) if semantic_weight is not None else None,
            semantic_weight_source=semantic_weight_source,
        )

    normalized_latent = rendered_latent / semantic_weight.clamp_min(1e-6)
    rendered_feature = gaussians.semantic_training_feature_map(normalized_latent)
    if rendered_feature is None or rendered_feature.numel() == 0:
        return _semantic_skip(
            "empty_decoded_semantic",
            rendered_latent_shape=tuple(rendered_latent.shape),
            semantic_weight_shape=tuple(semantic_weight.shape),
            semantic_weight_source=semantic_weight_source,
        )

    region_id_map = region_teacher["region_id_map"].long()
    region_features = region_teacher["region_features"]
    teacher_quality_map = _teacher_pixel_quality_map(region_teacher, region_id_map)
    semantic_weight_map = semantic_weight[0] if semantic_weight.ndim == 3 else semantic_weight
    valid_mask = region_teacher["valid_mask"] & (region_id_map >= 0)
    valid_mask &= teacher_quality_map > 0
    valid_mask &= semantic_weight_map > float(getattr(opt, "semantic_min_alpha", 0.05))
    min_pixels = int(getattr(opt, "semantic_min_pixels", 64))
    valid_pixels = int(valid_mask.sum().item())
    if valid_pixels < min_pixels:
        return _semantic_skip(
            "too_few_valid_pixels",
            valid_pixels=valid_pixels,
            min_pixels=min_pixels,
            rendered_semantic_shape=tuple(rendered_feature.shape),
            rendered_latent_shape=tuple(rendered_latent.shape),
            teacher_feature_shape=tuple(region_features.shape),
            semantic_weight_shape=tuple(semantic_weight.shape),
            semantic_weight_source=semantic_weight_source,
            semantic_weight_max=float(semantic_weight_map.max().detach().item()) if semantic_weight_map.numel() > 0 else 0.0,
        )

    flat_region_ids = region_id_map.view(-1)
    flat_valid = valid_mask.view(-1)
    rendered_flat = rendered_feature.permute(1, 2, 0).reshape(-1, rendered_feature.shape[0])
    teacher_flat = region_features[flat_region_ids[flat_valid]]
    rendered_flat = rendered_flat[flat_valid]
    cosine = (rendered_flat * teacher_flat).sum(dim=1).clamp(-1.0, 1.0)
    teacher_quality = teacher_quality_map.view(-1)[flat_valid].detach()
    loss = (teacher_quality * (1.0 - cosine)).sum() / teacher_quality.sum().clamp_min(1e-6)

    return {
        "loss": loss,
        "valid_pixels": valid_pixels,
        "rendered_semantic_shape": tuple(rendered_feature.shape),
        "rendered_latent_shape": tuple(rendered_latent.shape),
        "teacher_feature_shape": tuple(region_features.shape),
        "child_gaussians": int(xyz.shape[0]),
        "anchor_indices_shape": tuple(prepared.get("anchor_indices", torch.empty(0, device=xyz.device)).shape),
        "semantic_selected_shape": tuple(latent_per_gaussian.shape),
        "semantic_weight_mean": float(semantic_weight_map[valid_mask].mean().detach().item()),
        "teacher_quality_mean": float(teacher_quality.mean().item()),
        "semantic_weight_source": semantic_weight_source,
    }




















def _iter_gaussian_named_parameters(gaussians):
    tensor_names = [
        "_anchor",
        "_offset",
        "_anchor_feat",
        "_opacity",
        "_scaling",
        "_rotation",
        "_semantic_feat",
        "_semantic_anchor",
        "_semantic_offset_residual",
        "_semantic_offset_gate",
    ]
    for name in tensor_names:
        value = getattr(gaussians, name, None)
        if torch.is_tensor(value):
            yield name, value
    module_names = [
        "mlp_opacity",
        "mlp_cov",
        "mlp_color",
        "mlp_feature_bank",
        "embedding_appearance",
        "semantic_decoder",
        "offset_residual_proj",
    ]
    for module_name in module_names:
        module = getattr(gaussians, module_name, None)
        if module is None:
            continue
        for param_name, param in module.named_parameters():
            yield f"{module_name}.{param_name}", param


def configure_semantic_baseline_trainables(gaussians, logger=None):
    trainable_names = []
    frozen_names = []
    use_anchor_offset = bool(getattr(gaussians, "use_anchor_offset_semantic", False))
    residual_enabled = (
        use_anchor_offset
        and int(getattr(gaussians, "semantic_residual_dim", 0)) > 0
    )
    decoder_trainable = bool(getattr(gaussians, "semantic_decoder_trainable", True))
    for name, param in _iter_gaussian_named_parameters(gaussians):
        if use_anchor_offset:
            is_semantic = (
                name == "_semantic_anchor"
                or (
                    residual_enabled
                    and (
                        name in {"_semantic_offset_residual", "_semantic_offset_gate"}
                        or name.startswith("offset_residual_proj.")
                    )
                )
                or (decoder_trainable and name.startswith("semantic_decoder."))
            )
        else:
            is_semantic = (
                name == "_semantic_feat"
                or (decoder_trainable and name.startswith("semantic_decoder."))
            )
        param.requires_grad_(is_semantic)
        if is_semantic:
            trainable_names.append((name, tuple(param.shape), int(param.numel())))
        else:
            frozen_names.append(name)

    if logger is not None:
        mode = "Anchor-Offset" if residual_enabled else "Anchor-only"
        logger.info(f"[Semantic Baseline] {mode}; frozen geometry/RGB parameters; trainable parameters are:")
        for name, shape, count in trainable_names:
            logger.info(f"  trainable {name}: shape={shape}, params={count}")
        logger.info(f"[Semantic Baseline] Frozen parameter tensors: {len(frozen_names)}")
    if not trainable_names:
        raise RuntimeError("[Semantic Baseline] No semantic parameters are trainable.")


def _grad_norm_for_parameters(parameters):
    sq_sum = 0.0
    found = False
    for param in parameters:
        if param is None or param.grad is None:
            continue
        grad = param.grad.detach()
        sq_sum += float(grad.pow(2).sum().item())
        found = True
    return sq_sum ** 0.5 if found else 0.0


def semantic_latent_grad_norm(gaussians):
    if getattr(gaussians, "use_anchor_offset_semantic", False):
        return _grad_norm_for_parameters([
            getattr(gaussians, "_semantic_anchor", None),
            getattr(gaussians, "_semantic_offset_residual", None),
            getattr(gaussians, "_semantic_offset_gate", None),
        ])
    return _grad_norm_for_parameters([getattr(gaussians, "_semantic_feat", None)])


def semantic_anchor_grad_norm(gaussians):
    if getattr(gaussians, "use_anchor_offset_semantic", False):
        return _grad_norm_for_parameters([getattr(gaussians, "_semantic_anchor", None)])
    return _grad_norm_for_parameters([getattr(gaussians, "_semantic_feat", None)])


def semantic_offset_residual_grad_norm(gaussians):
    if not getattr(gaussians, "use_anchor_offset_semantic", False):
        return 0.0
    return _grad_norm_for_parameters([getattr(gaussians, "_semantic_offset_residual", None)])


def semantic_offset_gate_grad_norm(gaussians):
    if not getattr(gaussians, "use_anchor_offset_semantic", False):
        return 0.0
    return _grad_norm_for_parameters([getattr(gaussians, "_semantic_offset_gate", None)])


def semantic_decoder_grad_norm(gaussians):
    decoder = getattr(gaussians, "semantic_decoder", None)
    if decoder is None:
        return 0.0
    return _grad_norm_for_parameters(decoder.parameters())


def _resolve_instance_teacher_root(dataset):
    teacher_value = str(getattr(dataset, "instance_geometry_teacher", "") or "").strip()
    if not teacher_value:
        teacher_value = str(getattr(dataset, "region_teachers", "") or "").strip()
    teacher_root = Path(teacher_value)
    if not teacher_root.is_absolute():
        teacher_root = Path(dataset.source_path) / teacher_root
    return teacher_root.expanduser().resolve()


def _instance_geometry_residual_mode(dataset):
    mode = str(
        getattr(dataset, "instance_geometry_residual_mode", "sparse") or "sparse"
    ).strip().lower()
    if mode != "sparse":
        raise RuntimeError(
            "This release supports only geometry-driven sparse residuals, "
            f"got {mode!r}."
        )
    return mode


def configure_instance_geometry_semantics(dataset, opt, gaussians, logger=None, reuse_existing=False):
    if not bool(getattr(dataset, "use_instance_geometry_semantic", False)):
        return None
    if not bool(getattr(dataset, "use_anchor_offset_semantic", False)):
        raise RuntimeError("--use_instance_geometry_semantic requires --use_anchor_offset_semantic.")
    residual_mode = _instance_geometry_residual_mode(dataset)

    teacher_root = _resolve_instance_teacher_root(dataset)
    if not is_instance_level_teacher_dir(teacher_root):
        raise RuntimeError(
            "Instance geometry semantics require an instance-level compressed geometry-reference teacher. "
            "Check --instance_geometry_teacher or --region_teachers: "
            f"{teacher_root}"
        )
    store = get_instance_teacher_store(
        teacher_root,
        map_mode=getattr(dataset, "instance_teacher_map_mode", "auto"),
    )
    if int(store.feature_dim) != int(gaussians.semantic_latent_dim):
        raise RuntimeError(
            "Direct instance geometry supervision requires teacher features and semantic anchors in the same "
            f"latent space, got teacher={store.feature_dim}, anchor={gaussians.semantic_latent_dim}."
        )

    has_saved_assignment = bool(
        gaussians._semantic_anchor_instance_index.numel() == gaussians.get_anchor.shape[0]
        and torch.any(gaussians._semantic_anchor_instance_index >= 0).item()
    )
    if reuse_existing:
        if not has_saved_assignment:
            raise RuntimeError(
                "The checkpoint does not contain an instance-geometry assignment. Start a new semantic run from "
                "--start_model_path, or resume a checkpoint produced by the geometry-gated implementation."
            )
        assignment_stats = {
            "num_anchors": int(gaussians.get_anchor.shape[0]),
            "num_residual_anchors": int(gaussians._semantic_offset_residual.shape[0]),
            "residual_anchor_ratio": float(
                gaussians._semantic_offset_residual.shape[0] / max(gaussians.get_anchor.shape[0], 1)
            ),
            "residual_allocation_mode": (
                "none"
                if int(gaussians.semantic_residual_dim) == 0
                else residual_mode
            ),
            "reused_from_checkpoint": True,
        }
    else:
        binding = prepare_instance_geometry_binding(
            teacher_root=teacher_root,
            teacher_instance_ids=store.instance_ids,
            source_root=dataset.source_path,
            binding_override=getattr(dataset, "instance_geometry_binding", ""),
            auto_prepare=bool(getattr(dataset, "instance_geometry_auto_prepare", True)),
            object_dict_override=getattr(dataset, "instance_geometry_object_dict", ""),
            point_cloud_override=getattr(dataset, "instance_geometry_point_cloud", ""),
            log_callback=lambda message: _log_info(logger, message),
        )
        max_distance = float(getattr(dataset, "instance_geometry_max_distance", 0.08))
        if max_distance <= 0.0:
            raise RuntimeError("--instance_geometry_max_distance must be positive.")
        with torch.no_grad():
            anchor = gaussians.get_anchor.detach()
            offset_scale = gaussians.get_scaling.detach()[:, None, :3]
            child = anchor[:, None, :] + gaussians._offset.detach() * offset_scale
            assignment = classify_anchor_geometry(
                anchor.cpu().numpy(),
                child.cpu().numpy(),
                binding,
                num_instances=store.num_instances,
                max_distance=max_distance,
                interior_purity=float(getattr(dataset, "instance_geometry_interior_purity", 0.90)),
                min_support_fraction=float(
                    getattr(dataset, "instance_geometry_min_support_fraction", 0.50)
                ),
                min_query_confidence=float(
                    getattr(dataset, "instance_geometry_min_query_confidence", 0.50)
                ),
                unknown_uses_residual=bool(
                    getattr(dataset, "instance_geometry_unknown_residual", True)
                ),
            )
        interior_count = int(np.sum(assignment.states == GEOMETRY_STATE_INTERIOR))
        if interior_count == 0:
            raise RuntimeError(
                "Instance geometry assignment produced no interior anchors. Check that the binding point cloud and "
                "geometry 3DGS use the same coordinate system, then adjust --instance_geometry_max_distance or "
                "the purity/support thresholds."
            )
        geometry_residual_indices = assignment.residual_anchor_indices
        if int(gaussians.semantic_residual_dim) == 0:
            effective_residual_indices = np.empty(0, dtype=np.int64)
            model_residual_indices = effective_residual_indices
            residual_allocation_mode = "none"
        else:
            effective_residual_indices = geometry_residual_indices
            model_residual_indices = effective_residual_indices
            residual_allocation_mode = "sparse"
        gaussians.configure_geometry_gated_semantics(
            model_residual_indices,
            assignment.instance_indices,
            assignment.states,
            assignment.confidence,
        )
        if bool(getattr(dataset, "instance_geometry_initialize_interior", False)):
            interior = gaussians._semantic_anchor_geometry_state == GEOMETRY_STATE_INTERIOR
            target_ids = gaussians._semantic_anchor_instance_index[interior]
            targets = store.instance_features.to(gaussians.get_model_device())[target_ids]
            with torch.no_grad():
                gaussians._semantic_anchor[interior] = targets
        assignment_stats = dict(assignment.stats)
        assignment_stats["geometry_gated_num_residual_anchors"] = int(
            geometry_residual_indices.shape[0]
        )
        assignment_stats["geometry_gated_residual_anchor_ratio"] = float(
            geometry_residual_indices.shape[0] / max(anchor.shape[0], 1)
        )
        assignment_stats["num_residual_anchors"] = int(effective_residual_indices.shape[0])
        assignment_stats["residual_anchor_ratio"] = float(
            effective_residual_indices.shape[0] / max(anchor.shape[0], 1)
        )
        assignment_stats["residual_allocation_mode"] = residual_allocation_mode
        assignment_stats["binding_dir"] = str(binding.root)
        assignment_stats["geometry_reference_teacher"] = str(teacher_root)
        assignment_stats["render_teacher"] = str(
            Path(str(getattr(dataset, "region_teachers", "") or "")).expanduser()
        )
        assignment_stats["binding_points"] = int(binding.points_xyz.shape[0])
        assignment_stats["reused_from_checkpoint"] = False
        assignment_dir = Path(dataset.model_path)
        assignment_dir.mkdir(parents=True, exist_ok=True)
        assignment_path = assignment_dir / "instance_geometry_assignment.npz"
        np.savez_compressed(
            assignment_path,
            instance_indices=assignment.instance_indices.astype(np.int32),
            states=assignment.states,
            confidence=assignment.confidence.astype(np.float16),
            support_fraction=assignment.support_fraction.astype(np.float16),
            residual_anchor_indices=effective_residual_indices.astype(np.int32),
        )
        assignment_stats["assignment_path"] = str(assignment_path)
        with (assignment_dir / "instance_geometry_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(assignment_stats, handle, indent=2, ensure_ascii=False)

    if logger is not None:
        logger.info(f"[Instance Geometry] assignment={json.dumps(assignment_stats, ensure_ascii=False)}")
    return {
        "instance_features": store.instance_features.to(gaussians.get_model_device()),
        "stats": assignment_stats,
    }


def compute_instance_geometry_anchor_loss(gaussians, context):
    if context is None:
        return None
    states = gaussians._semantic_anchor_geometry_state
    instance_indices = gaussians._semantic_anchor_instance_index
    confidence = gaussians._semantic_anchor_geometry_confidence
    valid = (states == GEOMETRY_STATE_INTERIOR) & (instance_indices >= 0)
    if not bool(valid.any().item()):
        return None
    anchors = F.normalize(gaussians._semantic_anchor[valid], dim=-1, eps=1e-6)
    targets = context["instance_features"][instance_indices[valid]]
    weights = confidence[valid].detach().float().clamp_min(1e-4)
    cosine = (anchors * targets).sum(dim=-1).clamp(-1.0, 1.0)
    loss = (weights * (1.0 - cosine)).sum() / weights.sum().clamp_min(1e-6)
    return {
        "loss": loss,
        "interior_anchors": int(valid.sum().item()),
        "confidence_mean": float(weights.mean().item()),
        "cosine_mean": float(cosine.detach().mean().item()),
    }


def log_warning(logger, message):
    if logger is not None:
        logger.warning(message)
    else:
        print(message)


def semantic_loss_requested(dataset, opt):
    return bool(
        getattr(dataset, "use_semantic_features", False)
        and getattr(opt, "lambda_semantic", 0.0) > 0.0
    )


def _semantic_weights_requested(opt):
    return bool(getattr(opt, "lambda_semantic", 0.0) > 0.0)


def _resolve_depth_dir(dataset):
    depth_dir = getattr(dataset, "depths", "") or ""
    if not depth_dir:
        return ""
    if os.path.isabs(depth_dir):
        return depth_dir
    return os.path.join(dataset.source_path, depth_dir)


def configure_depth_loading_mode(dataset, opt, logger=None):
    valid_modes = {"auto", "off", "load", "loss", "semantic"}
    depth_mode = str(getattr(dataset, "depth_mode", "auto") or "auto").lower()
    if depth_mode not in valid_modes:
        raise RuntimeError(
            f"Unsupported --depth_mode '{depth_mode}'. Expected one of {sorted(valid_modes)}."
        )
    dataset.depth_mode = depth_mode

    semantic_3dgs_enabled = bool(getattr(dataset, "semantic_3dgs", False))
    semantic_weights_requested = _semantic_weights_requested(opt)
    if semantic_3dgs_enabled:
        dataset.use_semantic_features = True
    elif getattr(dataset, "use_semantic_features", False) and semantic_weights_requested:
        semantic_3dgs_enabled = True
        dataset.semantic_3dgs = True
        log_warning(
            logger,
            "[Mode] --use_semantic_features with --lambda_semantic is treated as semantic 3DGS. "
            "Prefer --semantic_3dgs for new runs.",
        )

    if semantic_weights_requested and not getattr(dataset, "use_semantic_features", False):
        raise RuntimeError(
            "[Semantic Training] Semantic loss weights were enabled, but --use_semantic_features is disabled. "
            "Enable --semantic_3dgs/--use_semantic_features or set --lambda_semantic 0."
        )
    if getattr(dataset, "use_semantic_features", False) and not semantic_weights_requested:
        log_warning(
            logger,
            "[Semantic Training] --use_semantic_features is enabled, but --lambda_semantic is 0. "
            "Semantic latent/decoder parameters will be created but receive no direct semantic supervision.",
        )

    depth_path_requested = bool(getattr(dataset, "depths", "") or "")
    depth_dir = _resolve_depth_dir(dataset)

    if depth_path_requested and not os.path.isdir(depth_dir):
        raise RuntimeError(
            f"[Depth] A depth directory was requested, but it does not exist: {depth_dir}"
        )
    if opt.lambda_depth < 0.0:
        if depth_path_requested and depth_mode == "loss":
            opt.lambda_depth = AUTO_DEPTH_LAMBDA
            log_warning(
                logger,
                f"[Depth] --lambda_depth is auto; using {AUTO_DEPTH_LAMBDA} because --depth_mode loss was requested.",
            )
        elif semantic_3dgs_enabled:
            opt.lambda_depth = 0.0
            log_warning(
                logger,
                "[Depth] --lambda_depth is auto; semantic training uses alpha-rendered distillation only. "
                "Set --depth_mode loss and --lambda_depth > 0 only for geometry-stage depth supervision.",
            )
        elif depth_path_requested and depth_mode != "load":
            opt.lambda_depth = AUTO_DEPTH_LAMBDA
            log_warning(
                logger,
                f"[Depth] --lambda_depth is auto; using {AUTO_DEPTH_LAMBDA} because --depths is provided.",
            )
        else:
            opt.lambda_depth = 0.0

    if not depth_path_requested and opt.lambda_depth > 0.0:
        raise RuntimeError(
            f"[Depth] --lambda_depth={opt.lambda_depth} was requested, but --depths is empty. "
            "Provide --depths <depth_dir> or set --lambda_depth 0."
        )

    if depth_mode == "load" and opt.lambda_depth > 0.0:
        log_warning(
            logger,
            "[Depth] --depth_mode load only loads depth maps; disabling ordinary depth loss "
            f"that was requested by lambda_depth={opt.lambda_depth}.",
        )
        opt.lambda_depth = 0.0

    depth_loss_enabled = opt.lambda_depth > 0.0 and depth_path_requested and depth_mode != "load"
    semantic_training_enabled = semantic_loss_requested(dataset, opt)
    if depth_mode == "off" and depth_loss_enabled:
        raise RuntimeError(
            "--depth_mode off conflicts with enabled depth-dependent training (--lambda_depth > 0)."
        )
    if depth_mode == "loss" and not depth_loss_enabled:
        raise RuntimeError("--depth_mode loss requires --depths and --lambda_depth > 0.")
    if depth_mode == "semantic":
        log_warning(
            logger,
            "[Depth] --depth_mode semantic is deprecated for the alpha-blending baseline; depth will not be loaded for semantic loss.",
        )

    should_load_depth = depth_mode in {"load", "loss", "semantic"}
    if depth_mode == "auto":
        should_load_depth = depth_path_requested and depth_loss_enabled
    elif depth_mode == "semantic":
        should_load_depth = False

    if should_load_depth:
        if not depth_path_requested:
            raise RuntimeError(
                "[Depth] Depth-dependent training is enabled, but --depths is empty. "
                "Provide a depth directory or disable depth-dependent losses."
            )
    else:
        if depth_path_requested:
            log_warning(
                logger,
                "[Depth] Skipping depth map loading because no active depth-dependent loss needs it. "
                "Use --depth_mode load if you intentionally want to load depth maps without a loss.",
            )
        dataset.depths = ""

    if logger is not None:
        if semantic_training_enabled:
            mode_name = "Semantic Scaffold-GS"
        elif depth_loss_enabled:
            mode_name = "Depth-supervised Scaffold-GS"
        else:
            mode_name = "RGB-only Scaffold-GS"
        logger.info(
            f"[Mode] {mode_name}; depth_load={should_load_depth}; "
            f"lambda_depth={opt.lambda_depth}; semantic_3dgs={bool(getattr(dataset, 'semantic_3dgs', False))}"
        )

    return semantic_training_enabled, should_load_depth


def count_valid_depth_pixels(camera):
    depth_mask = getattr(camera, "depth_mask", None)
    depth = getattr(camera, "depth", None)
    if depth is None or depth_mask is None:
        return 0
    valid = depth_mask > 0.5
    valid &= torch.isfinite(depth)
    valid &= depth > 0
    return int(valid.sum().item())


def save_training_checkpoint(scene, gaussians, iteration, logger=None, prefix="chkpnt", save_point_cloud=False):
    checkpoint_path = os.path.join(scene.model_path, f"{prefix}{iteration}.pth")
    torch.save((gaussians.capture(), iteration), checkpoint_path)
    if save_point_cloud:
        scene.save(iteration)
    if logger is not None:
        logger.info(f"[ITER {iteration}] Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def add_optional_loss(base_loss, term):
    if term is None:
        return base_loss
    return term if base_loss is None else base_loss + term


def configure_semantic_finetune_only(opt, logger=None):
    if not getattr(opt, "semantic_finetune_only", False):
        return
    if logger is not None:
        logger.info(
            "[Semantic Finetune] Freezing geometry/RGB optimizer groups; only semantic parameters "
            "and semantic_decoder will update. Disabling ordinary RGB/depth optimization and densification."
        )
    opt.lambda_depth = 0.0
    opt.update_until = 0
    opt.position_lr_init = 0.0
    opt.position_lr_final = 0.0
    opt.offset_lr_init = 0.0
    opt.offset_lr_final = 0.0
    opt.feature_lr = 0.0
    opt.opacity_lr = 0.0
    opt.scaling_lr = 0.0
    opt.rotation_lr = 0.0
    opt.mlp_opacity_lr_init = 0.0
    opt.mlp_opacity_lr_final = 0.0
    opt.mlp_cov_lr_init = 0.0
    opt.mlp_cov_lr_final = 0.0
    opt.mlp_color_lr_init = 0.0
    opt.mlp_color_lr_final = 0.0
    opt.mlp_featurebank_lr_init = 0.0
    opt.mlp_featurebank_lr_final = 0.0
    opt.appearance_lr_init = 0.0
    opt.appearance_lr_final = 0.0


def training(
    dataset,
    opt,
    pipe,
    dataset_name,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    wandb=None,
    logger=None,
    ply_path=None,
    disable_gui=False,
    start_model_path=None,
    start_point_cloud_iteration=None,
):
    first_iter = 0
    REGION_TEACHER_CACHE.clear()
    configure_semantic_finetune_only(opt, logger=logger)
    semantic_training_requested, _ = configure_depth_loading_mode(dataset, opt, logger=logger)
    if semantic_training_requested and not getattr(opt, "semantic_finetune_only", False):
        raise RuntimeError(
            "[Semantic Training] Semantic Gaussian training is now a second-stage-only workflow. "
            "First train geometry/RGB 3DGS, then rerun with --semantic_3dgs --semantic_finetune_only "
            "and initialize from --start_model_path plus --start_point_cloud_iteration."
        )
    start_from_point_cloud = start_point_cloud_iteration is not None and int(start_point_cloud_iteration) > 0
    if getattr(opt, "semantic_finetune_only", False) and start_from_point_cloud and not start_model_path:
        raise RuntimeError(
            "--semantic_finetune_only now keeps geometry and semantic outputs in separate directories. "
            "Provide --start_model_path for the frozen geometry model, and use --model_path for the semantic output."
        )
    if getattr(opt, "semantic_finetune_only", False) and not checkpoint and not start_from_point_cloud:
        raise RuntimeError(
            "--semantic_finetune_only is a second-stage mode. Provide --start_checkpoint, or provide "
            "--start_model_path plus --start_point_cloud_iteration to initialize from a saved point_cloud."
        )
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                              dataset.use_semantic_features, dataset.semantic_latent_dim, dataset.semantic_feature_dim,
                              semantic_teacher_dim=getattr(dataset, "semantic_teacher_dim", 0),
                              semantic_codec_path=getattr(dataset, "semantic_codec_path", ""),
                              sh_degree=dataset.sh_degree,
                              use_anchor_offset_semantic=getattr(dataset, "use_anchor_offset_semantic", False),
                              semantic_residual_dim=getattr(dataset, "semantic_residual_dim", 4),
                              semantic_gate_init=getattr(dataset, "semantic_gate_init", -4.0),
                              semantic_storage_dtype=getattr(dataset, "semantic_storage_dtype", "float16"))
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    train_cameras = scene.getTrainCameras()
    depth_ready_train_cameras = [
        cam for cam in train_cameras
        if getattr(cam, "depth", None) is not None and getattr(cam, "depth_mask", None) is not None
    ]
    missing_depth_train_cameras = [
        cam for cam in train_cameras
        if getattr(cam, "depth", None) is None or getattr(cam, "depth_mask", None) is None
    ]
    depth_train_cameras = len(depth_ready_train_cameras)
    region_teacher_train_cameras = sum(1 for cam in train_cameras if getattr(cam, "region_teacher_path", None) is not None)
    missing_region_teacher_train_cameras = [
        cam for cam in train_cameras
        if getattr(cam, "region_teacher_path", None) is None
    ]
    semantic_training_enabled = semantic_training_requested
    if logger is not None:
        logger.info(f"Depth-enabled train cameras: {depth_train_cameras}/{len(train_cameras)}")
        if dataset.use_semantic_features:
            logger.info(f"Region-teacher train cameras: {region_teacher_train_cameras}/{len(train_cameras)}")
        if opt.lambda_depth > 0.0:
            depth_duration = int(getattr(opt, "depth_duration_iters", 0))
            depth_end_desc = (
                "train_end"
                if depth_duration <= 0
                else str(int(opt.depth_start_iter) + depth_duration - 1)
            )
            logger.info(
                "[Depth Training] schedule: "
                f"start_iter={opt.depth_start_iter}, duration_iters={depth_duration}, "
                f"end_iter={depth_end_desc}, ramp_iters={opt.depth_ramp_iters}, "
                f"target_lambda={opt.lambda_depth}, alpha_min={getattr(opt, 'depth_alpha_min', 0.0)}, "
                f"huber_delta={getattr(opt, 'depth_residual_clip', 0.0)}"
            )
    if opt.lambda_depth > 0.0 and depth_train_cameras == 0:
        raise RuntimeError(
            "[Depth Training] --lambda_depth is enabled, but no training camera loaded valid depth/depth_mask. "
            "Please check --depths, depth filenames, image/depth stem matching, and depth format before training."
        )
    if opt.lambda_depth > 0.0 and missing_depth_train_cameras:
        missing_examples = ", ".join(cam.image_name for cam in missing_depth_train_cameras[:5])
        message = (
            "[Depth Training] Some training cameras do not have depth/depth_mask: "
            f"{len(missing_depth_train_cameras)}/{len(train_cameras)} missing. Examples: {missing_examples}"
        )
        if getattr(dataset, "depth_allow_missing", False):
            if logger is not None:
                logger.warning(message + " These views will skip depth loss because --depth_allow_missing is enabled.")
        else:
            raise RuntimeError(message + " Add --depth_allow_missing if this is intentional.")
    if opt.lambda_depth > 0.0 and depth_ready_train_cameras:
        weak_depth_cameras = [
            cam for cam in depth_ready_train_cameras
            if count_valid_depth_pixels(cam) < int(getattr(opt, "depth_min_valid_pixels", 512))
        ]
        if weak_depth_cameras:
            weak_examples = ", ".join(
                f"{cam.image_name}({count_valid_depth_pixels(cam)})"
                for cam in weak_depth_cameras[:5]
            )
            raise RuntimeError(
                "[Depth Training] Some training cameras have too few valid depth pixels for depth loss. "
                f"Required at least {int(getattr(opt, 'depth_min_valid_pixels', 512))}; "
                f"found {len(weak_depth_cameras)}/{len(depth_ready_train_cameras)} weak maps. "
                f"Examples: {weak_examples}"
            )
    if opt.lambda_depth > 0.0 and depth_ready_train_cameras and logger is not None:
        example_depth_cam = depth_ready_train_cameras[0]
        native_depth_projection = (
            getattr(dataset, "depth_use_native_resolution", False)
            and getattr(example_depth_cam, "depth_full_proj_transform", None) is not None
            and (
                int(getattr(example_depth_cam, "depth_height", example_depth_cam.image_height)) != int(example_depth_cam.image_height)
                or int(getattr(example_depth_cam, "depth_width", example_depth_cam.image_width)) != int(example_depth_cam.image_width)
            )
        )
        logger.info(
            "[Depth Training] Example camera resolution: "
            f"rgb={example_depth_cam.image_width}x{example_depth_cam.image_height}, "
            f"depth={example_depth_cam.depth_width}x{example_depth_cam.depth_height}, "
            f"native_depth_projection={native_depth_projection}"
        )
    if semantic_training_enabled and missing_region_teacher_train_cameras:
        missing_examples = ", ".join(cam.image_name for cam in missing_region_teacher_train_cameras[:5])
        raise RuntimeError(
            "[Semantic Training] Semantic Gaussian training is enabled, but not all training cameras have "
            "a validated semantic teacher. "
            f"Found {len(missing_region_teacher_train_cameras)}/{len(train_cameras)} training cameras missing region teachers. "
            "All training cameras must provide teacher maps before semantic distillation can start. "
            f"Example cameras without teachers: {missing_examples}. "
            "Check --region_teachers and the compressed teacher frame data."
        )
    gaussians.training_setup(opt)
    instance_geometry_context = None
    if checkpoint:
        try:
            model_params, first_iter = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            model_params, first_iter = torch.load(checkpoint, map_location="cpu")
        gaussians.restore(model_params, opt)
        gaussians.train()
        instance_geometry_context = configure_instance_geometry_semantics(
            dataset, opt, gaussians, logger=logger, reuse_existing=True
        )
        if getattr(opt, "semantic_finetune_only", False) and first_iter >= opt.iterations:
            raise RuntimeError(
                "--semantic_finetune_only loaded a checkpoint at iteration "
                f"{first_iter}, but --iterations is {opt.iterations}. "
                "Set --iterations larger than the checkpoint iteration, e.g. 40000 for "
                "a 30000-iteration checkpoint plus 10000 semantic finetune steps."
            )
    elif start_from_point_cloud:
        source_model_path = start_model_path or dataset.model_path
        source_model_path = os.path.abspath(source_model_path)
        source_iteration = int(start_point_cloud_iteration)
        source_point_cloud_dir = os.path.join(source_model_path, "point_cloud", f"iteration_{source_iteration}")
        source_ply = os.path.join(source_point_cloud_dir, "point_cloud.ply")
        if not os.path.exists(source_ply):
            raise RuntimeError(f"Saved point cloud not found for second-stage initialization: {source_ply}")
        gaussians.load_ply_sparse_gaussian(source_ply)
        gaussians.load_mlp_checkpoints(source_point_cloud_dir)
        instance_geometry_context = configure_instance_geometry_semantics(
            dataset, opt, gaussians, logger=logger, reuse_existing=False
        )
        gaussians.training_setup(opt)
        gaussians.train()
        first_iter = source_iteration
        if logger is not None:
            logger.info(
                "[Semantic Finetune] Initialized from saved point cloud "
                f"{source_ply}; continuing from iteration {first_iter}."
            )
        if getattr(opt, "semantic_finetune_only", False) and first_iter >= opt.iterations:
            raise RuntimeError(
                "--semantic_finetune_only loaded point_cloud iteration "
                f"{first_iter}, but --iterations is {opt.iterations}. "
                "Set --iterations larger than the source iteration."
            )
    if semantic_training_enabled and getattr(opt, "semantic_finetune_only", False):
        configure_semantic_baseline_trainables(gaussians, logger=logger)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=gaussians.get_anchor.device)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Train",
        dynamic_ncols=True,
    )
    first_iter += 1
    last_completed_iteration = first_iter - 1
    try:
        for iteration in range(first_iter, opt.iterations + 1):
            # network gui not available in scaffold-gs yet
            if not disable_gui:
                if network_gui.conn == None:
                    network_gui.try_connect()
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, _remote_request = network_gui.receive()
                        if custom_cam != None:
                            net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        network_gui.send(net_image_bytes, dataset.source_path)
                        if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                            break
                    except Exception as e:
                        network_gui.conn = None

            iter_start.record()

            lr_iteration = iteration
            if semantic_training_enabled and getattr(opt, "semantic_finetune_only", False):
                lr_iteration = iteration - first_iter + 1
            gaussians.update_learning_rate(lr_iteration)

            
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            region_teacher_available = getattr(viewpoint_cam, "region_teacher_path", None) is not None
            view_has_depth = (
                getattr(viewpoint_cam, "depth", None) is not None
                and getattr(viewpoint_cam, "depth_mask", None) is not None
            )
            semantic_baseline_enabled = (
                semantic_training_enabled
                and region_teacher_available
                and getattr(opt, "lambda_semantic", 0.0) > 0.0
            )
            prepared_base = prepare_visible_gaussians(
                viewpoint_cam,
                gaussians,
                visible_mask=voxel_visible_mask,
                render_semantic_features=False,
            )
            depth_weight = get_depth_loss_weight(opt, iteration)
            use_native_depth_camera = (
                depth_weight > 0.0
                and view_has_depth
                and getattr(dataset, "depth_use_native_resolution", False)
                and getattr(viewpoint_cam, "depth_full_proj_transform", None) is not None
                and (
                    int(getattr(viewpoint_cam, "depth_height", viewpoint_cam.image_height)) != int(viewpoint_cam.image_height)
                    or int(getattr(viewpoint_cam, "depth_width", viewpoint_cam.image_width)) != int(viewpoint_cam.image_width)
                )
            )
            need_inline_depth_render = depth_weight > 0.0 and view_has_depth and not use_native_depth_camera
            render_pkg = rasterize_precomputed_gaussians(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                prepared_base,
                retain_grad=retain_grad,
                render_depth=need_inline_depth_render,
                render_alpha=need_inline_depth_render,
            )
            
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            offset_selection_mask = render_pkg.get("selection_mask", None)
            scaling = render_pkg["scaling"]
            opacity = render_pkg.get("neural_opacity", render_pkg.get("opacity", None))
            rendered_depth = render_pkg.get("depth")
            rendered_alpha = render_pkg.get("alpha")

            gt_image = viewpoint_cam.original_image.to(image.device)
            Ll1 = l1_loss(image, gt_image)

            ssim_loss = (1.0 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
            loss = None if getattr(opt, "semantic_finetune_only", False) else rgb_loss
            depth_loss = None
            depth_front_loss = None
            depth_back_loss = None
            depth_surface_loss = None
            if depth_weight > 0.0 and view_has_depth:
                depth_render_pkg = rasterize_precomputed_gaussians(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    background,
                    prepared_base,
                    retain_grad=False,
                    render_depth=True,
                    render_alpha=True,
                    use_depth_camera=use_native_depth_camera,
                ) if use_native_depth_camera else None
                if depth_render_pkg is not None:
                    rendered_depth = depth_render_pkg.get("depth")
                    rendered_alpha = depth_render_pkg.get("alpha")
                depth_loss = rendered_depth_loss(viewpoint_cam, rendered_depth, rendered_alpha, opt)
                if depth_loss is not None:
                    loss = add_optional_loss(loss, depth_weight * depth_loss)
                depth_front_loss = front_floater_opacity_loss(viewpoint_cam, rendered_depth, rendered_alpha, opt)
                if depth_front_loss is not None:
                    loss = add_optional_loss(loss, depth_weight * opt.depth_front_weight * depth_front_loss)
                depth_back_loss = back_floater_opacity_loss(viewpoint_cam, rendered_depth, rendered_alpha, opt)
                if depth_back_loss is not None:
                    loss = add_optional_loss(loss, depth_weight * opt.depth_back_weight * depth_back_loss)
                depth_surface_loss = gaussian_surface_depth_loss(
                    viewpoint_cam,
                    prepared_base,
                    visibility_filter,
                    opt,
                    use_depth_camera=use_native_depth_camera,
                    rendered_depth=rendered_depth,
                    rendered_alpha=rendered_alpha,
                )
                if depth_surface_loss is not None:
                    loss = add_optional_loss(loss, depth_weight * opt.depth_surface_weight * depth_surface_loss)
            semantic_loss = None
            semantic_gate_loss = None
            semantic_residual_loss = None
            instance_geometry_outputs = None
            semantic_offset_stats = {}
            semantic_debug = None
            semantic_skip_reason = None
            prepared_semantic = None
            if semantic_baseline_enabled:
                prepared_semantic = prepare_visible_gaussians(
                    viewpoint_cam,
                    gaussians,
                    visible_mask=voxel_visible_mask,
                    render_semantic_features=True,
                )

            if semantic_baseline_enabled:
                region_teacher = load_region_teacher(
                    viewpoint_cam,
                    expected_dim=gaussians.semantic_teacher_dim,
                    target_hw=tuple(image.shape[-2:]),
                    device=background.device,
                )
                if region_teacher is not None and prepared_semantic is not None:
                    semantic_outputs = compute_alpha_blended_semantic_loss(
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        background,
                        prepared_semantic,
                        region_teacher,
                        opt,
                    )
                    if semantic_outputs is not None:
                        semantic_debug = semantic_outputs
                        semantic_skip_reason = semantic_outputs.get("skip_reason", None)
                    if semantic_outputs is not None and semantic_outputs["loss"] is not None:
                        semantic_loss = semantic_outputs["loss"]
                        semantic_skip_reason = None
                        loss = add_optional_loss(loss, opt.lambda_semantic * semantic_loss)
                        if getattr(dataset, "use_anchor_offset_semantic", False):
                            semantic_gate_loss, semantic_residual_loss, semantic_offset_stats = gaussians.semantic_offset_regularization()
                            if semantic_gate_loss is not None:
                                loss = add_optional_loss(loss, getattr(opt, "lambda_semantic_gate", 1e-3) * semantic_gate_loss)
                            if semantic_residual_loss is not None:
                                loss = add_optional_loss(loss, getattr(opt, "lambda_semantic_residual", 1e-4) * semantic_residual_loss)
                            semantic_debug.update(semantic_offset_stats)
                            semantic_debug["use_anchor_offset_semantic"] = True
                            semantic_debug["L_gate"] = (
                                float(semantic_gate_loss.detach().item()) if semantic_gate_loss is not None else 0.0
                            )
                            semantic_debug["L_residual"] = (
                                float(semantic_residual_loss.detach().item()) if semantic_residual_loss is not None else 0.0
                            )
                elif region_teacher is None:
                    semantic_skip_reason = "missing_region_teacher"
                elif prepared_semantic is None:
                    semantic_skip_reason = "missing_prepared_semantic"

            if instance_geometry_context is not None and float(getattr(opt, "lambda_instance_geometry", 0.0)) > 0.0:
                instance_geometry_outputs = compute_instance_geometry_anchor_loss(
                    gaussians, instance_geometry_context
                )
                if instance_geometry_outputs is not None:
                    loss = add_optional_loss(
                        loss,
                        float(getattr(opt, "lambda_instance_geometry", 0.1))
                        * instance_geometry_outputs["loss"],
                    )
                    if semantic_debug is None:
                        semantic_debug = {}
                    semantic_debug.update(
                        {
                            "L_instance_geometry": float(
                                instance_geometry_outputs["loss"].detach().item()
                            ),
                            "instance_geometry_interior_anchors": instance_geometry_outputs[
                                "interior_anchors"
                            ],
                            "instance_geometry_confidence_mean": instance_geometry_outputs[
                                "confidence_mean"
                            ],
                            "instance_geometry_cosine_mean": instance_geometry_outputs["cosine_mean"],
                        }
                    )

            did_backward = loss is not None and torch.is_tensor(loss) and bool(loss.requires_grad)
            if did_backward:
                loss.backward()
            semantic_latent_grad = semantic_latent_grad_norm(gaussians) if semantic_training_enabled else 0.0
            semantic_anchor_grad = semantic_anchor_grad_norm(gaussians) if semantic_training_enabled else 0.0
            semantic_offset_residual_grad = semantic_offset_residual_grad_norm(gaussians) if semantic_training_enabled else 0.0
            semantic_offset_gate_grad = semantic_offset_gate_grad_norm(gaussians) if semantic_training_enabled else 0.0
            semantic_decoder_grad = semantic_decoder_grad_norm(gaussians) if semantic_training_enabled else 0.0
            
            iter_end.record()

            with torch.no_grad():
                loss_for_log = loss if loss is not None else torch.zeros((), device=image.device)
                # Progress bar
                ema_loss_for_log = 0.4 * loss_for_log.item() + 0.6 * ema_loss_for_log

                if iteration % 10 == 0:
                    progress_info = {"L": f"{ema_loss_for_log:.7f}"}
                    if opt.lambda_depth <= 0.0:
                        progress_info["D"] = "off"
                    elif depth_loss is not None:
                        progress_info["D"] = f"{depth_loss.item():.5f}"
                    else:
                        progress_info["D"] = "skip"
                    if semantic_training_enabled:
                        progress_info["S"] = f"{semantic_loss.item():.5f}" if semantic_loss is not None else "skip"
                        if instance_geometry_outputs is not None:
                            progress_info["G"] = f"{instance_geometry_outputs['loss'].item():.5f}"
                    progress_bar.set_postfix(progress_info)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                if logger is not None and iteration % 1000 == 0:
                    depth_log = "off"
                    if opt.lambda_depth > 0.0:
                        depth_log = f"{depth_loss.item():.6f}" if depth_loss is not None else "skip"
                    depth_front_log = "off"
                    if opt.lambda_depth > 0.0 and opt.depth_front_weight > 0.0:
                        depth_front_log = f"{depth_front_loss.item():.6f}" if depth_front_loss is not None else "skip"
                    depth_back_log = "off"
                    if opt.lambda_depth > 0.0 and opt.depth_back_weight > 0.0:
                        depth_back_log = f"{depth_back_loss.item():.6f}" if depth_back_loss is not None else "skip"
                    depth_surface_log = "off"
                    if opt.lambda_depth > 0.0 and opt.depth_surface_weight > 0.0:
                        depth_surface_log = f"{depth_surface_loss.item():.6f}" if depth_surface_loss is not None else "skip"
                    semantic_log = "off"
                    if dataset.use_semantic_features and getattr(opt, "lambda_semantic", 0.0) > 0.0:
                        if semantic_loss is not None:
                            semantic_log = f"{semantic_loss.item():.6f}"
                        elif semantic_skip_reason:
                            semantic_log = f"skip:{semantic_skip_reason}"
                        else:
                            semantic_log = "skip"
                    anchor_offset_log = ""
                    if getattr(dataset, "use_anchor_offset_semantic", False):
                        gate_log = f"{semantic_gate_loss.item():.6f}" if semantic_gate_loss is not None else "skip"
                        residual_log = f"{semantic_residual_loss.item():.6f}" if semantic_residual_loss is not None else "skip"
                        anchor_offset_log = f", L_gate={gate_log}, L_residual={residual_log}"
                    instance_geometry_log = ""
                    if instance_geometry_context is not None:
                        geometry_loss_log = (
                            f"{instance_geometry_outputs['loss'].item():.6f}"
                            if instance_geometry_outputs is not None
                            else "skip"
                        )
                        instance_geometry_log = f", L_instance_geometry={geometry_loss_log}"
                    logger.info(
                        f"[ITER {iteration}] loss={loss_for_log.item():.6f}, l1={Ll1.item():.6f}, "
                        f"ssim={ssim_loss.item():.6f}, depth={depth_log}, "
                        f"front={depth_front_log}, back={depth_back_log}, surface={depth_surface_log}, "
                        f"semantic={semantic_log}, semantic_latent_grad={semantic_latent_grad:.6e}, "
                        f"semantic_decoder_grad={semantic_decoder_grad:.6e}{anchor_offset_log}"
                        f"{instance_geometry_log}"
                    )

                semantic_debug_interval = int(getattr(opt, "semantic_debug_interval", 1000))
                semantic_debug_due = (
                    semantic_debug is not None
                    and logger is not None
                    and semantic_debug_interval > 0
                    and (iteration == first_iter or iteration % semantic_debug_interval == 0)
                )
                if semantic_debug_due:
                    common_debug = (
                        "[Semantic Debug] "
                        f"teacher_feature_shape={semantic_debug.get('teacher_feature_shape')}, "
                        f"rendered_latent_shape={semantic_debug.get('rendered_latent_shape')}, "
                        f"valid_pixels={semantic_debug.get('valid_pixels')}, "
                        f"semantic_weight_mean={semantic_debug.get('semantic_weight_mean')}, "
                        f"semantic_weight_source={semantic_debug.get('semantic_weight_source')}, "
                        f"residual_is_sparse={semantic_debug.get('semantic_residual_is_sparse')}, "
                        f"residual_anchor_ratio={semantic_debug.get('semantic_residual_anchor_ratio')}, "
                        f"gate_mean={semantic_debug.get('gate_mean')}, "
                        f"residual_norm={semantic_debug.get('residual_norm')}, "
                        f"L_gate={semantic_debug.get('L_gate')}, "
                        f"L_residual={semantic_debug.get('L_residual')}, "
                        f"L_instance_geometry={semantic_debug.get('L_instance_geometry')}, "
                        f"interior_anchors={semantic_debug.get('instance_geometry_interior_anchors')}, "
                        f"geometry_confidence_mean={semantic_debug.get('instance_geometry_confidence_mean')}, "
                        f"geometry_cosine_mean={semantic_debug.get('instance_geometry_cosine_mean')}, "
                        f"anchor_grad={semantic_anchor_grad:.6e}, "
                        f"residual_grad={semantic_offset_residual_grad:.6e}, "
                        f"gate_grad={semantic_offset_gate_grad:.6e}"
                    )
                    if semantic_loss is not None:
                        logger.info(f"{common_debug}, loss={semantic_loss.item():.6f}, total={loss_for_log.item():.6f}")
                    else:
                        logger.info(f"{common_debug}, skip_reason={semantic_debug.get('skip_reason', 'unknown')}")

                # Log and save
                training_report(tb_writer, dataset_name, iteration, Ll1, loss_for_log, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
                if (iteration in saving_iterations):
                    logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # densification
                if iteration < opt.update_until and iteration > opt.start_stat:
                    # add statis
                    densify_update_filter = visibility_filter
                    if depth_weight > 0.0 and view_has_depth:
                        densify_update_filter = depth_aware_densification_filter(
                            viewpoint_cam,
                            prepared_base,
                            visibility_filter,
                            opt,
                            use_depth_camera=use_native_depth_camera,
                            rendered_depth=rendered_depth,
                            rendered_alpha=rendered_alpha,
                        )
                    gaussians.training_statis(viewspace_point_tensor, opacity, densify_update_filter, offset_selection_mask, voxel_visible_mask)
                    
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()
                        
                # Optimizer step
                if iteration < opt.iterations and did_backward:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                saved_checkpoint_this_iter = False
                if (iteration in checkpoint_iterations):
                    logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                    save_training_checkpoint(scene, gaussians, iteration, logger=logger, prefix="chkpnt", save_point_cloud=False)
                    saved_checkpoint_this_iter = True
                if (
                    getattr(opt, "auto_checkpoint_interval", 0) > 0
                    and iteration < opt.iterations
                    and iteration % opt.auto_checkpoint_interval == 0
                    and not saved_checkpoint_this_iter
                ):
                    save_training_checkpoint(
                        scene,
                        gaussians,
                        iteration,
                        logger=logger,
                        prefix="autochkpnt",
                        save_point_cloud=getattr(opt, "auto_checkpoint_save_point_cloud", False),
                    )

            last_completed_iteration = iteration
    except KeyboardInterrupt:
        progress_bar.close()
        if getattr(opt, "auto_checkpoint_on_interrupt", False) and last_completed_iteration > 0:
            log_warning(logger, f"[Interrupt] Training interrupted. Saving recovery checkpoint at iteration {last_completed_iteration}.")
            save_training_checkpoint(
                scene,
                gaussians,
                last_completed_iteration,
                logger=logger,
                prefix="interrupt_chkpnt",
                save_point_cloud=getattr(opt, "auto_checkpoint_save_point_cloud", False),
            )
        raise

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    if getattr(args, "use_semantic_features", False) and getattr(args, "semantic_codec_path", ""):
        source_codec = Path(args.semantic_codec_path).resolve()
        target_codec = (Path(args.model_path) / "semantic_codec.pth").resolve()
        if source_codec != target_codec:
            shutil.copy2(source_codec, target_codec)
            source_metadata = source_codec.with_suffix(".json")
            if source_metadata.is_file():
                shutil.copy2(source_metadata, target_codec.with_suffix(".json"))
        args.semantic_codec_path = str(target_codec)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            *renderArgs,
                            visible_mask=voxel_visible_mask,
                            render_depth=False,
                            render_alpha=False,
                        )["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to(image.device), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            visible_mask=voxel_visible_mask,
            render_depth=False,
            render_alpha=False,
        )
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        dataset.depths = ""
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                              dataset.use_semantic_features, dataset.semantic_latent_dim, dataset.semantic_feature_dim,
                              semantic_teacher_dim=getattr(dataset, "semantic_teacher_dim", 0),
                              semantic_codec_path=getattr(dataset, "semantic_codec_path", ""),
                              sh_degree=dataset.sh_degree,
                              use_anchor_offset_semantic=getattr(dataset, "use_anchor_offset_semantic", False),
                              semantic_residual_dim=getattr(dataset, "semantic_residual_dim", 4),
                              semantic_gate_init=getattr(dataset, "semantic_gate_init", -4.0),
                              semantic_storage_dtype=getattr(dataset, "semantic_storage_dtype", "float16"))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=gaussians.get_anchor.device)
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = _compute_fps(t_train_list)
            if train_fps is not None:
                logger.info(f'Train FPS: \033[1;35m{train_fps:.5f}\033[0m')
                if wandb is not None:
                    wandb.log({"train_fps": train_fps})
            else:
                logger.info('Train FPS: unavailable (insufficient timing samples)')

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = _compute_fps(t_test_list)
            if test_fps is not None:
                logger.info(f'Test FPS: \033[1;35m{test_fps:.5f}\033[0m')
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps, 0)
                if wandb is not None:
                    wandb.log({"test_fps": test_fps})
            else:
                logger.info('Test FPS: unavailable (insufficient timing samples)')
    
    return visible_count


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_fn = get_lpips_fn(eval_device)
        render_names = sorted(os.listdir(renders_dir))
        per_view_count_path = method_dir / "per_view_count.json"
        visible_count_by_name = None
        if per_view_count_path.exists():
            with open(per_view_count_path, "r", encoding="utf-8") as fp:
                visible_count_by_name = json.load(fp)
        elif visible_count is not None:
            visible_count_values = torch.tensor(visible_count).tolist()
            visible_count_by_name = {name: vc for vc, name in zip(visible_count_values, render_names)}

        ssims = []
        psnrs = []
        lpipss = []
        image_names = []

        for image_name, render_tensor, gt_tensor in tqdm(
            read_image_pairs(renders_dir, gt_dir, eval_device),
            desc="Metric evaluation progress",
            total=len(render_names),
        ):
            image_names.append(image_name)
            ssims.append(ssim(render_tensor, gt_tensor))
            psnrs.append(psnr(render_tensor, gt_tensor))
            lpipss.append(lpips_fn(render_tensor, gt_tensor).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            if visible_count_by_name:
                tb_writer.add_scalar(
                    f'{dataset_name}/VISIBLE_NUMS',
                    torch.tensor(list(visible_count_by_name.values()), dtype=torch.float32).mean().item(),
                    0,
                )
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": (
                                                        {name: visible_count_by_name[name] for name in image_names if visible_count_by_name is not None and name in visible_count_by_name}
                                                    )})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger_name = f"scaffold_gs.{os.path.abspath(path)}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.propagate = False
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_model_path", type=str, default="")
    parser.add_argument("--start_point_cloud_iteration", type=int, default=0)
    parser.add_argument("--geometry_model_path", type=str, default="")
    parser.add_argument("--geometry_iterations", type=int, default=30000)
    parser.add_argument("--semantic_iterations", type=int, default=10000)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--disable_gui", action="store_true", default=False)
    parser.add_argument(
        "--backup_runtime_code",
        action="store_true",
        default=False,
        help="Copy the repository source tree into <model_path>/backup before training.",
    )
    args = parser.parse_args(sys.argv[1:])
    resolved_model_args = lp.extract(args)
    args.source_path = resolved_model_args.source_path
    args.semantic_teacher_dim = resolved_model_args.semantic_teacher_dim
    args.semantic_codec_path = resolved_model_args.semantic_codec_path
    if getattr(args, "semantic_3dgs", False):
        args.use_semantic_features = True
    if bool(getattr(args, "use_instance_geometry_semantic", False)):
        if not bool(getattr(args, "semantic_3dgs", False)):
            raise RuntimeError("--use_instance_geometry_semantic requires --semantic_3dgs.")
        if not bool(getattr(args, "use_anchor_offset_semantic", False)):
            raise RuntimeError("--use_instance_geometry_semantic requires --use_anchor_offset_semantic.")
        if float(getattr(args, "lambda_instance_geometry", 0.0)) < 0.0:
            raise RuntimeError("--lambda_instance_geometry must be non-negative.")
        _instance_geometry_residual_mode(args)
    if int(getattr(args, "semantic_residual_dim", 0)) < 0:
        raise RuntimeError("--semantic_residual_dim must be non-negative.")

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')
    if args.skip_final_render and not args.skip_final_eval and not args.eval_existing_renders:
        raise RuntimeError(
            "--skip_final_render was set while final RGB evaluation is still enabled. "
            "This could evaluate stale or missing render outputs. Add --skip_final_eval for pure training, "
            "or add --eval_existing_renders if you intentionally want to evaluate renders that already exist."
        )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logger.info(
        f"CUDA_VISIBLE_DEVICES={visible_devices if visible_devices else '<unset>'} "
        f"(source={CUDA_VISIBLE_DEVICES_SOURCE})"
    )
    if args.gpu != '-1':
        logger.info(f'using GPU argument {args.gpu}')

    

    if args.backup_runtime_code:
        try:
            saveRuntimeCode(os.path.join(args.model_path, "backup"))
        except Exception:
            logger.exception("save code failed")
    else:
        logger.info("Runtime code backup disabled. Pass --backup_runtime_code to enable it.")
        
    source_path_obj = Path(args.source_path)
    model_path_obj = Path(args.model_path)
    dataset = source_path_obj.name or source_path_obj.resolve().name
    exp_name = model_path_obj.parent.name or model_path_obj.name
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.disable_gui:
        logger.info("Network GUI disabled by --disable_gui.")
    else:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    final_args = args
    semantic_requested = bool(getattr(args, "semantic_3dgs", False))
    has_start_geometry = bool(getattr(args, "start_model_path", "")) and int(getattr(args, "start_point_cloud_iteration", 0)) > 0
    has_start_checkpoint = bool(getattr(args, "start_checkpoint", None))
    continue_existing_geometry = semantic_requested and (
        bool(getattr(args, "semantic_finetune_only", False))
        or has_start_geometry
        or has_start_checkpoint
    )

    if semantic_requested and args.warmup:
        raise RuntimeError("--warmup is not supported for the two-stage semantic workflow.")

    if semantic_requested:
        if has_start_geometry and os.path.abspath(args.start_model_path) == os.path.abspath(args.model_path):
            raise RuntimeError(
                "[Semantic Workflow] Geometry and semantic outputs must use separate directories. "
                "--start_model_path points to the frozen geometry model, while --model_path must point to "
                "a different semantic output directory."
            )
        if (
            bool(getattr(args, "semantic_finetune_only", False))
            and int(getattr(args, "start_point_cloud_iteration", 0)) > 0
            and not getattr(args, "start_model_path", "")
            and not has_start_checkpoint
        ):
            raise RuntimeError(
                "[Semantic Workflow] --semantic_finetune_only with --start_point_cloud_iteration requires "
                "--start_model_path so geometry and semantic outputs remain separated."
            )
        ensure_semantic_teachers(args, logger=logger)

    if semantic_requested and (has_start_geometry or has_start_checkpoint) and not getattr(args, "semantic_finetune_only", False):
        logger.warning(
            "[Semantic Workflow] Existing geometry initialization was provided with --semantic_3dgs; "
            "enabling --semantic_finetune_only automatically."
        )
        args.semantic_finetune_only = True
        continue_existing_geometry = True

    if semantic_requested and not continue_existing_geometry:
        geometry_iterations = int(getattr(args, "geometry_iterations", 30000))
        if geometry_iterations <= 0:
            raise RuntimeError("--geometry_iterations must be positive for automatic two-stage semantic training.")
        final_iterations = int(getattr(args, "iterations", 0))
        if final_iterations <= geometry_iterations:
            semantic_iterations = int(getattr(args, "semantic_iterations", 10000))
            if semantic_iterations <= 0:
                raise RuntimeError("--semantic_iterations must be positive when --iterations <= --geometry_iterations.")
            final_iterations = geometry_iterations + semantic_iterations
            logger.warning(
                "[Semantic Workflow] --iterations is not larger than --geometry_iterations; "
                f"using final semantic iteration {final_iterations} "
                f"({geometry_iterations}+{semantic_iterations})."
            )

        geometry_model_path = getattr(args, "geometry_model_path", "") or (args.model_path.rstrip("/\\") + "_geometry")
        if os.path.abspath(geometry_model_path) == os.path.abspath(args.model_path):
            raise RuntimeError("--geometry_model_path must be different from --model_path.")

        logger.info(
            "[Semantic Workflow] Starting continuous two-stage training: "
            f"geometry_model_path={geometry_model_path}, geometry_iterations={geometry_iterations}, "
            f"semantic_model_path={args.model_path}, final_iterations={final_iterations}."
        )

        geometry_args = _clone_namespace(
            args,
            model_path=geometry_model_path,
            iterations=geometry_iterations,
            semantic_3dgs=False,
            use_semantic_features=False,
            semantic_finetune_only=False,
            start_checkpoint=None,
            start_model_path="",
            start_point_cloud_iteration=0,
            lambda_semantic=0.0,
            save_iterations=_unique_iterations(args.save_iterations, geometry_iterations),
            test_iterations=_unique_iterations(args.test_iterations, geometry_iterations),
        )
        training(
            lp.extract(geometry_args),
            op.extract(geometry_args),
            pp.extract(geometry_args),
            dataset,
            geometry_args.test_iterations,
            geometry_args.save_iterations,
            geometry_args.checkpoint_iterations,
            geometry_args.start_checkpoint,
            geometry_args.debug_from,
            wandb,
            logger,
            disable_gui=geometry_args.disable_gui,
            start_model_path=geometry_args.start_model_path,
            start_point_cloud_iteration=geometry_args.start_point_cloud_iteration,
        )

        torch.cuda.empty_cache()

        semantic_args = _clone_namespace(
            args,
            iterations=final_iterations,
            semantic_3dgs=True,
            use_semantic_features=True,
            semantic_finetune_only=True,
            start_checkpoint=None,
            start_model_path=geometry_model_path,
            start_point_cloud_iteration=geometry_iterations,
            save_iterations=_unique_iterations(args.save_iterations, final_iterations),
            test_iterations=_unique_iterations(args.test_iterations, final_iterations),
        )
        logger.info("[Semantic Workflow] Geometry stage complete. Starting frozen semantic stage.")
        training(
            lp.extract(semantic_args),
            op.extract(semantic_args),
            pp.extract(semantic_args),
            dataset,
            semantic_args.test_iterations,
            semantic_args.save_iterations,
            semantic_args.checkpoint_iterations,
            semantic_args.start_checkpoint,
            semantic_args.debug_from,
            wandb,
            logger,
            disable_gui=semantic_args.disable_gui,
            start_model_path=semantic_args.start_model_path,
            start_point_cloud_iteration=semantic_args.start_point_cloud_iteration,
        )
        final_args = semantic_args
    else:
        args.save_iterations = _unique_iterations(args.save_iterations, args.iterations)
        if semantic_requested:
            args.test_iterations = _unique_iterations(args.test_iterations, args.iterations)
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            dataset,
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            wandb,
            logger,
            disable_gui=args.disable_gui,
            start_model_path=args.start_model_path,
            start_point_cloud_iteration=args.start_point_cloud_iteration,
        )
        if args.warmup:
            logger.info("\n Warmup finished! Reboot from last checkpoints")
            new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
            training(
                lp.extract(args),
                op.extract(args),
                pp.extract(args),
                dataset,
                args.test_iterations,
                args.save_iterations,
                args.checkpoint_iterations,
                args.start_checkpoint,
                args.debug_from,
                wandb=wandb,
                logger=logger,
                ply_path=new_ply_path,
                disable_gui=args.disable_gui,
                start_model_path=args.start_model_path,
                start_point_cloud_iteration=args.start_point_cloud_iteration,
            )

    # All done
    logger.info("\nTraining complete.")

    visible_count = None
    if final_args.skip_final_render:
        logger.info("\nSkipping final rendering because --skip_final_render is enabled.")
    else:
        logger.info(f'\nStarting Rendering~')
        visible_count = render_sets(lp.extract(final_args), -1, pp.extract(final_args), wandb=wandb, logger=logger)
        logger.info("\nRendering complete.")

    if final_args.skip_final_eval:
        logger.info("\nSkipping final RGB evaluation because --skip_final_eval is enabled.")
    else:
        logger.info("\n Starting evaluation...")
        evaluate(final_args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
        logger.info("\nEvaluating complete.")

    semantic_dataset = lp.extract(final_args)
    if final_args.skip_final_semantic_eval:
        logger.info("\nSkipping final semantic evaluation because --skip_final_semantic_eval is enabled.")
    elif semantic_benchmark_enabled(semantic_dataset):
        logger.info("\nStarting semantic evaluation...")
        semantic_prompt_path = (
            semantic_dataset.semantic_eval_prompt_path
            or "<missing>"
        )
        semantic_gt_2d = semantic_dataset.semantic_gt_2d or "<missing>"
        semantic_gt_3d = semantic_dataset.semantic_gt_3d or "<missing>"
        semantic_split = semantic_dataset.semantic_eval_split or "<unset>"
        try:
            run_semantic_benchmark_for_saved_model(
                semantic_dataset,
                -1,
                pp.extract(final_args),
                logger=logger,
                wandb=wandb,
            )
            logger.info("\nSemantic evaluation complete.")
        except Exception:
            logger.exception(
                "Semantic evaluation failed. "
                f"prompt_path={semantic_prompt_path}, "
                f"semantic_gt_2d={semantic_gt_2d}, "
                f"semantic_gt_3d={semantic_gt_3d}, "
                f"split={semantic_split}"
            )
            if getattr(semantic_dataset, "strict_semantic_eval", False):
                raise
