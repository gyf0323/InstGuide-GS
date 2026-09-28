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

from scene.cameras import Camera
import numpy as np
import os
import torch
import torch.nn.functional as F
from utils.semantic_feature_utils import find_region_teacher_path
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def _resize_depth_and_mask(depth_tensor, mask_tensor, resolution):
    target_hw = (int(resolution[1]), int(resolution[0]))
    if tuple(depth_tensor.shape[-2:]) == target_hw:
        return depth_tensor, mask_tensor

    weighted_depth = F.interpolate(
        (depth_tensor * mask_tensor).unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    soft_mask = F.interpolate(
        mask_tensor.unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    hard_mask = F.interpolate(
        mask_tensor.unsqueeze(0),
        size=target_hw,
        mode="nearest",
    )
    resized_mask = (hard_mask.squeeze(0) > 0.5).float()
    resized_depth = weighted_depth.squeeze(0) / soft_mask.squeeze(0).clamp_min(1e-6)
    resized_depth = resized_depth * resized_mask
    return resized_depth.contiguous(), resized_mask.contiguous()


def _load_legacy_la_depth(depth_image, resolution):
    depth_channel = depth_image.getchannel("L")
    mask_channel = depth_image.getchannel("A")
    gt_depth = PILtoTorch(depth_channel, resolution)[0:1, ...]
    gt_depth_mask = PILtoTorch(mask_channel, resolution)[0:1, ...]
    return gt_depth.contiguous(), gt_depth_mask.contiguous(), True


def _load_metric_depth(depth_image, resolution, args):
    depth_array = np.array(depth_image)
    if depth_array.ndim == 3:
        if depth_array.shape[2] == 1:
            depth_array = depth_array[..., 0]
        else:
            raise RuntimeError(
                f"Unsupported metric depth shape {depth_array.shape}; expected a single-channel depth map."
            )

    raw_dtype = depth_array.dtype
    depth_array = depth_array.astype(np.float32, copy=False)
    if np.issubdtype(raw_dtype, np.integer):
        depth_array *= float(getattr(args, "depth_metric_scale", 0.001))

    invalid_value = float(getattr(args, "depth_invalid_value", 0.0))
    valid_mask = np.isfinite(depth_array) & (depth_array > invalid_value)
    depth_array = np.where(valid_mask, depth_array, 0.0)

    gt_depth = torch.from_numpy(depth_array).unsqueeze(0)
    gt_depth_mask = torch.from_numpy(valid_mask.astype(np.float32, copy=False)).unsqueeze(0)
    if resolution is not None:
        gt_depth, gt_depth_mask = _resize_depth_and_mask(gt_depth, gt_depth_mask, resolution)
    return gt_depth, gt_depth_mask, False


def load_depth_map(depth_image, resolution, args, keep_native_resolution=False):
    metric_resolution = None if keep_native_resolution else resolution
    if depth_image.mode == "LA":
        # Legacy LA depth encodes warped depth and mask together; keep the old
        # RGB-resolution path for compatibility with existing datasets.
        return _load_legacy_la_depth(depth_image, resolution)
    if depth_image.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
        return _load_metric_depth(depth_image, metric_resolution, args)

    depth_array = np.array(depth_image)
    if depth_array.ndim == 2 and depth_array.dtype.itemsize >= 2:
        return _load_metric_depth(depth_image, metric_resolution, args)
    if depth_array.ndim == 3 and depth_array.shape[2] == 2:
        return _load_legacy_la_depth(depth_image.convert("LA"), resolution)
    if depth_array.ndim == 2 and np.issubdtype(depth_array.dtype, np.floating):
        return _load_metric_depth(depth_image, metric_resolution, args)

    return _load_legacy_la_depth(depth_image.convert("LA"), resolution)

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    # print(f'gt_image: {gt_image.shape}')
    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    gt_depth = None
    gt_depth_mask = None
    depth_is_warped = False
    if getattr(cam_info, "depth_path", None) is not None:
        if getattr(cam_info, "depth", None) is not None:
            depth_image = cam_info.depth
        else:
            with Image.open(cam_info.depth_path) as depth_file:
                depth_image = depth_file.copy()
        gt_depth, gt_depth_mask, depth_is_warped = load_depth_map(
            depth_image,
            resolution,
            args,
            keep_native_resolution=getattr(args, "depth_use_native_resolution", False),
        )

    region_teacher_path = None
    if getattr(args, "use_semantic_features", False):
        region_teacher_dir = getattr(args, "region_teachers", "")
        if region_teacher_dir:
            if not os.path.isabs(region_teacher_dir):
                region_teacher_dir = os.path.join(args.source_path, region_teacher_dir)
            region_teacher_path = find_region_teacher_path(
                region_teacher_dir,
                cam_info.image_name,
                instance_teacher_map_mode=getattr(args, "instance_teacher_map_mode", "auto"),
            )

    cx = getattr(cam_info, "cx", None)
    cy = getattr(cam_info, "cy", None)
    if cx is not None:
        cx = float(cx) * float(resolution[0]) / float(orig_w)
    if cy is not None:
        cy = float(cy) * float(resolution[1]) / float(orig_h)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  depth=gt_depth, depth_mask=gt_depth_mask,
                  depth_is_warped=depth_is_warped,
                  image_name=cam_info.image_name, uid=id,
                  cx=cx, cy=cy,
                  region_teacher_path=region_teacher_path,
                  data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    if hasattr(camera, "cx") and hasattr(camera, "cy"):
        camera_entry["cx"] = camera.cx
        camera_entry["cy"] = camera.cy
    return camera_entry
