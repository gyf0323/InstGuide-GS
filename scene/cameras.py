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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 depth, depth_mask, depth_is_warped, image_name, uid, cx=None, cy=None,
                 region_teacher_path=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name
        self.region_teacher_path = region_teacher_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.depth = depth.to(self.data_device) if depth is not None else None
        self.depth_mask = depth_mask.to(self.data_device) if depth_mask is not None else None
        self.depth_is_warped = bool(depth_is_warped)
        self.depth_height = int(self.depth.shape[1]) if self.depth is not None else None
        self.depth_width = int(self.depth.shape[2]) if self.depth is not None else None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale), dtype=torch.float32, device=self.data_device
        ).transpose(0, 1).contiguous()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
            width=self.image_width,
            height=self.image_height,
            cx=self.cx,
            cy=self.cy,
        ).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.depth_projection_matrix = None
        self.depth_full_proj_transform = None
        self.depth_cx = None
        self.depth_cy = None
        if self.depth is not None:
            sx = float(self.depth_width) / float(self.image_width)
            sy = float(self.depth_height) / float(self.image_height)
            self.depth_cx = self.cx * sx if self.cx is not None else None
            self.depth_cy = self.cy * sy if self.cy is not None else None
            self.depth_projection_matrix = getProjectionMatrix(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx,
                fovY=self.FoVy,
                width=self.depth_width,
                height=self.depth_height,
                cx=self.depth_cx,
                cy=self.depth_cy,
            ).transpose(0, 1).to(self.data_device)
            self.depth_full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(self.depth_projection_matrix.unsqueeze(0))
            ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, uid=0):
        self.uid = int(uid)
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
