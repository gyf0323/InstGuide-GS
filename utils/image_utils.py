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

from PIL import Image
import torch
import torchvision.transforms.functional as tf

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def warpped_depth(depth):
    return torch.where(depth < 10.0, depth / 10.0, 2.0 - 10.0 / depth) / 2.0

def unwarpped_depth(depth):
    return torch.where(depth < 0.5, 2.0 * depth * 10.0, 10.0 / (1.0 - depth) / 2.0)


def read_image_pairs(renders_dir, gt_dir, device):
    for fname in sorted(os.listdir(renders_dir)):
        with Image.open(renders_dir / fname) as render:
            render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device)
        with Image.open(gt_dir / fname) as gt:
            gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device)
        yield fname, render_tensor, gt_tensor
