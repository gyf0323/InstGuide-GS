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
from pathlib import Path
import traceback
import torch
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, read_image_pairs
from argparse import ArgumentParser

LPIPS_FN_CACHE = {}


def get_lpips_fn(device):
    torch_device = torch.device(device)
    cache_key = str(torch_device)
    if cache_key not in LPIPS_FN_CACHE:
        LPIPS_FN_CACHE[cache_key] = lpips.LPIPS(net='vgg').to(torch_device)
    return LPIPS_FN_CACHE[cache_key]

def evaluate(model_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = get_lpips_fn(device)

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                ssims = []
                psnrs = []
                lpipss = []
                image_names = []
                render_names = sorted(os.listdir(renders_dir))

                for image_name, render_tensor, gt_tensor in tqdm(
                    read_image_pairs(renders_dir, gt_dir, device),
                    desc="Metric evaluation progress",
                    total=len(render_names),
                ):
                    image_names.append(image_name)
                    ssims.append(ssim(render_tensor, gt_tensor))
                    psnrs.append(psnr(render_tensor, gt_tensor))
                    lpipss.append(lpips_fn(render_tensor, gt_tensor).detach())

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as exc:
            print(f"Unable to compute metrics for model {scene_dir}: {exc}")
            traceback.print_exc()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
