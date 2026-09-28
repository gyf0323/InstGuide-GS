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
import torch

import numpy as np

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.system_utils import configure_visible_devices
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


SELECTED_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES_SOURCE = configure_visible_devices(sys.argv[1:])


def _compute_fps(timings, warmup=5):
    if not timings:
        return None
    effective_timings = timings[warmup:] if len(timings) > warmup else timings
    if not effective_timings:
        return None
    mean_time = float(np.mean(effective_timings))
    if mean_time <= 0.0:
        return None
    return 1.0 / mean_time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)

    per_view_dict = {}
    # debug = 0
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()
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
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        per_view_dict['{0:05d}'.format(idx) + ".png"] = int((render_pkg["radii"] > 0).sum().item())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    fps = _compute_fps(t_list)
    if fps is not None:
        print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')
    else:
        print("Test FPS: unavailable (insufficient timing samples)")

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
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
                              semantic_gate_init=getattr(dataset, "semantic_gate_init", -4.0))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=gaussians.get_anchor.device)
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default='-1')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        "Using GPU visibility:",
        visible_devices if visible_devices else "<unset>",
        f"(source={CUDA_VISIBLE_DEVICES_SOURCE})",
    )

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args, for_inference=True), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
