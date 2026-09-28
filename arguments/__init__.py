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

import argparse
from argparse import ArgumentParser, Namespace
import ast
import sys
import os

class GroupParams:
    pass


class BooleanFlagAction(argparse.Action):
    def __init__(self, option_strings, dest, default=None, **kwargs):
        normalized_option_strings = list(option_strings)
        for option in list(normalized_option_strings):
            if option.startswith("--") and not option.startswith("--no-"):
                normalized_option_strings.append("--no-" + option[2:])
        super().__init__(option_strings=normalized_option_strings, dest=dest, nargs=0, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, not str(option_string).startswith("--no-"))

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action=BooleanFlagAction)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action=BooleanFlagAction)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.005 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = 1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False

        self.semantic_3dgs = False
        self.use_semantic_features = False
        self.region_teachers = ""
        self.semantic_teacher_allow_resize = False
        self.teacher_clip_model = ""
        self.semantic_latent_dim = 12
        self.semantic_feature_dim = 512
        self.semantic_teacher_dim = 12
        self.semantic_codec_path = ""
        self.semantic_storage_dtype = "float16"
        self.instance_teacher_map_mode = "direct"
        self.use_anchor_offset_semantic = True
        self.semantic_residual_dim = 4
        self.semantic_gate_init = -4.0
        self.use_instance_geometry_semantic = False
        self.instance_geometry_residual_mode = "sparse"
        self.instance_geometry_teacher = ""
        self.instance_geometry_binding = ""
        self.instance_geometry_auto_prepare = True
        self.instance_geometry_object_dict = ""
        self.instance_geometry_point_cloud = ""
        self.instance_geometry_max_distance = 0.08
        self.instance_geometry_interior_purity = 0.90
        self.instance_geometry_min_support_fraction = 0.50
        self.instance_geometry_min_query_confidence = 0.50
        self.instance_geometry_unknown_residual = True
        self.instance_geometry_initialize_interior = True
        self.semantic_eval_prompt_path = ""
        self.semantic_raw_class_path = ""
        self.semantic_benchmark_map_path = ""
        self.semantic_gt_2d = ""
        self.semantic_gt_3d = ""
        self.semantic_gt_info = ""
        self.semantic_3d_eval_mode = "anchor"
        self.semantic_3d_child_knn = 8
        self.semantic_3d_child_knn_weight = "inverse_distance"
        self.semantic_eval_ignore_label = 255
        self.semantic_eval_exclude_classes = ""
        self.semantic_eval_mesh_classes_only = False
        self.semantic_eval_split = "test"
        self.strict_semantic_eval = False
        self.skip_final_render = False
        self.skip_final_eval = False
        self.skip_final_semantic_eval = False
        self.eval_existing_renders = False
        self.depth_metric_scale = 0.001
        self.depth_invalid_value = 0.0
        self.depth_use_native_resolution = True
        self.depth_allow_missing = False
        self.depth_mode = "auto"
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args, *, for_inference=False):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if getattr(g, "model_path", ""):
            g.model_path = os.path.abspath(g.model_path)
        configured_codec_path = str(getattr(g, "semantic_codec_path", "") or "")
        model_codec_path = (
            os.path.join(g.model_path, "semantic_codec.pth")
            if getattr(g, "model_path", "")
            else ""
        )
        if for_inference:
            # Teachers supervise training, but are never inference dependencies.
            g.region_teachers = ""
            g.instance_geometry_teacher = ""
            g.semantic_codec_path = ""
            semantic_enabled = bool(g.semantic_3dgs or g.use_semantic_features)
            compressed = bool(configured_codec_path) or (
                int(g.semantic_teacher_dim or 0) not in {0, int(g.semantic_feature_dim)}
            )
            configured_codec_path = ""
            if semantic_enabled and model_codec_path and os.path.isfile(model_codec_path):
                configured_codec_path = model_codec_path
            elif semantic_enabled and compressed:
                raise RuntimeError(
                    "Compressed semantic inference requires the model-owned semantic_codec.pth: "
                    f"{model_codec_path}. Teacher paths are not used for inference."
                )
        if not configured_codec_path and not for_inference:
            if model_codec_path and os.path.isfile(model_codec_path):
                configured_codec_path = model_codec_path
            else:
                teacher_dir = str(getattr(g, "region_teachers", "") or "")
                if teacher_dir:
                    if not os.path.isabs(teacher_dir):
                        teacher_dir = os.path.join(g.source_path, teacher_dir)
                    teacher_codec_path = os.path.join(teacher_dir, "semantic_codec.pth")
                    if os.path.isfile(teacher_codec_path):
                        configured_codec_path = teacher_codec_path
        if configured_codec_path:
            configured_codec_path = os.path.abspath(configured_codec_path)
            if not os.path.isfile(configured_codec_path) and model_codec_path and os.path.isfile(model_codec_path):
                configured_codec_path = model_codec_path
            g.semantic_codec_path = configured_codec_path
        if getattr(g, "semantic_codec_path", ""):
            from utils.semantic_codec import load_semantic_codec

            _, codec_payload = load_semantic_codec(g.semantic_codec_path, map_location="cpu")
            codec_metadata = codec_payload["metadata"]
            codec_input_dim = int(codec_metadata["input_dim"])
            codec_latent_dim = int(codec_metadata["latent_dim"])
            if int(g.semantic_feature_dim) != codec_input_dim:
                raise RuntimeError(
                    "--semantic_feature_dim must match the codec query dimension: "
                    f"configured={g.semantic_feature_dim}, codec={codec_input_dim}."
                )
            if int(g.semantic_latent_dim) != codec_latent_dim:
                raise RuntimeError(
                    "--semantic_latent_dim must match the codec compressed dimension: "
                    f"configured={g.semantic_latent_dim}, codec={codec_latent_dim}."
                )
            if int(g.semantic_teacher_dim or 0) not in {0, codec_latent_dim}:
                raise RuntimeError(
                    "--semantic_teacher_dim must match the codec compressed dimension: "
                    f"configured={g.semantic_teacher_dim}, codec={codec_latent_dim}."
                )
            g.semantic_teacher_dim = codec_latent_dim
        elif int(getattr(g, "semantic_teacher_dim", 0) or 0) == 0:
            g.semantic_teacher_dim = int(g.semantic_feature_dim)
        if getattr(g, "semantic_3dgs", False):
            g.use_semantic_features = True
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0001
        self.position_lr_final = 0.000001
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        # -1 means "auto": enable a conservative default only when --depths is provided.
        self.lambda_depth = -1.0
        self.depth_start_iter = 3000
        # 0 means keep depth supervision active until the end of training.
        self.depth_duration_iters = 0
        self.depth_ramp_iters = 3000
        self.depth_min_valid_pixels = 512
        self.depth_alpha_min = 0.1
        self.depth_grad_filter_threshold = 0.05
        self.depth_grad_filter_dilation = 1
        self.depth_residual_clip = 0.3
        self.depth_charbonnier_eps = 0.001
        self.depth_front_weight = 0.0
        self.depth_front_margin = 0.05
        self.depth_front_alpha_min = 0.02
        self.depth_back_weight = 0.5
        self.depth_back_margin = 0.05
        self.depth_back_alpha_min = 0.02
        self.depth_surface_weight = 0.2
        self.depth_surface_margin = 0.05
        self.depth_surface_mode = "back"
        self.depth_surface_min_opacity = 0.01
        self.depth_densify_gate = True
        self.depth_densify_mode = "back"
        self.depth_densify_front_margin = 0.08
        self.depth_densify_back_margin = 0.08
        self.semantic_lr = 0.005
        self.semantic_decoder_lr = 0.001
        self.lambda_semantic = 0.0
        self.lambda_semantic_gate = 1e-3
        self.lambda_semantic_residual = 1e-4
        self.lambda_instance_geometry = 0.1
        self.semantic_min_alpha = 0.05
        self.semantic_min_pixels = 64
        self.semantic_debug_interval = 200
        self.semantic_finetune_only = False
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002
        self.auto_checkpoint_interval = 10000
        self.auto_checkpoint_save_point_cloud = False
        self.auto_checkpoint_on_interrupt = True

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError, OSError):
        print("Config file not found at")
        pass
    args_cfgfile = _safe_parse_namespace(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def _safe_parse_namespace(cfgfile_string: str) -> Namespace:
    cfgfile_string = (cfgfile_string or "").strip()
    if not cfgfile_string:
        return Namespace()

    try:
        expr = ast.parse(cfgfile_string, mode="eval").body
    except SyntaxError as exc:
        raise RuntimeError("cfg_args is not a valid Namespace expression.") from exc

    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != "Namespace":
        raise RuntimeError("cfg_args must be a Namespace(...) expression.")
    if expr.args:
        raise RuntimeError("cfg_args Namespace(...) may only contain keyword arguments.")

    kwargs = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            raise RuntimeError("cfg_args does not support **kwargs unpacking.")
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError) as exc:
            raise RuntimeError(f"cfg_args contains a non-literal value for '{keyword.arg}'.") from exc
    return Namespace(**kwargs)
