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

from typing import NamedTuple
import torch.nn as nn
import torch

_REBUILD_EXTENSION_HINT = (
    "Rebuild the diff-gaussian-rasterization CUDA extension after the "
    "RGB+depth+semantic+alpha rasterizer changes, for example with "
    "`python setup.py build_ext --inplace` inside submodules/diff-gaussian-rasterization."
)

try:
    from . import _C
except ImportError as exc:
    raise ImportError(
        "Could not import diff_gaussian_rasterization._C. "
        + _REBUILD_EXTENSION_HINT
    ) from exc

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def _checked_extension_outputs(outputs, expected_count, function_name):
    output_count = len(outputs) if isinstance(outputs, tuple) else "non-tuple"
    if not isinstance(outputs, tuple) or len(outputs) != expected_count:
        raise RuntimeError(
            f"{function_name} returned {output_count} values, but this Python wrapper expects "
            f"{expected_count}. The compiled CUDA extension is likely stale. {_REBUILD_EXTENSION_HINT}"
        )
    return outputs

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    semantic_features,
    semantic_bg,
    lord_depth,
    lord_depth_grad,
    lord_valid,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    render_color_and_depth=True,
    render_depth=True,
    render_alpha=True,
    lord_sigma_alpha=0.0,
    lord_sigma_beta=0.0,
    lord_sigma_gamma=0.0,
    lord_kappa=0.0,
    lord_tau_d=1.0,
    lord_branch_mode=0,
    return_topk_contrib=False,
    topk_contrib_k=0,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        semantic_features,
        semantic_bg,
        lord_depth,
        lord_depth_grad,
        lord_valid,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        render_color_and_depth,
        render_depth,
        render_alpha,
        lord_sigma_alpha,
        lord_sigma_beta,
        lord_sigma_gamma,
        lord_kappa,
        lord_tau_d,
        lord_branch_mode,
        return_topk_contrib,
        topk_contrib_k,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        semantic_features,
        semantic_bg,
        lord_depth,
        lord_depth_grad,
        lord_valid,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        render_color_and_depth,
        render_depth,
        render_alpha,
        lord_sigma_alpha,
        lord_sigma_beta,
        lord_sigma_gamma,
        lord_kappa,
        lord_tau_d,
        lord_branch_mode,
        return_topk_contrib,
        topk_contrib_k,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            semantic_features,
            semantic_bg,
            lord_depth,
            lord_depth_grad,
            lord_valid,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            render_color_and_depth,
            render_depth,
            render_alpha,
            lord_sigma_alpha,
            lord_sigma_beta,
            lord_sigma_gamma,
            lord_kappa,
            lord_tau_d,
            lord_branch_mode,
            bool(return_topk_contrib),
            int(topk_contrib_k),
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                expected_outputs = 13 if return_topk_contrib else 10
                outputs = _checked_extension_outputs(_C.rasterize_gaussians(*args), expected_outputs, "_C.rasterize_gaussians")
                if return_topk_contrib:
                    num_rendered, color, depth, feature, feature_weight, alpha, radii, geomBuffer, binningBuffer, imgBuffer, topk_ids, topk_alpha, topk_depth = outputs
                else:
                    num_rendered, color, depth, feature, feature_weight, alpha, radii, geomBuffer, binningBuffer, imgBuffer = outputs
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            expected_outputs = 13 if return_topk_contrib else 10
            outputs = _checked_extension_outputs(_C.rasterize_gaussians(*args), expected_outputs, "_C.rasterize_gaussians")
            if return_topk_contrib:
                num_rendered, color, depth, feature, feature_weight, alpha, radii, geomBuffer, binningBuffer, imgBuffer, topk_ids, topk_alpha, topk_depth = outputs
            else:
                num_rendered, color, depth, feature, feature_weight, alpha, radii, geomBuffer, binningBuffer, imgBuffer = outputs

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.render_color_and_depth = render_color_and_depth
        ctx.render_depth = render_depth
        ctx.render_alpha = render_alpha
        ctx.lord_sigma_alpha = lord_sigma_alpha
        ctx.lord_sigma_beta = lord_sigma_beta
        ctx.lord_sigma_gamma = lord_sigma_gamma
        ctx.lord_kappa = lord_kappa
        ctx.lord_tau_d = lord_tau_d
        ctx.lord_branch_mode = lord_branch_mode
        ctx.return_topk_contrib = bool(return_topk_contrib)
        ctx.save_for_backward(
            colors_precomp,
            semantic_features,
            semantic_bg,
            lord_depth,
            lord_depth_grad,
            lord_valid,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        if return_topk_contrib:
            ctx.mark_non_differentiable(topk_ids, topk_alpha, topk_depth)
            return color, depth, feature, feature_weight, alpha, radii, topk_ids, topk_alpha, topk_depth
        return color, depth, feature, feature_weight, alpha, radii

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_out_depth,
        grad_out_feature,
        grad_out_feature_weight,
        grad_out_alpha,
        _,
        grad_topk_ids=None,
        grad_topk_alpha=None,
        grad_topk_depth=None,
    ):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        render_color_and_depth = ctx.render_color_and_depth
        render_depth = ctx.render_depth
        (
            colors_precomp,
            semantic_features,
            semantic_bg,
            lord_depth,
            lord_depth_grad,
            lord_valid,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        if (
            grad_out_color is None
            or not torch.is_tensor(grad_out_color)
            or grad_out_color.numel() == 0
            or grad_out_color.ndim != 3
        ):
            grad_out_color = torch.zeros(
                (
                    raster_settings.bg.shape[0],
                    raster_settings.image_height,
                    raster_settings.image_width,
                ),
                device=means3D.device,
                dtype=means3D.dtype,
            )
        if not render_depth:
            grad_out_depth = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        elif (
            grad_out_depth is None
            or not torch.is_tensor(grad_out_depth)
            or grad_out_depth.numel() == 0
            or grad_out_depth.ndim != 3
        ):
            grad_out_depth = torch.zeros(
                (1, raster_settings.image_height, raster_settings.image_width),
                device=means3D.device,
                dtype=means3D.dtype,
            )
        if grad_out_feature is None:
            if semantic_features.numel() > 0:
                grad_out_feature = torch.zeros(
                    (
                        semantic_features.shape[1],
                        raster_settings.image_height,
                        raster_settings.image_width,
                    ),
                    device=means3D.device,
                    dtype=means3D.dtype,
                )
            else:
                grad_out_feature = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if (
            grad_out_feature_weight is None
            or not torch.is_tensor(grad_out_feature_weight)
            or grad_out_feature_weight.numel() == 0
            or grad_out_feature_weight.ndim != 3
        ):
            if semantic_features.numel() > 0 and ctx.lord_branch_mode != 0:
                grad_out_feature_weight = torch.zeros(
                    (1, raster_settings.image_height, raster_settings.image_width),
                    device=means3D.device,
                    dtype=means3D.dtype,
                )
            else:
                grad_out_feature_weight = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                semantic_features,
                semantic_bg,
                lord_depth,
                lord_depth_grad,
                lord_valid,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth,
                grad_out_feature,
                grad_out_feature_weight,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,
                render_color_and_depth,
                ctx.lord_sigma_alpha,
                ctx.lord_sigma_beta,
                ctx.lord_sigma_gamma,
                ctx.lord_kappa,
                ctx.lord_tau_d,
                ctx.lord_branch_mode)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                outputs = _checked_extension_outputs(_C.rasterize_gaussians_backward(*args), 9, "_C.rasterize_gaussians_backward")
                grad_means2D, grad_colors_precomp, grad_semantic_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = outputs
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             outputs = _checked_extension_outputs(_C.rasterize_gaussians_backward(*args), 9, "_C.rasterize_gaussians_backward")
             grad_means2D, grad_colors_precomp, grad_semantic_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = outputs

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_semantic_features,
            None,
            None,
            None,
            None,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        semantic_features=None,
        semantic_bg=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        render_color_and_depth=True,
        render_depth=True,
        render_alpha=True,
        lord_depth=None,
        lord_depth_grad=None,
        lord_valid=None,
        lord_sigma_alpha=0.0,
        lord_sigma_beta=0.0,
        lord_sigma_gamma=0.0,
        lord_kappa=0.0,
        lord_tau_d=1.0,
        lord_branch_mode=0,
        return_topk_contrib=False,
        topk_contrib_k=0,
    ):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if colors_precomp is None:
            colors_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if semantic_features is None:
            semantic_features = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if semantic_bg is None:
            semantic_bg = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if lord_depth is None:
            lord_depth = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if lord_depth_grad is None:
            lord_depth_grad = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if lord_valid is None:
            lord_valid = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

        if scales is None:
            scales = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if rotations is None:
            rotations = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if cov3D_precomp is None:
            cov3D_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            semantic_features,
            semantic_bg,
            lord_depth,
            lord_depth_grad,
            lord_valid,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
            render_color_and_depth,
            render_depth,
            render_alpha,
            lord_sigma_alpha,
            lord_sigma_beta,
            lord_sigma_gamma,
            lord_kappa,
            lord_tau_d,
            lord_branch_mode,
            bool(return_topk_contrib),
            int(topk_contrib_k),
        )

    def visible_filter(self, means3D, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if rotations is None:
            rotations = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        if cov3D_precomp is None:
            cov3D_precomp = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = _C.rasterize_aussians_filter(means3D,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.prefiltered,
            raster_settings.debug)
        return  radii
    
    
