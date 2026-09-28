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
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, render_semantic_features=False, semantic_anchor_override=None):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    visible_anchor_indices = torch.nonzero(visible_mask, as_tuple=False).squeeze(1)
    semantic_anchor_feat = None
    semantic_child_feat = None
    if pc.use_semantic_features and render_semantic_features:
        if semantic_anchor_override is not None:
            semantic_anchor_feat = semantic_anchor_override[visible_mask]
        elif getattr(pc, "use_anchor_offset_semantic", False):
            semantic_child = pc.get_semantic_child_latents(visible_anchor_indices)
            if semantic_child is not None and semantic_child.numel() > 0:
                semantic_child_feat = semantic_child.reshape(-1, pc.semantic_latent_dim)
        else:
            semantic_anchor_feat = pc._semantic_feat[visible_mask]

    ## get view properties for anchor
    camera_center = viewpoint_camera.camera_center.to(anchor.device)
    ob_view = anchor - camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_uid = int(getattr(viewpoint_camera, "uid", 0))
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * camera_uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    semantic_feature = None
    if semantic_child_feat is not None:
        semantic_feature = semantic_child_feat[mask]
    elif semantic_anchor_feat is not None:
        semantic_feature = repeat(semantic_anchor_feat, 'n c -> (n k) c', k=pc.n_offsets)[mask]
    anchor_indices = repeat(visible_anchor_indices, 'n -> (n k)', k=pc.n_offsets)[mask]
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, semantic_feature, neural_opacity, mask, anchor_indices
    else:
        return xyz, color, opacity, scaling, rot, semantic_feature, anchor_indices


def prepare_visible_gaussians(
    viewpoint_camera,
    pc: GaussianModel,
    visible_mask=None,
    render_semantic_features=False,
    semantic_anchor_override=None,
):
    is_training = pc.get_color_mlp.training
    generated = generate_neural_gaussians(
        viewpoint_camera,
        pc,
        visible_mask,
        is_training=is_training,
        render_semantic_features=render_semantic_features,
        semantic_anchor_override=semantic_anchor_override,
    )
    if is_training:
        xyz, color, opacity, scaling, rot, semantic_feature, neural_opacity, mask, anchor_indices = generated
    else:
        xyz, color, opacity, scaling, rot, semantic_feature, anchor_indices = generated
        neural_opacity = None
        mask = None
    return {
        "xyz": xyz,
        "color": color,
        "opacity": opacity,
        "scaling": scaling,
        "rotation": rot,
        "semantic_feature": semantic_feature,
        "neural_opacity": neural_opacity,
        "selection_mask": mask,
        "anchor_indices": anchor_indices,
        "is_training": is_training,
    }


def build_rasterizer(
    viewpoint_camera,
    pipe,
    bg_color,
    scaling_modifier=1.0,
    image_height_override=None,
    image_width_override=None,
    use_depth_camera=False,
    sh_degree=1,
):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if use_depth_camera:
        if getattr(viewpoint_camera, "depth_full_proj_transform", None) is None:
            raise RuntimeError("Depth-camera rasterization requested, but this view has no depth projection.")
        default_height = int(viewpoint_camera.depth_height)
        default_width = int(viewpoint_camera.depth_width)
        proj_source = viewpoint_camera.depth_full_proj_transform
    else:
        default_height = int(viewpoint_camera.image_height)
        default_width = int(viewpoint_camera.image_width)
        proj_source = viewpoint_camera.full_proj_transform

    image_height = int(image_height_override) if image_height_override is not None else default_height
    image_width = int(image_width_override) if image_width_override is not None else default_width

    viewmatrix = viewpoint_camera.world_view_transform.to(bg_color.device)
    projmatrix = proj_source.to(bg_color.device)
    campos = viewpoint_camera.camera_center.to(bg_color.device)
    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=int(sh_degree),
        campos=campos,
        prefiltered=False,
        debug=pipe.debug
    )
    return GaussianRasterizer(raster_settings=raster_settings), image_height, image_width


def rasterize_precomputed_gaussians(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    prepared,
    scaling_modifier=1.0,
    retain_grad=False,
    image_height_override=None,
    image_width_override=None,
    use_depth_camera=False,
    semantic_only=False,
    render_depth=False,
    render_alpha=False,
    semantic_feature_override=None,
    return_topk_contrib=False,
    topk_contrib_k=0,
):
    xyz = prepared["xyz"]
    color = prepared["color"]
    opacity = prepared["opacity"]
    scaling = prepared["scaling"]
    rot = prepared["rotation"]
    semantic_feature = prepared["semantic_feature"] if semantic_feature_override is None else semantic_feature_override
    is_training = prepared["is_training"]

    screenspace_points = torch.zeros_like(
        xyz,
        dtype=pc.get_anchor.dtype,
        requires_grad=True,
        device=xyz.device,
    ) + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

    rasterizer, _, _ = build_rasterizer(
        viewpoint_camera,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        image_height_override=image_height_override,
        image_width_override=image_width_override,
        use_depth_camera=use_depth_camera,
        sh_degree=getattr(pc, "active_sh_degree", getattr(pc, "max_sh_degree", 1)),
    )

    semantic_bg = None
    if semantic_feature is not None:
        semantic_bg = torch.zeros((semantic_feature.shape[1],), dtype=semantic_feature.dtype, device=semantic_feature.device)

    raster_outputs = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        semantic_features=semantic_feature,
        semantic_bg=semantic_bg,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
        render_color_and_depth=not semantic_only,
        render_depth=render_depth,
        render_alpha=render_alpha,
        return_topk_contrib=return_topk_contrib,
        topk_contrib_k=topk_contrib_k,
    )
    topk_ids = topk_alpha = topk_depth = None
    if return_topk_contrib:
        (
            rendered_image,
            rendered_depth,
            rendered_semantic,
            rendered_semantic_weight,
            rendered_alpha,
            radii,
            topk_ids,
            topk_alpha,
            topk_depth,
        ) = raster_outputs
    else:
        rendered_image, rendered_depth, rendered_semantic, rendered_semantic_weight, rendered_alpha, radii = raster_outputs
    if rendered_depth is not None and torch.is_tensor(rendered_depth) and rendered_depth.numel() == 0:
        rendered_depth = None
    if rendered_alpha is not None and torch.is_tensor(rendered_alpha) and rendered_alpha.numel() == 0:
        rendered_alpha = None
    if rendered_semantic_weight is not None and torch.is_tensor(rendered_semantic_weight) and rendered_semantic_weight.numel() == 0:
        rendered_semantic_weight = None

    result = {
        "render": rendered_image,
        "depth": rendered_depth,
        "semantic": rendered_semantic,
        "semantic_weight": rendered_semantic_weight,
        "alpha": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    if return_topk_contrib:
        result["topk_ids"] = topk_ids
        result["topk_alpha"] = topk_alpha
        result["topk_depth"] = topk_depth
    if is_training:
        result["selection_mask"] = prepared["selection_mask"]
        result["neural_opacity"] = prepared["neural_opacity"]
        result["scaling"] = scaling
    return result

def render(
    viewpoint_camera,
    pc : GaussianModel,
    pipe,
    bg_color : torch.Tensor,
    scaling_modifier = 1.0,
    visible_mask=None,
    retain_grad=False,
    render_semantic_features=None,
    image_height_override=None,
    image_width_override=None,
    use_depth_camera=False,
    semantic_only=False,
    render_depth=False,
    render_alpha=False,
    semantic_anchor_override=None,
    return_topk_contrib=False,
    topk_contrib_k=0,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    if render_semantic_features is None:
        render_semantic_features = is_training

    prepared = prepare_visible_gaussians(
        viewpoint_camera,
        pc,
        visible_mask=visible_mask,
        render_semantic_features=render_semantic_features,
        semantic_anchor_override=semantic_anchor_override,
    )

    return rasterize_precomputed_gaussians(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        prepared,
        scaling_modifier=scaling_modifier,
        retain_grad=retain_grad,
        image_height_override=image_height_override,
        image_width_override=image_width_override,
        use_depth_camera=use_depth_camera,
        semantic_only=semantic_only,
        render_depth=render_depth,
        render_alpha=render_alpha,
        return_topk_contrib=return_topk_contrib,
        topk_contrib_k=topk_contrib_k,
    )


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        pc.get_anchor,
        dtype=pc.get_anchor.dtype,
        requires_grad=True,
        device=pc.get_anchor.device,
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    viewmatrix = viewpoint_camera.world_view_transform.to(bg_color.device)
    projmatrix = viewpoint_camera.full_proj_transform.to(bg_color.device)
    campos = viewpoint_camera.camera_center.to(bg_color.device)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=int(getattr(pc, "active_sh_degree", getattr(pc, "max_sh_degree", 1))),
        campos=campos,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
