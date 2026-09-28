/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cstdio>
#include <tuple>
#include <string>
	
pybind11::tuple
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
	const torch::Tensor& semantic_features,
	const torch::Tensor& semantic_background,
	const torch::Tensor& lord_depth,
	const torch::Tensor& lord_depth_grad,
	const torch::Tensor& lord_valid,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const bool render_color_and_depth,
	const bool render_depth,
	const bool render_alpha,
	const float lord_sigma_alpha,
	const float lord_sigma_beta,
	const float lord_sigma_gamma,
	const float lord_kappa,
	const float lord_tau_d,
	const int lord_branch_mode,
	const bool return_topk_contrib,
	const int topk_contrib_k);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& semantic_features,
	const torch::Tensor& semantic_background,
	const torch::Tensor& lord_depth,
	const torch::Tensor& lord_depth_grad,
	const torch::Tensor& lord_valid,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& dL_dout_feature,
	const torch::Tensor& dL_dout_feature_weight,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const bool render_color_and_depth,
	const float lord_sigma_alpha,
	const float lord_sigma_beta,
	const float lord_sigma_gamma,
	const float lord_kappa,
	const float lord_tau_d,
	const int lord_branch_mode);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);


torch::Tensor
RasterizeGaussiansfilterCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const bool prefiltered,
	const bool debug);
