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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const int topk_contrib_k)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::empty({0}, float_opts);
  torch::Tensor out_depth = torch::empty({0}, float_opts);
  torch::Tensor out_feature = torch::empty({0}, float_opts);
  torch::Tensor out_feature_weight = torch::empty({0}, float_opts);
  torch::Tensor out_alpha = torch::empty({0}, float_opts);
  torch::Tensor topk_ids = torch::empty({0}, int_opts);
  torch::Tensor topk_alpha = torch::empty({0}, float_opts);
  torch::Tensor topk_depth = torch::empty({0}, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  if (return_topk_contrib) {
    if (topk_contrib_k <= 0 || topk_contrib_k > MAX_TOPK_CONTRIB) {
      AT_ERROR("topk_contrib_k must be in [1, MAX_TOPK_CONTRIB]");
    }
    topk_ids = torch::full({H, W, topk_contrib_k}, -1, int_opts);
    topk_alpha = torch::full({H, W, topk_contrib_k}, 0.0, float_opts);
    topk_depth = torch::full({H, W, topk_contrib_k}, 0.0, float_opts);
  }
  if (render_color_and_depth) {
    out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
    if (render_depth) {
      out_depth = torch::full({1, H, W}, 0.0, float_opts);
    }
  }
  if (render_alpha) {
    out_alpha = torch::full({1, H, W}, 0.0, float_opts);
  }
  const bool render_semantic = semantic_features.numel() > 0;
  int semantic_channels = 0;
  if (render_semantic) {
    if (semantic_features.ndimension() != 2 || semantic_features.size(0) != P) {
      AT_ERROR("semantic_features must have dimensions (num_points, semantic_channels)");
    }
    semantic_channels = semantic_features.size(1);
    if (semantic_channels <= 0 || semantic_channels > 1024) {
      AT_ERROR("semantic_channels must be in [1, 1024]");
    }
    if (semantic_background.numel() != semantic_channels) {
      AT_ERROR("semantic_background must have semantic_channels values");
    }
    out_feature = torch::full({semantic_channels, H, W}, 0.0, float_opts);
    if (lord_branch_mode != 0) {
      out_feature_weight = torch::full({1, H, W}, 0.0, float_opts);
    }
  }
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		render_semantic ? semantic_features.contiguous().data<float>() : nullptr,
		render_semantic ? semantic_background.contiguous().data<float>() : nullptr,
		semantic_channels,
		(lord_branch_mode != 0 && lord_depth.numel() > 0) ? lord_depth.contiguous().data<float>() : nullptr,
		(lord_branch_mode != 0 && lord_depth_grad.numel() > 0) ? lord_depth_grad.contiguous().data<float>() : nullptr,
		(lord_branch_mode != 0 && lord_valid.numel() > 0) ? lord_valid.contiguous().data<float>() : nullptr,
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		render_color_and_depth ? out_color.contiguous().data<float>() : nullptr,
		(render_color_and_depth && render_depth && out_depth.numel() > 0) ? out_depth.contiguous().data<float>() : nullptr,
		render_semantic ? out_feature.contiguous().data<float>() : nullptr,
		(render_semantic && lord_branch_mode != 0) ? out_feature_weight.contiguous().data<float>() : nullptr,
		(render_alpha && out_alpha.numel() > 0) ? out_alpha.contiguous().data<float>() : nullptr,
		(return_topk_contrib && topk_ids.numel() > 0) ? topk_ids.contiguous().data<int>() : nullptr,
		(return_topk_contrib && topk_alpha.numel() > 0) ? topk_alpha.contiguous().data<float>() : nullptr,
		(return_topk_contrib && topk_depth.numel() > 0) ? topk_depth.contiguous().data<float>() : nullptr,
		return_topk_contrib ? topk_contrib_k : 0,
		lord_sigma_alpha,
		lord_sigma_beta,
		lord_sigma_gamma,
		lord_kappa,
		lord_tau_d,
		lord_branch_mode,
		render_color_and_depth,
		radii.contiguous().data<int>(),
		debug);
  }
  if (return_topk_contrib) {
    return pybind11::make_tuple(rendered, out_color, out_depth, out_feature, out_feature_weight, out_alpha, radii, geomBuffer, binningBuffer, imgBuffer, topk_ids, topk_alpha, topk_depth);
  }
  return pybind11::make_tuple(rendered, out_color, out_depth, out_feature, out_feature_weight, out_alpha, radii, geomBuffer, binningBuffer, imgBuffer);
}

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
	const int lord_branch_mode) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dsemantic_features = torch::empty({0}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  const bool render_semantic = semantic_features.numel() > 0;
  int semantic_channels = 0;
  if (render_semantic) {
    if (semantic_features.ndimension() != 2 || semantic_features.size(0) != P) {
      AT_ERROR("semantic_features must have dimensions (num_points, semantic_channels)");
    }
    semantic_channels = semantic_features.size(1);
    if (semantic_channels <= 0 || semantic_channels > 1024) {
      AT_ERROR("semantic_channels must be in [1, 1024]");
    }
    if (semantic_background.numel() != semantic_channels) {
      AT_ERROR("semantic_background must have semantic_channels values");
    }
    dL_dsemantic_features = torch::zeros({P, semantic_channels}, means3D.options());
  }
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  render_semantic ? semantic_features.contiguous().data<float>() : nullptr,
	  render_semantic ? semantic_background.contiguous().data<float>() : nullptr,
	  semantic_channels,
	  (lord_branch_mode != 0 && lord_depth.numel() > 0) ? lord_depth.contiguous().data<float>() : nullptr,
	  (lord_branch_mode != 0 && lord_depth_grad.numel() > 0) ? lord_depth_grad.contiguous().data<float>() : nullptr,
	  (lord_branch_mode != 0 && lord_valid.numel() > 0) ? lord_valid.contiguous().data<float>() : nullptr,
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depth.numel() > 0 ? dL_dout_depth.contiguous().data<float>() : nullptr,
	  render_semantic ? dL_dout_feature.contiguous().data<float>() : nullptr,
	  (render_semantic && lord_branch_mode != 0 && dL_dout_feature_weight.numel() > 0) ? dL_dout_feature_weight.contiguous().data<float>() : nullptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  render_semantic ? dL_dsemantic_features.contiguous().data<float>() : nullptr,
	  dL_ddepths.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  lord_sigma_alpha,
	  lord_sigma_beta,
	  lord_sigma_gamma,
	  lord_kappa,
	  lord_tau_d,
	  lord_branch_mode,
	  render_color_and_depth,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dsemantic_features, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}


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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);


  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  

  if(P != 0)
  {
	  int M = 0;

	  CudaRasterizer::Rasterizer::visible_filter(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, M,
		W, H,
		means3D.contiguous().data<float>(),
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		radii.contiguous().data<int>(),
		debug);
  }
  return radii;
}
