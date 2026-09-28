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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* semantic_features,
			const float* semantic_background,
			const int semantic_channels,
			const float* lord_depth,
			const float* lord_depth_grad,
			const float* lord_valid,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			float* out_feature,
			float* out_feature_weight,
			float* out_alpha,
			int* topk_ids,
			float* topk_alpha,
			float* topk_depth,
			int topk_contrib_k,
			float lord_sigma_alpha,
			float lord_sigma_beta,
			float lord_sigma_gamma,
			float lord_kappa,
			float lord_tau_d,
			int lord_branch_mode,
			bool render_color_and_depth,
			int* radii = nullptr,
			bool debug = false);


		static void visible_filter(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int M,
			const int width, int height,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			int* radii,
			bool debug);
		
		
		
		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* semantic_features,
			const float* semantic_background,
			const int semantic_channels,
			const float* lord_depth,
			const float* lord_depth_grad,
			const float* lord_valid,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dpix_depth,
			const float* dL_dpix_feature,
			const float* dL_dpix_feature_weight,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dsemantic_feature,
			float* dL_ddepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float lord_sigma_alpha,
			float lord_sigma_beta,
			float lord_sigma_gamma,
			float lord_kappa,
			float lord_tau_d,
			int lord_branch_mode,
			bool render_color_and_depth,
			bool debug);
	};
};

#endif
