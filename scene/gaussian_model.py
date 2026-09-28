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
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.semantic_codec import load_semantic_decoder
from scene.embedding import Embedding


def scatter_mean_2d(src, index, dim_size):
    out = torch.zeros((dim_size, src.shape[1]), device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand(-1, src.shape[1]), src)

    count = torch.zeros((dim_size, 1), device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1), torch.ones((src.shape[0], 1), device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0)


class GaussianModel:

    @staticmethod
    def _to_device_tensor(tensor, device):
        if tensor is None:
            return None
        if isinstance(tensor, nn.Parameter):
            tensor = tensor.detach()
        return tensor.to(device)

    @staticmethod
    def _to_parameter(tensor, device, requires_grad):
        tensor = GaussianModel._to_device_tensor(tensor, device)
        if tensor is None:
            return None
        return nn.Parameter(tensor.requires_grad_(requires_grad))

    @staticmethod
    def _move_optimizer_state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    @staticmethod
    def _jit_load_robust(path, map_location=None):
        try:
            return torch.jit.load(path, map_location=map_location)
        except (RuntimeError, OSError, ValueError, FileNotFoundError):
            with open(path, "rb") as handle:
                return torch.jit.load(handle, map_location=map_location)

    @staticmethod
    def _torch_load_robust(path, map_location=None):
        try:
            return torch.load(path, map_location=map_location)
        except (RuntimeError, OSError, ValueError, FileNotFoundError):
            with open(path, "rb") as handle:
                return torch.load(handle, map_location=map_location)

    @staticmethod
    def _restore_fp32_compute_tensors(value):
        """Upcast storage-only half tensors before semantic inference."""
        if torch.is_tensor(value):
            if value.dtype in {torch.float16, torch.bfloat16}:
                return value.float()
            return value
        if isinstance(value, dict):
            return value.__class__(
                (key, GaussianModel._restore_fp32_compute_tensors(item))
                for key, item in value.items()
            )
        if isinstance(value, list):
            return [GaussianModel._restore_fp32_compute_tensors(item) for item in value]
        if isinstance(value, tuple):
            return tuple(GaussianModel._restore_fp32_compute_tensors(item) for item in value)
        return value

    @staticmethod
    def _clear_cuda_cache_if_needed(device):
        if device is not None and torch.device(device).type == "cuda":
            torch.cuda.empty_cache()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 use_semantic_features : bool = False,
                 semantic_latent_dim : int = 12,
                 semantic_feature_dim : int = 512,
                 semantic_teacher_dim : int = 0,
                 semantic_codec_path : str = "",
                 sh_degree : int = 3,
                 use_anchor_offset_semantic : bool = False,
                 semantic_residual_dim : int = 4,
                 semantic_gate_init : float = -4.0,
                 semantic_storage_dtype : str = "float16",
                 ):

        self.active_sh_degree = int(sh_degree)
        self.max_sh_degree = int(sh_degree)
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.use_semantic_features = use_semantic_features
        self.semantic_latent_dim = semantic_latent_dim
        self.semantic_feature_dim = semantic_feature_dim
        self.semantic_codec_path = str(semantic_codec_path or "")
        self.semantic_teacher_dim = int(semantic_teacher_dim or 0)
        self.uses_compressed_semantic_teacher = bool(self.semantic_codec_path)
        self.semantic_decoder_trainable = not self.uses_compressed_semantic_teacher
        self.use_anchor_offset_semantic = bool(use_anchor_offset_semantic)
        self.semantic_residual_dim = int(semantic_residual_dim)
        if self.semantic_residual_dim < 0:
            raise ValueError(
                "semantic_residual_dim must be non-negative, got "
                f"{self.semantic_residual_dim}."
            )
        self.semantic_gate_init = float(semantic_gate_init)
        semantic_storage_dtype = str(semantic_storage_dtype or "float32").strip().lower()
        semantic_storage_aliases = {
            "float": "float32",
            "fp32": "float32",
            "half": "float16",
            "fp16": "float16",
        }
        self.semantic_storage_dtype = semantic_storage_aliases.get(
            semantic_storage_dtype, semantic_storage_dtype
        )
        if self.semantic_storage_dtype not in {"float32", "float16"}:
            raise ValueError(
                "semantic_storage_dtype must be 'float32' or 'float16', got "
                f"{semantic_storage_dtype!r}."
            )

        self._default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._semantic_feat = torch.empty(0)
        self._semantic_anchor = torch.empty(0)
        self._semantic_offset_residual = torch.empty(0)
        self._semantic_offset_gate = torch.empty(0)
        self._semantic_residual_anchor_indices = torch.empty(0, dtype=torch.long)
        self._semantic_residual_row_for_anchor = torch.empty(0, dtype=torch.long)
        self._semantic_anchor_instance_index = torch.empty(0, dtype=torch.long)
        self._semantic_anchor_geometry_state = torch.empty(0, dtype=torch.uint8)
        self._semantic_anchor_geometry_confidence = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).to(self._default_device)

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).to(self._default_device)

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).to(self._default_device)

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).to(self._default_device)

        self.semantic_decoder = None
        self.semantic_codec_metadata = {}
        self.offset_residual_proj = None
        if self.use_semantic_features:
            if self.uses_compressed_semantic_teacher:
                decoder, codec_payload = load_semantic_decoder(
                    self.semantic_codec_path,
                    map_location=self._default_device,
                )
                self.semantic_codec_metadata = dict(codec_payload["metadata"])
                codec_input_dim = int(self.semantic_codec_metadata["input_dim"])
                codec_latent_dim = int(self.semantic_codec_metadata["latent_dim"])
                if codec_input_dim != int(self.semantic_feature_dim):
                    raise RuntimeError(
                        "Semantic codec query dimension mismatch: "
                        f"codec input_dim={codec_input_dim}, semantic_feature_dim={self.semantic_feature_dim}."
                    )
                if codec_latent_dim != int(self.semantic_latent_dim):
                    raise RuntimeError(
                        "Semantic codec latent dimension mismatch: "
                        f"codec latent_dim={codec_latent_dim}, semantic_latent_dim={self.semantic_latent_dim}."
                    )
                if self.semantic_teacher_dim not in {0, codec_latent_dim}:
                    raise RuntimeError(
                        "Compressed teacher dimension mismatch: "
                        f"semantic_teacher_dim={self.semantic_teacher_dim}, codec latent_dim={codec_latent_dim}."
                    )
                self.semantic_teacher_dim = codec_latent_dim
                self.semantic_decoder = decoder.to(self._default_device)
            else:
                if self.semantic_teacher_dim not in {0, int(self.semantic_feature_dim)}:
                    raise RuntimeError(
                        "A teacher dimension different from semantic_feature_dim requires --semantic_codec_path."
                    )
                self.semantic_teacher_dim = int(self.semantic_feature_dim)
                self.semantic_decoder = nn.Sequential(
                    nn.Linear(self.semantic_latent_dim, self.semantic_latent_dim * 2),
                    nn.ReLU(True),
                    nn.Linear(self.semantic_latent_dim * 2, self.semantic_feature_dim),
                ).to(self._default_device)
            if self.use_anchor_offset_semantic and self.semantic_residual_dim > 0:
                self.offset_residual_proj = nn.Linear(
                    self.semantic_residual_dim,
                    self.semantic_latent_dim,
                ).to(self._default_device)

    def get_model_device(self):
        modules = [
            self.mlp_opacity,
            self.mlp_cov,
            self.mlp_color,
            self.semantic_decoder,
            self.offset_residual_proj,
            self.mlp_feature_bank if self.use_feat_bank else None,
            self.embedding_appearance if self.appearance_dim > 0 else None,
        ]
        for module in modules:
            if module is None:
                continue
            params = list(module.parameters())
            if params:
                return params[0].device
        if torch.is_tensor(self._anchor) and self._anchor.numel() > 0:
            return self._anchor.device
        return self._default_device


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.semantic_decoder is not None:
            self.semantic_decoder.eval()
        if self.offset_residual_proj is not None:
            self.offset_residual_proj.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.semantic_decoder is not None and self.semantic_decoder_trainable:
            self.semantic_decoder.train()
        elif self.semantic_decoder is not None:
            self.semantic_decoder.eval()
        if self.offset_residual_proj is not None:
            self.offset_residual_proj.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def _ensure_offset_residual_proj(self, device):
        if (
            not (self.use_semantic_features and self.use_anchor_offset_semantic)
            or self.semantic_residual_dim == 0
        ):
            self.offset_residual_proj = None
            return
        if self.offset_residual_proj is None:
            self.offset_residual_proj = nn.Linear(
                self.semantic_residual_dim,
                self.semantic_latent_dim,
            ).to(device)
        else:
            self.offset_residual_proj = self.offset_residual_proj.to(device)

    def _init_anchor_offset_semantic_parameters(
        self,
        num_anchors,
        device,
        semantic_anchor=None,
        semantic_offset_residual=None,
        semantic_offset_gate=None,
        offset_residual_proj_state=None,
        semantic_residual_anchor_indices=None,
        semantic_anchor_instance_index=None,
        semantic_anchor_geometry_state=None,
        semantic_anchor_geometry_confidence=None,
    ):
        self._ensure_offset_residual_proj(device)
        anchor = self._to_device_tensor(semantic_anchor, device)
        if anchor is None:
            anchor = torch.empty((num_anchors, self.semantic_latent_dim), dtype=torch.float32, device=device)
            anchor.normal_(mean=0.0, std=0.01)
        if tuple(anchor.shape) != (num_anchors, self.semantic_latent_dim):
            raise RuntimeError(
                "Anchor-offset semantic anchor shape mismatch: "
                f"expected {(num_anchors, self.semantic_latent_dim)}, got {tuple(anchor.shape)}."
            )

        sparse_indices = (
            torch.empty(0, dtype=torch.long, device=device)
            if self.semantic_residual_dim == 0
            else self._to_device_tensor(semantic_residual_anchor_indices, device)
        )
        if sparse_indices is not None:
            sparse_indices = sparse_indices.long().reshape(-1)
            if sparse_indices.numel() > 0:
                if int(sparse_indices.min().item()) < 0 or int(sparse_indices.max().item()) >= int(num_anchors):
                    raise RuntimeError("Sparse semantic residual anchor indices are outside the anchor range.")
                if sparse_indices.numel() > 1 and not bool(torch.all(sparse_indices[1:] > sparse_indices[:-1]).item()):
                    raise RuntimeError(
                        "Sparse semantic residual anchor indices must be unique and strictly increasing so their "
                        "rows stay aligned with residual tensors."
                    )
            residual_anchor_count = int(sparse_indices.numel())
            row_for_anchor = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
            if residual_anchor_count > 0:
                row_for_anchor[sparse_indices] = torch.arange(residual_anchor_count, device=device)
        else:
            residual_anchor_count = int(num_anchors)
            sparse_indices = torch.empty(0, dtype=torch.long, device=device)
            row_for_anchor = torch.empty(0, dtype=torch.long, device=device)

        residual = self._to_device_tensor(semantic_offset_residual, device)
        expected_residual_shape = (residual_anchor_count, self.n_offsets, self.semantic_residual_dim)
        if residual is None or tuple(residual.shape) != expected_residual_shape:
            residual = torch.empty(expected_residual_shape, dtype=torch.float32, device=device)
            residual.normal_(mean=0.0, std=0.01)

        gate = self._to_device_tensor(semantic_offset_gate, device)
        expected_gate_shape = (residual_anchor_count, self.n_offsets, 1)
        if gate is None or tuple(gate.shape) != expected_gate_shape:
            gate = torch.full(expected_gate_shape, self.semantic_gate_init, dtype=torch.float32, device=device)

        self._semantic_anchor = nn.Parameter(anchor.requires_grad_(True))
        self._semantic_offset_residual = nn.Parameter(residual.requires_grad_(True))
        self._semantic_offset_gate = nn.Parameter(gate.requires_grad_(True))
        self._semantic_residual_anchor_indices = sparse_indices
        self._semantic_residual_row_for_anchor = row_for_anchor
        self._semantic_anchor_instance_index = self._geometry_assignment_tensor(
            semantic_anchor_instance_index, num_anchors, device, torch.long, -1
        )
        self._semantic_anchor_geometry_state = self._geometry_assignment_tensor(
            semantic_anchor_geometry_state, num_anchors, device, torch.uint8, 0
        )
        self._semantic_anchor_geometry_confidence = self._geometry_assignment_tensor(
            semantic_anchor_geometry_confidence, num_anchors, device, torch.float32, 0.0
        )
        self._semantic_feat = torch.empty(0, device=device)
        if offset_residual_proj_state is not None and self.offset_residual_proj is not None:
            self.offset_residual_proj.load_state_dict(offset_residual_proj_state)

    def _geometry_assignment_tensor(self, value, num_anchors, device, dtype, default):
        if value is None:
            return torch.empty(0, dtype=dtype, device=device)
        if isinstance(value, nn.Parameter):
            value = value.detach()
        tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
        if tensor.shape[0] != int(num_anchors):
            raise RuntimeError(
                f"Semantic geometry assignment expected {num_anchors} rows, got {tensor.shape[0]}."
            )
        return tensor.contiguous()

    @property
    def semantic_residual_is_sparse(self):
        return bool(self._semantic_residual_row_for_anchor.numel() == self._semantic_anchor.shape[0])

    def configure_geometry_gated_semantics(
        self,
        residual_anchor_indices,
        anchor_instance_index,
        anchor_geometry_state,
        anchor_geometry_confidence,
    ):
        if not (self.use_semantic_features and self.use_anchor_offset_semantic):
            raise RuntimeError("Geometry-gated semantic residuals require Anchor-Offset semantics.")
        device = self.get_model_device()
        num_anchors = int(self._semantic_anchor.shape[0])
        if self.semantic_residual_dim == 0:
            active = torch.empty(0, dtype=torch.long, device=device)
        elif residual_anchor_indices is None:
            active = None
        else:
            active = torch.as_tensor(residual_anchor_indices, dtype=torch.long, device=device).reshape(-1)
            if active.numel() > 0:
                active = torch.unique(active, sorted=True)
            if active.numel() > 0 and (
                int(active.min().item()) < 0 or int(active.max().item()) >= num_anchors
            ):
                raise RuntimeError("Geometry-gated residual indices are outside the anchor range.")

        current_residual = self._semantic_offset_residual.detach()
        current_gate = self._semantic_offset_gate.detach()
        active_count = num_anchors if active is None else int(active.numel())
        new_shape = (active_count, self.n_offsets, self.semantic_residual_dim)
        gate_shape = (active_count, self.n_offsets, 1)
        new_residual = torch.empty(new_shape, dtype=torch.float32, device=device)
        new_gate = torch.full(gate_shape, self.semantic_gate_init, dtype=torch.float32, device=device)
        if active_count > 0:
            new_residual.normal_(mean=0.0, std=0.01)
            if active is None and self.semantic_residual_is_sparse:
                reusable_positions = torch.nonzero(
                    self._semantic_residual_row_for_anchor >= 0,
                    as_tuple=False,
                ).squeeze(1)
                if reusable_positions.numel() > 0:
                    old_rows = self._semantic_residual_row_for_anchor[reusable_positions]
                    new_residual[reusable_positions] = current_residual[old_rows]
                    new_gate[reusable_positions] = current_gate[old_rows]
            elif active is None and current_residual.shape[0] == num_anchors:
                new_residual.copy_(current_residual)
                new_gate.copy_(current_gate)
            elif self.semantic_residual_is_sparse:
                old_rows = self._semantic_residual_row_for_anchor[active]
                reusable = old_rows >= 0
                if reusable.any():
                    new_residual[reusable] = current_residual[old_rows[reusable]]
                    new_gate[reusable] = current_gate[old_rows[reusable]]
            elif current_residual.shape[0] == num_anchors:
                new_residual.copy_(current_residual[active])
                new_gate.copy_(current_gate[active])

        self._init_anchor_offset_semantic_parameters(
            num_anchors,
            device,
            semantic_anchor=self._semantic_anchor.detach(),
            semantic_offset_residual=new_residual,
            semantic_offset_gate=new_gate,
            semantic_residual_anchor_indices=active,
            semantic_anchor_instance_index=anchor_instance_index,
            semantic_anchor_geometry_state=anchor_geometry_state,
            semantic_anchor_geometry_confidence=anchor_geometry_confidence,
        )

    def get_semantic_anchor_latent(self):
        if not self.use_semantic_features:
            return None
        if self.use_anchor_offset_semantic:
            return self._semantic_anchor
        return self._semantic_feat

    def get_semantic_child_latents(self, anchor_indices=None):
        if not self.use_semantic_features:
            return None
        selected_indices = None
        if anchor_indices is not None:
            selected_indices = torch.as_tensor(
                anchor_indices, dtype=torch.long, device=self._semantic_anchor.device
            ).reshape(-1)
        if not self.use_anchor_offset_semantic:
            anchor_latent = self._semantic_feat
            if anchor_latent is None or anchor_latent.numel() == 0:
                return anchor_latent
            if selected_indices is not None:
                anchor_latent = anchor_latent[selected_indices]
            return anchor_latent[:, None, :].expand(-1, self.n_offsets, -1)
        selected_anchor = self._semantic_anchor if selected_indices is None else self._semantic_anchor[selected_indices]
        base = selected_anchor[:, None, :]
        if self.semantic_residual_dim == 0:
            return F.normalize(base.expand(-1, self.n_offsets, -1), dim=-1, eps=1e-6)
        self._ensure_offset_residual_proj(self.get_model_device())
        if not self.semantic_residual_is_sparse:
            selected_residual = (
                self._semantic_offset_residual
                if selected_indices is None
                else self._semantic_offset_residual[selected_indices]
            )
            selected_gate = (
                self._semantic_offset_gate
                if selected_indices is None
                else self._semantic_offset_gate[selected_indices]
            )
            residual_flat = selected_residual.reshape(-1, self.semantic_residual_dim)
            residual = self.offset_residual_proj(residual_flat).view(
                selected_anchor.shape[0], self.n_offsets, self.semantic_latent_dim
            )
            return F.normalize(base + torch.sigmoid(selected_gate) * residual, dim=-1, eps=1e-6)

        if selected_indices is None:
            selected_indices = torch.arange(self._semantic_anchor.shape[0], device=base.device)
        sparse_rows = self._semantic_residual_row_for_anchor[selected_indices]
        active_positions = torch.nonzero(sparse_rows >= 0, as_tuple=False).squeeze(1)
        delta = torch.zeros(
            (selected_anchor.shape[0], self.n_offsets, self.semantic_latent_dim),
            dtype=selected_anchor.dtype,
            device=selected_anchor.device,
        )
        if active_positions.numel() > 0:
            active_rows = sparse_rows[active_positions]
            active_residual = self._semantic_offset_residual[active_rows]
            projected = self.offset_residual_proj(
                active_residual.reshape(-1, self.semantic_residual_dim)
            ).view(active_rows.shape[0], self.n_offsets, self.semantic_latent_dim)
            active_delta = torch.sigmoid(self._semantic_offset_gate[active_rows]) * projected
            delta = delta.index_copy(0, active_positions, active_delta)
        return F.normalize(base + delta, dim=-1, eps=1e-6)

    def semantic_offset_regularization(self):
        if not (self.use_semantic_features and self.use_anchor_offset_semantic):
            return None, None, {}
        if self._semantic_offset_residual.numel() == 0:
            zero = self._semantic_anchor.sum() * 0.0
            return zero, zero, {
                "semantic_anchor_shape": tuple(self._semantic_anchor.shape),
                "semantic_offset_residual_shape": tuple(self._semantic_offset_residual.shape),
                "semantic_residual_is_sparse": self.semantic_residual_is_sparse,
                "semantic_residual_anchor_ratio": 0.0,
                "gate_mean": 0.0,
                "gate_min": 0.0,
                "gate_max": 0.0,
                "residual_norm": 0.0,
            }
        self._ensure_offset_residual_proj(self.get_model_device())
        residual_flat = self._semantic_offset_residual.reshape(-1, self.semantic_residual_dim)
        residual = self.offset_residual_proj(residual_flat).view(
            self._semantic_offset_residual.shape[0],
            self.n_offsets,
            self.semantic_latent_dim,
        )
        gate = torch.sigmoid(self._semantic_offset_gate)
        residual_norm_sq = residual.pow(2).sum(dim=-1, keepdim=True)
        loss_gate = gate.mean()
        loss_residual = (gate * residual_norm_sq).mean()
        stats = {
            "semantic_anchor_shape": tuple(self._semantic_anchor.shape),
            "semantic_offset_residual_shape": tuple(self._semantic_offset_residual.shape),
            "semantic_residual_is_sparse": self.semantic_residual_is_sparse,
            "semantic_residual_anchor_ratio": float(
                self._semantic_offset_residual.shape[0] / max(self._semantic_anchor.shape[0], 1)
            ),
            "gate_mean": float(gate.mean().detach().item()),
            "gate_min": float(gate.min().detach().item()),
            "gate_max": float(gate.max().detach().item()),
            "residual_norm": float(residual.norm(dim=-1).mean().detach().item()),
        }
        return loss_gate, loss_residual, stats

    def capture(self):
        semantic_anchor_latent = self.get_semantic_anchor_latent() if self.use_semantic_features else None
        model_state = {
            "anchor": self._anchor,
            "offset": self._offset,
            "anchor_feat": self._anchor_feat,
            "semantic_feat": semantic_anchor_latent,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "max_radii2D": self.max_radii2D,
            "mlp_opacity": self.mlp_opacity.state_dict(),
            "mlp_cov": self.mlp_cov.state_dict(),
            "mlp_color": self.mlp_color.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "spatial_lr_scale": self.spatial_lr_scale,
            "use_anchor_offset_semantic": self.use_anchor_offset_semantic,
            "semantic_residual_dim": self.semantic_residual_dim,
            "semantic_gate_init": self.semantic_gate_init,
            "semantic_latent_dim": self.semantic_latent_dim,
            "semantic_feature_dim": self.semantic_feature_dim,
            "semantic_teacher_dim": self.semantic_teacher_dim,
            "uses_compressed_semantic_teacher": self.uses_compressed_semantic_teacher,
            "semantic_codec_metadata": self.semantic_codec_metadata,
        }
        if self.use_feat_bank:
            model_state["mlp_feature_bank"] = self.mlp_feature_bank.state_dict()
        if self.appearance_dim > 0 and self.embedding_appearance is not None:
            model_state["embedding_appearance"] = self.embedding_appearance.state_dict()
        if self.semantic_decoder is not None:
            model_state["semantic_decoder"] = self.semantic_decoder.state_dict()
        if self.use_semantic_features and self.use_anchor_offset_semantic:
            model_state["semantic_anchor"] = self._semantic_anchor
            model_state["semantic_offset_residual"] = self._semantic_offset_residual
            model_state["semantic_offset_gate"] = self._semantic_offset_gate
            model_state["semantic_residual_is_sparse"] = self.semantic_residual_is_sparse
            if self.semantic_residual_is_sparse:
                model_state["semantic_residual_anchor_indices"] = self._semantic_residual_anchor_indices
            if self._semantic_anchor_instance_index.numel() == self._semantic_anchor.shape[0]:
                model_state["semantic_anchor_instance_index"] = self._semantic_anchor_instance_index
                model_state["semantic_anchor_geometry_state"] = self._semantic_anchor_geometry_state
                model_state["semantic_anchor_geometry_confidence"] = self._semantic_anchor_geometry_confidence
            if self.offset_residual_proj is not None:
                model_state["offset_residual_proj"] = self.offset_residual_proj.state_dict()
        return model_state
    
    def restore(self, model_args, training_args):
        if not isinstance(model_args, dict):
            raise RuntimeError("Unsupported legacy checkpoint format for this semantic Scaffold-GS build.")
        device = next(self.mlp_opacity.parameters()).device
        checkpoint_latent_dim = int(model_args.get("semantic_latent_dim", self.semantic_latent_dim))
        checkpoint_feature_dim = int(model_args.get("semantic_feature_dim", self.semantic_feature_dim))
        checkpoint_teacher_dim = int(model_args.get("semantic_teacher_dim", checkpoint_feature_dim))
        if checkpoint_latent_dim != int(self.semantic_latent_dim):
            raise RuntimeError(
                f"Checkpoint semantic latent dim {checkpoint_latent_dim} != configured {self.semantic_latent_dim}."
            )
        if checkpoint_feature_dim != int(self.semantic_feature_dim):
            raise RuntimeError(
                f"Checkpoint semantic query dim {checkpoint_feature_dim} != configured {self.semantic_feature_dim}."
            )
        if checkpoint_teacher_dim != int(self.semantic_teacher_dim):
            raise RuntimeError(
                f"Checkpoint semantic teacher dim {checkpoint_teacher_dim} != configured {self.semantic_teacher_dim}."
            )
        if bool(model_args.get("use_anchor_offset_semantic", False)):
            self.use_anchor_offset_semantic = True
            self.semantic_residual_dim = int(model_args.get("semantic_residual_dim", self.semantic_residual_dim))
            self.semantic_gate_init = float(model_args.get("semantic_gate_init", self.semantic_gate_init))
        self._anchor = self._to_parameter(model_args["anchor"], device, True)
        self._offset = self._to_parameter(model_args["offset"], device, True)
        self._anchor_feat = self._to_parameter(model_args["anchor_feat"], device, True)
        semantic_feat = model_args.get("semantic_feat", None)
        if self.use_semantic_features and self.use_anchor_offset_semantic:
            semantic_anchor = model_args.get("semantic_anchor", None)
            if semantic_anchor is None:
                semantic_anchor = semantic_feat
            self._init_anchor_offset_semantic_parameters(
                self._anchor.shape[0],
                device,
                semantic_anchor=semantic_anchor,
                semantic_offset_residual=model_args.get("semantic_offset_residual", None),
                semantic_offset_gate=model_args.get("semantic_offset_gate", None),
                offset_residual_proj_state=model_args.get("offset_residual_proj", None),
                semantic_residual_anchor_indices=model_args.get("semantic_residual_anchor_indices", None),
                semantic_anchor_instance_index=model_args.get("semantic_anchor_instance_index", None),
                semantic_anchor_geometry_state=model_args.get("semantic_anchor_geometry_state", None),
                semantic_anchor_geometry_confidence=model_args.get("semantic_anchor_geometry_confidence", None),
            )
        else:
            if self.use_semantic_features and semantic_feat is None:
                semantic_feat = torch.zeros((self._anchor.shape[0], self.semantic_latent_dim), device=device)
            self._semantic_feat = (
                self._to_parameter(semantic_feat, device, True)
                if semantic_feat is not None
                else torch.empty(0, device=device)
            )
            self._semantic_anchor = torch.empty(0, device=device)
            self._semantic_offset_residual = torch.empty(0, device=device)
            self._semantic_offset_gate = torch.empty(0, device=device)
            self._semantic_residual_anchor_indices = torch.empty(0, dtype=torch.long, device=device)
            self._semantic_residual_row_for_anchor = torch.empty(0, dtype=torch.long, device=device)
            self._semantic_anchor_instance_index = torch.empty(0, dtype=torch.long, device=device)
            self._semantic_anchor_geometry_state = torch.empty(0, dtype=torch.uint8, device=device)
            self._semantic_anchor_geometry_confidence = torch.empty(0, device=device)
        self._scaling = self._to_parameter(model_args["scaling"], device, True)
        self._rotation = self._to_parameter(model_args["rotation"], device, False)
        self._opacity = self._to_parameter(model_args["opacity"], device, False)
        self.max_radii2D = self._to_device_tensor(model_args["max_radii2D"], device)
        self.spatial_lr_scale = model_args["spatial_lr_scale"]
        self.mlp_opacity.load_state_dict(model_args["mlp_opacity"])
        self.mlp_cov.load_state_dict(model_args["mlp_cov"])
        self.mlp_color.load_state_dict(model_args["mlp_color"])
        if self.use_feat_bank and "mlp_feature_bank" in model_args:
            self.mlp_feature_bank.load_state_dict(model_args["mlp_feature_bank"])
        if self.appearance_dim > 0 and self.embedding_appearance is not None and "embedding_appearance" in model_args:
            self.embedding_appearance.load_state_dict(model_args["embedding_appearance"])
        if self.semantic_decoder is not None and "semantic_decoder" in model_args:
            self.semantic_decoder.load_state_dict(model_args["semantic_decoder"])
        self.training_setup(training_args)
        if model_args.get("optimizer", None) is not None:
            try:
                self.optimizer.load_state_dict(model_args["optimizer"])
                self._move_optimizer_state_to_device(self.optimizer, device)
            except (ValueError, RuntimeError) as exc:
                print(
                    "[Semantic Restore] Skipping optimizer state because checkpoint optimizer "
                    f"groups do not match the current model: {exc}"
                )

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).to(self.get_model_device())

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_semantic_decoder(self):
        return self.semantic_decoder

    def decode_semantic_features(self, semantic_latent=None):
        if not self.use_semantic_features or self.semantic_decoder is None:
            return None
        if semantic_latent is None:
            semantic_latent = self.get_semantic_anchor_latent()
        if semantic_latent is None or semantic_latent.numel() == 0:
            return semantic_latent
        if self.uses_compressed_semantic_teacher:
            semantic_latent = F.normalize(semantic_latent, dim=-1, eps=1e-6)
        return F.normalize(self.semantic_decoder(semantic_latent), dim=-1, eps=1e-6)

    def semantic_training_features(self, semantic_latent=None):
        if not self.use_semantic_features:
            return None
        if semantic_latent is None:
            semantic_latent = self.get_semantic_anchor_latent()
        if semantic_latent is None or semantic_latent.numel() == 0:
            return semantic_latent
        if self.uses_compressed_semantic_teacher:
            return F.normalize(semantic_latent, dim=-1, eps=1e-6)
        return self.decode_semantic_features(semantic_latent)

    def semantic_training_feature_map(self, semantic_feature_map):
        if not self.use_semantic_features:
            return None
        if semantic_feature_map is None or semantic_feature_map.numel() == 0:
            return semantic_feature_map
        if semantic_feature_map.ndim != 3 or semantic_feature_map.shape[0] != self.semantic_latent_dim:
            raise RuntimeError(
                f"Expected semantic feature map with shape [{self.semantic_latent_dim}, H, W], "
                f"got {tuple(semantic_feature_map.shape)}."
            )
        if self.uses_compressed_semantic_teacher:
            return F.normalize(semantic_feature_map, dim=0, eps=1e-6)
        return self.decode_semantic_feature_map(semantic_feature_map)

    def decode_semantic_feature_map(self, semantic_feature_map):
        if not self.use_semantic_features or self.semantic_decoder is None:
            return None
        if semantic_feature_map is None or semantic_feature_map.numel() == 0:
            return semantic_feature_map
        if semantic_feature_map.ndim != 3 or semantic_feature_map.shape[0] != self.semantic_latent_dim:
            raise RuntimeError(
                f"Expected semantic feature map with shape [{self.semantic_latent_dim}, H, W], "
                f"got {tuple(semantic_feature_map.shape)}."
            )

        height, width = semantic_feature_map.shape[-2:]
        latent = semantic_feature_map.permute(1, 2, 0).reshape(-1, self.semantic_latent_dim)
        if self.uses_compressed_semantic_teacher:
            latent = F.normalize(latent, dim=-1, eps=1e-6)
        decoded = self.semantic_decoder(latent)
        decoded = decoded.view(height, width, self.semantic_feature_dim).permute(2, 0, 1).contiguous()
        return F.normalize(decoded, dim=0, eps=1e-6)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        self._clear_cuda_cache_if_needed(new_anchor.device)
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]
        device = self.get_model_device()

        if self.voxel_size <= 0:
            init_points = torch.tensor(points, dtype=torch.float32, device=device)
            init_dist = distCUDA2(init_points).float().to(device)
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            self._clear_cuda_cache_if_needed(device)

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points), dtype=torch.float32, device=device)
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3), dtype=torch.float32, device=device)
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim), dtype=torch.float32, device=device)
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().to(device), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device=device))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        if self.use_semantic_features:
            if self.use_anchor_offset_semantic:
                self._init_anchor_offset_semantic_parameters(fused_point_cloud.shape[0], device)
            else:
                semantic_feat = torch.zeros(
                    (fused_point_cloud.shape[0], self.semantic_latent_dim),
                    dtype=torch.float32,
                    device=device,
                )
                self._semantic_feat = nn.Parameter(semantic_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device=device)


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        device = self.get_anchor.device if self.get_anchor.numel() > 0 else self.get_model_device()

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device=device)

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device=device)
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device=device)
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device=device)

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        if self.use_semantic_features:
            if self.use_anchor_offset_semantic:
                l.append(
                    {'params': [self._semantic_anchor], 'lr': training_args.semantic_lr, "name": "semantic_anchor"}
                )
                if self.semantic_residual_dim > 0:
                    self._ensure_offset_residual_proj(device)
                    l.extend([
                        {'params': [self._semantic_offset_residual], 'lr': training_args.semantic_lr, "name": "semantic_offset_residual"},
                        {'params': [self._semantic_offset_gate], 'lr': training_args.semantic_lr, "name": "semantic_offset_gate"},
                        {'params': self.offset_residual_proj.parameters(), 'lr': training_args.semantic_lr, "name": "offset_residual_proj"},
                    ])
                if self.semantic_decoder_trainable:
                    l.append(
                        {'params': self.semantic_decoder.parameters(), 'lr': training_args.semantic_decoder_lr, "name": "semantic_decoder"}
                    )
            else:
                l.extend([
                    {'params': [self._semantic_feat], 'lr': training_args.semantic_lr, "name": "semantic_feat"},
                ])
                if self.semantic_decoder_trainable:
                    l.append(
                        {'params': self.semantic_decoder.parameters(), 'lr': training_args.semantic_decoder_lr, "name": "semantic_decoder"}
                    )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
        if self.use_semantic_features:
            self.semantic_scheduler_args = get_expon_lr_func(
                lr_init=training_args.semantic_lr,
                lr_final=max(training_args.semantic_lr * 0.1, 1e-6),
                lr_delay_mult=0.01,
                max_steps=training_args.iterations,
            )
            if self.semantic_decoder_trainable:
                self.semantic_decoder_scheduler_args = get_expon_lr_func(
                    lr_init=training_args.semantic_decoder_lr,
                    lr_final=max(training_args.semantic_decoder_lr * 0.1, 1e-6),
                    lr_delay_mult=0.01,
                    max_steps=training_args.iterations,
                )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_semantic_features and param_group["name"] in {
                "semantic_feat",
                "semantic_anchor",
                "semantic_offset_residual",
                "semantic_offset_gate",
                "offset_residual_proj",
            }:
                lr = self.semantic_scheduler_args(iteration)
                param_group['lr'] = lr
            if (
                self.use_semantic_features
                and self.semantic_decoder_trainable
                and param_group["name"] == "semantic_decoder"
            ):
                lr = self.semantic_decoder_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self, include_semantic=True):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        semantic_latent = self.get_semantic_anchor_latent()
        if include_semantic and self.use_semantic_features and semantic_latent is not None and semantic_latent.numel() > 0:
            for i in range(semantic_latent.shape[1]):
                l.append('f_semantic_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def _anchor_offset_semantic_sidecar_path(self, ply_path):
        return os.path.join(os.path.dirname(ply_path), "semantic_anchor_offset.pt")

    def _semantic_storage_value(self, value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if self.semantic_storage_dtype == "float16" and value.is_floating_point():
                value = value.to(dtype=torch.float16)
            return value
        if isinstance(value, dict):
            return value.__class__(
                (key, self._semantic_storage_value(item)) for key, item in value.items()
            )
        if isinstance(value, list):
            return [self._semantic_storage_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._semantic_storage_value(item) for item in value)
        return value

    def _save_anchor_offset_semantic_sidecar(self, ply_path):
        if not self.use_semantic_features:
            return
        sidecar_path = self._anchor_offset_semantic_sidecar_path(ply_path)
        if not self.use_anchor_offset_semantic:
            if self.semantic_storage_dtype != "float16":
                return
            semantic_anchor = self.get_semantic_anchor_latent()
            if semantic_anchor is None or semantic_anchor.numel() == 0:
                raise RuntimeError("FP16 semantic storage requires a non-empty semantic anchor tensor.")
            torch.save(
                {
                    "version": 1,
                    "use_anchor_offset_semantic": False,
                    "storage_dtype": "float16",
                    "compute_dtype": "float32",
                    "storage_format": "legacy_semantic_sidecar_v1",
                    "semantic_latent_dim": self.semantic_latent_dim,
                    "semantic_feature_dim": self.semantic_feature_dim,
                    "semantic_teacher_dim": self.semantic_teacher_dim,
                    "uses_compressed_semantic_teacher": self.uses_compressed_semantic_teacher,
                    "semantic_anchor": self._semantic_storage_value(semantic_anchor),
                },
                sidecar_path,
            )
            return
        payload = {
            "version": 3,
            "use_anchor_offset_semantic": True,
            "storage_dtype": self.semantic_storage_dtype,
            "compute_dtype": "float32",
            "storage_format": "semantic_sidecar_v2",
            "semantic_latent_dim": self.semantic_latent_dim,
            "semantic_feature_dim": self.semantic_feature_dim,
            "semantic_teacher_dim": self.semantic_teacher_dim,
            "uses_compressed_semantic_teacher": self.uses_compressed_semantic_teacher,
            "semantic_residual_dim": self.semantic_residual_dim,
            "semantic_gate_init": self.semantic_gate_init,
            "n_offsets": self.n_offsets,
            "semantic_offset_residual": self._semantic_storage_value(self._semantic_offset_residual),
            "semantic_offset_gate": self._semantic_storage_value(self._semantic_offset_gate),
            "semantic_residual_is_sparse": self.semantic_residual_is_sparse,
            "semantic_residual_anchor_indices": (
                self._semantic_residual_anchor_indices.detach().to(dtype=torch.int32, device="cpu")
                if self.semantic_residual_is_sparse
                else None
            ),
            "offset_residual_proj": (
                self._semantic_storage_value(self.offset_residual_proj.state_dict())
                if self.offset_residual_proj is not None
                else None
            ),
        }
        if self.semantic_storage_dtype == "float16":
            semantic_anchor = self.get_semantic_anchor_latent()
            if semantic_anchor is None or semantic_anchor.numel() == 0:
                raise RuntimeError("FP16 semantic storage requires a non-empty semantic anchor tensor.")
            payload["semantic_anchor"] = self._semantic_storage_value(semantic_anchor)
        torch.save(payload, sidecar_path)

    def _load_anchor_offset_semantic_sidecar(self, ply_path):
        sidecar_path = self._anchor_offset_semantic_sidecar_path(ply_path)
        if not os.path.exists(sidecar_path):
            return None
        payload = self._torch_load_robust(sidecar_path, map_location=self.get_model_device())
        if isinstance(payload, dict) and payload.get("storage_dtype") in {"float16", "bfloat16"}:
            payload = self._restore_fp32_compute_tensors(payload)
        return payload

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        semantic_in_sidecar = (
            self.semantic_storage_dtype == "float16"
            and self.use_semantic_features
        )

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        semantic_feat = None
        semantic_latent = self.get_semantic_anchor_latent()
        if self.use_semantic_features and semantic_latent is not None and semantic_latent.numel() > 0:
            semantic_feat = semantic_latent.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, 'f4')
            for attribute in self.construct_list_of_attributes(
                include_semantic=not semantic_in_sidecar
            )
        ]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes_list = [anchor, normals, offset, anchor_feat]
        if semantic_feat is not None and not semantic_in_sidecar:
            attributes_list.append(semantic_feat)
        attributes_list.extend([opacities, scale, rotation])
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        self._save_anchor_offset_semantic_sidecar(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        semantic_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_semantic_feat")]
        semantic_feat_names = sorted(semantic_feat_names, key = lambda x: int(x.split('_')[-1]))
        semantic_feats = np.zeros((anchor.shape[0], self.semantic_latent_dim))
        if len(semantic_feat_names) > 0:
            semantic_feats = np.zeros((anchor.shape[0], len(semantic_feat_names)))
            for idx, attr_name in enumerate(semantic_feat_names):
                semantic_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        if semantic_feats.shape[1] != self.semantic_latent_dim:
            if len(semantic_feat_names) == 0:
                semantic_feats = np.zeros((anchor.shape[0], self.semantic_latent_dim))
            else:
                raise RuntimeError(
                    f"Loaded semantic latent dim {semantic_feats.shape[1]} does not match configured dim {self.semantic_latent_dim}."
                )

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        device = self.get_model_device()
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float32, device=device).requires_grad_(True))
        if self.use_semantic_features:
            semantic_feats_tensor = torch.tensor(semantic_feats, dtype=torch.float32, device=device)
            if self.use_anchor_offset_semantic:
                sidecar = self._load_anchor_offset_semantic_sidecar(path)
                semantic_anchor = semantic_feats_tensor if len(semantic_feat_names) > 0 else None
                semantic_offset_residual = None
                semantic_offset_gate = None
                offset_residual_proj_state = None
                semantic_residual_anchor_indices = None
                semantic_anchor_instance_index = None
                semantic_anchor_geometry_state = None
                semantic_anchor_geometry_confidence = None
                if sidecar is not None:
                    semantic_anchor = sidecar.get("semantic_anchor", semantic_anchor)
                    semantic_offset_residual = sidecar.get("semantic_offset_residual", None)
                    semantic_offset_gate = sidecar.get("semantic_offset_gate", None)
                    offset_residual_proj_state = sidecar.get("offset_residual_proj", None)
                    semantic_residual_anchor_indices = sidecar.get("semantic_residual_anchor_indices", None)
                    semantic_anchor_instance_index = sidecar.get("semantic_anchor_instance_index", None)
                    semantic_anchor_geometry_state = sidecar.get("semantic_anchor_geometry_state", None)
                    semantic_anchor_geometry_confidence = sidecar.get("semantic_anchor_geometry_confidence", None)
                self._init_anchor_offset_semantic_parameters(
                    anchor.shape[0],
                    device,
                    semantic_anchor=semantic_anchor,
                    semantic_offset_residual=semantic_offset_residual,
                    semantic_offset_gate=semantic_offset_gate,
                    offset_residual_proj_state=offset_residual_proj_state,
                    semantic_residual_anchor_indices=semantic_residual_anchor_indices,
                    semantic_anchor_instance_index=semantic_anchor_instance_index,
                    semantic_anchor_geometry_state=semantic_anchor_geometry_state,
                    semantic_anchor_geometry_confidence=semantic_anchor_geometry_confidence,
                )
            else:
                sidecar = self._load_anchor_offset_semantic_sidecar(path)
                if sidecar is not None:
                    if bool(sidecar.get("use_anchor_offset_semantic", False)):
                        raise RuntimeError(
                            "Loaded an Anchor-Offset semantic sidecar for a legacy shared-Anchor model."
                        )
                    sidecar_semantic = sidecar.get("semantic_anchor", None)
                    if sidecar_semantic is None:
                        raise RuntimeError(
                            f"Legacy FP16 semantic sidecar is missing semantic_anchor: {path}"
                        )
                    semantic_feats_tensor = torch.as_tensor(
                        sidecar_semantic, dtype=torch.float32, device=device
                    )
                    expected_shape = (anchor.shape[0], self.semantic_latent_dim)
                    if tuple(semantic_feats_tensor.shape) != expected_shape:
                        raise RuntimeError(
                            "Legacy FP16 semantic sidecar shape mismatch: "
                            f"expected {expected_shape}, got {tuple(semantic_feats_tensor.shape)}."
                        )
                self._semantic_feat = nn.Parameter(semantic_feats_tensor.requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float32, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float32, device=device).requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device=device).requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'decoder' in group['name'] or \
                'semantic' in group['name'] or \
                'offset_residual_proj' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'decoder' in group['name'] or \
                'semantic' in group['name'] or \
                'offset_residual_proj' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        if self.use_semantic_features and not self.use_anchor_offset_semantic:
            self._semantic_feat = optimizable_tensors["semantic_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device=candidate_mask.device)], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float32, device=candidate_anchor.device))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                if self.use_semantic_features and not self.use_anchor_offset_semantic:
                    new_semantic_feat = self._semantic_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.semantic_latent_dim])[candidate_mask]
                    # Semantic latents are not ordinary activations; per-dimension max can splice
                    # unrelated anchors into a mixed vector, so average contributors per new voxel.
                    new_semantic_feat = scatter_mean_2d(new_semantic_feat, inverse_indices, selected_grid_coords_unique.shape[0])[remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                if self.use_semantic_features and not self.use_anchor_offset_semantic:
                    d["semantic_feat"] = new_semantic_feat
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device=self.anchor_demon.device).float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device=self.opacity_accum.device).float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                self._clear_cuda_cache_if_needed(candidate_anchor.device)
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                if self.use_semantic_features and not self.use_anchor_offset_semantic:
                    self._semantic_feat = optimizable_tensors["semantic_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device=self.opacity_accum.device).float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device=self.anchor_demon.device).float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device=self.get_anchor.device)

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        device = self.get_model_device()
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim, device=device),))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim, device=device),))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim, device=device),))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1, device=device),))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long, device=device),))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

            if self.semantic_decoder is not None:
                decoder_was_training = self.semantic_decoder.training
                self.semantic_decoder.eval()
                semantic_decoder = torch.jit.trace(self.semantic_decoder, (torch.rand(1, self.semantic_latent_dim, device=device),))
                semantic_decoder.save(os.path.join(path, 'semantic_decoder.pt'))
                if decoder_was_training and self.semantic_decoder_trainable:
                    self.semantic_decoder.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                checkpoint = {
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }
                if self.semantic_decoder is not None:
                    checkpoint['semantic_decoder'] = self.semantic_decoder.state_dict()
                if self.offset_residual_proj is not None:
                    checkpoint['offset_residual_proj'] = self.offset_residual_proj.state_dict()
                torch.save(checkpoint, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                checkpoint = {
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }
                if self.semantic_decoder is not None:
                    checkpoint['semantic_decoder'] = self.semantic_decoder.state_dict()
                if self.offset_residual_proj is not None:
                    checkpoint['offset_residual_proj'] = self.offset_residual_proj.state_dict()
                torch.save(checkpoint, os.path.join(path, 'checkpoints.pth'))
            else:
                checkpoint = {
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }
                if self.semantic_decoder is not None:
                    checkpoint['semantic_decoder'] = self.semantic_decoder.state_dict()
                if self.offset_residual_proj is not None:
                    checkpoint['offset_residual_proj'] = self.offset_residual_proj.state_dict()
                torch.save(checkpoint, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        device = self.get_model_device()
        if mode == 'split':
            self.mlp_opacity = self._jit_load_robust(os.path.join(path, 'opacity_mlp.pt')).to(device)
            self.mlp_cov = self._jit_load_robust(os.path.join(path, 'cov_mlp.pt')).to(device)
            self.mlp_color = self._jit_load_robust(os.path.join(path, 'color_mlp.pt')).to(device)
            if self.use_feat_bank:
                self.mlp_feature_bank = self._jit_load_robust(os.path.join(path, 'feature_bank_mlp.pt')).to(device)
            if self.appearance_dim > 0:
                self.embedding_appearance = self._jit_load_robust(os.path.join(path, 'embedding_appearance.pt')).to(device)
            semantic_decoder_path = os.path.join(path, 'semantic_decoder.pt')
            if self.semantic_decoder is not None and os.path.exists(semantic_decoder_path):
                self.semantic_decoder = self._jit_load_robust(
                    semantic_decoder_path, map_location="cpu"
                ).float().to(device)
                if not self.semantic_decoder_trainable:
                    self.semantic_decoder.eval()
                    for parameter in self.semantic_decoder.parameters():
                        parameter.requires_grad_(False)
        elif mode == 'unite':
            checkpoint = self._torch_load_robust(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
            if self.semantic_decoder is not None and 'semantic_decoder' in checkpoint:
                self.semantic_decoder.load_state_dict(checkpoint['semantic_decoder'])
            if self.offset_residual_proj is not None and 'offset_residual_proj' in checkpoint:
                self.offset_residual_proj.load_state_dict(checkpoint['offset_residual_proj'])
        else:
            raise NotImplementedError
