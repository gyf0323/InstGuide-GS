from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except ImportError:
    class _ProgressFallback:
        def __init__(self, iterable, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **_kwargs):
            return None

    def tqdm(iterable, **kwargs):
        return _ProgressFallback(iterable, **kwargs)


CODEC_FORMAT_VERSION = 1


def _normalize_dims(values: Iterable[int]) -> Tuple[int, ...]:
    dims = tuple(int(value) for value in values)
    if any(value <= 0 for value in dims):
        raise ValueError("Codec hidden dimensions must be positive, got {}.".format(dims))
    return dims


def _make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dims = [int(input_dim), *_normalize_dims(hidden_dims), int(output_dim)]
    layers: List[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index + 1 < len(dims) - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class SemanticFeatureCodec(nn.Module):
    """Scaffold-GS-compatible normalized autoencoder for CLIP-family features."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden_dims: Sequence[int] = (256, 128),
        decoder_hidden_dims: Sequence[int] = (128, 256),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden_dims = _normalize_dims(encoder_hidden_dims)
        self.decoder_hidden_dims = _normalize_dims(decoder_hidden_dims)
        if self.input_dim <= 0 or self.latent_dim <= 0:
            raise ValueError(
                "Codec dimensions must be positive, got input={}, latent={}.".format(
                    self.input_dim,
                    self.latent_dim,
                )
            )
        self.encoder = _make_mlp(self.input_dim, self.encoder_hidden_dims, self.latent_dim)
        self.decoder = _make_mlp(self.latent_dim, self.decoder_hidden_dims, self.input_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1, eps=1e-6)
        return F.normalize(self.encoder(features), dim=-1, eps=1e-6)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = F.normalize(latents, dim=-1, eps=1e-6)
        return F.normalize(self.decoder(latents), dim=-1, eps=1e-6)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(features)
        return latents, self.decode(latents)

    def metadata(self) -> Dict[str, object]:
        return {
            "format_version": CODEC_FORMAT_VERSION,
            "codec_type": "mlp",
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "encoder_hidden_dims": list(self.encoder_hidden_dims),
            "decoder_hidden_dims": list(self.decoder_hidden_dims),
            "normalization": "l2",
        }


class SceneFeatureCodec(nn.Module):
    """Scene-wise normalized language-feature autoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden_dims: Sequence[int] = (256, 128, 64),
        decoder_hidden_dims: Sequence[int] = (64, 128, 256, 256),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden_dims = _normalize_dims(encoder_hidden_dims)
        self.decoder_hidden_dims = _normalize_dims(decoder_hidden_dims)
        if self.input_dim <= 0 or self.latent_dim <= 0:
            raise ValueError(
                "Codec dimensions must be positive, got input={}, latent={}.".format(
                    self.input_dim,
                    self.latent_dim,
                )
            )

        encoder_dims = [self.input_dim, *self.encoder_hidden_dims, self.latent_dim]
        encoder_layers: List[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(encoder_dims[:-1], encoder_dims[1:])):
            if index > 0:
                encoder_layers.append(nn.BatchNorm1d(in_dim))
                encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Linear(in_dim, out_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_dims = [self.latent_dim, *self.decoder_hidden_dims, self.input_dim]
        decoder_layers: List[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(decoder_dims[:-1], decoder_dims[1:])):
            if index > 0:
                decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(in_dim, out_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1, eps=1e-6)
        return F.normalize(self.encoder(features), dim=-1, eps=1e-6)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = F.normalize(latents, dim=-1, eps=1e-6)
        return F.normalize(self.decoder(latents), dim=-1, eps=1e-6)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(features)
        return latents, self.decode(latents)

    def metadata(self) -> Dict[str, object]:
        return {
            "format_version": CODEC_FORMAT_VERSION,
            "codec_type": "langsplat",
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "encoder_hidden_dims": list(self.encoder_hidden_dims),
            "decoder_hidden_dims": list(self.decoder_hidden_dims),
            "normalization": "l2",
        }


def save_semantic_codec(path: Path, codec: nn.Module, training_metadata: Optional[dict] = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": codec.metadata(),
        "state_dict": codec.state_dict(),
        "training_metadata": dict(training_metadata or {}),
    }
    torch.save(payload, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": payload["metadata"],
                "training_metadata": payload["training_metadata"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def load_semantic_codec(path: Path, map_location: torch.device) -> Tuple[SemanticFeatureCodec, dict]:
    codec_path = Path(path)
    if not codec_path.is_file():
        raise FileNotFoundError("Semantic codec checkpoint not found: {}".format(codec_path))
    try:
        payload = torch.load(codec_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(codec_path, map_location=map_location)
    if not isinstance(payload, dict) or "metadata" not in payload or "state_dict" not in payload:
        raise RuntimeError("Invalid semantic codec checkpoint: {}".format(codec_path))
    metadata = dict(payload["metadata"])
    version = int(metadata.get("format_version", 0))
    if version != CODEC_FORMAT_VERSION:
        raise RuntimeError(
            "Unsupported semantic codec format version {}; expected {}.".format(
                version,
                CODEC_FORMAT_VERSION,
            )
        )
    codec_type = str(metadata.get("codec_type", "mlp"))
    if codec_type == "langsplat":
        codec = SceneFeatureCodec(
            input_dim=int(metadata["input_dim"]),
            latent_dim=int(metadata["latent_dim"]),
            encoder_hidden_dims=metadata.get("encoder_hidden_dims", (256, 128, 64)),
            decoder_hidden_dims=metadata.get("decoder_hidden_dims", (64, 128, 256, 256)),
        )
    elif codec_type == "mlp":
        codec = SemanticFeatureCodec(
            input_dim=int(metadata["input_dim"]),
            latent_dim=int(metadata["latent_dim"]),
            encoder_hidden_dims=metadata.get("encoder_hidden_dims", (256, 128)),
            decoder_hidden_dims=metadata.get("decoder_hidden_dims", (128, 256)),
        )
    else:
        raise RuntimeError("Unsupported semantic codec type: {}".format(codec_type))
    codec.load_state_dict(payload["state_dict"], strict=True)
    return codec, payload


@dataclass
class TeacherFeatureFile:
    feature_path: Path
    segmentation_path: Path
    weight_path: Optional[Path]
    features: np.ndarray
    weights: np.ndarray


def parse_dims(value: str) -> Tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return ()
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def discover_teacher_files(input_dir: Path) -> List[TeacherFeatureFile]:
    files: List[TeacherFeatureFile] = []
    for feature_path in sorted(input_dir.glob("*_f.npy")):
        stem = feature_path.name[:-6]
        segmentation_path = input_dir / "{}_s.npy".format(stem)
        weight_path = input_dir / "{}_w.npy".format(stem)
        if not segmentation_path.is_file():
            raise FileNotFoundError("Missing segmentation map for {}: {}".format(feature_path.name, segmentation_path))

        features = np.asarray(np.load(feature_path), dtype=np.float32)
        if features.ndim != 2 or features.shape[1] == 0:
            raise RuntimeError("Expected [R,C] features in {}, got {}.".format(feature_path, features.shape))

        segmentation = np.asarray(np.load(segmentation_path)).squeeze()
        if segmentation.ndim != 2:
            raise RuntimeError("Expected [H,W] segmentation in {}, got {}.".format(segmentation_path, segmentation.shape))

        if weight_path.is_file():
            pixel_weights = np.asarray(np.load(weight_path), dtype=np.float32).squeeze()
            if pixel_weights.shape != segmentation.shape:
                raise RuntimeError(
                    "Weight/segmentation shape mismatch for {}: {} vs {}.".format(
                        stem,
                        pixel_weights.shape,
                        segmentation.shape,
                    )
                )
        else:
            pixel_weights = np.ones(segmentation.shape, dtype=np.float32)
            weight_path = None

        region_weights = np.ones((features.shape[0],), dtype=np.float32)
        for region_id in range(features.shape[0]):
            selected = segmentation == region_id
            if np.any(selected):
                region_weights[region_id] = float(np.mean(pixel_weights[selected]))
        region_weights = np.clip(region_weights, 1e-3, None)

        files.append(
            TeacherFeatureFile(
                feature_path=feature_path,
                segmentation_path=segmentation_path,
                weight_path=weight_path,
                features=features,
                weights=region_weights,
            )
        )

    if not files:
        raise FileNotFoundError("No *_f.npy teacher features found under {}.".format(input_dir))
    input_dims = {int(item.features.shape[1]) for item in files}
    if len(input_dims) != 1:
        raise RuntimeError("Teacher feature dimensions are inconsistent: {}.".format(sorted(input_dims)))
    if sum(int(item.features.shape[0]) for item in files) == 0:
        raise RuntimeError("Teacher contains no region features under {}.".format(input_dir))
    return files


def split_files(files: List[TeacherFeatureFile], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(len(files)))
    random.Random(seed).shuffle(indices)
    if len(indices) <= 1 or val_fraction <= 0:
        return indices, []
    val_count = max(1, int(round(len(indices) * val_fraction)))
    val_count = min(val_count, len(indices) - 1)
    return indices[val_count:], indices[:val_count]


def deduplicate_feature_tensor(
    features: torch.Tensor,
    weights: torch.Tensor,
    decimals: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    if features.shape[0] <= 1:
        return features, weights, {"before": int(features.shape[0]), "after": int(features.shape[0])}

    feature_array = features.cpu().numpy().astype(np.float32, copy=False)
    weight_array = weights.cpu().numpy().astype(np.float32, copy=False)
    rounded = np.round(feature_array, int(decimals))
    unique_rounded, inverse = np.unique(rounded, axis=0, return_inverse=True)
    if unique_rounded.shape[0] == feature_array.shape[0]:
        return features, weights, {"before": int(features.shape[0]), "after": int(features.shape[0])}

    grouped_weights = np.bincount(inverse, weights=weight_array).astype(np.float32)
    grouped_features = np.zeros((unique_rounded.shape[0], feature_array.shape[1]), dtype=np.float32)
    np.add.at(grouped_features, inverse, feature_array * weight_array[:, None])
    grouped_features = grouped_features / np.maximum(grouped_weights[:, None], 1e-8)
    grouped_features = grouped_features / np.maximum(
        np.linalg.norm(grouped_features, axis=1, keepdims=True),
        1e-8,
    )
    grouped_weights = np.clip(grouped_weights, 1e-3, None)
    grouped_weights = grouped_weights / np.maximum(float(grouped_weights.mean()), 1e-8)
    return (
        torch.from_numpy(grouped_features.astype(np.float32, copy=False)),
        torch.from_numpy(grouped_weights.astype(np.float32, copy=False)),
        {"before": int(features.shape[0]), "after": int(grouped_features.shape[0])},
    )


def stack_files(
    files: List[TeacherFeatureFile],
    indices: Sequence[int],
    deduplicate: bool,
    deduplicate_decimals: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    features = np.concatenate([files[index].features for index in indices], axis=0)
    weights = np.concatenate([files[index].weights for index in indices], axis=0)
    feature_tensor = F.normalize(torch.from_numpy(features), dim=-1, eps=1e-6)
    weight_tensor = torch.from_numpy(weights).float()
    if weight_tensor.numel() > 0:
        weight_tensor = weight_tensor / weight_tensor.mean().clamp_min(1e-6)
    stats = {"before": int(feature_tensor.shape[0]), "after": int(feature_tensor.shape[0])}
    if deduplicate:
        feature_tensor, weight_tensor, stats = deduplicate_feature_tensor(
            feature_tensor,
            weight_tensor,
            int(deduplicate_decimals),
        )
    return feature_tensor, weight_tensor, stats


def codec_loss(
    codec: nn.Module,
    features: torch.Tensor,
    weights: torch.Tensor,
    loss_type: str,
    langsplat_cosine_weight: float,
    similarity_weight: float,
    decoded_similarity_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    pair_sample_count: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    latents, reconstructed = codec(features)
    cosine = 1.0 - (reconstructed * features).sum(dim=-1).clamp(-1.0, 1.0)
    mse = (reconstructed - features).pow(2).mean(dim=-1)
    if loss_type == "langsplat":
        reconstruction = ((mse + float(langsplat_cosine_weight) * cosine) * weights).sum() / weights.sum().clamp_min(1e-6)
        return reconstruction, {
            "l2_loss": float((mse * weights).sum().detach().item() / weights.sum().clamp_min(1e-6).item()),
            "cosine_loss": float((cosine * weights).sum().detach().item() / weights.sum().clamp_min(1e-6).item()),
            "latent_similarity_loss": 0.0,
            "decoded_similarity_loss": 0.0,
            "contrastive_loss": 0.0,
        }

    reconstruction = ((cosine + 0.1 * mse) * weights).sum() / weights.sum().clamp_min(1e-6)
    if loss_type == "reconstruction":
        return reconstruction, {
            "l2_loss": float((mse * weights).sum().detach().item() / weights.sum().clamp_min(1e-6).item()),
            "cosine_loss": float((cosine * weights).sum().detach().item() / weights.sum().clamp_min(1e-6).item()),
            "latent_similarity_loss": 0.0,
            "decoded_similarity_loss": 0.0,
            "contrastive_loss": 0.0,
        }

    latent_similarity = features.new_zeros(())
    decoded_similarity = features.new_zeros(())
    if (similarity_weight > 0 or decoded_similarity_weight > 0) and features.shape[0] > 1 and pair_sample_count > 1:
        sample_count = min(int(features.shape[0]), int(pair_sample_count))
        sample_indices = torch.randperm(features.shape[0], device=features.device)[:sample_count]
        source_similarity = features[sample_indices] @ features[sample_indices].T
        if similarity_weight > 0:
            latent_similarity_matrix = latents[sample_indices] @ latents[sample_indices].T
            latent_similarity = F.smooth_l1_loss(latent_similarity_matrix, source_similarity)
        if decoded_similarity_weight > 0:
            decoded_similarity_matrix = reconstructed[sample_indices] @ reconstructed[sample_indices].T
            decoded_similarity = F.smooth_l1_loss(decoded_similarity_matrix, source_similarity)

    contrastive = features.new_zeros(())
    if contrastive_weight > 0 and features.shape[0] > 1:
        temperature = max(float(contrastive_temperature), 1e-4)
        labels = torch.arange(features.shape[0], device=features.device)
        logits_forward = reconstructed @ features.T / temperature
        logits_backward = features @ reconstructed.T / temperature
        forward_loss = F.cross_entropy(logits_forward, labels, reduction="none")
        backward_loss = F.cross_entropy(logits_backward, labels, reduction="none")
        contrastive = ((0.5 * (forward_loss + backward_loss)) * weights).sum() / weights.sum().clamp_min(1e-6)

    total = (
        reconstruction
        + float(similarity_weight) * latent_similarity
        + float(decoded_similarity_weight) * decoded_similarity
        + float(contrastive_weight) * contrastive
    )
    return total, {
        "cosine_loss": float((cosine * weights).sum().detach().item() / weights.sum().clamp_min(1e-6).item()),
        "latent_similarity_loss": float(latent_similarity.detach().item()),
        "decoded_similarity_loss": float(decoded_similarity.detach().item()),
        "contrastive_loss": float(contrastive.detach().item()),
    }


@torch.no_grad()
def evaluate_codec(
    codec: nn.Module,
    features: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    batch_size: int,
) -> Optional[Dict[str, float]]:
    if features is None or weights is None or features.numel() == 0:
        return None
    codec.eval()
    device = next(codec.parameters()).device
    total_weight = 0.0
    cosine_sum = 0.0
    retrieval_correct = 0.0
    retrieval_total = 0.0
    reference = features.to(device)
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].to(device)
        batch_weights = weights[start : start + batch_size].to(device)
        reconstructed = codec.decode(codec.encode(batch))
        cosine = (reconstructed * batch).sum(dim=-1).clamp(-1.0, 1.0)
        cosine_sum += float((cosine * batch_weights).sum().item())
        total_weight += float(batch_weights.sum().item())
        similarities = reconstructed @ reference.T
        predicted = torch.argmax(similarities, dim=1)
        target = torch.arange(start, start + batch.shape[0], device=device)
        retrieval_correct += float((predicted == target).float().sum().item())
        retrieval_total += float(batch.shape[0])
    return {
        "reconstruction_cosine": cosine_sum / max(total_weight, 1e-6),
        "self_retrieval_top1": retrieval_correct / max(retrieval_total, 1.0),
    }


def build_codec(args: argparse.Namespace, input_dim: int) -> nn.Module:
    codec_arch = str(args.codec_arch)
    if codec_arch == "langsplat":
        return SceneFeatureCodec(
            input_dim=int(input_dim),
            latent_dim=int(args.compressed_dim),
            encoder_hidden_dims=parse_dims(args.encoder_hidden_dims),
            decoder_hidden_dims=parse_dims(args.decoder_hidden_dims),
        )
    if codec_arch == "mlp":
        return SemanticFeatureCodec(
            input_dim=int(input_dim),
            latent_dim=int(args.compressed_dim),
            encoder_hidden_dims=parse_dims(args.encoder_hidden_dims),
            decoder_hidden_dims=parse_dims(args.decoder_hidden_dims),
        )
    raise ValueError("Unsupported --codec_arch: {}".format(codec_arch))


def train_codec(args: argparse.Namespace, files: List[TeacherFeatureFile], device: torch.device) -> Tuple[nn.Module, dict]:
    train_indices, val_indices = split_files(files, args.val_fraction, args.seed)
    if sum(int(files[index].features.shape[0]) for index in train_indices) == 0:
        train_indices = list(range(len(files)))
        val_indices = []

    train_features, train_weights, train_dedup_stats = stack_files(
        files,
        train_indices,
        deduplicate=bool(args.deduplicate_features),
        deduplicate_decimals=int(args.deduplicate_decimals),
    )
    val_features = None
    val_weights = None
    val_dedup_stats = {"before": 0, "after": 0}
    if val_indices:
        val_features, val_weights, val_dedup_stats = stack_files(
            files,
            val_indices,
            deduplicate=bool(args.deduplicate_features),
            deduplicate_decimals=int(args.deduplicate_decimals),
        )

    codec = build_codec(args, int(train_features.shape[1])).to(device)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_features, train_weights),
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=str(args.codec_arch) == "langsplat" and train_features.shape[0] > int(args.batch_size),
    )

    optimizer = torch.optim.AdamW(codec.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = None
    best_score = -float("inf")
    history = []
    progress = tqdm(range(1, int(args.epochs) + 1), desc="Train semantic codec", ascii=True, dynamic_ncols=True)

    for epoch in progress:
        codec.train()
        epoch_loss = 0.0
        batch_count = 0
        for features, weights in loader:
            if str(args.codec_arch) == "langsplat" and features.shape[0] <= 1:
                continue
            features = features.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _loss_parts = codec_loss(
                codec,
                features,
                weights,
                loss_type=str(args.loss_type),
                langsplat_cosine_weight=float(args.langsplat_cosine_weight),
                similarity_weight=float(args.similarity_weight),
                decoded_similarity_weight=float(args.decoded_similarity_weight),
                contrastive_weight=float(args.contrastive_weight),
                contrastive_temperature=float(args.contrastive_temperature),
                pair_sample_count=int(args.pair_sample_count),
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batch_count += 1

        train_metrics = evaluate_codec(codec, train_features, train_weights, int(args.eval_batch_size))
        val_metrics = evaluate_codec(codec, val_features, val_weights, int(args.eval_batch_size))
        train_cosine = None if train_metrics is None else train_metrics["reconstruction_cosine"]
        val_cosine = None if val_metrics is None else val_metrics["reconstruction_cosine"]
        selection_metrics = val_metrics if val_metrics is not None else train_metrics
        selection_score = None if selection_metrics is None else selection_metrics["reconstruction_cosine"]
        history.append(
            {
                "epoch": int(epoch),
                "loss": float(epoch_loss / max(batch_count, 1)),
                "train_cosine": train_cosine,
                "val_cosine": val_cosine,
                "train_self_retrieval_top1": None
                if train_metrics is None
                else train_metrics["self_retrieval_top1"],
                "val_self_retrieval_top1": None if val_metrics is None else val_metrics["self_retrieval_top1"],
            }
        )
        progress.set_postfix(loss=history[-1]["loss"], cosine=selection_score)
        if selection_score is not None and selection_score > best_score:
            best_score = float(selection_score)
            best_state = {key: value.detach().cpu() for key, value in codec.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Codec training did not produce a valid checkpoint.")
    codec.load_state_dict(best_state)
    codec.to(device).eval()
    return codec, {
        "train_files": len(train_indices),
        "val_files": len(val_indices),
        "train_regions": int(train_features.shape[0]),
        "val_regions": int(val_features.shape[0]) if val_features is not None else 0,
        "deduplicate_features": bool(args.deduplicate_features),
        "deduplicate_decimals": int(args.deduplicate_decimals),
        "codec_arch": str(args.codec_arch),
        "encoder_hidden_dims": list(parse_dims(args.encoder_hidden_dims)),
        "decoder_hidden_dims": list(parse_dims(args.decoder_hidden_dims)),
        "train_regions_before_dedup": int(train_dedup_stats["before"]),
        "train_regions_after_dedup": int(train_dedup_stats["after"]),
        "val_regions_before_dedup": int(val_dedup_stats["before"]),
        "val_regions_after_dedup": int(val_dedup_stats["after"]),
        "loss_weights": {
            "loss_type": str(args.loss_type),
            "langsplat_cosine_weight": float(args.langsplat_cosine_weight),
            "similarity_weight": float(args.similarity_weight),
            "decoded_similarity_weight": float(args.decoded_similarity_weight),
            "contrastive_weight": float(args.contrastive_weight),
            "contrastive_temperature": float(args.contrastive_temperature),
            "pair_sample_count": int(args.pair_sample_count),
        },
        "best_reconstruction_cosine": best_score,
        "history": history,
    }


def validate_output_paths(
    files: List[TeacherFeatureFile],
    output_dir: Path,
    codec_path: Path,
    force: bool,
    reuse_codec: bool,
) -> None:
    if force:
        return
    conflicts: List[Path] = []
    for item in files:
        stem = item.feature_path.name[:-6]
        for suffix in ("_f.npy", "_s.npy", "_w.npy"):
            path = output_dir / "{}{}".format(stem, suffix)
            if path.exists():
                conflicts.append(path)
    if not reuse_codec:
        for path in (codec_path, codec_path.with_suffix(".json")):
            if path.exists():
                conflicts.append(path)
    summary_path = output_dir / "semantic_compression_summary.json"
    if summary_path.exists():
        conflicts.append(summary_path)
    if conflicts:
        examples = ", ".join(str(path) for path in conflicts[:5])
        raise FileExistsError(
            "Compression outputs already exist. Pass --force to overwrite them. Examples: {}".format(examples)
        )


@torch.no_grad()
def write_compressed_teacher(
    files: List[TeacherFeatureFile],
    output_dir: Path,
    codec: nn.Module,
    batch_size: int,
    force: bool,
) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(codec.parameters()).device
    rows = []
    for item in tqdm(files, desc="Encode teacher features", ascii=True, dynamic_ncols=True):
        stem = item.feature_path.name[:-6]
        output_feature_path = output_dir / "{}_f.npy".format(stem)
        output_segmentation_path = output_dir / "{}_s.npy".format(stem)
        output_weight_path = output_dir / "{}_w.npy".format(stem)

        for path in (output_feature_path, output_segmentation_path):
            if path.exists() and not force:
                raise FileExistsError("Output already exists; pass --force to overwrite: {}".format(path))

        features = torch.from_numpy(item.features)
        encoded_parts = []
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size].to(device)
            encoded_parts.append(codec.encode(batch).cpu())
        if encoded_parts:
            encoded = torch.cat(encoded_parts, dim=0).numpy().astype(np.float32, copy=False)
        else:
            encoded = np.empty((0, codec.latent_dim), dtype=np.float32)

        np.save(output_feature_path, encoded)
        shutil.copy2(item.segmentation_path, output_segmentation_path)
        if item.weight_path is not None:
            shutil.copy2(item.weight_path, output_weight_path)
        elif force and output_weight_path.exists():
            output_weight_path.unlink()

        rows.append(
            {
                "image_id": stem,
                "regions": int(encoded.shape[0]),
                "input_dim": int(item.features.shape[1]),
                "compressed_dim": int(encoded.shape[1]),
            }
        )
    return rows


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress teacher features with a configurable semantic codec."
    )
    parser.add_argument("--input_teacher_dir", required=True, help="Directory containing source *_s.npy, *_f.npy, *_w.npy.")
    parser.add_argument("--output_teacher_dir", required=True, help="Directory where compressed teacher files are written.")
    parser.add_argument("--compressed_dim", required=True, type=int, help="Target latent dimension, e.g. 16, 32, 64.")
    parser.add_argument("--codec_path", default="", help="Optional codec checkpoint path. Defaults to output_teacher_dir/semantic_codec.pth.")
    parser.add_argument("--reuse_codec", action="store_true", help="Reuse an existing codec instead of training a new one.")
    parser.add_argument("--codec_arch", choices=("langsplat", "mlp"), default="langsplat")
    parser.add_argument("--loss_type", choices=("langsplat", "reconstruction", "enhanced"), default="langsplat")
    parser.add_argument("--encoder_hidden_dims", default="256,128,64", help="Comma-separated encoder hidden dims.")
    parser.add_argument("--decoder_hidden_dims", default="64,128,256,256", help="Comma-separated decoder hidden dims.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--langsplat_cosine_weight", type=float, default=0.001)
    parser.add_argument("--similarity_weight", type=float, default=0.0)
    parser.add_argument("--decoded_similarity_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.04)
    parser.add_argument("--pair_sample_count", type=int, default=256)
    parser.add_argument("--deduplicate_features", dest="deduplicate_features", action="store_true")
    parser.add_argument("--no_deduplicate_features", dest="deduplicate_features", action="store_false")
    parser.set_defaults(deduplicate_features=False)
    parser.add_argument("--deduplicate_decimals", type=int, default=6)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.compressed_dim) <= 0:
        raise ValueError("--compressed_dim must be positive.")
    if int(args.epochs) <= 0 and not args.reuse_codec:
        raise ValueError("--epochs must be positive when training a codec.")
    if int(args.batch_size) <= 0 or int(args.eval_batch_size) <= 0:
        raise ValueError("--batch_size and --eval_batch_size must be positive.")
    if int(args.pair_sample_count) <= 1:
        raise ValueError("--pair_sample_count must be greater than 1.")
    if int(args.deduplicate_decimals) < 0:
        raise ValueError("--deduplicate_decimals must be non-negative.")
    if float(args.contrastive_temperature) <= 0:
        raise ValueError("--contrastive_temperature must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_dir = Path(args.input_teacher_dir).resolve()
    output_dir = Path(args.output_teacher_dir).resolve()
    codec_path = Path(args.codec_path).resolve() if args.codec_path else output_dir / "semantic_codec.pth"
    if input_dir == output_dir:
        raise ValueError("Input and output teacher directories must be different.")

    files = discover_teacher_files(input_dir)
    input_dim = int(files[0].features.shape[1])
    if int(args.compressed_dim) >= input_dim:
        raise ValueError(
            "--compressed_dim must be smaller than the source feature dimension ({}), got {}.".format(
                input_dim,
                args.compressed_dim,
            )
        )

    validate_output_paths(files, output_dir, codec_path, bool(args.force), bool(args.reuse_codec))
    device = resolve_device(args.device)

    if args.reuse_codec:
        codec, checkpoint = load_semantic_codec(codec_path, map_location=device)
        metadata = dict(checkpoint["metadata"])
        if int(metadata["input_dim"]) != input_dim or int(metadata["latent_dim"]) != int(args.compressed_dim):
            raise RuntimeError(
                "Reused codec dimensions do not match: codec={}->{}, requested={}->{}.".format(
                    metadata["input_dim"],
                    metadata["latent_dim"],
                    input_dim,
                    args.compressed_dim,
                )
            )
        codec = codec.to(device).eval()
        training_metadata = dict(checkpoint.get("training_metadata", {}))
    else:
        codec, training_metadata = train_codec(args, files, device)
        save_semantic_codec(codec_path, codec, training_metadata=training_metadata)

    rows = write_compressed_teacher(
        files,
        output_dir,
        codec,
        batch_size=int(args.eval_batch_size),
        force=bool(args.force),
    )

    summary = {
        "input_teacher_dir": str(input_dir),
        "output_teacher_dir": str(output_dir),
        "codec_path": str(codec_path),
        "codec_format": "TieredTeacher semantic codec v{}".format(CODEC_FORMAT_VERSION),
        "codec_arch": str(getattr(args, "codec_arch", "")),
        "loss_type": str(getattr(args, "loss_type", "")),
        "input_dim": int(input_dim),
        "compressed_dim": int(args.compressed_dim),
        "num_images": len(rows),
        "num_regions": int(sum(row["regions"] for row in rows)),
        "training": training_metadata,
        "files": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_compression_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("[TieredTeacherCompression] Teacher: {}".format(output_dir))
    print("[TieredTeacherCompression] Codec: {}".format(codec_path))
    print("[TieredTeacherCompression] Dimensions: {} -> {}".format(input_dim, args.compressed_dim))


if __name__ == "__main__":
    main()
