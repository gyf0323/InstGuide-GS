from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


CODEC_FORMAT_VERSION = 1


def _normalize_dims(values: Iterable[int]) -> tuple[int, ...]:
    dims = tuple(int(value) for value in values)
    if any(value <= 0 for value in dims):
        raise ValueError(f"Codec hidden dimensions must be positive, got {dims}.")
    return dims


def _make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dims = [int(input_dim), *_normalize_dims(hidden_dims), int(output_dim)]
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index + 1 < len(dims) - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _make_langsplat_encoder(input_dim: int, hidden_dims: Sequence[int], latent_dim: int) -> nn.Sequential:
    dims = [int(input_dim), *_normalize_dims(hidden_dims), int(latent_dim)]
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        if index == 0:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.BatchNorm1d(in_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


def _make_langsplat_decoder(latent_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dims = [int(latent_dim), *_normalize_dims(hidden_dims), int(output_dim)]
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        if index > 0:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class SemanticFeatureCodec(nn.Module):
    """Configurable normalized autoencoder for CLIP-family semantic features."""

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
                f"Codec dimensions must be positive, got input={self.input_dim}, latent={self.latent_dim}."
            )
        self.encoder = _make_mlp(self.input_dim, self.encoder_hidden_dims, self.latent_dim)
        self.decoder = _make_mlp(self.latent_dim, self.decoder_hidden_dims, self.input_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1, eps=1e-6)
        return F.normalize(self.encoder(features), dim=-1, eps=1e-6)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = F.normalize(latents, dim=-1, eps=1e-6)
        return F.normalize(self.decoder(latents), dim=-1, eps=1e-6)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(features)
        return latents, self.decode(latents)

    def metadata(self) -> dict:
        return {
            "format_version": CODEC_FORMAT_VERSION,
            "codec_type": "semantic_feature_mlp",
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "encoder_hidden_dims": list(self.encoder_hidden_dims),
            "decoder_hidden_dims": list(self.decoder_hidden_dims),
            "normalization": "l2",
        }


class LangSplatSemanticFeatureCodec(nn.Module):
    """LangSplat autoencoder layout used by instance-level compressed teachers."""

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
                f"Codec dimensions must be positive, got input={self.input_dim}, latent={self.latent_dim}."
            )
        self.encoder = _make_langsplat_encoder(self.input_dim, self.encoder_hidden_dims, self.latent_dim)
        self.decoder = _make_langsplat_decoder(self.latent_dim, self.decoder_hidden_dims, self.input_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(features), dim=-1, eps=1e-6)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.decoder(latents), dim=-1, eps=1e-6)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(features)
        return latents, self.decode(latents)

    def metadata(self) -> dict:
        return {
            "format_version": CODEC_FORMAT_VERSION,
            "codec_type": "langsplat",
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "encoder_hidden_dims": list(self.encoder_hidden_dims),
            "decoder_hidden_dims": list(self.decoder_hidden_dims),
            "normalization": "l2",
        }


def save_semantic_codec(
    path,
    codec: SemanticFeatureCodec,
    *,
    training_metadata: dict | None = None,
) -> Path:
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


def load_semantic_codec(path, map_location="cpu") -> tuple[nn.Module, dict]:
    codec_path = Path(path)
    if not codec_path.is_file():
        raise FileNotFoundError(f"Semantic codec checkpoint not found: {codec_path}")
    try:
        payload = torch.load(codec_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(codec_path, map_location=map_location)
    if not isinstance(payload, dict) or "metadata" not in payload or "state_dict" not in payload:
        raise RuntimeError(f"Invalid semantic codec checkpoint: {codec_path}")
    metadata = dict(payload["metadata"])
    version = int(metadata.get("format_version", 0))
    if version != CODEC_FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported semantic codec format version {version}; expected {CODEC_FORMAT_VERSION}."
        )
    codec_type = str(metadata.get("codec_type", "semantic_feature_mlp") or "semantic_feature_mlp").lower()
    if codec_type == "langsplat":
        codec = LangSplatSemanticFeatureCodec(
            input_dim=int(metadata["input_dim"]),
            latent_dim=int(metadata["latent_dim"]),
            encoder_hidden_dims=metadata.get("encoder_hidden_dims", (256, 128, 64)),
            decoder_hidden_dims=metadata.get("decoder_hidden_dims", (64, 128, 256, 256)),
        )
    else:
        codec = SemanticFeatureCodec(
            input_dim=int(metadata["input_dim"]),
            latent_dim=int(metadata["latent_dim"]),
            encoder_hidden_dims=metadata.get("encoder_hidden_dims", (256, 128)),
            decoder_hidden_dims=metadata.get("decoder_hidden_dims", (128, 256)),
        )
    codec.load_state_dict(payload["state_dict"], strict=True)
    return codec, payload


def load_semantic_decoder(path, map_location="cpu") -> tuple[nn.Module, dict]:
    codec, payload = load_semantic_codec(path, map_location=map_location)
    decoder = codec.decoder
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder, payload
