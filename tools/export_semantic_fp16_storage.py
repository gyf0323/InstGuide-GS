from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData, PlyElement


SEMANTIC_PREFIX = "f_semantic_feat_"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _half_floating_tensors(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.detach().cpu().to(dtype=torch.float16)
        return value.detach().cpu()
    if isinstance(value, dict):
        return value.__class__((key, _half_floating_tensors(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_half_floating_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_half_floating_tensors(item) for item in value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tree_tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tree_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_tensor_bytes(item) for item in value)
    return 0


def _semantic_property_key(name: str) -> int:
    return int(name[len(SEMANTIC_PREFIX) :])


def _convert_point_cloud(src: Path, dst: Path, latent_dim: int) -> tuple[torch.Tensor, dict[str, Any]]:
    ply = PlyData.read(src)
    if not ply.elements or ply.elements[0].name != "vertex":
        raise RuntimeError(f"Expected the first PLY element to be vertex: {src}")

    vertex = ply.elements[0]
    names = list(vertex.data.dtype.names or ())
    semantic_names = sorted(
        (name for name in names if name.startswith(SEMANTIC_PREFIX)),
        key=_semantic_property_key,
    )
    if len(semantic_names) != latent_dim:
        raise RuntimeError(
            f"Expected {latent_dim} semantic PLY columns, found {len(semantic_names)} in {src}"
        )

    semantic_anchor_np = np.stack(
        [np.asarray(vertex.data[name], dtype=np.float32) for name in semantic_names],
        axis=1,
    )
    if not np.isfinite(semantic_anchor_np).all():
        raise RuntimeError(f"Semantic anchor contains non-finite values: {src}")

    retained_names = [name for name in names if name not in set(semantic_names)]
    retained_dtype = [
        (name, vertex.data.dtype.fields[name][0])
        for name in retained_names
    ]
    retained = np.empty(vertex.count, dtype=retained_dtype)
    for name in retained_names:
        retained[name] = vertex.data[name]

    vertex_comments = list(getattr(vertex, "comments", []))
    compact_vertex = PlyElement.describe(
        retained,
        vertex.name,
        comments=vertex_comments,
    )
    compact_ply = PlyData(
        [compact_vertex, *list(ply.elements[1:])],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=list(ply.comments),
        obj_info=list(ply.obj_info),
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    compact_ply.write(dst)

    roundtrip = semantic_anchor_np.astype(np.float16).astype(np.float32)
    abs_error = np.abs(roundtrip - semantic_anchor_np)
    original_norm = np.linalg.norm(semantic_anchor_np, axis=1)
    roundtrip_norm = np.linalg.norm(roundtrip, axis=1)
    cosine_denom = np.maximum(original_norm * roundtrip_norm, 1e-12)
    cosine = np.sum(semantic_anchor_np * roundtrip, axis=1) / cosine_denom
    nonzero = (original_norm > 0) & (roundtrip_norm > 0)
    mean_cosine = float(cosine[nonzero].mean()) if nonzero.any() else 1.0

    semantic_anchor = torch.from_numpy(semantic_anchor_np).to(dtype=torch.float16)
    stats = {
        "vertex_count": int(vertex.count),
        "semantic_latent_dim": int(latent_dim),
        "semantic_fp32_payload_bytes": int(semantic_anchor_np.nbytes),
        "semantic_fp16_payload_bytes": int(semantic_anchor.numel() * semantic_anchor.element_size()),
        "roundtrip_max_abs_error": float(abs_error.max(initial=0.0)),
        "roundtrip_mean_abs_error": float(abs_error.mean()),
        "roundtrip_mean_cosine": mean_cosine,
        "retained_property_count": len(retained_names),
    }
    return semantic_anchor, stats


def _convert_sidecar(src: Path, dst: Path, semantic_anchor: torch.Tensor) -> dict[str, Any]:
    payload = _torch_load(src)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Semantic sidecar is not a dictionary: {src}")
    if not bool(payload.get("use_anchor_offset_semantic", False)):
        raise RuntimeError(f"Semantic sidecar is not Anchor-Offset format: {src}")
    latent_dim = int(payload.get("semantic_latent_dim", 0))
    if tuple(semantic_anchor.shape[1:]) != (latent_dim,):
        raise RuntimeError(
            f"Sidecar latent dim {latent_dim} does not match PLY semantic anchor {tuple(semantic_anchor.shape)}"
        )

    original_tensor_bytes = _tree_tensor_bytes(payload)
    converted = _half_floating_tensors(payload)
    converted["semantic_anchor"] = semantic_anchor.cpu()
    converted["storage_dtype"] = "float16"
    converted["compute_dtype"] = "float32"
    converted["storage_format"] = "semantic_sidecar_fp16_v1"
    converted["source_sidecar_sha256"] = _sha256(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, dst)
    return {
        "original_tensor_bytes_excluding_ply_anchor": int(original_tensor_bytes),
        "converted_tensor_bytes_including_ply_anchor": int(_tree_tensor_bytes(converted)),
    }


def _convert_codec(src: Path, dst: Path) -> dict[str, Any]:
    payload = _torch_load(src)
    if not isinstance(payload, dict) or "state_dict" not in payload or "metadata" not in payload:
        raise RuntimeError(f"Invalid semantic codec: {src}")
    original_tensor_bytes = _tree_tensor_bytes(payload["state_dict"])
    converted = dict(payload)
    converted["state_dict"] = _half_floating_tensors(payload["state_dict"])
    converted["storage_dtype"] = "float16"
    converted["compute_dtype"] = "float32"
    converted["source_codec_sha256"] = _sha256(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, dst)
    return {
        "original_tensor_bytes": int(original_tensor_bytes),
        "converted_tensor_bytes": int(_tree_tensor_bytes(converted["state_dict"])),
    }


def _convert_script_module(src: Path, dst: Path) -> dict[str, Any]:
    module = torch.jit.load(str(src), map_location="cpu")
    original_tensor_bytes = sum(
        int(parameter.numel() * parameter.element_size())
        for parameter in module.parameters()
    ) + sum(
        int(buffer.numel() * buffer.element_size())
        for buffer in module.buffers()
    )
    module = module.half()
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(module, str(dst))
    converted_tensor_bytes = sum(
        int(parameter.numel() * parameter.element_size())
        for parameter in module.parameters()
    ) + sum(
        int(buffer.numel() * buffer.element_size())
        for buffer in module.buffers()
    )
    return {
        "original_tensor_bytes": int(original_tensor_bytes),
        "converted_tensor_bytes": int(converted_tensor_bytes),
    }


def _copy_required(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return rows


def _manifest_size(rows: list[dict[str, Any]]) -> int:
    return sum(int(row["bytes"]) for row in rows)


def export_model(source: Path, output: Path, iteration: int, overwrite: bool) -> Path:
    # Keep Windows junction paths intact: torch.jit.load can fail after an ASCII
    # junction is resolved back to a path containing non-ASCII characters.
    source = Path(os.path.abspath(source))
    output = Path(os.path.abspath(output))
    if source == output:
        raise RuntimeError("Output must differ from source; in-place conversion is forbidden.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    source_iteration = source / "point_cloud" / f"iteration_{iteration}"
    output_iteration = output / "point_cloud" / f"iteration_{iteration}"
    source_ply = source_iteration / "point_cloud.ply"
    source_sidecar = source_iteration / "semantic_anchor_offset.pt"
    source_decoder = source_iteration / "semantic_decoder.pt"
    source_codec = source / "semantic_codec.pth"
    source_codec_json = source / "semantic_codec.json"
    source_cfg = source / "cfg_args"
    required = (
        source_ply,
        source_sidecar,
        source_decoder,
        source_codec,
        source_codec_json,
        source_cfg,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    sidecar_payload = _torch_load(source_sidecar)
    latent_dim = int(sidecar_payload.get("semantic_latent_dim", 0))
    if latent_dim <= 0:
        raise RuntimeError(f"Invalid semantic_latent_dim in {source_sidecar}: {latent_dim}")

    output.mkdir(parents=True, exist_ok=False)
    _copy_required(source_cfg, output / "cfg_args")
    _copy_required(source_codec_json, output / "semantic_codec.json")

    for path in sorted(source_iteration.iterdir()):
        if not path.is_file() or path.name in {
            "point_cloud.ply",
            "semantic_anchor_offset.pt",
            "semantic_decoder.pt",
        }:
            continue
        _copy_required(path, output_iteration / path.name)

    semantic_anchor, ply_stats = _convert_point_cloud(
        source_ply,
        output_iteration / "point_cloud.ply",
        latent_dim,
    )
    sidecar_stats = _convert_sidecar(
        source_sidecar,
        output_iteration / "semantic_anchor_offset.pt",
        semantic_anchor,
    )
    decoder_stats = _convert_script_module(
        source_decoder,
        output_iteration / "semantic_decoder.pt",
    )
    codec_stats = _convert_codec(source_codec, output / "semantic_codec.pth")

    source_files = [
        source_cfg,
        source_codec,
        source_codec_json,
        *sorted(item for item in source_iteration.iterdir() if item.is_file()),
    ]
    source_size = sum(int(path.stat().st_size) for path in source_files)
    output_rows = _file_manifest(output)
    output_size_without_manifest = _manifest_size(output_rows)
    report = {
        "format": "scaffold_semantic_fp16_storage_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": str(source),
        "output_model": str(output),
        "iteration": int(iteration),
        "storage_dtype": "float16",
        "compute_dtype": "float32",
        "geometry_storage_changed": False,
        "semantic_ply_columns_moved_to_sidecar": True,
        "source_inference_bytes": int(source_size),
        "fp16_inference_bytes_before_manifest": int(output_size_without_manifest),
        "saved_bytes_before_manifest": int(source_size - output_size_without_manifest),
        "saved_fraction_before_manifest": float(
            (source_size - output_size_without_manifest) / source_size
        ),
        "ply": ply_stats,
        "sidecar": sidecar_stats,
        "decoder": decoder_stats,
        "codec": codec_stats,
        "files_before_manifest": output_rows,
    }
    report_path = output / "fp16_storage_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Scaffold-GS Anchor-Offset semantic model with FP16 semantic storage "
            "and unchanged geometry storage."
        )
    )
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=32000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report_path = export_model(
        args.source_model,
        args.output_model,
        args.iteration,
        args.overwrite,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"manifest: {report_path}")
    print(f"source bytes: {report['source_inference_bytes']}")
    print(f"fp16 bytes  : {report['fp16_inference_bytes_before_manifest']}")
    print(f"saved bytes : {report['saved_bytes_before_manifest']}")
    print(f"saved ratio : {report['saved_fraction_before_manifest']:.4%}")
    print(f"anchor cosine after FP16 roundtrip: {report['ply']['roundtrip_mean_cosine']:.9f}")


if __name__ == "__main__":
    main()
