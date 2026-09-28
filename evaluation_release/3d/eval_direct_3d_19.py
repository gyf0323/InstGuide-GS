from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


FULL_CLASS_NAMES = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
)

# ScanNet-v2 IDs for the 19-class protocol; NYU40 otherfurniture is excluded.
FULL_NYU40_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36)

CLASS_PROTOCOL_IDS = {
    "scannet19": FULL_NYU40_IDS,
    "scannet15": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34),
    "scannet10": (1, 2, 4, 5, 6, 7, 8, 9, 10, 33),
    "scannetpp100": None,
}

TEXT_PROMPT_TEMPLATES = {
    "single": ("{label}",),
    "maskclustering": (
        "{label}",
        "a photo of a {label}.",
        "an indoor photo of a {label}.",
        "a photo of the {label} in a room.",
    ),
}


def text_prompt_metadata(mode: str) -> dict:
    if mode not in TEXT_PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported prompt mode: {mode}")
    return {
        "promptMode": mode,
        "positiveTemplates": list(TEXT_PROMPT_TEMPLATES[mode]),
        "templatesPerClass": len(TEXT_PROMPT_TEMPLATES[mode]),
        "promptAggregation": "per-prompt L2 -> equal embedding mean per class -> L2",
        "promptSelectionUsesGt": False,
    }

FULL_PALETTE = np.asarray(
    [
        (174, 199, 232),
        (152, 223, 138),
        (31, 119, 180),
        (255, 187, 120),
        (188, 189, 34),
        (140, 86, 75),
        (255, 152, 150),
        (214, 39, 40),
        (197, 176, 213),
        (148, 103, 189),
        (196, 156, 148),
        (23, 190, 207),
        (247, 182, 210),
        (219, 219, 141),
        (255, 127, 14),
        (158, 218, 229),
        (44, 160, 44),
        (112, 128, 144),
        (227, 119, 194),
    ],
    dtype=np.uint8,
)

CLASS_PROTOCOL = "scannet19"
CLASS_NAMES = FULL_CLASS_NAMES
NYU40_IDS = FULL_NYU40_IDS
CLASS_IDS = FULL_NYU40_IDS
CLASS_ID_NAMESPACE = "nyu40"
PALETTE = FULL_PALETTE


def build_name_palette(class_names) -> np.ndarray:
    colors = []
    for class_name in class_names:
        digest = hashlib.sha256(normalize_name(class_name).encode("utf-8")).digest()
        rgb = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.uint16)
        colors.append((rgb // 2 + 96).astype(np.uint8))
    return np.asarray(colors, dtype=np.uint8)


def configure_class_protocol(protocol: str, top100_path: str = "") -> None:
    global CLASS_PROTOCOL, CLASS_NAMES, NYU40_IDS, CLASS_IDS, CLASS_ID_NAMESPACE, PALETTE

    if protocol not in CLASS_PROTOCOL_IDS:
        raise RuntimeError(f"Unsupported class protocol: {protocol}")
    if protocol == "scannetpp100":
        top100 = require_file(top100_path, "ScanNet++ official Top-100 class list")
        class_names = tuple(
            line.strip()
            for line in top100.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
        normalized = [normalize_name(name) for name in class_names]
        if len(class_names) != 100:
            raise RuntimeError(f"Expected 100 ScanNet++ benchmark classes, found {len(class_names)}: {top100}")
        if len(set(normalized)) != len(normalized):
            raise RuntimeError(f"Duplicate normalized class names in ScanNet++ Top-100 list: {top100}")
        CLASS_PROTOCOL = protocol
        CLASS_NAMES = class_names
        NYU40_IDS = ()
        CLASS_IDS = tuple(range(len(class_names)))
        CLASS_ID_NAMESPACE = "scannetpp_top100_index"
        PALETTE = build_name_palette(class_names)
        return

    index_by_id = {nyu_id: index for index, nyu_id in enumerate(FULL_NYU40_IDS)}
    selected_ids = CLASS_PROTOCOL_IDS[protocol]
    selected_indices = [index_by_id[nyu_id] for nyu_id in selected_ids]
    CLASS_PROTOCOL = protocol
    NYU40_IDS = tuple(selected_ids)
    CLASS_IDS = tuple(selected_ids)
    CLASS_ID_NAMESPACE = "nyu40"
    CLASS_NAMES = tuple(FULL_CLASS_NAMES[index] for index in selected_indices)
    PALETTE = FULL_PALETTE[selected_indices]

# Conservative ScanNet++ raw/Top-100 name aliases. Classes that are separate
# NYU40 categories but absent from the 19-class protocol (for example blinds,
# shelves, and ceiling) are intentionally not folded into a target class.
ALIASES = {
    "wall": ("wall", "shower wall"),
    "floor": ("floor",),
    "cabinet": (
        "cabinet",
        "storage cabinet",
        "kitchen cabinet",
        "base cabinet",
        "wall cabinet",
    ),
    "bed": ("bed",),
    "chair": ("chair", "office chair"),
    "sofa": ("sofa", "couch"),
    "table": ("table", "coffee table", "dining table", "side table"),
    "door": ("door", "doorframe", "door frame"),
    "window": ("window", "window frame", "windowsill", "window sill"),
    "bookshelf": ("bookshelf", "bookcase"),
    "picture": ("picture", "painting", "poster"),
    "counter": ("counter", "kitchen counter", "countertop", "counter top"),
    "desk": ("desk",),
    "curtain": ("curtain",),
    "refrigerator": ("refrigerator", "fridge"),
    "shower curtain": ("shower curtain", "showercurtain"),
    "toilet": ("toilet",),
    "sink": ("sink",),
    "bathtub": ("bathtub", "bath tub"),
}


def torch_load(path, map_location="cpu"):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def torchscript_load(path, map_location="cpu"):
    """Load TorchScript checkpoints even when PyTorch rejects a Unicode path."""
    import torch

    source = require_file(path, "TorchScript checkpoint")
    try:
        return torch.jit.load(str(source), map_location=map_location)
    except RuntimeError as error:
        if "open file failed" not in str(error):
            raise
        with tempfile.TemporaryDirectory(prefix="direct3d19_") as temp_dir:
            local_copy = Path(temp_dir) / "model.pt"
            shutil.copyfile(source, local_copy)
            return torch.jit.load(str(local_copy), map_location=map_location)


def normalize_name(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{label} not found: {value}")
    return value


def read_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines()]


def load_benchmark_map(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"class", "semantic_map_to"}.issubset(reader.fieldnames):
            raise RuntimeError(f"Invalid ScanNet++ benchmark map: {path}")
        for row in reader:
            source = normalize_name(row.get("class", ""))
            target = normalize_name(row.get("semantic_map_to", ""))
            if source:
                mapping[source] = target
    return mapping


def alias_index() -> dict[str, int]:
    if CLASS_PROTOCOL == "scannetpp100":
        return {normalize_name(class_name): index for index, class_name in enumerate(CLASS_NAMES)}

    result: dict[str, int] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        for alias in ALIASES[class_name]:
            key = normalize_name(alias)
            previous = result.get(key)
            if previous is not None and previous != class_index:
                raise RuntimeError(f"Ambiguous protocol-class alias: {alias}")
            result[key] = class_index
    return result


def read_ply_vertex(path: str | Path):
    from plyfile import PlyData

    return PlyData.read(str(path), mmap=True)["vertex"].data


def ply_xyz(vertex) -> np.ndarray:
    return np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float32, copy=False)


def sorted_fields(names, prefix: str) -> list[str]:
    result = [name for name in names if name.startswith(prefix)]
    return sorted(result, key=lambda name: int(name.rsplit("_", 1)[1]))


def ply_matrix(vertex, prefix: str) -> np.ndarray:
    fields = sorted_fields(vertex.dtype.names or (), prefix)
    if not fields:
        raise RuntimeError(f"PLY has no fields beginning with {prefix!r}")
    return np.column_stack([vertex[name] for name in fields]).astype(np.float32, copy=False)


def load_gt(args):
    mesh_path = require_file(args.gt_mesh, "semantic GT mesh")
    raw_class_path = require_file(args.raw_classes, "ScanNet++ raw class list")
    benchmark_map_path = require_file(args.benchmark_map, "ScanNet++ benchmark map")

    vertex = read_ply_vertex(mesh_path)
    if "label" not in (vertex.dtype.names or ()):
        raise RuntimeError(f"GT mesh has no vertex.label field: {mesh_path}")
    xyz = ply_xyz(vertex)
    raw_labels = np.asarray(vertex["label"], dtype=np.int64)
    raw_names = read_lines(raw_class_path)
    benchmark_map = load_benchmark_map(benchmark_map_path)
    aliases = alias_index()

    raw_to_target: dict[int, int] = {}
    mapped_rows = []
    for raw_id, raw_name in enumerate(raw_names):
        raw_key = normalize_name(raw_name)
        benchmark_key = benchmark_map.get(raw_key, "")
        if CLASS_PROTOCOL == "scannetpp100":
            if raw_key not in benchmark_map:
                raise RuntimeError(f"Raw class is missing from the benchmark map: {raw_name}")
            target = aliases.get(benchmark_key or raw_key)
            source = "official_top100_map"
        else:
            target = aliases.get(raw_key)
            source = "raw_exact_or_alias"
            if target is None and benchmark_key:
                target = aliases.get(benchmark_key)
                source = "official_top100_then_alias"
        if target is None:
            continue
        raw_to_target[raw_id] = int(target)
        mapped_rows.append(
            {
                "raw_id": raw_id,
                "raw_name": raw_name,
                "benchmark_name": benchmark_key,
                "target_index": int(target),
                "target_name": CLASS_NAMES[target],
                "mapping_source": source,
            }
        )

    target = np.full(raw_labels.shape, -1, dtype=np.int16)
    for raw_id, target_id in raw_to_target.items():
        target[raw_labels == raw_id] = target_id
    valid = target >= 0
    if not np.any(valid):
        raise RuntimeError(f"No GT vertices map to the configured {CLASS_PROTOCOL} classes")

    raw_nonvoid = (raw_labels >= 0) & (raw_labels < len(raw_names))
    present = sorted(int(value) for value in np.unique(target[valid]))
    stats = {
        "mesh": str(mesh_path),
        "meshVertexCount": int(len(target)),
        "rawClassCount": int(len(raw_names)),
        "mappedRawClassCount": int(len(raw_to_target)),
        "rawNonVoidVertexCount": int(raw_nonvoid.sum()),
        "evaluatedVertexCount": int(valid.sum()),
        "mappedFractionOfRawNonVoid": float(valid.sum() / max(int(raw_nonvoid.sum()), 1)),
        "presentClassIndices": present,
        "presentClassNames": [CLASS_NAMES[index] for index in present],
        "mappingPolicy": (
            "official ScanNet++ semantic_map_to target; blank maps to raw name; explicit exclusions remain ignored"
            if CLASS_PROTOCOL == "scannetpp100"
            else "raw-name alias first, then official ScanNet++ Top-100 mapping plus conservative alias"
        ),
        "rawClasses": str(raw_class_path),
        "benchmarkMap": str(benchmark_map_path),
        "benchmarkMapSha256": sha256_file(benchmark_map_path),
        "mappedRows": mapped_rows,
        "classProtocol": CLASS_PROTOCOL,
        "targetClassCount": len(CLASS_NAMES),
        "targetNyu40Ids": list(NYU40_IDS),
        "targetClassIds": list(CLASS_IDS),
        "targetClassIdNamespace": CLASS_ID_NAMESPACE,
    }
    if CLASS_PROTOCOL == "scannetpp100":
        top100_path = require_file(args.top100_classes, "ScanNet++ official Top-100 class list")
        stats["top100Classes"] = str(top100_path)
        stats["top100ClassesSha256"] = sha256_file(top100_path)
    return xyz, raw_labels, target, valid, stats


def expand_prompts(class_names, mode: str):
    if mode not in TEXT_PROMPT_TEMPLATES:
        raise RuntimeError(f"Unsupported prompt mode: {mode}")
    prompts, owners = [], []
    for index, name in enumerate(class_names):
        for template in TEXT_PROMPT_TEMPLATES[mode]:
            prompts.append(template.format(label=name))
            owners.append(index)
    return prompts, owners


def group_prompt_features(features, owners, class_count):
    import torch
    import torch.nn.functional as F

    owner_tensor = torch.as_tensor(owners, device=features.device)
    grouped = []
    for index in range(class_count):
        grouped.append(F.normalize(features[owner_tensor == index].mean(0), dim=0, eps=1e-6))
    return torch.stack(grouped)


def encode_alphaclip(args, device):
    import torch
    import torch.nn.functional as F

    package = Path(args.alphaclip_package).expanduser().resolve()
    alpha_weight = require_file(args.alphaclip_checkpoint, "Alpha-CLIP weight")
    base_root = Path(args.alphaclip_download_root).expanduser().resolve()
    base_weight = require_file(base_root / "ViT-B-16.pt", "Alpha-CLIP base weight")
    if not package.is_dir():
        raise FileNotFoundError(f"Alpha-CLIP package not found: {package}")
    sys.path.insert(0, str(package))
    os.environ["ALPHACLIP_PACKAGE_PATH"] = str(package)
    os.environ["ALPHACLIP_DOWNLOAD_ROOT"] = str(base_root)
    import alpha_clip

    model, _ = alpha_clip.load(
        args.alphaclip_model,
        alpha_vision_ckpt_pth=str(alpha_weight),
        device=str(device),
        download_root=str(base_root),
    )
    model.eval()
    prompts, owners = expand_prompts(CLASS_NAMES, args.prompt_mode)
    encoded = []
    with torch.no_grad():
        for start in range(0, len(prompts), args.text_batch_size):
            tokens = alpha_clip.tokenize(prompts[start : start + args.text_batch_size]).to(device)
            encoded.append(F.normalize(model.encode_text(tokens).float(), dim=-1, eps=1e-6))
    features = group_prompt_features(torch.cat(encoded), owners, len(CLASS_NAMES))
    metadata = {
        "family": "Alpha-CLIP",
        "model": args.alphaclip_model,
        "baseWeight": str(base_weight),
        "baseWeightSha256": sha256_file(base_weight),
        "alphaWeight": str(alpha_weight),
        "alphaWeightSha256": sha256_file(alpha_weight),
        **text_prompt_metadata(args.prompt_mode),
        "featureDim": int(features.shape[1]),
    }
    del model
    torch.cuda.empty_cache()
    return features, metadata


def load_ours(args, device):
    import torch
    import torch.nn.functional as F

    point_cloud = require_file(args.point_cloud, "Ours Scaffold-GS anchor point cloud")
    iteration_dir = point_cloud.parent
    sidecar_path = require_file(iteration_dir / "semantic_anchor_offset.pt", "Ours anchor-offset semantics")
    decoder_path = require_file(iteration_dir / "semantic_decoder.pt", "Ours semantic decoder")
    vertex = read_ply_vertex(point_cloud)
    anchor = ply_xyz(vertex)
    offset_flat = ply_matrix(vertex, "f_offset_")
    scale_raw = ply_matrix(vertex, "scale_")
    sidecar = torch_load(sidecar_path, map_location="cpu")
    n_offsets = int(sidecar["n_offsets"])
    latent_dim = int(sidecar["semantic_latent_dim"])
    residual_dim = int(sidecar["semantic_residual_dim"])
    if offset_flat.shape[1] != n_offsets * 3:
        raise RuntimeError(f"Ours offset fields do not match n_offsets={n_offsets}: {offset_flat.shape}")
    offsets = offset_flat.reshape(len(anchor), 3, n_offsets).transpose(0, 2, 1)
    scaling = np.exp(scale_raw[:, :3]).astype(np.float32)
    child_xyz = (anchor[:, None, :] + offsets * scaling[:, None, :]).reshape(-1, 3).astype(np.float32)

    anchor_values = sidecar.get("semantic_anchor")
    if anchor_values is None:
        anchor_values = ply_matrix(vertex, "f_semantic_feat_")
    semantic_anchor = torch.as_tensor(anchor_values, dtype=torch.float32)
    if semantic_anchor.shape != (len(anchor), latent_dim):
        raise RuntimeError(f"Ours semantic anchor shape mismatch: {tuple(semantic_anchor.shape)}")
    row_for_anchor = torch.full((len(anchor),), -1, dtype=torch.long)
    residual_is_sparse = bool(sidecar.get("semantic_residual_is_sparse", False))
    residual = torch.as_tensor(sidecar["semantic_offset_residual"], dtype=torch.float32)
    gate = torch.as_tensor(sidecar["semantic_offset_gate"], dtype=torch.float32)
    if residual.shape != (len(residual), n_offsets, residual_dim) or gate.shape != (len(residual), n_offsets, 1):
        raise RuntimeError("Ours residual/gate shapes do not match the sidecar dimensions")
    if residual_dim > 0:
        if residual_is_sparse:
            active = torch.as_tensor(sidecar["semantic_residual_anchor_indices"], dtype=torch.long)
            if (active.ndim != 1 or len(active) != len(residual)
                    or torch.unique(active).numel() != active.numel()
                    or bool(((active < 0) | (active >= len(anchor))).any())):
                raise RuntimeError("Ours sparse residual anchor indices are invalid")
            row_for_anchor[active] = torch.arange(len(active), dtype=torch.long)
        else:
            if len(residual) != len(anchor):
                raise RuntimeError("Dense Ours residual rows do not match anchor count")
            row_for_anchor[:] = torch.arange(len(anchor), dtype=torch.long)
    projection = sidecar.get("offset_residual_proj")
    if residual_dim > 0 and not isinstance(projection, dict):
        raise RuntimeError("Ours residual projection state is missing")
    projection_weight = None if residual_dim == 0 else torch.as_tensor(projection["weight"], dtype=torch.float32)
    projection_bias = None if residual_dim == 0 else torch.as_tensor(projection["bias"], dtype=torch.float32)
    tensors = [semantic_anchor, residual, gate]
    if residual_dim > 0:
        if projection_weight.shape != (latent_dim, residual_dim) or projection_bias.shape != (latent_dim,):
            raise RuntimeError("Ours residual projection shape does not match latent/residual dimensions")
        tensors.extend([projection_weight, projection_bias])
    if not np.isfinite(child_xyz).all() or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise RuntimeError("Ours geometry or semantic parameters contain NaN or Inf")

    text, text_meta = encode_alphaclip(args, device)
    decoder = torchscript_load(decoder_path, map_location=device).float().eval()
    if args.geometry_transfer == "knn_score":
        carrier_output = np.empty((len(child_xyz), len(CLASS_NAMES)), dtype=np.float32)
    elif args.geometry_transfer == "knn_latent":
        carrier_output = np.empty((len(child_xyz), latent_dim), dtype=np.float32)
    else:
        carrier_output = np.empty(len(child_xyz), dtype=np.int16)
    anchor_chunk = max(1, args.classify_chunk_size // n_offsets)
    with torch.no_grad():
        for start in range(0, len(anchor), anchor_chunk):
            end = min(start + anchor_chunk, len(anchor))
            base = semantic_anchor[start:end, None, :].expand(-1, n_offsets, -1).clone()
            if residual_dim > 0:
                rows = row_for_anchor[start:end]
                active_position = torch.nonzero(rows >= 0, as_tuple=False).squeeze(1)
                if active_position.numel():
                    source_rows = rows[active_position]
                    projected = F.linear(residual[source_rows], projection_weight, projection_bias)
                    delta = torch.sigmoid(gate[source_rows]) * projected
                    base[active_position] += delta
            latent = F.normalize(base.reshape(-1, latent_dim), dim=1, eps=1e-6).to(device)
            output_slice = slice(start * n_offsets, end * n_offsets)
            if args.geometry_transfer == "knn_latent":
                carrier_output[output_slice] = latent.float().cpu().numpy()
                continue
            decoded = F.normalize(decoder(latent).float(), dim=1, eps=1e-6)
            scores = decoded @ text.T
            if args.geometry_transfer == "knn_score":
                carrier_output[output_slice] = scores.float().cpu().numpy()
            else:
                carrier_output[output_slice] = scores.argmax(1).to(torch.int16).cpu().numpy()
    if args.geometry_transfer == "knn_latent":
        runtime_output = {
            "latents": carrier_output,
            "decoder": decoder,
            "textFeatures": text,
            "normalizeFusedLatent": True,
        }
    else:
        runtime_output = carrier_output
    return child_xyz, runtime_output, {
        "methodAdapter": "Ours all child Gaussians -> anchor latent plus gated projected child residual -> trained decoder -> Alpha-CLIP argmax",
        "pointCloud": str(point_cloud),
        "semanticSidecar": str(sidecar_path),
        "semanticDecoder": str(decoder_path),
        "anchorCount": int(len(anchor)),
        "offsetsPerAnchor": n_offsets,
        "carrierCount": int(len(child_xyz)),
        "semanticLatentDim": latent_dim,
        "semanticResidualDim": residual_dim,
        "residualIsSparse": residual_is_sparse,
        "residualAnchorCount": int((row_for_anchor >= 0).sum()),
        "childOpacityFilter": False,
        "carrierOutput": {
            "knn_score": "class_scores",
            "knn_latent": "normalized_semantic_latent",
        }.get(args.geometry_transfer, "hard_class"),
        "latentTransfer": (
            {
                "carrierNormalization": "L2 after anchor plus gated projected child residual",
                "fusedLatentNormalization": True,
                "decoderInput": "L2-normalized inverse-distance fused child latent",
                "nativeEvidence": "Scaffold-GS semantic path normalizes child and decoder-input latents",
            }
            if args.geometry_transfer == "knn_latent"
            else None
        ),
        "semanticOperationOrder": (
            "3D kNN latent fusion -> L2 normalization -> trained decoder -> "
            "L2 normalization -> text cosine argmax"
            if args.geometry_transfer == "knn_latent"
            else "trained decoder per carrier -> text classification -> geometry transfer"
        ),
        "note": "All stored child Gaussians are used to avoid selecting a camera-dependent opacity mask.",
        "textEncoder": text_meta,
    }


def transfer_to_gt(carrier_xyz, carrier_labels, gt_xyz, query_chunk_size):
    from scipy.spatial import cKDTree

    carrier_xyz = np.asarray(carrier_xyz, dtype=np.float32)
    carrier_labels = np.asarray(carrier_labels, dtype=np.int16).reshape(-1)
    if carrier_xyz.shape != (len(carrier_labels), 3):
        raise RuntimeError(f"Carrier mismatch: xyz={carrier_xyz.shape}, labels={carrier_labels.shape}")
    if not np.isfinite(carrier_xyz).all():
        raise RuntimeError("Carrier xyz contains NaN or Inf")
    tree = cKDTree(carrier_xyz)
    pred = np.empty(len(gt_xyz), dtype=np.int16)
    distance = np.empty(len(gt_xyz), dtype=np.float32)
    for start in range(0, len(gt_xyz), query_chunk_size):
        end = min(start + query_chunk_size, len(gt_xyz))
        try:
            dist, index = tree.query(gt_xyz[start:end], k=1, workers=-1)
        except TypeError:
            dist, index = tree.query(gt_xyz[start:end], k=1)
        pred[start:end] = carrier_labels[index]
        distance[start:end] = dist.astype(np.float32)
    return pred, distance


def _knn_weights(distance, mode):
    distance = np.asarray(distance, dtype=np.float32)
    if distance.ndim != 2:
        raise RuntimeError(f"kNN distances must be 2D, got {distance.shape}")
    if mode == "uniform":
        weights = np.ones_like(distance, dtype=np.float32)
    elif mode == "inverse_distance":
        exact = distance <= 1e-8
        has_exact = exact.any(axis=1, keepdims=True)
        inverse = 1.0 / np.maximum(distance, 1e-8)
        weights = np.where(has_exact, exact.astype(np.float32), inverse)
    elif mode == "gaussian_rbf":
        sigma = np.maximum(distance[:, -1:], 1e-8)
        weights = np.exp(-(distance * distance) / (2.0 * sigma * sigma)).astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported kNN score weight mode: {mode}")
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def transfer_scores_to_gt(
    carrier_xyz,
    carrier_scores,
    gt_xyz,
    query_chunk_size,
    neighbor_count=8,
    weight_mode="inverse_distance",
    zero_score_is_invalid=False,
):
    """Interpolate native 3D carrier class scores at GT mesh vertices."""
    from scipy.spatial import cKDTree

    carrier_xyz = np.asarray(carrier_xyz, dtype=np.float32)
    carrier_scores = np.asarray(carrier_scores, dtype=np.float32)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32)
    if carrier_scores.ndim != 2 or carrier_scores.shape != (len(carrier_xyz), len(CLASS_NAMES)):
        raise RuntimeError(
            "Carrier score mismatch: "
            f"xyz={carrier_xyz.shape}, scores={carrier_scores.shape}, classes={len(CLASS_NAMES)}"
        )
    if not np.isfinite(carrier_xyz).all() or not np.isfinite(carrier_scores).all():
        raise RuntimeError("Carrier xyz or class scores contain NaN or Inf")
    k = max(1, min(int(neighbor_count), len(carrier_xyz)))
    chunk_size = max(1, int(query_chunk_size))
    tree = cKDTree(carrier_xyz)
    pred = np.empty(len(gt_xyz), dtype=np.int16)
    nearest_distance = np.empty(len(gt_xyz), dtype=np.float32)
    support_distance = np.empty(len(gt_xyz), dtype=np.float32)
    for start in range(0, len(gt_xyz), chunk_size):
        end = min(start + chunk_size, len(gt_xyz))
        try:
            distance, index = tree.query(gt_xyz[start:end], k=k, workers=-1)
        except TypeError:
            distance, index = tree.query(gt_xyz[start:end], k=k)
        distance = np.asarray(distance, dtype=np.float32)
        index = np.asarray(index, dtype=np.int64)
        if distance.ndim == 1:
            distance = distance[:, None]
            index = index[:, None]
        weights = _knn_weights(distance, weight_mode)
        neighbor_scores = carrier_scores[index]
        fused_scores = np.einsum("nk,nkc->nc", weights, neighbor_scores, optimize=True)
        pred[start:end] = fused_scores.argmax(axis=1).astype(np.int16)
        if zero_score_is_invalid:
            pred[start:end][fused_scores.max(axis=1) == 0] = -1
        nearest_distance[start:end] = distance[:, 0]
        support_distance[start:end] = distance[:, -1]
    return pred, nearest_distance, support_distance


def transfer_latents_to_gt(
    carrier_xyz,
    carrier_latents,
    gt_xyz,
    decoder,
    text_features,
    device,
    query_chunk_size,
    neighbor_count=8,
    weight_mode="inverse_distance",
    normalize_fused_latent=True,
):
    """Fuse native 3D latents at GT vertices, then decode and classify once."""
    import torch
    import torch.nn.functional as F
    from scipy.spatial import cKDTree

    carrier_xyz = np.asarray(carrier_xyz, dtype=np.float32)
    carrier_latents = np.asarray(carrier_latents, dtype=np.float32)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32)
    if carrier_latents.ndim != 2 or len(carrier_latents) != len(carrier_xyz):
        raise RuntimeError(
            "Carrier latent mismatch: "
            f"xyz={carrier_xyz.shape}, latents={carrier_latents.shape}"
        )
    if int(text_features.shape[0]) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Text class count {text_features.shape[0]} != protocol class count {len(CLASS_NAMES)}"
        )
    if not np.isfinite(carrier_xyz).all() or not np.isfinite(carrier_latents).all():
        raise RuntimeError("Carrier xyz or semantic latents contain NaN or Inf")

    k = max(1, min(int(neighbor_count), len(carrier_xyz)))
    chunk_size = max(1, int(query_chunk_size))
    tree = cKDTree(carrier_xyz)
    pred = np.empty(len(gt_xyz), dtype=np.int16)
    nearest_distance = np.empty(len(gt_xyz), dtype=np.float32)
    support_distance = np.empty(len(gt_xyz), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(gt_xyz), chunk_size):
            end = min(start + chunk_size, len(gt_xyz))
            try:
                distance, index = tree.query(gt_xyz[start:end], k=k, workers=-1)
            except TypeError:
                distance, index = tree.query(gt_xyz[start:end], k=k)
            distance = np.asarray(distance, dtype=np.float32)
            index = np.asarray(index, dtype=np.int64)
            if distance.ndim == 1:
                distance = distance[:, None]
                index = index[:, None]
            weights = _knn_weights(distance, weight_mode)
            neighbor_latents = torch.as_tensor(
                carrier_latents[index], dtype=torch.float32, device=device
            )
            torch_weights = torch.as_tensor(
                weights, dtype=torch.float32, device=device
            )
            fused_latent = torch.einsum("nk,nkd->nd", torch_weights, neighbor_latents)
            if normalize_fused_latent:
                fused_latent = F.normalize(fused_latent, dim=-1, eps=1e-6)
            decoded = F.normalize(decoder(fused_latent).float(), dim=-1, eps=1e-6)
            scores = decoded @ text_features.T
            pred[start:end] = scores.argmax(dim=1).to(torch.int16).cpu().numpy()
            nearest_distance[start:end] = distance[:, 0]
            support_distance[start:end] = distance[:, -1]
    return pred, nearest_distance, support_distance


def compute_metrics(gt, pred):
    valid = gt >= 0
    gt_valid = gt[valid].astype(np.int64)
    pred_valid = pred[valid].astype(np.int64)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    paired = (pred_valid >= 0) & (pred_valid < len(CLASS_NAMES))
    np.add.at(confusion, (gt_valid[paired], pred_valid[paired]), 1)
    gt_count = np.bincount(gt_valid, minlength=len(CLASS_NAMES)).astype(np.int64)
    pred_count = np.bincount(pred_valid[paired], minlength=len(CLASS_NAMES)).astype(np.int64)
    tp = np.diag(confusion)
    union = gt_count + pred_count - tp
    present = gt_count > 0
    iou = np.divide(tp, union, out=np.full(len(CLASS_NAMES), np.nan), where=union > 0)
    acc = np.divide(tp, gt_count, out=np.full(len(CLASS_NAMES), np.nan), where=gt_count > 0)
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "class_index": index,
                "benchmark_id": CLASS_IDS[index],
                "id_namespace": CLASS_ID_NAMESPACE,
                "nyu40_id": NYU40_IDS[index] if CLASS_ID_NAMESPACE == "nyu40" else None,
                "class_name": name,
                "present_in_gt": bool(present[index]),
                "gt_count": int(gt_count[index]),
                "pred_count": int(pred_count[index]),
                "true_positive": int(tp[index]),
                "iou": None if math.isnan(float(iou[index])) else float(iou[index]),
                "accuracy": None if math.isnan(float(acc[index])) else float(acc[index]),
            }
        )
    return {
        "mIoU": float(np.nanmean(iou[present])),
        "mAcc": float(np.nanmean(acc[present])),
        "overallAcc": float((gt_valid == pred_valid).sum() / max(len(gt_valid), 1)),
        "evaluatedVertexCount": int(len(gt_valid)),
        "predictedVertexCount": int(paired.sum()),
        "predictionCoverage": float(paired.sum() / max(len(gt_valid), 1)),
        "presentClassCount": int(present.sum()),
        "presentClassNames": [CLASS_NAMES[index] for index in np.flatnonzero(present)],
        "perClass": per_class,
        "confusion": confusion,
    }


def distance_summary(distance, valid):
    values = np.asarray(distance[valid], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "fractionOver2p5cm": float((values > 0.025).mean()),
        "fractionOver5cm": float((values > 0.05).mean()),
        "fractionOver10cm": float((values > 0.10).mean()),
    }


def write_ply(path: Path, xyz, labels, label_field):
    from plyfile import PlyData, PlyElement

    labels = np.asarray(labels, dtype=np.int16)
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(CLASS_NAMES))
    colors[valid] = PALETTE[labels[valid]]
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        (label_field, "i2"),
    ]
    output = np.empty(len(labels), dtype=dtype)
    output["x"], output["y"], output["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    output["red"], output["green"], output["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    output[label_field] = labels
    PlyData([PlyElement.describe(output, "vertex")], text=False).write(str(path))


def save_results(
    args,
    gt_xyz,
    gt_target,
    gt_valid,
    gt_stats,
    pred,
    distance,
    carrier_meta,
    elapsed,
    support_distance=None,
):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metrics = compute_metrics(gt_target, pred)
    confusion = metrics.pop("confusion")
    summary = {
        "protocol": (
            "ScanNet++ official Top-100 vocabulary direct 3D single-label per-vertex evaluation v2"
            if CLASS_PROTOCOL == "scannetpp100"
            else f"ScanNet++-to-ScanNet{len(CLASS_NAMES)} direct 3D point evaluation v2"
        ),
        "classProtocol": CLASS_PROTOCOL,
        "classCount": len(CLASS_NAMES),
        "method": args.method,
        "predictionVocabulary": list(CLASS_NAMES),
        "allProtocolClassesCompeteInArgmax": True,
        "all19ClassesCompeteInArgmax": len(CLASS_NAMES) == 19,
        "metricClassScope": (
            "GT-present classes among the official ScanNet++ Top-100 targets"
            if CLASS_PROTOCOL == "scannetpp100"
            else f"GT-present classes among mapped ScanNet {len(CLASS_NAMES)} targets"
        ),
        "groundTruthMultiplicity": "single_label_vertex_field",
        "topK": 1,
        "geometryTransfer": {
            "knn_score": (
                "k-nearest native 3D semantic carrier class-score interpolation; "
                "argmax after fusion"
            ),
            "knn_latent": (
                "k-nearest native 3D semantic latent interpolation; apply the method-native "
                "latent normalization policy, then decode once at each GT vertex before text argmax"
            ),
        }.get(
            args.geometry_transfer,
            "one nearest native semantic carrier hard label in Euclidean 3D; no distance rejection",
        ),
        "geometryTransferConfig": {
            "mode": args.geometry_transfer,
            "neighborCount": int(args.knn_k) if args.geometry_transfer in {"knn_score", "knn_latent"} else 1,
            "weight": args.knn_weight if args.geometry_transfer in {"knn_score", "knn_latent"} else None,
            "scoreThenArgmax": args.geometry_transfer == "knn_score",
            "latentThenDecodeThenArgmax": args.geometry_transfer == "knn_latent",
            "carrierLatentNormalization": (
                carrier_meta.get("latentTransfer", {}).get("carrierNormalization")
                if args.geometry_transfer == "knn_latent"
                else None
            ),
            "fusedLatentNormalization": (
                carrier_meta.get("latentTransfer", {}).get("fusedLatentNormalization")
                if args.geometry_transfer == "knn_latent"
                else None
            ),
            "cameraProjectionUsed": False,
            "twoDimensionalPredictionUsed": False,
        },
        "uses2DSemanticPrediction": False,
        "gt": gt_stats,
        "carrier": carrier_meta,
        "nearestDistanceMeters": distance_summary(distance, gt_valid),
        "metrics": metrics,
        "elapsedSeconds": float(elapsed),
    }
    if support_distance is not None:
        summary["knnSupportDistanceMeters"] = distance_summary(support_distance, gt_valid)
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol_slug = "scannetpp_top100" if CLASS_PROTOCOL == "scannetpp100" else f"scannet{len(CLASS_NAMES)}"
    np.save(output / f"gt_{protocol_slug}_index.npy", gt_target)
    np.save(output / f"pred_{protocol_slug}_index.npy", pred)
    np.save(output / "nearest_distance_m.npy", distance)
    if support_distance is not None:
        np.save(output / "knn_support_distance_m.npy", support_distance)
    np.savetxt(output / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    with (output / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["perClass"][0].keys()))
        writer.writeheader()
        writer.writerows(metrics["perClass"])
    write_ply(output / f"gt_{protocol_slug}.ply", gt_xyz, gt_target, "gt_class")
    write_ply(output / f"pred_{protocol_slug}.ply", gt_xyz, pred, "pred_class")
    print(json.dumps({"method": args.method, **{k: metrics[k] for k in ("mIoU", "mAcc", "overallAcc", "predictionCoverage")}}, indent=2))
    print(f"outputs: {output}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Direct 3D ScanNet++ Top-100 or ScanNet 19/15/10 evaluation for InstGuide-GS"
    )
    parser.add_argument("--method", default="ours", choices=("ours",))
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--gt_mesh", required=True)
    parser.add_argument("--raw_classes", required=True)
    parser.add_argument("--benchmark_map", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--class_protocol",
        choices=tuple(CLASS_PROTOCOL_IDS),
        default="scannet19",
    )
    parser.add_argument(
        "--top100_classes",
        default="",
        help="Official ScanNet++ semantic_benchmark/top100.txt; required by scannetpp100.",
    )
    parser.add_argument("--point_cloud", default="")
    parser.add_argument("--alphaclip_checkpoint", default="")
    parser.add_argument("--alphaclip_package", default="")
    parser.add_argument("--alphaclip_download_root", default="")
    parser.add_argument("--alphaclip_model", default="ViT-B/16")
    parser.add_argument("--prompt_mode", choices=("single", "maskclustering"), default="single")
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--classify_chunk_size", type=int, default=131072)
    parser.add_argument("--query_chunk_size", type=int, default=262144)
    parser.add_argument(
        "--geometry_transfer",
        choices=("nearest_label", "knn_score", "knn_latent"),
        default="nearest_label",
        help=(
            "nearest_label reproduces the existing hard-label transfer. knn_score performs "
            "camera-free 3D interpolation of native carrier class scores before argmax. "
            "knn_latent fuses semantic latents before the "
            "trained nonlinear decoder, retaining the native normalization policy."
        ),
    )
    parser.add_argument("--knn_k", type=int, default=1)
    parser.add_argument(
        "--knn_weight",
        choices=("inverse_distance", "gaussian_rbf", "uniform"),
        default="inverse_distance",
    )
    parser.add_argument(
        "--score_query_chunk_size",
        type=int,
        default=16384,
        help="GT vertex chunk size for kNN score interpolation.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.knn_k < 1:
        raise ValueError("--knn_k must be at least 1")
    if args.score_query_chunk_size < 1:
        raise ValueError("--score_query_chunk_size must be at least 1")
    configure_class_protocol(args.class_protocol, args.top100_classes)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to decode and classify the semantic carriers")
    device = torch.device("cuda")
    gt_xyz, _raw_gt, gt_target, gt_valid, gt_stats = load_gt(args)
    start = time.perf_counter()
    carrier_xyz, carrier_output, carrier_meta = load_ours(args, device)
    support_distance = None
    if args.geometry_transfer == "knn_score":
        pred, distance, support_distance = transfer_scores_to_gt(
            carrier_xyz,
            carrier_output,
            gt_xyz,
            args.score_query_chunk_size,
            neighbor_count=args.knn_k,
            weight_mode=args.knn_weight,
            zero_score_is_invalid=bool(carrier_meta.get("zeroScoreIsInvalid", False)),
        )
    elif args.geometry_transfer == "knn_latent":
        pred, distance, support_distance = transfer_latents_to_gt(
            carrier_xyz,
            carrier_output["latents"],
            gt_xyz,
            carrier_output["decoder"],
            carrier_output["textFeatures"],
            device,
            args.score_query_chunk_size,
            neighbor_count=args.knn_k,
            weight_mode=args.knn_weight,
            normalize_fused_latent=bool(carrier_output.get("normalizeFusedLatent", False)),
        )
    else:
        pred, distance = transfer_to_gt(
            carrier_xyz, carrier_output, gt_xyz, args.query_chunk_size
        )
    elapsed = time.perf_counter() - start
    save_results(
        args,
        gt_xyz,
        gt_target,
        gt_valid,
        gt_stats,
        pred,
        distance,
        carrier_meta,
        elapsed,
        support_distance=support_distance,
    )


if __name__ == "__main__":
    main()
