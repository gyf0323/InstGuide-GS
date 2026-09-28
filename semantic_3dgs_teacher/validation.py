"""Read-only checks for portable, direct-index instance teachers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_frame_directory(root, summary, name):
    root = Path(root)
    frame_data = summary.get("frame_data", {})
    raw = frame_data.get(name + "_dir", "") if isinstance(frame_data, dict) else ""
    candidates = []
    if raw:
        path = Path(raw)
        candidates.append(path if path.is_absolute() else root / path)
    candidates.append(root / name)
    for path in candidates:
        if path.is_dir():
            return path
    raise ValueError(f"Missing {name}: {candidates}")


def _frame_array(directory, frame, allow_npz):
    path = directory / (frame + ".npy")
    if path.is_file():
        array = np.load(path, allow_pickle=False)
    elif allow_npz and path.with_suffix(".npz").is_file():
        path = path.with_suffix(".npz")
        with np.load(path, allow_pickle=False) as archive:
            array = archive["data"]
    else:
        raise ValueError(f"Missing frame array: {path}")
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not array.size:
        raise ValueError(f"Expected nonempty [H,W] or [1,H,W] array: {path}")
    return array


def validate_teacher(root, *, expected_dim=None, require_codec=False,
                     expected_frames=None, allow_npz=True):
    """Raise on incomplete data; expected_frames optionally maps stems to (H, W).

    The generator emits direct instance indices. Legacy region-remapped teachers
    remain handled by the runtime adapter, not this release-pipeline check.
    """
    root = Path(root)
    summary = _json(root / "summary.json")
    features = np.load(root / "instance_features.npy", allow_pickle=False)
    ids = np.load(root / "instance_ids.npy", allow_pickle=False)
    if features.ndim != 2 or not all(features.shape):
        raise ValueError("instance_features.npy must be a nonempty [N,D] table")
    if not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
        raise ValueError("Instance features must be finite floating-point values")
    if np.any(np.linalg.norm(features.astype(np.float32), axis=1) < 1e-6):
        raise ValueError("Instance features contain a zero semantic vector")
    if ids.ndim != 1 or len(ids) != len(features) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("instance_ids.npy must be integer [N], aligned with features")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate instance IDs")
    if expected_dim is not None and features.shape[1] != int(expected_dim):
        raise ValueError(f"Teacher dimension {features.shape[1]} != requested {expected_dim}")
    if not (root / "object_dict.npy").is_file():
        raise ValueError(f"Missing geometry sidecar: {root / 'object_dict.npy'}")
    # object_dict is the existing trusted MaskClustering pickle-based contract.
    objects = np.load(root / "object_dict.npy", allow_pickle=True).item()
    if not isinstance(objects, dict) or not objects:
        raise ValueError("object_dict.npy must contain nonempty instance records")
    objects = {int(key): value for key, value in objects.items()}
    for instance_id in ids:
        record = objects.get(int(instance_id))
        if not isinstance(record, dict) or "point_ids" not in record:
            raise ValueError(f"Missing geometry membership for instance {instance_id}")
        points = np.asarray(record["point_ids"])
        if points.ndim != 1 or not points.size or not np.issubdtype(points.dtype, np.integer) or np.any(points < 0):
            raise ValueError(f"Invalid point indices for instance {instance_id}")

    if require_codec:
        from semantic_3dgs_teacher.codec_utils import load_semantic_codec

        codec, _ = load_semantic_codec(root / "semantic_codec.pth", map_location="cpu")
        if int(codec.latent_dim) != features.shape[1]:
            raise ValueError("Codec latent dimension does not match instance features")
        if int(codec.input_dim) != 512:
            raise ValueError("The Alpha-CLIP teacher must decode to 512 dimensions")

    table = _json(root / "frame_table.json")
    if not isinstance(table, list) or not table:
        raise ValueError("frame_table.json must list the exported observations")
    rows = {}
    for row in table:
        frame = str(row.get("frame_id", ""))
        if not frame or Path(frame).name != frame or frame in rows:
            raise ValueError(f"Invalid or duplicate frame ID: {frame!r}")
        rows[frame] = row
    declared = summary.get("teacher_files", {}).get("num_images")
    if declared is not None and int(declared) != len(rows):
        raise ValueError("summary.json frame count does not match frame_table.json")
    frames = rows if expected_frames is None else expected_frames
    if not frames:
        raise ValueError("No frames selected for teacher validation")
    missing = sorted(set(frames) - set(rows))
    if missing:
        raise ValueError(f"Teacher missing selected frames: {missing[:10]}")
    maps = resolve_frame_directory(root, summary, "frame_instance_maps")
    weights = resolve_frame_directory(root, summary, "frame_weights")
    for frame in frames:
        labels = _frame_array(maps, frame, allow_npz)
        quality = _frame_array(weights, frame, allow_npz)
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Noninteger instance map: {frame}")
        if labels.min() < -1 or labels.max() >= len(ids):
            raise ValueError(f"Instance index out of range: {frame}")
        if labels.shape != quality.shape:
            raise ValueError(f"Map/weight shape mismatch: {frame}")
        shape = tuple(rows[frame].get("shape", ()))
        if len(shape) == 3 and shape[0] == 1:
            shape = shape[1:]
        if shape and shape != labels.shape:
            raise ValueError(f"Frame metadata shape mismatch: {frame}")
        if expected_frames is not None and tuple(expected_frames[frame]) != labels.shape:
            raise ValueError(f"RGB/teacher shape mismatch: {frame}")
        if not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
            raise ValueError(f"Frame weights must be finite and in [0,1]: {frame}")
    return {"instances": len(ids), "dimension": int(features.shape[1]),
            "checked_frames": len(frames), "listed_frames": len(rows),
            "frame_instance_maps_dir": str(maps), "frame_weights_dir": str(weights)}
