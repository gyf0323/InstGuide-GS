from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np


GEOMETRY_STATE_UNKNOWN = 0
GEOMETRY_STATE_INTERIOR = 1
GEOMETRY_STATE_BOUNDARY = 2
GEOMETRY_STATE_NAMES = {
    GEOMETRY_STATE_UNKNOWN: "unknown",
    GEOMETRY_STATE_INTERIOR: "interior",
    GEOMETRY_STATE_BOUNDARY: "boundary",
}
GEOMETRY_BINDING_REQUIRED_FILES = (
    "points_xyz.npy",
    "point_instance_ids.npy",
    "source_point_ids.npy",
    "binding_meta.json",
)


@dataclass(frozen=True)
class InstanceGeometryBinding:
    root: Path
    points_xyz: np.ndarray
    instance_indices: np.ndarray
    raw_instance_ids: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class AnchorGeometryAssignment:
    instance_indices: np.ndarray
    states: np.ndarray
    confidence: np.ndarray
    support_fraction: np.ndarray
    residual_anchor_indices: np.ndarray
    stats: dict


def resolve_geometry_binding_dir(teacher_root, override="") -> Path:
    raw = str(override or "").strip()
    return Path(raw).expanduser().resolve() if raw else Path(teacher_root).resolve() / "geometry_binding"


def geometry_binding_missing_files(binding_dir):
    root = Path(binding_dir).expanduser().resolve()
    return [name for name in GEOMETRY_BINDING_REQUIRED_FILES if not (root / name).is_file()]


def geometry_binding_is_complete(binding_dir) -> bool:
    return not geometry_binding_missing_files(binding_dir)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _load_ply_xyz(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError(
            "Reading the MaskClustering source PLY requires the 'plyfile' package. "
            "Install it in the training environment before using automatic instance-geometry preparation."
        ) from exc

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise RuntimeError(f"PLY has no vertex element: {path}")
    vertex = ply["vertex"]
    missing = [name for name in ("x", "y", "z") if name not in vertex.data.dtype.names]
    if missing:
        raise RuntimeError(f"PLY is missing coordinates {missing}: {path}")
    return np.column_stack(
        [np.asarray(vertex[name], dtype=np.float32) for name in ("x", "y", "z")]
    )


def build_point_instance_owners(num_points, object_dict, valid_teacher_ids):
    owners = np.full(int(num_points), -1, dtype=np.int64)
    conflicts = np.zeros(int(num_points), dtype=bool)
    skipped_instances = []
    invalid_point_ids = 0
    for raw_id, payload in object_dict.items():
        instance_id = int(raw_id)
        if instance_id not in valid_teacher_ids:
            skipped_instances.append(instance_id)
            continue
        if not isinstance(payload, dict):
            raise RuntimeError(f"object_dict entry {raw_id!r} must be a dict.")
        point_ids = np.asarray(payload.get("point_ids", []), dtype=np.int64).reshape(-1)
        valid = (point_ids >= 0) & (point_ids < int(num_points))
        invalid_point_ids += int((~valid).sum())
        point_ids = np.unique(point_ids[valid])
        if point_ids.size == 0:
            continue
        existing = owners[point_ids]
        collision = (existing >= 0) & (existing != instance_id)
        conflicts[point_ids[collision]] = True
        assignable = existing < 0
        owners[point_ids[assignable]] = instance_id
    owners[conflicts] = -2
    return owners, {
        "conflicting_points": int(conflicts.sum()),
        "invalid_point_ids": int(invalid_point_ids),
        "skipped_object_instance_ids": sorted(skipped_instances),
    }


def export_instance_geometry_binding(
    teacher_dir,
    object_dict_path,
    point_cloud_path,
    output_dir=None,
    coordinate_system="geometry_3dgs_world",
    overwrite=False,
):
    teacher_root = Path(teacher_dir).expanduser().resolve()
    object_path = Path(object_dict_path).expanduser().resolve()
    point_path = Path(point_cloud_path).expanduser().resolve()
    target_root = (
        Path(output_dir).expanduser().resolve()
        if str(output_dir or "").strip()
        else teacher_root / "geometry_binding"
    )
    instance_ids_path = teacher_root / "instance_ids.npy"
    if not instance_ids_path.is_file():
        raise FileNotFoundError(f"Teacher instance_ids.npy not found: {instance_ids_path}")
    if not object_path.is_file():
        raise FileNotFoundError(f"object_dict.npy not found: {object_path}")
    if not point_path.is_file():
        raise FileNotFoundError(f"Point cloud PLY not found: {point_path}")
    if target_root.exists() and not target_root.is_dir():
        raise RuntimeError(f"Geometry binding output path is not a directory: {target_root}")
    if target_root.exists() and any(target_root.iterdir()) and not bool(overwrite):
        raise RuntimeError(
            f"Geometry binding output directory is not empty: {target_root}. "
            "Remove the incomplete cache or use overwrite explicitly."
        )

    teacher_ids = np.asarray(np.load(instance_ids_path)).reshape(-1)
    if teacher_ids.size == 0 or not np.issubdtype(teacher_ids.dtype, np.integer):
        raise RuntimeError(f"Teacher instance_ids.npy must contain a non-empty integer vector: {instance_ids_path}")
    teacher_ids = teacher_ids.astype(np.int64, copy=False)
    if np.unique(teacher_ids).shape[0] != teacher_ids.shape[0]:
        raise RuntimeError(f"Teacher instance_ids.npy contains duplicate IDs: {instance_ids_path}")
    valid_teacher_ids = {int(value) for value in teacher_ids.tolist()}

    object_dict = np.load(object_path, allow_pickle=True).item()
    if not isinstance(object_dict, dict):
        raise RuntimeError(f"object_dict.npy must contain a dict: {object_path}")
    points = _load_ply_xyz(point_path)
    owners, diagnostics = build_point_instance_owners(points.shape[0], object_dict, valid_teacher_ids)
    finite = np.isfinite(points).all(axis=1)
    nonfinite_labeled = int(np.sum((owners >= 0) & (~finite)))
    selected = (owners >= 0) & finite
    if not np.any(selected):
        raise RuntimeError(
            "No finite point-instance memberships matched the teacher instance IDs. Check that object_dict.npy "
            "and instance_ids.npy belong to the same teacher run and that the exact source PLY was selected."
        )

    present_ids = sorted({int(value) for value in owners[selected].tolist()})
    missing_teacher_ids = sorted(valid_teacher_ids.difference(present_ids))
    metadata = {
        "format": "maskclustering_instance_geometry_binding",
        "format_version": 2,
        "coordinate_system": str(coordinate_system),
        "teacher_dir": str(teacher_root),
        "object_dict_path": str(object_path),
        "source_point_cloud": str(point_path),
        "source_point_count": int(points.shape[0]),
        "exported_point_count": int(selected.sum()),
        "teacher_instance_count": int(teacher_ids.shape[0]),
        "teacher_instance_ids": [int(value) for value in teacher_ids.tolist()],
        "exported_instance_count": int(len(present_ids)),
        "exported_raw_instance_ids": present_ids,
        "missing_teacher_instance_ids": missing_teacher_ids,
        "nonfinite_labeled_points": nonfinite_labeled,
        **diagnostics,
    }

    target_root.parent.mkdir(parents=True, exist_ok=True)
    working_root = target_root.parent / (
        f".{target_root.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        working_root.mkdir(parents=False, exist_ok=False)
        np.save(
            working_root / "points_xyz.npy",
            np.ascontiguousarray(points[selected], dtype=np.float32),
        )
        np.save(
            working_root / "point_instance_ids.npy",
            np.ascontiguousarray(owners[selected], dtype=np.int64),
        )
        np.save(
            working_root / "source_point_ids.npy",
            np.flatnonzero(selected).astype(np.int64),
        )
        with (working_root / "binding_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        validate_instance_geometry_binding(working_root, teacher_ids)
        if target_root.exists():
            if any(target_root.iterdir()):
                if not bool(overwrite):
                    raise RuntimeError(f"Geometry binding output became non-empty: {target_root}")
                shutil.rmtree(target_root)
            else:
                target_root.rmdir()
        working_root.replace(target_root)
    finally:
        if working_root.exists():
            shutil.rmtree(working_root, ignore_errors=True)
    return metadata


def _resolve_geometry_source_file(
    teacher_root,
    source_root,
    value,
    defaults,
    flag_name,
    description,
):
    raw = str(value or "").strip()
    candidates = []
    if raw:
        requested = Path(raw).expanduser()
        if requested.is_absolute():
            candidates.append(requested)
        else:
            candidates.extend((source_root / requested, teacher_root / requested, Path.cwd() / requested))
    else:
        candidates.extend(defaults)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(resolved)
        if resolved.is_file():
            return resolved

    searched = "\n  - ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(
        f"Could not locate the {description} required for automatic instance-geometry preparation. "
        f"Searched:\n  - {searched}\nSet --{flag_name} to the exact file path."
    )


def prepare_instance_geometry_binding(
    teacher_root,
    teacher_instance_ids,
    source_root,
    binding_override="",
    auto_prepare=True,
    object_dict_override="",
    point_cloud_override="",
    log_callback=None,
) -> InstanceGeometryBinding:
    teacher_root = Path(teacher_root).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    binding_dir = resolve_geometry_binding_dir(teacher_root, override=binding_override)

    def emit(message):
        if log_callback is not None:
            log_callback(message)

    if geometry_binding_is_complete(binding_dir):
        binding = validate_instance_geometry_binding(binding_dir, teacher_instance_ids)
        emit(
            f"[Instance Geometry] Reusing cached geometry binding: {binding.root} "
            f"(points={binding.points_xyz.shape[0]})"
        )
        return binding

    missing = geometry_binding_missing_files(binding_dir)
    if binding_dir.exists():
        if not binding_dir.is_dir():
            raise RuntimeError(f"Instance geometry binding path is not a directory: {binding_dir}")
        if any(binding_dir.iterdir()):
            raise RuntimeError(
                f"Instance geometry binding cache is incomplete under {binding_dir}; missing: "
                f"{', '.join(missing)}. Remove or repair this partial cache before training."
            )
    if not bool(auto_prepare):
        raise FileNotFoundError(
            f"Instance geometry binding is missing under {binding_dir}; missing: {', '.join(missing)}. "
            "Automatic preparation is disabled by --no-instance_geometry_auto_prepare."
        )

    object_path = _resolve_geometry_source_file(
        teacher_root,
        source_root,
        object_dict_override,
        defaults=(
            teacher_root / "object_dict.npy",
            source_root / "output" / "object" / teacher_root.name / "object_dict.npy",
        ),
        flag_name="instance_geometry_object_dict",
        description="MaskClustering object_dict.npy",
    )
    point_cloud_path = _resolve_geometry_source_file(
        teacher_root,
        source_root,
        point_cloud_override,
        defaults=(source_root / "mesh_aligned_0.05.ply",),
        flag_name="instance_geometry_point_cloud",
        description="exact MaskClustering source point cloud",
    )
    emit(
        "[Instance Geometry] Geometry binding cache is missing; preparing it automatically. "
        f"object_dict={object_path}, point_cloud={point_cloud_path}, output={binding_dir}"
    )
    try:
        metadata = export_instance_geometry_binding(
            teacher_dir=teacher_root,
            object_dict_path=object_path,
            point_cloud_path=point_cloud_path,
            output_dir=binding_dir,
        )
    except PermissionError as exc:
        raise RuntimeError(
            f"Cannot write the automatic geometry binding cache under {binding_dir}. "
            "Choose a writable path with --instance_geometry_binding."
        ) from exc

    binding = validate_instance_geometry_binding(binding_dir, teacher_instance_ids)
    emit(
        "[Instance Geometry] Automatic geometry binding preparation complete: "
        f"points={metadata['exported_point_count']}, instances={metadata['exported_instance_count']}, "
        f"conflicts_excluded={metadata['conflicting_points']}, output={binding.root}"
    )
    return binding


def load_instance_geometry_binding(binding_dir, teacher_instance_ids) -> InstanceGeometryBinding:
    root = Path(binding_dir).expanduser().resolve()
    points_path = root / "points_xyz.npy"
    ids_path = root / "point_instance_ids.npy"
    if not points_path.is_file() or not ids_path.is_file():
        raise FileNotFoundError(
            "Instance geometry binding requires points_xyz.npy and point_instance_ids.npy under "
            f"{root}. Generate them with tools/export_instance_geometry_binding.py."
        )

    points = np.asarray(np.load(points_path), dtype=np.float32)
    raw_ids = np.asarray(np.load(ids_path)).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"points_xyz.npy must have shape [N,3], got {points.shape}: {points_path}")
    if raw_ids.shape[0] != points.shape[0]:
        raise RuntimeError(
            f"point_instance_ids.npy has {raw_ids.shape[0]} rows but points_xyz.npy has {points.shape[0]}: {root}"
        )
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise RuntimeError(f"point_instance_ids.npy must contain integers, got {raw_ids.dtype}: {ids_path}")
    raw_ids = raw_ids.astype(np.int64, copy=False)

    teacher_ids = np.asarray(teacher_instance_ids).reshape(-1)
    raw_to_index = {int(raw_id): index for index, raw_id in enumerate(teacher_ids.tolist())}
    instance_indices = np.asarray([raw_to_index.get(int(raw_id), -1) for raw_id in raw_ids], dtype=np.int64)
    finite = np.isfinite(points).all(axis=1)
    valid = finite & (instance_indices >= 0)
    if not np.any(valid):
        raise RuntimeError(
            "Geometry binding contains no finite points whose raw instance IDs occur in instance_ids.npy. "
            f"binding={root}"
        )

    metadata = _read_json(root / "binding_meta.json")
    metadata = dict(metadata)
    metadata["loaded_points"] = int(valid.sum())
    metadata["dropped_nonfinite_points"] = int((~finite).sum())
    metadata["dropped_unknown_instance_points"] = int((finite & (instance_indices < 0)).sum())
    return InstanceGeometryBinding(
        root=root,
        points_xyz=np.ascontiguousarray(points[valid], dtype=np.float32),
        instance_indices=np.ascontiguousarray(instance_indices[valid], dtype=np.int64),
        raw_instance_ids=np.ascontiguousarray(raw_ids[valid]),
        metadata=metadata,
    )


def validate_instance_geometry_binding(binding_dir, teacher_instance_ids) -> InstanceGeometryBinding:
    root = Path(binding_dir).expanduser().resolve()
    missing = geometry_binding_missing_files(root)
    if missing:
        raise RuntimeError(
            f"Geometry binding is incomplete under {root}; missing files: {', '.join(missing)}"
        )

    teacher_ids = np.asarray(teacher_instance_ids).reshape(-1)
    if teacher_ids.size == 0 or not np.issubdtype(teacher_ids.dtype, np.integer):
        raise RuntimeError("teacher_instance_ids must be a non-empty integer vector.")
    teacher_ids = teacher_ids.astype(np.int64, copy=False)
    if np.unique(teacher_ids).shape[0] != teacher_ids.shape[0]:
        raise RuntimeError("teacher_instance_ids contains duplicate IDs.")

    binding = load_instance_geometry_binding(root, teacher_ids)
    if int(binding.metadata.get("dropped_unknown_instance_points", 0)) != 0:
        raise RuntimeError(
            "Geometry binding contains point instance IDs that are absent from the current teacher. "
            f"binding={root}"
        )

    metadata = binding.metadata
    metadata_teacher_count = metadata.get("teacher_instance_count")
    if metadata_teacher_count is not None and int(metadata_teacher_count) != int(teacher_ids.shape[0]):
        raise RuntimeError(
            "Geometry binding teacher instance count does not match the current teacher: "
            f"binding={metadata_teacher_count}, teacher={teacher_ids.shape[0]}, root={root}"
        )
    metadata_teacher_ids = metadata.get("teacher_instance_ids")
    if metadata_teacher_ids is not None:
        recorded = np.asarray(metadata_teacher_ids, dtype=np.int64).reshape(-1)
        if not np.array_equal(recorded, teacher_ids):
            raise RuntimeError(
                "Geometry binding was generated for a different teacher instance ID table. "
                f"binding={root}"
            )
    return binding


class VoxelInstanceIndex:
    """Dependency-free local instance voting for fixed Scaffold-GS anchor geometry."""

    _NEIGHBOR_OFFSETS = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=np.int64)

    def __init__(self, points_xyz, instance_indices, num_instances, cell_size):
        self.points = np.asarray(points_xyz, dtype=np.float32)
        self.labels = np.asarray(instance_indices, dtype=np.int64).reshape(-1)
        self.num_instances = int(num_instances)
        self.cell_size = float(cell_size)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or self.points.shape[0] != self.labels.shape[0]:
            raise ValueError("points_xyz and instance_indices must have shapes [N,3] and [N].")
        if self.points.shape[0] == 0 or self.num_instances <= 0 or self.cell_size <= 0.0:
            raise ValueError("VoxelInstanceIndex requires points, positive num_instances, and positive cell_size.")
        if np.any(self.labels < 0) or np.any(self.labels >= self.num_instances):
            raise ValueError("instance_indices contain values outside [0, num_instances).")

        cells = np.floor(self.points / self.cell_size).astype(np.int64)
        self.min_cell = cells.min(axis=0) - 1
        self.max_cell = cells.max(axis=0) + 1
        self.dims = self.max_cell - self.min_cell + 1
        total_cells = int(self.dims[0]) * int(self.dims[1]) * int(self.dims[2])
        if total_cells <= 0 or total_cells >= np.iinfo(np.int64).max:
            raise RuntimeError(f"Geometry voxel key range is invalid: dims={self.dims.tolist()}")

        point_keys, in_bounds = self._encode_cells(cells)
        if not bool(np.all(in_bounds)):
            raise RuntimeError("Internal geometry voxel indexing error: source points fell outside their own bounds.")
        self.keys, inverse, self.cell_counts = np.unique(
            point_keys, return_inverse=True, return_counts=True
        )

        label_stride = self.num_instances
        pair_keys = inverse.astype(np.int64) * label_stride + self.labels
        unique_pairs, pair_counts = np.unique(pair_keys, return_counts=True)
        pair_cells = unique_pairs // label_stride
        pair_labels = unique_pairs % label_stride
        group_cells, group_starts = np.unique(pair_cells, return_index=True)
        max_counts = np.maximum.reduceat(pair_counts, group_starts)
        max_for_pair = max_counts[np.searchsorted(group_cells, pair_cells)]
        candidates = np.flatnonzero(pair_counts == max_for_pair)
        candidate_cells = pair_cells[candidates]
        _, first_candidate = np.unique(candidate_cells, return_index=True)
        selected = candidates[first_candidate]

        self.cell_labels = np.full(self.keys.shape[0], -1, dtype=np.int64)
        self.cell_dominant_counts = np.zeros(self.keys.shape[0], dtype=np.int64)
        self.cell_labels[pair_cells[selected]] = pair_labels[selected]
        self.cell_dominant_counts[pair_cells[selected]] = pair_counts[selected]
        self.cell_purity = self.cell_dominant_counts.astype(np.float32) / np.maximum(
            self.cell_counts.astype(np.float32), 1.0
        )
        self.cell_centroids = np.stack(
            [
                np.bincount(inverse, weights=self.points[:, axis], minlength=self.keys.shape[0])
                / self.cell_counts
                for axis in range(3)
            ],
            axis=1,
        ).astype(np.float32)

    def _encode_cells(self, cells):
        cells = np.asarray(cells, dtype=np.int64)
        local = cells - self.min_cell
        in_bounds = np.all((local >= 0) & (local < self.dims), axis=-1)
        safe = np.where(in_bounds[..., None], local, 0)
        keys = (safe[..., 0] * self.dims[1] + safe[..., 1]) * self.dims[2] + safe[..., 2]
        return keys.astype(np.int64, copy=False), in_bounds

    def query(self, query_xyz, max_distance, chunk_size=0):
        queries = np.asarray(query_xyz, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"query_xyz must have shape [Q,3], got {queries.shape}")
        max_distance = float(max_distance)
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive.")

        labels_out = np.full(queries.shape[0], -1, dtype=np.int64)
        confidence_out = np.zeros(queries.shape[0], dtype=np.float32)
        nearest_out = np.full(queries.shape[0], np.inf, dtype=np.float32)
        if chunk_size <= 0:
            target_score_values = 2_000_000
            chunk_size = max(512, min(32768, target_score_values // max(self.num_instances, 1)))

        eps = max(self.cell_size * 0.05, 1e-6)
        for start in range(0, queries.shape[0], chunk_size):
            end = min(start + chunk_size, queries.shape[0])
            query = queries[start:end]
            query_cells = np.floor(query / self.cell_size).astype(np.int64)
            neighbor_cells = query_cells[:, None, :] + self._NEIGHBOR_OFFSETS[None, :, :]
            neighbor_keys, in_bounds = self._encode_cells(neighbor_cells)
            flat_keys = neighbor_keys.reshape(-1)
            positions = np.searchsorted(self.keys, flat_keys)
            safe_positions = np.minimum(positions, self.keys.shape[0] - 1)
            hits = in_bounds.reshape(-1) & (positions < self.keys.shape[0])
            hits &= self.keys[safe_positions] == flat_keys

            cell_labels = np.full(flat_keys.shape[0], -1, dtype=np.int64)
            cell_purity = np.zeros(flat_keys.shape[0], dtype=np.float32)
            cell_centroids = np.zeros((flat_keys.shape[0], 3), dtype=np.float32)
            cell_labels[hits] = self.cell_labels[safe_positions[hits]]
            cell_purity[hits] = self.cell_purity[safe_positions[hits]]
            cell_centroids[hits] = self.cell_centroids[safe_positions[hits]]

            repeated_queries = np.repeat(query, self._NEIGHBOR_OFFSETS.shape[0], axis=0)
            distances = np.linalg.norm(repeated_queries - cell_centroids, axis=1).astype(np.float32)
            valid = hits & (cell_labels >= 0) & (distances <= max_distance)
            inverse_distance = np.zeros_like(distances)
            inverse_distance[valid] = 1.0 / (distances[valid] + eps)
            vote_weights = inverse_distance * cell_purity

            query_count = end - start
            scores = np.zeros((query_count, self.num_instances), dtype=np.float32)
            row_ids = np.repeat(np.arange(query_count, dtype=np.int64), self._NEIGHBOR_OFFSETS.shape[0])
            np.add.at(scores, (row_ids[valid], cell_labels[valid]), vote_weights[valid])
            totals = scores.sum(axis=1)
            top_labels = scores.argmax(axis=1)
            top_scores = scores[np.arange(query_count), top_labels]

            quality_numerator = np.zeros(query_count, dtype=np.float32)
            quality_denominator = np.zeros(query_count, dtype=np.float32)
            np.add.at(quality_numerator, row_ids[valid], inverse_distance[valid] * cell_purity[valid])
            np.add.at(quality_denominator, row_ids[valid], inverse_distance[valid])
            local_quality = quality_numerator / np.maximum(quality_denominator, 1e-8)
            vote_purity = top_scores / np.maximum(totals, 1e-8)
            local_confidence = vote_purity * local_quality

            distance_matrix = distances.reshape(query_count, -1)
            valid_matrix = valid.reshape(query_count, -1)
            local_nearest = np.min(np.where(valid_matrix, distance_matrix, np.inf), axis=1)
            has_support = totals > 0.0
            supported_rows = np.flatnonzero(has_support)
            labels_out[start + supported_rows] = top_labels[supported_rows]
            confidence_out[start + supported_rows] = local_confidence[supported_rows]
            nearest_out[start:end] = local_nearest

        return labels_out, confidence_out, nearest_out


def classify_anchor_geometry(
    anchor_xyz,
    child_xyz,
    binding: InstanceGeometryBinding,
    num_instances,
    max_distance,
    interior_purity=0.90,
    min_support_fraction=0.50,
    min_query_confidence=0.50,
    unknown_uses_residual=True,
) -> AnchorGeometryAssignment:
    anchors = np.asarray(anchor_xyz, dtype=np.float32)
    children = np.asarray(child_xyz, dtype=np.float32)
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError(f"anchor_xyz must have shape [N,3], got {anchors.shape}")
    if children.ndim != 3 or children.shape[0] != anchors.shape[0] or children.shape[2] != 3:
        raise ValueError(f"child_xyz must have shape [N,K,3], got {children.shape}")
    if not 0.0 < float(interior_purity) <= 1.0:
        raise ValueError("interior_purity must be in (0,1].")
    if not 0.0 <= float(min_support_fraction) <= 1.0:
        raise ValueError("min_support_fraction must be in [0,1].")

    spatial_index = VoxelInstanceIndex(
        binding.points_xyz,
        binding.instance_indices,
        num_instances=num_instances,
        cell_size=float(max_distance),
    )
    samples = np.concatenate([anchors[:, None, :], children], axis=1)
    sample_labels, sample_confidence, _ = spatial_index.query(
        samples.reshape(-1, 3), max_distance=float(max_distance)
    )
    sample_labels = sample_labels.reshape(samples.shape[:2])
    sample_confidence = sample_confidence.reshape(samples.shape[:2])
    sample_valid = sample_labels >= 0
    support_fraction = sample_valid.mean(axis=1).astype(np.float32)

    anchor_instance = np.full(anchors.shape[0], -1, dtype=np.int64)
    consensus = np.zeros(anchors.shape[0], dtype=np.float32)
    mean_confidence = np.zeros(anchors.shape[0], dtype=np.float32)
    chunk_size = max(512, min(32768, 2_000_000 // max(int(num_instances), 1)))
    for start in range(0, anchors.shape[0], chunk_size):
        end = min(start + chunk_size, anchors.shape[0])
        labels = sample_labels[start:end]
        confidence = sample_confidence[start:end]
        valid = labels >= 0
        scores = np.zeros((end - start, int(num_instances)), dtype=np.float32)
        rows = np.repeat(np.arange(end - start, dtype=np.int64), labels.shape[1])
        flat_valid = valid.reshape(-1)
        np.add.at(
            scores,
            (rows[flat_valid], labels.reshape(-1)[flat_valid]),
            confidence.reshape(-1)[flat_valid],
        )
        totals = scores.sum(axis=1)
        top = scores.argmax(axis=1)
        top_scores = scores[np.arange(end - start), top]
        has_support = totals > 0.0
        supported_rows = np.flatnonzero(has_support)
        anchor_instance[start + supported_rows] = top[supported_rows]
        consensus[start + supported_rows] = top_scores[supported_rows] / totals[supported_rows]
        valid_counts = valid.sum(axis=1)
        mean_confidence[start:end] = confidence.sum(axis=1) / np.maximum(valid_counts, 1)

    geometry_confidence = (consensus * mean_confidence).astype(np.float32)
    enough_support = support_fraction >= float(min_support_fraction)
    interior = (
        enough_support
        & (consensus >= float(interior_purity))
        & (mean_confidence >= float(min_query_confidence))
        & (anchor_instance >= 0)
    )
    boundary = enough_support & (~interior) & (anchor_instance >= 0)
    states = np.full(anchors.shape[0], GEOMETRY_STATE_UNKNOWN, dtype=np.uint8)
    states[interior] = GEOMETRY_STATE_INTERIOR
    states[boundary] = GEOMETRY_STATE_BOUNDARY
    anchor_instance[~enough_support] = -1

    residual_mask = boundary.copy()
    if bool(unknown_uses_residual):
        residual_mask |= states == GEOMETRY_STATE_UNKNOWN
    residual_indices = np.flatnonzero(residual_mask).astype(np.int64)
    state_counts = {
        GEOMETRY_STATE_NAMES[state]: int(np.sum(states == state))
        for state in GEOMETRY_STATE_NAMES
    }
    total = max(int(states.shape[0]), 1)
    stats = {
        "num_anchors": int(states.shape[0]),
        "num_residual_anchors": int(residual_indices.shape[0]),
        "residual_anchor_ratio": float(residual_indices.shape[0] / total),
        "state_counts": state_counts,
        "state_ratios": {name: float(count / total) for name, count in state_counts.items()},
        "mean_support_fraction": float(support_fraction.mean()),
        "mean_geometry_confidence": float(geometry_confidence.mean()),
        "max_distance": float(max_distance),
        "interior_purity": float(interior_purity),
        "min_support_fraction": float(min_support_fraction),
        "min_query_confidence": float(min_query_confidence),
        "unknown_uses_residual": bool(unknown_uses_residual),
    }
    return AnchorGeometryAssignment(
        instance_indices=np.ascontiguousarray(anchor_instance),
        states=np.ascontiguousarray(states),
        confidence=np.ascontiguousarray(geometry_confidence),
        support_fraction=np.ascontiguousarray(support_fraction),
        residual_anchor_indices=np.ascontiguousarray(residual_indices),
        stats=stats,
    )
