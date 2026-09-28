from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


INSTANCE_TEACHER_FORMAT = "maskclustering_mlp_compressed_instance_level_teacher"
DIRECT_INSTANCE_MAP_FORMATS = {
    "maskclustering_mlp_compressed_instance_level_teacher",
    "maskclustering_final_compressed_instance_level_teacher",
    "maskclustering_decoded_instance_level_teacher_for_visualization",
}
INSTANCE_TEACHER_MAP_MODES = {"auto", "direct", "remap"}
INSTANCE_TEACHER_REF_PREFIX = "instance_teacher|"

_INSTANCE_TEACHER_STORES = {}


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _squeeze_hw(array: np.ndarray, path: Path, field_name: str) -> np.ndarray:
    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"{field_name} must have shape [H,W] or [1,H,W], got {array.shape}: {path}")
    return array


def _resolve_summary_dir(root: Path, summary: dict, key: str, fallback_name: str) -> Path:
    frame_data = summary.get("frame_data", {}) if isinstance(summary.get("frame_data", {}), dict) else {}
    raw_value = str(frame_data.get(key, "") or "").strip()
    candidates = []
    if raw_value:
        raw_path = Path(raw_value)
        candidates.append(raw_path if raw_path.is_absolute() else root / raw_path)
    candidates.append(root / fallback_name)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _normalize_map_mode(map_mode) -> str:
    mode = str(map_mode or "auto").strip().lower()
    if mode not in INSTANCE_TEACHER_MAP_MODES:
        raise ValueError(
            f"Unsupported instance teacher map mode {map_mode!r}. "
            f"Expected one of {sorted(INSTANCE_TEACHER_MAP_MODES)}."
        )
    return mode


def _resolve_map_mode(summary: dict, requested_mode="auto") -> str:
    requested_mode = _normalize_map_mode(requested_mode)
    if requested_mode != "auto":
        return requested_mode

    frame_data = summary.get("frame_data", {}) if isinstance(summary.get("frame_data", {}), dict) else {}
    summary_mode = str(
        frame_data.get("map_value_mode", "")
        or summary.get("map_value_mode", "")
        or summary.get("frame_instance_map_value_mode", "")
        or ""
    ).strip().lower()
    if summary_mode in {"instance_index", "instance", "direct"}:
        return "direct"
    if summary_mode in {"region_id", "region", "remap"}:
        return "remap"

    teacher_format = str(summary.get("format", "") or "")
    if teacher_format in DIRECT_INSTANCE_MAP_FORMATS:
        return "direct"
    if "instance_level_teacher" in teacher_format:
        return "direct"
    return "remap"


def is_instance_teacher_ref(value) -> bool:
    return isinstance(value, str) and value.startswith(INSTANCE_TEACHER_REF_PREFIX)


def _has_instance_teacher_core_files(root: Path) -> bool:
    return (
        (root / "summary.json").is_file()
        and (root / "instance_features.npy").is_file()
        and (root / "semantic_codec.pth").is_file()
    )


def _is_supported_instance_teacher_summary(root: Path, summary: dict) -> bool:
    teacher_format = str(summary.get("format", "") or "")
    if teacher_format == INSTANCE_TEACHER_FORMAT:
        return True
    return _has_instance_teacher_core_files(root)


def make_instance_teacher_ref(root, frame_id, map_mode="auto") -> str:
    root_path = Path(root).resolve()
    frame = Path(str(frame_id)).stem
    mode = _normalize_map_mode(map_mode)
    return f"{INSTANCE_TEACHER_REF_PREFIX}{root_path}|{frame}|{mode}"


def parse_instance_teacher_ref(value) -> tuple[Path, str, str]:
    if not is_instance_teacher_ref(value):
        raise ValueError(f"Not an instance teacher reference: {value}")
    body = str(value)[len(INSTANCE_TEACHER_REF_PREFIX):]
    parts = body.rsplit("|", 2)
    if len(parts) == 3 and parts[2] in INSTANCE_TEACHER_MAP_MODES:
        root_text, frame_id, map_mode = parts
    else:
        root_text, frame_id = body.rsplit("|", 1)
        map_mode = "auto"
    return Path(root_text), frame_id, map_mode


def is_instance_level_teacher_dir(path) -> bool:
    root = Path(path)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = _read_json(summary_path)
    except Exception:
        return False
    return _is_supported_instance_teacher_summary(root, summary)


def get_instance_teacher_store(root, map_mode="auto"):
    root_path = Path(root).resolve()
    mode = _normalize_map_mode(map_mode)
    cache_key = (str(root_path), mode)
    if cache_key not in _INSTANCE_TEACHER_STORES:
        _INSTANCE_TEACHER_STORES[cache_key] = InstanceTeacherStore(root_path, map_mode=mode)
    return _INSTANCE_TEACHER_STORES[cache_key]


def find_instance_teacher_ref(root, image_name, map_mode="auto"):
    if not is_instance_level_teacher_dir(root):
        return None
    frame_id = Path(str(image_name)).stem
    store = get_instance_teacher_store(root, map_mode=map_mode)
    if not store.has_frame(frame_id):
        return None
    return make_instance_teacher_ref(store.root, frame_id, map_mode=map_mode)


def load_instance_teacher_from_ref(ref, expected_dim):
    root, frame_id, map_mode = parse_instance_teacher_ref(ref)
    return get_instance_teacher_store(root, map_mode=map_mode).load_frame(frame_id, expected_dim)


class InstanceTeacherStore:
    def __init__(self, root, map_mode="auto"):
        self.root = Path(root).resolve()
        self.summary_path = self.root / "summary.json"
        if not self.summary_path.is_file():
            raise FileNotFoundError(f"Instance teacher summary.json not found: {self.summary_path}")
        self.summary = _read_json(self.summary_path)
        self.requested_map_mode = _normalize_map_mode(map_mode)
        self.map_mode = _resolve_map_mode(self.summary, self.requested_map_mode)
        if not _is_supported_instance_teacher_summary(self.root, self.summary):
            raise RuntimeError(
                "Unsupported instance teacher directory. Expected "
                f"format={INSTANCE_TEACHER_FORMAT!r}, or the core files "
                "summary.json, instance_features.npy, and semantic_codec.pth: "
                f"{self.summary_path}"
            )

        feature_path = self.root / "instance_features.npy"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Instance teacher feature file not found: {feature_path}")
        features = torch.as_tensor(np.load(feature_path), dtype=torch.float32)
        if features.ndim != 2:
            raise RuntimeError(f"instance_features.npy must have shape [N,C], got {tuple(features.shape)}")
        self.instance_features = F.normalize(features, dim=-1, eps=1e-6).contiguous()

        ids_path = self.root / "instance_ids.npy"
        self.instance_ids = np.load(ids_path) if ids_path.is_file() else np.arange(int(features.shape[0]))

        frame_table_path = self.root / "frame_table.json"
        self.frame_table_rows = _read_json(frame_table_path) if frame_table_path.is_file() else []
        self.frame_table = {
            str(row.get("frame_id", "") or ""): row
            for row in self.frame_table_rows
            if str(row.get("frame_id", "") or "")
        }

        mapping_path = self.root / "frame_region_to_instance.json"
        self.frame_region_to_instance = _read_json(mapping_path) if mapping_path.is_file() else {}

        self.frame_instance_maps_dir = _resolve_summary_dir(
            self.root,
            self.summary,
            "frame_instance_maps_dir",
            "frame_instance_maps",
        )
        self.frame_weights_dir = _resolve_summary_dir(
            self.root,
            self.summary,
            "frame_weights_dir",
            "frame_weights",
        )

    @property
    def feature_dim(self) -> int:
        return int(self.instance_features.shape[1])

    @property
    def num_instances(self) -> int:
        return int(self.instance_features.shape[0])

    def has_frame(self, frame_id) -> bool:
        frame = Path(str(frame_id)).stem
        return frame in self.frame_table or frame in self.frame_region_to_instance

    def frame_shape(self, frame_id):
        row = self.frame_table.get(Path(str(frame_id)).stem, {})
        shape = row.get("shape", None)
        if not shape:
            return None
        shape = tuple(int(value) for value in shape)
        if len(shape) == 3 and shape[0] == 1:
            return shape[1], shape[2]
        if len(shape) == 2:
            return shape
        return None

    def frame_map_path(self, frame_id) -> Path:
        return self.frame_instance_maps_dir / f"{Path(str(frame_id)).stem}.npy"

    def frame_weight_path(self, frame_id) -> Path:
        return self.frame_weights_dir / f"{Path(str(frame_id)).stem}.npy"

    def frame_issue(self, frame_id, expected_dim, image_hw=None):
        frame = Path(str(frame_id)).stem
        if int(expected_dim) != self.feature_dim:
            return f"stale: instance feature dim {self.feature_dim} != expected {int(expected_dim)}"
        if not self.has_frame(frame):
            return f"missing: frame {frame} not listed in instance teacher"
        if not self.frame_instance_maps_dir.is_dir():
            return f"missing: frame_instance_maps_dir does not exist: {self.frame_instance_maps_dir}"
        map_path = self.frame_map_path(frame)
        if not map_path.is_file():
            return f"missing: frame instance map does not exist: {map_path}"
        if not self.frame_weights_dir.is_dir():
            return f"missing: frame_weights_dir does not exist: {self.frame_weights_dir}"
        weight_path = self.frame_weight_path(frame)
        if not weight_path.is_file():
            return f"missing: frame weight map does not exist: {weight_path}"
        frame_shape = self.frame_shape(frame)
        if image_hw is not None and frame_shape is not None and tuple(frame_shape) != tuple(image_hw):
            return f"stale: frame map shape {tuple(frame_shape)} != image shape {tuple(image_hw)}"
        return None

    def _remap_frame_regions(self, frame_id: str, region_map: np.ndarray) -> np.ndarray:
        mapping = self.frame_region_to_instance.get(frame_id, None)
        if not mapping:
            remapped = np.full(region_map.shape, -1, dtype=np.int64)
            valid = (region_map >= 0) & (region_map < self.num_instances)
            remapped[valid] = region_map[valid].astype(np.int64, copy=False)
            return remapped

        max_region_id = max(int(region_id) for region_id in mapping.keys()) if mapping else -1
        max_map_id = int(region_map[region_map >= 0].max()) if np.any(region_map >= 0) else -1
        lut_size = max(max_region_id, max_map_id) + 1
        lut = np.full((max(lut_size, 0),), -1, dtype=np.int64)
        for region_id, record in mapping.items():
            instance_index = int(record.get("instance_index", -1))
            if 0 <= instance_index < self.num_instances:
                lut[int(region_id)] = instance_index

        remapped = np.full(region_map.shape, -1, dtype=np.int64)
        valid = (region_map >= 0) & (region_map < lut.shape[0])
        remapped[valid] = lut[region_map[valid].astype(np.int64, copy=False)]
        return remapped

    def _direct_frame_regions(self, region_map: np.ndarray) -> np.ndarray:
        remapped = np.full(region_map.shape, -1, dtype=np.int64)
        valid = (region_map >= 0) & (region_map < self.num_instances)
        remapped[valid] = region_map[valid].astype(np.int64, copy=False)
        return remapped

    def _frame_regions_to_instance_indices(self, frame_id: str, region_map: np.ndarray) -> np.ndarray:
        if self.map_mode == "direct":
            return self._direct_frame_regions(region_map)
        return self._remap_frame_regions(frame_id, region_map)

    def _frame_region_quality(self, frame_id: str) -> torch.Tensor:
        quality = torch.ones((self.num_instances,), dtype=torch.float32)
        mapping = self.frame_region_to_instance.get(frame_id, None)
        if not mapping:
            return quality
        counts = torch.zeros((self.num_instances,), dtype=torch.float32)
        sums = torch.zeros((self.num_instances,), dtype=torch.float32)
        for record in mapping.values():
            instance_index = int(record.get("instance_index", -1))
            if 0 <= instance_index < self.num_instances:
                confidence = float(record.get("teacher_confidence", 1.0))
                sums[instance_index] += max(0.0, min(confidence, 1.0))
                counts[instance_index] += 1.0
        observed = counts > 0
        quality[observed] = sums[observed] / counts[observed].clamp_min(1.0)
        return quality

    def load_frame(self, frame_id, expected_dim):
        frame = Path(str(frame_id)).stem
        issue = self.frame_issue(frame, expected_dim)
        if issue is not None:
            raise RuntimeError(f"Instance teacher frame is not usable for {frame}: {issue}")

        map_path = self.frame_map_path(frame)
        region_map_np = _squeeze_hw(np.load(map_path), map_path, "frame instance map").astype(np.int64, copy=False)
        instance_index_map_np = self._frame_regions_to_instance_indices(frame, region_map_np)
        region_id_map = torch.as_tensor(instance_index_map_np, dtype=torch.int64).contiguous()

        valid = (region_id_map >= 0) & (region_id_map < self.num_instances)
        region_id_map = torch.where(valid, region_id_map, torch.full_like(region_id_map, -1)).contiguous()

        weight_path = self.frame_weight_path(frame)
        weight_np = _squeeze_hw(np.load(weight_path), weight_path, "frame weight map").astype(np.float32, copy=False)
        if tuple(weight_np.shape) != tuple(region_id_map.shape):
            raise RuntimeError(
                f"frame weight map shape {tuple(weight_np.shape)} does not match "
                f"frame map shape {tuple(region_id_map.shape)}: {weight_path}"
            )
        pixel_quality = torch.as_tensor(weight_np, dtype=torch.float32).clamp(0.0, 1.0)
        pixel_quality = torch.where(valid, pixel_quality, torch.zeros_like(pixel_quality)).contiguous()

        if bool(valid.any().item()):
            region_area = torch.bincount(region_id_map[valid].long(), minlength=self.num_instances).float()
        else:
            region_area = torch.zeros((self.num_instances,), dtype=torch.float32)

        region_quality = self._frame_region_quality(frame)
        return {
            "image_name": frame,
            "region_id_map": region_id_map,
            "region_features": self.instance_features,
            "region_quality": region_quality.contiguous(),
            "clip_confidence": region_quality.contiguous(),
            "has_clip_confidence": False,
            "region_area": region_area.contiguous(),
            "pixel_quality": pixel_quality,
            "teacher_format": "instance_level",
            "instance_teacher_map_mode": self.map_mode,
        }
