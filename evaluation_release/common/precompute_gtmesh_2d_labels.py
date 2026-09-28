"""Pre-rasterize semantic GT meshes into reusable 2D evaluation labels.

This script is intentionally independent of trained ILGS models, language
teachers, object masks, OpenCLIP, and the custom Gaussian CUDA extensions.  It
reads COLMAP text cameras and the semantic GT assets shipped with the eight
evaluation scenes, then writes deterministic per-view label maps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


IGNORE_INDEX = -1
PNG_IGNORE_INDEX = np.uint16(65535)
INT64_MAX = torch.iinfo(torch.int64).max
TRIANGLE_ID_MASK = (1 << 32) - 1
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png"}
DEFAULT_EXCLUDED_DIRS = {"js"}


@dataclass(frozen=True)
class ColmapIntrinsics:
    camera_id: int
    model: str
    width: int
    height: int
    params: Tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    @property
    def stem(self) -> str:
        return Path(self.name).stem


@dataclass
class RasterCamera:
    image_name: str
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    znear: float = 0.01
    zfar: float = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-rasterize ScanNet++/Replica semantic GT meshes into 2D labels."
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--exclude_dirs", nargs="*", default=sorted(DEFAULT_EXCLUDED_DIRS))
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument(
        "--cs_target_height",
        type=int,
        default=1080,
        help="Output height for cs* scenes; 1080 matches the trained images_1080 models. Use 0 for native.",
    )
    parser.add_argument(
        "--replica_target_height",
        type=int,
        default=0,
        help="Output height for Replica scenes. Zero keeps the native 1200x680 images.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max_views", type=int, default=0, help="Limit views per scene for a smoke test.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--replica_object_overrides",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with explicit per-scene Replica object_id to class_id corrections. "
            "Corrections are validated and recorded in the output manifest; the source GT is never modified."
        ),
    )
    parser.add_argument("--mixed_face_policy", choices=("ignore", "majority"), default="ignore")
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--face_chunk_size", type=int, default=200000)
    parser.add_argument("--pair_chunk_size", type=int, default=2048)
    parser.add_argument("--max_face_tile_pairs", type=int, default=2000000)
    parser.add_argument("--barycentric_epsilon", type=float, default=1e-5)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build."
        )
    return torch.device(value)


def parse_cameras_txt(path: Path) -> Dict[int, ColmapIntrinsics]:
    cameras: Dict[int, ColmapIntrinsics] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id = int(fields[0])
        model = fields[1]
        if model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise RuntimeError(f"Unsupported COLMAP camera model {model!r} in {path}")
        cameras[camera_id] = ColmapIntrinsics(
            camera_id=camera_id,
            model=model,
            width=int(fields[2]),
            height=int(fields[3]),
            params=tuple(float(value) for value in fields[4:]),
        )
    if not cameras:
        raise RuntimeError(f"No COLMAP cameras found in {path}")
    return cameras


def parse_images_txt(path: Path) -> List[ColmapImage]:
    images: List[ColmapImage] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            image_id = int(fields[0])
            qvec = np.asarray([float(value) for value in fields[1:5]], dtype=np.float64)
            tvec = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
            camera_id = int(fields[8])
        except ValueError:
            continue
        images.append(
            ColmapImage(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=fields[9],
            )
        )
        if index < len(lines):
            index += 1  # COLMAP stores the 2D point observations on the next line.
    if not images:
        raise RuntimeError(f"No COLMAP images found in {path}")
    return sorted(images, key=lambda item: item.stem)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float32,
    )


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def get_world_to_view(r_transposed: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[:3, :3] = r_transposed.transpose()
    matrix[:3, 3] = translation
    matrix[3, 3] = 1.0
    return matrix


def get_projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> torch.Tensor:
    tan_half_y = math.tan(fov_y / 2.0)
    tan_half_x = math.tan(fov_x / 2.0)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right
    matrix = torch.zeros((4, 4), dtype=torch.float32)
    matrix[0, 0] = 2.0 * znear / (right - left)
    matrix[1, 1] = 2.0 * znear / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


def make_raster_camera(
    image: ColmapImage,
    intrinsics: ColmapIntrinsics,
    width: int,
    height: int,
    device: torch.device,
) -> RasterCamera:
    if intrinsics.model == "SIMPLE_PINHOLE":
        focal_x = focal_y = intrinsics.params[0]
    else:
        focal_x, focal_y = intrinsics.params[:2]
    fov_x = focal2fov(focal_x, intrinsics.width)
    fov_y = focal2fov(focal_y, intrinsics.height)
    r_transposed = qvec2rotmat(image.qvec).transpose()
    world_view = torch.from_numpy(get_world_to_view(r_transposed, image.tvec)).transpose(0, 1).to(device)
    projection = get_projection_matrix(0.01, 100.0, fov_x, fov_y).transpose(0, 1).to(device)
    return RasterCamera(
        image_name=image.stem,
        image_width=width,
        image_height=height,
        world_view_transform=world_view,
        full_proj_transform=world_view @ projection,
    )


def select_views(images: Sequence[ColmapImage], split: str, llffhold: int) -> List[ColmapImage]:
    if llffhold <= 0:
        raise ValueError("--llffhold must be positive")
    if split == "all":
        return list(images)
    want_test = split == "test"
    return [image for index, image in enumerate(images) if (index % llffhold == 0) == want_test]


def discover_scenes(data_root: Path, requested: Optional[Sequence[str]], excluded: Iterable[str]) -> List[Path]:
    excluded_names = {name.lower() for name in excluded}
    if requested:
        candidates = [data_root / name for name in requested]
    else:
        candidates = sorted((path for path in data_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    scenes = []
    for scene in candidates:
        if scene.name.lower() in excluded_names:
            continue
        if not scene.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene}")
        if (scene / "sparse" / "0" / "cameras.txt").exists() and (
            scene / "sparse" / "0" / "images.txt"
        ).exists():
            scenes.append(scene)
    if not scenes:
        raise RuntimeError(f"No supported scenes found under {data_root}")
    return scenes


def detect_dataset(scene: Path) -> Tuple[str, Path, Path]:
    scannet_mesh = scene / "semantic_gt_3d" / "mesh_aligned_0.05_semantic.ply"
    scannet_classes = scene / "semantic_classes.txt"
    replica_mesh = scene / "habitat" / "mesh_semantic.ply"
    replica_info = scene / "habitat" / "info_semantic.json"
    if scannet_mesh.exists() and scannet_classes.exists():
        return "scannetpp", scannet_mesh, scannet_classes
    if replica_mesh.exists() and replica_info.exists():
        return "replica", replica_mesh, replica_info
    raise RuntimeError(f"Cannot identify semantic GT assets for {scene}")


def load_prompt_bank(dataset_type: str, path: Path) -> Dict[str, List]:
    if dataset_type == "scannetpp":
        names = []
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.split("#", 1)[0].strip()
            if value:
                names.append(value)
        class_ids = list(range(len(names)))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("classes", [])
        class_ids = [int(entry["id"]) for entry in entries]
        names = [str(entry["name"]).replace("-", " ") for entry in entries]
    if not names or len(class_ids) != len(set(class_ids)):
        raise RuntimeError(f"Invalid class mapping: {path}")
    return {"class_ids": class_ids, "class_names": names}


def load_replica_object_overrides(
    path: Optional[Path], scene_name: str
) -> Tuple[Dict[int, Dict], Optional[Dict]]:
    if path is None:
        return {}, None
    if not path.is_file():
        raise FileNotFoundError(f"Replica object override file not found: {path}")
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    scene_entries = data.get("scenes", {}).get(scene_name, {})
    if not isinstance(scene_entries, dict):
        raise RuntimeError(f"Replica overrides for {scene_name} must be a JSON object")
    overrides: Dict[int, Dict] = {}
    for raw_object_id, raw_spec in scene_entries.items():
        object_id = int(raw_object_id)
        if object_id < 0 or not isinstance(raw_spec, dict) or "class_id" not in raw_spec:
            raise RuntimeError(f"Invalid Replica override {scene_name}/{raw_object_id}")
        overrides[object_id] = dict(raw_spec)
        overrides[object_id]["class_id"] = int(raw_spec["class_id"])
    return overrides, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "version": data.get("version", 1),
    }


def map_class_ids(raw_labels: np.ndarray, prompt_bank: Dict[str, List]) -> Tuple[np.ndarray, Dict]:
    id_to_index = {int(class_id): index for index, class_id in enumerate(prompt_bank["class_ids"])}
    mapped = np.full(raw_labels.shape, IGNORE_INDEX, dtype=np.int32)
    unmapped: Dict[str, int] = {}
    for raw_value in np.unique(raw_labels.astype(np.int64)):
        raw = int(raw_value)
        if raw in id_to_index:
            mapped[raw_labels == raw] = id_to_index[raw]
        elif 0 <= raw < len(prompt_bank["class_ids"]):
            mapped[raw_labels == raw] = raw
        else:
            unmapped[str(raw)] = int((raw_labels == raw).sum())
    return mapped, {"unmappedRawLabels": unmapped}


def triangulate_faces(face_vertex_lists: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    triangles: List[Tuple[int, int, int]] = []
    source_face_ids: List[int] = []
    for face_id, polygon in enumerate(face_vertex_lists):
        polygon = np.asarray(polygon, dtype=np.int64)
        for offset in range(1, len(polygon) - 1):
            triangles.append((int(polygon[0]), int(polygon[offset]), int(polygon[offset + 1])))
            source_face_ids.append(face_id)
    if not triangles:
        raise RuntimeError("Semantic GT mesh contains no triangulatable faces")
    return np.asarray(triangles, dtype=np.int64), np.asarray(source_face_ids, dtype=np.int64)


def load_semantic_mesh(
    dataset_type: str,
    mesh_path: Path,
    mapping_path: Path,
    prompt_bank: Dict[str, List],
    mixed_face_policy: str,
    replica_object_overrides: Optional[Dict[int, Dict]] = None,
    replica_override_source: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
    from plyfile import PlyData

    ply = PlyData.read(str(mesh_path))
    if "vertex" not in ply or "face" not in ply:
        raise RuntimeError(f"Semantic PLY must contain vertex and face elements: {mesh_path}")
    vertex_data = ply["vertex"].data
    vertices = np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).astype(
        np.float32, copy=False
    )
    face_data = ply["face"].data
    face_names = set(face_data.dtype.names or [])
    index_key = "vertex_indices" if "vertex_indices" in face_names else "vertex_index"
    if index_key not in face_names:
        raise RuntimeError(f"Face vertex indices missing from {mesh_path}")
    triangles, source_face_ids = triangulate_faces(face_data[index_key])

    mapping_info: Dict = {}
    mixed_faces = 0
    if dataset_type == "replica":
        if "object_id" not in face_names:
            raise RuntimeError(f"Replica mesh has no face.object_id: {mesh_path}")
        info = json.loads(mapping_path.read_text(encoding="utf-8"))
        object_entries = {int(item["id"]): item for item in info.get("objects", [])}
        id_to_label = info.get("id_to_label")
        if isinstance(id_to_label, list):
            object_to_class = {index: int(class_id) for index, class_id in enumerate(id_to_label)}
            replica_mapping_source = "info_semantic.id_to_label"
        else:
            object_to_class = {
                object_id: int(item["class_id"]) for object_id, item in object_entries.items()
            }
            replica_mapping_source = "info_semantic.objects"
        object_ids = np.asarray(face_data["object_id"], dtype=np.int64)
        object_id_counts = {
            int(object_id): int(count)
            for object_id, count in zip(*np.unique(object_ids, return_counts=True))
        }
        source_mismatches = []
        for object_id, item in object_entries.items():
            if object_id in object_to_class and object_to_class[object_id] != int(item["class_id"]):
                source_mismatches.append(
                    {
                        "objectId": object_id,
                        "idToLabelClassId": object_to_class[object_id],
                        "objectEntryClassId": int(item["class_id"]),
                    }
                )
        if source_mismatches:
            raise RuntimeError(
                f"Replica id_to_label disagrees with objects entries for {len(source_mismatches)} IDs"
            )

        class_id_to_name = {
            int(class_id): str(name)
            for class_id, name in zip(prompt_bank["class_ids"], prompt_bank["class_names"])
        }
        applied_overrides = []
        for object_id, spec in sorted((replica_object_overrides or {}).items()):
            if object_id not in object_id_counts:
                raise RuntimeError(
                    f"Replica override object_id={object_id} does not occur in {mesh_path}"
                )
            original_class_id = int(object_to_class.get(object_id, IGNORE_INDEX))
            expected_class_id = spec.get("expected_original_class_id")
            if expected_class_id is not None and original_class_id != int(expected_class_id):
                raise RuntimeError(
                    f"Replica override {object_id} expected original class {expected_class_id}, "
                    f"found {original_class_id}"
                )
            source_name = str(object_entries.get(object_id, {}).get("class_name", ""))
            expected_source_name = spec.get("expected_original_class_name")
            if expected_source_name is not None and source_name != str(expected_source_name):
                raise RuntimeError(
                    f"Replica override {object_id} expected source name {expected_source_name!r}, "
                    f"found {source_name!r}"
                )
            target_class_id = int(spec["class_id"])
            if target_class_id not in class_id_to_name:
                raise RuntimeError(
                    f"Replica override {object_id} targets unknown class_id={target_class_id}"
                )
            expected_target_name = spec.get("class_name")
            target_name = class_id_to_name[target_class_id]
            if expected_target_name is not None and target_name != str(expected_target_name).replace("-", " "):
                raise RuntimeError(
                    f"Replica override {object_id} target name mismatch: "
                    f"{expected_target_name!r} != {target_name!r}"
                )
            object_to_class[object_id] = target_class_id
            applied_overrides.append(
                {
                    "objectId": object_id,
                    "faceCount": object_id_counts[object_id],
                    "originalClassId": original_class_id,
                    "originalClassName": source_name,
                    "classId": target_class_id,
                    "className": target_name,
                    "reason": str(spec.get("reason", "")),
                }
            )

        raw_face_labels = np.asarray([object_to_class.get(int(value), -1) for value in object_ids], dtype=np.int64)
        mapped_faces, mapping_info = map_class_ids(raw_face_labels, prompt_bank)
        triangle_labels = mapped_faces[source_face_ids]
        label_source = f"face.object_id->{replica_mapping_source}"
        missing_ids = sorted(int(value) for value in np.unique(object_ids) if int(value) not in object_to_class)
        ignored_ids = [
            {
                "objectId": object_id,
                "faceCount": object_id_counts[object_id],
                "classId": int(object_to_class.get(object_id, IGNORE_INDEX)),
                "className": str(object_entries.get(object_id, {}).get("class_name", "")),
            }
            for object_id in sorted(object_id_counts)
            if int(object_to_class.get(object_id, IGNORE_INDEX)) < 0
        ]
        mapping_info["replicaMappingSource"] = replica_mapping_source
        mapping_info["missingObjectIds"] = missing_ids
        mapping_info["ignoredObjectIds"] = ignored_ids
        mapping_info["objectClassOverrides"] = applied_overrides
        mapping_info["objectClassOverrideSource"] = replica_override_source
    else:
        vertex_names = set(vertex_data.dtype.names or [])
        label_key = next(
            (key for key in ("label", "semantic", "semantic_label", "class_id", "category_id") if key in vertex_names),
            None,
        )
        if label_key is None:
            raise RuntimeError(f"ScanNet++ mesh has no supported vertex label: {mesh_path}")
        mapped_vertices, mapping_info = map_class_ids(np.asarray(vertex_data[label_key]), prompt_bank)
        triangle_vertex_labels = mapped_vertices[triangles]
        all_equal = (
            (triangle_vertex_labels[:, 0] == triangle_vertex_labels[:, 1])
            & (triangle_vertex_labels[:, 0] == triangle_vertex_labels[:, 2])
        )
        mixed_faces = int((~all_equal).sum())
        if mixed_face_policy == "majority":
            sorted_labels = np.sort(triangle_vertex_labels, axis=1)
            triangle_labels = np.where(
                sorted_labels[:, 0] == sorted_labels[:, 1],
                sorted_labels[:, 0],
                sorted_labels[:, 2],
            )
            all_distinct = (
                (sorted_labels[:, 0] != sorted_labels[:, 1])
                & (sorted_labels[:, 1] != sorted_labels[:, 2])
            )
            triangle_labels[all_distinct] = IGNORE_INDEX
        else:
            triangle_labels = triangle_vertex_labels[:, 0].copy()
            triangle_labels[~all_equal] = IGNORE_INDEX
        label_source = f"vertex.{label_key}"

    mesh_info = {
        "path": str(mesh_path.resolve()),
        "numVertices": int(vertices.shape[0]),
        "numTriangles": int(triangles.shape[0]),
        "labelSource": label_source,
        "mixedFacePolicy": mixed_face_policy,
        "numMixedTriangles": mixed_faces,
        "numValidLabelTriangles": int((triangle_labels >= 0).sum()),
    }
    return vertices, triangles, triangle_labels.astype(np.int64, copy=False), mesh_info, mapping_info


def project_mesh_vertices(vertices: torch.Tensor, camera: RasterCamera) -> Tuple[torch.Tensor, ...]:
    ones = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=vertices.device)
    vertices_h = torch.cat((vertices, ones), dim=1)
    clip = vertices_h @ camera.full_proj_transform
    view = vertices_h @ camera.world_view_transform
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-8)
    pixel_x = ((ndc[:, 0] + 1.0) * camera.image_width - 1.0) * 0.5
    pixel_y = ((ndc[:, 1] + 1.0) * camera.image_height - 1.0) * 0.5
    return pixel_x, pixel_y, view[:, 2]


def pack_depth_and_triangle(depth: torch.Tensor, triangle_ids: torch.Tensor) -> torch.Tensor:
    depth_bits = depth.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    triangle_ids = triangle_ids.to(torch.int64)
    if triangle_ids.numel() and int(triangle_ids.max().item()) > TRIANGLE_ID_MASK:
        raise RuntimeError("GT mesh exceeds the rasterizer's 32-bit triangle id range")
    return (depth_bits << 32) | triangle_ids


def rasterize_pair_batch(
    key_buffer: torch.Tensor,
    pair_triangle_ids: torch.Tensor,
    pair_tile_ids: torch.Tensor,
    triangles: torch.Tensor,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    view_z: torch.Tensor,
    width: int,
    height: int,
    tiles_x: int,
    tile_size: int,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    barycentric_epsilon: float,
) -> None:
    triangle_vertices = triangles[pair_triangle_ids]
    x0 = pixel_x[triangle_vertices[:, 0]].unsqueeze(1)
    x1 = pixel_x[triangle_vertices[:, 1]].unsqueeze(1)
    x2 = pixel_x[triangle_vertices[:, 2]].unsqueeze(1)
    y0 = pixel_y[triangle_vertices[:, 0]].unsqueeze(1)
    y1 = pixel_y[triangle_vertices[:, 1]].unsqueeze(1)
    y2 = pixel_y[triangle_vertices[:, 2]].unsqueeze(1)
    z0 = view_z[triangle_vertices[:, 0]].unsqueeze(1)
    z1 = view_z[triangle_vertices[:, 1]].unsqueeze(1)
    z2 = view_z[triangle_vertices[:, 2]].unsqueeze(1)
    tile_x = pair_tile_ids.remainder(tiles_x)
    tile_y = torch.div(pair_tile_ids, tiles_x, rounding_mode="floor")
    sample_x = tile_x.unsqueeze(1) * tile_size + local_x.unsqueeze(0)
    sample_y = tile_y.unsqueeze(1) * tile_size + local_y.unsqueeze(0)
    in_image = (sample_x < width) & (sample_y < height)
    sample_xf = sample_x.float()
    sample_yf = sample_y.float()
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegenerate = denominator.abs() > 1e-12
    safe_denominator = torch.where(nondegenerate, denominator, torch.ones_like(denominator))
    bary0 = ((y1 - y2) * (sample_xf - x2) + (x2 - x1) * (sample_yf - y2)) / safe_denominator
    bary1 = ((y2 - y0) * (sample_xf - x2) + (x0 - x2) * (sample_yf - y2)) / safe_denominator
    bary2 = 1.0 - bary0 - bary1
    inside = (
        in_image
        & nondegenerate
        & (bary0 >= -barycentric_epsilon)
        & (bary1 >= -barycentric_epsilon)
        & (bary2 >= -barycentric_epsilon)
    )
    if not bool(inside.any().item()):
        return
    inverse_z = bary0 / z0 + bary1 / z1 + bary2 / z2
    depth = inverse_z.reciprocal()
    inside &= torch.isfinite(depth) & (depth > 0)
    if not bool(inside.any().item()):
        return
    pixel_indices = (sample_y * width + sample_x)[inside].long()
    candidate_triangles = pair_triangle_ids.unsqueeze(1).expand_as(sample_x)[inside]
    packed = pack_depth_and_triangle(depth[inside], candidate_triangles)
    key_buffer.scatter_reduce_(0, pixel_indices, packed, reduce="amin", include_self=True)


def rasterize_triangle_subset(
    key_buffer: torch.Tensor,
    triangle_ids: torch.Tensor,
    triangles: torch.Tensor,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    view_z: torch.Tensor,
    width: int,
    height: int,
    znear: float,
    zfar: float,
    tile_size: int,
    pair_chunk_size: int,
    max_face_tile_pairs: int,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    barycentric_epsilon: float,
) -> int:
    if triangle_ids.numel() == 0:
        return 0
    triangle_vertices = triangles[triangle_ids]
    triangle_x = pixel_x[triangle_vertices]
    triangle_y = pixel_y[triangle_vertices]
    triangle_z = view_z[triangle_vertices]
    finite = torch.isfinite(triangle_x).all(dim=1) & torch.isfinite(triangle_y).all(dim=1)
    in_depth = (triangle_z > znear).all(dim=1) & (triangle_z < zfar).all(dim=1)
    min_x = torch.ceil(triangle_x.min(dim=1).values).long()
    max_x = torch.floor(triangle_x.max(dim=1).values).long()
    min_y = torch.ceil(triangle_y.min(dim=1).values).long()
    max_y = torch.floor(triangle_y.max(dim=1).values).long()
    intersects = (max_x >= 0) & (max_y >= 0) & (min_x < width) & (min_y < height)
    valid = finite & in_depth & intersects
    if not bool(valid.any().item()):
        return 0
    triangle_ids = triangle_ids[valid]
    min_x = min_x[valid].clamp(0, width - 1)
    max_x = max_x[valid].clamp(0, width - 1)
    min_y = min_y[valid].clamp(0, height - 1)
    max_y = max_y[valid].clamp(0, height - 1)
    valid_extent = (max_x >= min_x) & (max_y >= min_y)
    triangle_ids = triangle_ids[valid_extent]
    min_x = min_x[valid_extent]
    max_x = max_x[valid_extent]
    min_y = min_y[valid_extent]
    max_y = max_y[valid_extent]
    if triangle_ids.numel() == 0:
        return 0
    tile_min_x = torch.div(min_x, tile_size, rounding_mode="floor")
    tile_max_x = torch.div(max_x, tile_size, rounding_mode="floor")
    tile_min_y = torch.div(min_y, tile_size, rounding_mode="floor")
    tile_max_y = torch.div(max_y, tile_size, rounding_mode="floor")
    tile_widths = tile_max_x - tile_min_x + 1
    tile_counts = tile_widths * (tile_max_y - tile_min_y + 1)
    total_pairs = int(tile_counts.sum().item())
    if total_pairs > max_face_tile_pairs and triangle_ids.numel() > 1:
        midpoint = triangle_ids.numel() // 2
        return rasterize_triangle_subset(
            key_buffer,
            triangle_ids[:midpoint],
            triangles,
            pixel_x,
            pixel_y,
            view_z,
            width,
            height,
            znear,
            zfar,
            tile_size,
            pair_chunk_size,
            max_face_tile_pairs,
            local_x,
            local_y,
            barycentric_epsilon,
        ) + rasterize_triangle_subset(
            key_buffer,
            triangle_ids[midpoint:],
            triangles,
            pixel_x,
            pixel_y,
            view_z,
            width,
            height,
            znear,
            zfar,
            tile_size,
            pair_chunk_size,
            max_face_tile_pairs,
            local_x,
            local_y,
            barycentric_epsilon,
        )
    starts = torch.cumsum(tile_counts, dim=0) - tile_counts
    pair_triangle_ids = torch.repeat_interleave(triangle_ids, tile_counts)
    pair_starts = torch.repeat_interleave(starts, tile_counts)
    pair_tile_widths = torch.repeat_interleave(tile_widths, tile_counts)
    pair_tile_min_x = torch.repeat_interleave(tile_min_x, tile_counts)
    pair_tile_min_y = torch.repeat_interleave(tile_min_y, tile_counts)
    local_offsets = torch.arange(total_pairs, device=triangle_ids.device) - pair_starts
    pair_tile_x = pair_tile_min_x + local_offsets.remainder(pair_tile_widths)
    pair_tile_y = pair_tile_min_y + torch.div(local_offsets, pair_tile_widths, rounding_mode="floor")
    tiles_x = int(math.ceil(width / tile_size))
    pair_tile_ids = pair_tile_y * tiles_x + pair_tile_x
    for start in range(0, total_pairs, pair_chunk_size):
        rasterize_pair_batch(
            key_buffer,
            pair_triangle_ids[start : start + pair_chunk_size],
            pair_tile_ids[start : start + pair_chunk_size],
            triangles,
            pixel_x,
            pixel_y,
            view_z,
            width,
            height,
            tiles_x,
            tile_size,
            local_x,
            local_y,
            barycentric_epsilon,
        )
    return int(triangle_ids.numel())


@torch.inference_mode()
def rasterize_semantic_mesh(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    triangle_labels: torch.Tensor,
    camera: RasterCamera,
    tile_size: int,
    face_chunk_size: int,
    pair_chunk_size: int,
    max_face_tile_pairs: int,
    barycentric_epsilon: float,
) -> Tuple[torch.Tensor, Dict]:
    width, height = camera.image_width, camera.image_height
    pixel_x, pixel_y, view_z = project_mesh_vertices(vertices, camera)
    key_buffer = torch.full((height * width,), INT64_MAX, dtype=torch.int64, device=vertices.device)
    local_y, local_x = torch.meshgrid(
        torch.arange(tile_size, device=vertices.device, dtype=torch.long),
        torch.arange(tile_size, device=vertices.device, dtype=torch.long),
        indexing="ij",
    )
    local_x, local_y = local_x.reshape(-1), local_y.reshape(-1)
    processed = 0
    for start in range(0, triangles.shape[0], face_chunk_size):
        end = min(start + face_chunk_size, triangles.shape[0])
        triangle_ids = torch.arange(start, end, device=vertices.device, dtype=torch.long)
        processed += rasterize_triangle_subset(
            key_buffer,
            triangle_ids,
            triangles,
            pixel_x,
            pixel_y,
            view_z,
            width,
            height,
            camera.znear,
            camera.zfar,
            tile_size,
            pair_chunk_size,
            max_face_tile_pairs,
            local_x,
            local_y,
            barycentric_epsilon,
        )
    label_map = torch.full((height * width,), IGNORE_INDEX, dtype=torch.long, device=vertices.device)
    covered = key_buffer != INT64_MAX
    if bool(covered.any().item()):
        winning_triangles = key_buffer[covered] & TRIANGLE_ID_MASK
        label_map[covered] = triangle_labels[winning_triangles.long()]
    label_map = label_map.view(height, width)
    return label_map, {
        "numTriangles": int(triangles.shape[0]),
        "numProjectedTriangles": int(processed),
        "numCoveredPixels": int(covered.sum().item()),
        "numValidLabelPixels": int((label_map >= 0).sum().item()),
    }


def build_palette(class_names: Sequence[str]) -> np.ndarray:
    palette = np.zeros((len(class_names), 3), dtype=np.uint8)
    for index, name in enumerate(class_names):
        digest = hashlib.sha1(f"{index}:{name}".encode("utf-8")).digest()
        raw_color = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.uint16)
        palette[index] = (raw_color // 2 + 96).astype(np.uint8)
    return palette


def labels_to_rgb(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(palette))
    rgb[valid] = palette[labels[valid]]
    return rgb


def target_size(image: Image.Image, dataset_type: str, args: argparse.Namespace) -> Tuple[int, int]:
    target_height = args.cs_target_height if dataset_type == "scannetpp" else args.replica_target_height
    if target_height <= 0:
        return image.width, image.height
    return int(round(image.width * target_height / image.height)), int(target_height)


def save_palette(path: Path, prompt_bank: Dict[str, List], palette: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class_index", "class_id", "class_name", "r", "g", "b"))
        for index, (class_id, name) in enumerate(zip(prompt_bank["class_ids"], prompt_bank["class_names"])):
            writer.writerow((index, class_id, name, *[int(value) for value in palette[index]]))


def write_outputs(
    output_dir: Path,
    image_name: str,
    rgb_image: Image.Image,
    labels: np.ndarray,
    palette: np.ndarray,
    overlay_alpha: float,
) -> None:
    np.save(output_dir / "labels" / f"{image_name}.npy", labels.astype(np.int32, copy=False))
    png_labels = labels.astype(np.int64, copy=True)
    png_labels[png_labels < 0] = int(PNG_IGNORE_INDEX)
    Image.fromarray(png_labels.astype(np.uint16)).save(
        output_dir / "labels_png16" / f"{image_name}.png"
    )
    rgb_array = np.asarray(rgb_image.convert("RGB"), dtype=np.uint8)
    semantic_rgb = labels_to_rgb(labels, palette)
    overlay = rgb_array.copy()
    valid = labels >= 0
    blended = (
        rgb_array.astype(np.float32) * (1.0 - overlay_alpha)
        + semantic_rgb.astype(np.float32) * overlay_alpha
    ).astype(np.uint8)
    overlay[valid] = blended[valid]
    rgb_image.convert("RGB").save(output_dir / "rgb" / f"{image_name}.png")
    Image.fromarray(semantic_rgb, mode="RGB").save(output_dir / "semantic_rgb" / f"{image_name}.png")
    Image.fromarray(overlay, mode="RGB").save(output_dir / "overlay_rgb" / f"{image_name}.png")


def ensure_output_dirs(output_dir: Path) -> None:
    for name in ("labels", "labels_png16", "rgb", "semantic_rgb", "overlay_rgb"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def image_lookup(scene: Path) -> Dict[str, Path]:
    image_dir = scene / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    result = {
        path.name: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES
    }
    return result


def run_scene(scene: Path, args: argparse.Namespace, device: torch.device) -> Dict:
    started = time.perf_counter()
    output_dir = args.output_root / scene.name / args.split
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Use a fresh GT output directory to avoid reusing labels with changed settings: {output_dir}"
        )
    from tqdm import tqdm

    dataset_type, mesh_path, mapping_path = detect_dataset(scene)
    prompt_bank = load_prompt_bank(dataset_type, mapping_path)
    cameras = parse_cameras_txt(scene / "sparse" / "0" / "cameras.txt")
    colmap_images = parse_images_txt(scene / "sparse" / "0" / "images.txt")
    selected = select_views(colmap_images, args.split, args.llffhold)
    if args.max_views > 0:
        selected = selected[: args.max_views]
    lookup = image_lookup(scene)
    missing_images = [image.name for image in selected if image.name not in lookup]
    if missing_images:
        raise RuntimeError(f"{scene.name}: missing RGB files for {len(missing_images)} selected cameras")

    replica_overrides, replica_override_source = load_replica_object_overrides(
        args.replica_object_overrides if dataset_type == "replica" else None,
        scene.name,
    )
    vertices, triangles, triangle_labels, mesh_info, mapping_info = load_semantic_mesh(
        dataset_type,
        mesh_path,
        mapping_path,
        prompt_bank,
        args.mixed_face_policy,
        replica_object_overrides=replica_overrides,
        replica_override_source=replica_override_source,
    )
    vertices_gpu = torch.from_numpy(vertices).to(device=device, dtype=torch.float32)
    triangles_gpu = torch.from_numpy(triangles).to(device=device, dtype=torch.long)
    labels_gpu = torch.from_numpy(triangle_labels).to(device=device, dtype=torch.long)
    palette = build_palette(prompt_bank["class_names"])
    output_dir = args.output_root / scene.name / args.split
    ensure_output_dirs(output_dir)
    save_palette(output_dir / "semantic_palette.csv", prompt_bank, palette)

    per_view: Dict[str, Dict] = {}
    selected_names: List[str] = []
    output_size: Optional[Tuple[int, int]] = None
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    iterator = tqdm(selected, desc=f"{scene.name} [{dataset_type}]", dynamic_ncols=True)
    for image_record in iterator:
        image_path = lookup[image_record.name]
        with Image.open(image_path) as source_image:
            width, height = target_size(source_image, dataset_type, args)
            rgb_image = source_image.convert("RGB").resize((width, height), resampling)
        if output_size is None:
            output_size = (width, height)
        elif output_size != (width, height):
            raise RuntimeError(f"{scene.name}: inconsistent target image dimensions")
        selected_names.append(image_record.stem)
        label_path = output_dir / "labels" / f"{image_record.stem}.npy"
        if label_path.exists() and not args.overwrite:
            cached = np.load(label_path)
            if cached.shape != (height, width):
                raise RuntimeError(f"Cached label size mismatch: {label_path}")
            labels = cached.astype(np.int32, copy=False)
            stats = {
                "cacheHit": True,
                "numValidLabelPixels": int((labels >= 0).sum()),
                "coverage": float((labels >= 0).mean()),
            }
        else:
            camera = make_raster_camera(
                image_record,
                cameras[image_record.camera_id],
                width,
                height,
                device,
            )
            label_tensor, stats = rasterize_semantic_mesh(
                vertices_gpu,
                triangles_gpu,
                labels_gpu,
                camera,
                tile_size=args.tile_size,
                face_chunk_size=args.face_chunk_size,
                pair_chunk_size=args.pair_chunk_size,
                max_face_tile_pairs=args.max_face_tile_pairs,
                barycentric_epsilon=args.barycentric_epsilon,
            )
            labels = label_tensor.cpu().numpy().astype(np.int32, copy=False)
            stats["cacheHit"] = False
            stats["coverage"] = float((labels >= 0).mean())
        write_outputs(
            output_dir,
            image_record.stem,
            rgb_image,
            labels,
            palette,
            args.overlay_alpha,
        )
        per_view[image_record.stem] = stats

    if output_size is None:
        raise RuntimeError(f"No views selected for {scene.name}")
    elapsed = time.perf_counter() - started
    coverage_values = [float(values["coverage"]) for values in per_view.values()]
    manifest = {
        "protocol": "rasterized_gt_mesh_2d_gt_only",
        "createdAt": datetime.now().astimezone().isoformat(),
        "scene": scene.name,
        "datasetType": dataset_type,
        "split": args.split,
        "llffhold": args.llffhold,
        "selectionRule": "sorted image stem; test iff index % llffhold == 0",
        "numAllViews": len(colmap_images),
        "numSelectedViews": len(selected),
        "selectedViews": selected_names,
        "outputWidth": output_size[0],
        "outputHeight": output_size[1],
        "ignoreIndexNpy": IGNORE_INDEX,
        "ignoreIndexPng16": int(PNG_IGNORE_INDEX),
        "classMappingPath": str(mapping_path.resolve()),
        "numClasses": len(prompt_bank["class_ids"]),
        "mesh": mesh_info,
        "mapping": mapping_info,
        "device": str(device),
        "torchVersion": torch.__version__,
        "elapsedSeconds": elapsed,
        "meanValidCoverage": float(np.mean(coverage_values)),
        "rasterizer": {
            "tileSize": args.tile_size,
            "faceChunkSize": args.face_chunk_size,
            "pairChunkSize": args.pair_chunk_size,
            "maxFaceTilePairs": args.max_face_tile_pairs,
            "barycentricEpsilon": args.barycentric_epsilon,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "per_view_raster_stats.json").write_text(
        json.dumps(per_view, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "view_names.txt").write_text("\n".join(selected_names) + "\n", encoding="utf-8")
    return manifest


def write_summary(output_root: Path, rows: Sequence[Dict]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "precompute_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "scene",
                "dataset_type",
                "split",
                "all_views",
                "selected_views",
                "width",
                "height",
                "mean_valid_coverage",
                "elapsed_seconds",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row["scene"],
                    row["datasetType"],
                    row["split"],
                    row["numAllViews"],
                    row["numSelectedViews"],
                    row["outputWidth"],
                    row["outputHeight"],
                    row["meanValidCoverage"],
                    row["elapsedSeconds"],
                )
            )
    return path


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay_alpha must be between 0 and 1")
    device = resolve_device(args.device)
    scenes = discover_scenes(args.data_root, args.scenes, args.exclude_dirs)
    print(f"data_root   : {args.data_root}")
    print(f"output_root : {args.output_root}")
    print(f"device      : {device}")
    print(f"split       : {args.split}; llffhold={args.llffhold}")
    print("scenes      : " + " ".join(scene.name for scene in scenes))
    if device.type == "cpu":
        print("WARNING: CPU mode is supported for debugging but full eight-scene rasterization will be slow.")

    summaries = []
    total_start = time.perf_counter()
    for index, scene in enumerate(scenes, start=1):
        print(f"\n===== scene {index}/{len(scenes)}: {scene.name} =====")
        summaries.append(run_scene(scene, args, device))
    summary_path = write_summary(args.output_root, summaries)
    print(f"\nComplete in {time.perf_counter() - total_start:.1f}s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
