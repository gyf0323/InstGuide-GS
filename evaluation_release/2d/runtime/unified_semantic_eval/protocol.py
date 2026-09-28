from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


PROTOCOL_NAME = os.environ.get(
    "UNIFIED_PROTOCOL_NAME",
    "unified_precomputed_gt_2d_fixed_vocab_v1",
)


def _prediction_class_scope() -> str:
    scope = os.environ.get("UNIFIED_PREDICTION_CLASS_SCOPE", "all").strip()
    if scope not in {"gt_present", "mesh_present", "all"}:
        raise RuntimeError(
            "UNIFIED_PREDICTION_CLASS_SCOPE must be 'gt_present', "
            "'mesh_present', or 'all', "
            f"got {scope!r}"
        )
    return scope


def _requested_prediction_class_indices() -> list[int]:
    raw = os.environ.get("UNIFIED_PREDICTION_CLASS_INDICES", "").strip()
    if not raw:
        raise RuntimeError(
            "UNIFIED_PREDICTION_CLASS_INDICES is required for mesh_present scope"
        )
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "UNIFIED_PREDICTION_CLASS_INDICES must be a JSON integer list"
        ) from exc
    if not isinstance(values, list) or not values:
        raise RuntimeError(
            "UNIFIED_PREDICTION_CLASS_INDICES must be a non-empty JSON list"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise RuntimeError(
            "UNIFIED_PREDICTION_CLASS_INDICES may contain only integers"
        )
    if len(values) != len(set(values)):
        raise RuntimeError("UNIFIED_PREDICTION_CLASS_INDICES contains duplicates")
    return sorted(values)


def _requested_excluded_classes() -> list[str]:
    value = os.environ.get("UNIFIED_EXCLUDE_CLASSES", "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_name(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


@dataclass(frozen=True)
class SemanticClass:
    index: int
    class_id: str
    name: str
    color: tuple[int, int, int]


class UnifiedSemanticGT:
    """Validated view and class contract produced by precompute_gtmesh_2d_labels.py."""

    def __init__(self, split_dir: Path):
        self.split_dir = split_dir.resolve()
        self.manifest_path = self.split_dir / "manifest.json"
        self.palette_path = self.split_dir / "semantic_palette.csv"
        self.view_names_path = self.split_dir / "view_names.txt"
        self.labels_dir = self.split_dir / "labels"

        for required in (
            self.manifest_path,
            self.palette_path,
            self.view_names_path,
            self.labels_dir,
        ):
            if not required.exists():
                raise FileNotFoundError(f"Unified GT artifact is missing: {required}")

        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.classes = self._read_palette()
        self.class_names = [item.name for item in self.classes]
        self.class_ids = [item.class_id for item in self.classes]
        self.palette = np.asarray([item.color for item in self.classes], dtype=np.uint8)
        self.view_names = [
            line.strip()
            for line in self.view_names_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._validate_manifest()
        self._validate_label_files()
        self.present_class_indices = self._collect_present_classes()

    @classmethod
    def open(cls, gt_root: str | Path, scene: str = "", split: str = "test") -> "UnifiedSemanticGT":
        root = Path(gt_root).expanduser()
        candidates = []
        if scene:
            candidates.append(root / scene / split)
        candidates.extend((root / split, root))
        for candidate in candidates:
            if (candidate / "manifest.json").is_file():
                gt = cls(candidate)
                if (scene and gt.scene != scene) or gt.split != split:
                    raise RuntimeError(f"GT scene/split mismatch: requested {scene}/{split}, found {gt.scene}/{gt.split}")
                return gt
        raise FileNotFoundError(
            "Could not resolve unified GT split directory. Checked: "
            + ", ".join(str(path) for path in candidates)
        )

    @property
    def scene(self) -> str:
        return str(self.manifest.get("scene", self.split_dir.parent.name))

    @property
    def split(self) -> str:
        return str(self.manifest.get("split", self.split_dir.name))

    @property
    def width(self) -> int:
        return int(self.manifest["outputWidth"])

    @property
    def height(self) -> int:
        return int(self.manifest["outputHeight"])

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in (self.manifest_path, self.palette_path, self.view_names_path):
            digest.update(_sha256_file(path).encode("ascii"))
        return digest.hexdigest()

    def label_path(self, view_name: str) -> Path:
        return self.labels_dir / f"{view_name}.npy"

    def load_label(self, view_name: str) -> np.ndarray:
        path = self.label_path(view_name)
        if not path.is_file():
            raise FileNotFoundError(f"GT label is missing for view {view_name}: {path}")
        label = np.load(path, allow_pickle=False)
        self.validate_map(label, f"GT {view_name}", allow_invalid=True)
        return label.astype(np.int32, copy=False)

    def validate_map(self, label: np.ndarray, label_name: str, allow_invalid: bool) -> None:
        if label.ndim != 2:
            raise RuntimeError(f"{label_name} must be HxW, got {label.shape}")
        expected = (self.height, self.width)
        if tuple(label.shape) != expected:
            raise RuntimeError(f"{label_name} has shape {label.shape}; expected {expected}")
        if not np.issubdtype(label.dtype, np.integer):
            raise RuntimeError(f"{label_name} must contain integer class indices, got {label.dtype}")
        invalid_low = label < 0
        invalid_high = label >= self.num_classes
        if np.any(invalid_high):
            values = np.unique(label[invalid_high])[:12].tolist()
            raise RuntimeError(f"{label_name} contains out-of-range class indices: {values}")
        if not allow_invalid and np.any(invalid_low):
            raise RuntimeError(f"{label_name} contains invalid negative predictions")

    def class_index(self, selector: str) -> int:
        selector_token = _normalize_name(selector)
        matches = []
        for item in self.classes:
            if selector_token in {
                _normalize_name(item.name),
                _normalize_name(item.class_id),
                str(item.index),
            }:
                matches.append(item.index)
        if len(matches) != 1:
            raise RuntimeError(f"Class selector {selector!r} matched {len(matches)} classes")
        return matches[0]

    def resolve_class_indices(
        self,
        selectors: Sequence[str] | None,
        *,
        allow_missing: bool = False,
    ) -> list[int]:
        indices = []
        for selector in selectors or ():
            try:
                indices.append(self.class_index(selector))
            except RuntimeError:
                if not allow_missing:
                    raise
        return sorted(set(indices))

    def prediction_class_indices(
        self,
        exclude_classes: Sequence[str] | None = None,
    ) -> list[int]:
        expected_excluded = {
            _normalize_name(value) for value in _requested_excluded_classes()
        }
        actual_excluded = {
            _normalize_name(value) for value in (exclude_classes or ())
        }
        if expected_excluded and actual_excluded != expected_excluded:
            raise RuntimeError(
                "Adapter exclusion list differs from the fixed-vocabulary contract: "
                f"expected={sorted(expected_excluded)} actual={sorted(actual_excluded)}"
            )
        excluded = set(
            self.resolve_class_indices(exclude_classes, allow_missing=True)
        )
        prediction_scope = _prediction_class_scope()
        if prediction_scope == "all":
            candidates = range(self.num_classes)
        elif prediction_scope == "mesh_present":
            candidates = _requested_prediction_class_indices()
            out_of_range = [
                index for index in candidates if index < 0 or index >= self.num_classes
            ]
            if out_of_range:
                raise RuntimeError(
                    "Mesh-present prediction indices are outside the GT vocabulary: "
                    f"{out_of_range[:12]}"
                )
            missing_test_classes = sorted(
                set(self.present_class_indices) - set(candidates) - excluded
            )
            if missing_test_classes:
                raise RuntimeError(
                    "Mesh-present vocabulary omits classes found in the test GT: "
                    f"{missing_test_classes[:12]}"
                )
        else:
            candidates = self.present_class_indices
        selected = [index for index in candidates if index not in excluded]
        if not selected:
            raise RuntimeError("No prediction classes remain after applying exclusions")
        return selected

    def metadata(self) -> dict:
        return {
            "protocol": PROTOCOL_NAME,
            "scene": self.scene,
            "split": self.split,
            "numViews": len(self.view_names),
            "numClasses": self.num_classes,
            "width": self.width,
            "height": self.height,
            "presentClassIndices": self.present_class_indices,
            "presentClasses": [self.class_names[index] for index in self.present_class_indices],
            "fingerprint": self.fingerprint,
            "splitDir": str(self.split_dir),
        }

    def _read_palette(self) -> list[SemanticClass]:
        rows = []
        with self.palette_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    SemanticClass(
                        index=int(row["class_index"]),
                        class_id=str(row["class_id"]),
                        name=str(row["class_name"]),
                        color=(int(row["r"]), int(row["g"]), int(row["b"])),
                    )
                )
        rows.sort(key=lambda item: item.index)
        expected = list(range(len(rows)))
        actual = [item.index for item in rows]
        if actual != expected:
            raise RuntimeError(f"Palette class_index values are not contiguous: {actual[:12]}")
        return rows

    def _validate_manifest(self) -> None:
        if not self.view_names or len(set(self.view_names)) != len(self.view_names):
            raise RuntimeError("GT must contain non-empty, unique view names")
        if self.manifest.get("protocol") != "rasterized_gt_mesh_2d_gt_only":
            raise RuntimeError(f"Unsupported GT manifest protocol: {self.manifest.get('protocol')}")
        selected = [str(value) for value in self.manifest.get("selectedViews", [])]
        if selected != self.view_names:
            raise RuntimeError("manifest.json selectedViews does not match view_names.txt")
        if int(self.manifest.get("numSelectedViews", -1)) != len(self.view_names):
            raise RuntimeError("numSelectedViews does not match view_names.txt")
        if int(self.manifest.get("numClasses", -1)) != self.num_classes:
            raise RuntimeError("numClasses does not match semantic_palette.csv")

    def _validate_label_files(self) -> None:
        expected = {f"{name}.npy" for name in self.view_names}
        actual = {path.name for path in self.labels_dir.glob("*.npy")}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise RuntimeError(
                f"GT label/view mismatch: missing={missing[:8]} extra={extra[:8]}"
            )

    def _collect_present_classes(self) -> list[int]:
        present = set()
        for view_name in self.view_names:
            label = self.load_label(view_name)
            values = np.unique(label)
            present.update(int(value) for value in values if 0 <= int(value) < self.num_classes)
        if not present:
            raise RuntimeError("Unified GT contains no valid semantic classes")
        return sorted(present)


class PredictionWriter:
    def __init__(
        self,
        output_dir: str | Path,
        gt: UnifiedSemanticGT,
        method: str,
        metadata: Mapping[str, object] | None = None,
        overwrite: bool = False,
        exclude_classes: Sequence[str] | None = None,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.prediction_dir = self.output_dir / "predictions"
        self.gt = gt
        self.method = str(method)
        self.metadata_payload = dict(metadata or {})
        prediction_scope = _prediction_class_scope()
        requested_excluded = list(exclude_classes) if exclude_classes is not None else _requested_excluded_classes()
        prediction_indices = gt.prediction_class_indices(requested_excluded)
        self.metadata_payload.update(
            {
                "predictionVocabulary": os.environ.get(
                    "UNIFIED_VOCABULARY_NAME",
                    gt.manifest.get("fixedVocabulary", {}).get("name", "full_fixed_dataset_vocabulary"),
                ),
                "predictionClassCount": len(prediction_indices),
                "predictionClassIndices": prediction_indices,
                "metricClassScope": "gt_present_within_fixed_vocabulary",
                "vocabularySource": os.environ.get(
                    "UNIFIED_VOCABULARY_SOURCE", gt.manifest.get("fixedVocabulary", {}).get("source", "")
                ),
                "vocabularySha256": os.environ.get(
                    "UNIFIED_VOCABULARY_SHA256", gt.manifest.get("fixedVocabulary", {}).get("sourceSha256", "")
                ),
                "requestedExcludedClasses": requested_excluded,
            }
        )
        if prediction_scope == "all":
            self.metadata_payload.update(
                {
                    "predictionClassScope": "all_fixed_vocabulary_classes",
                    "groundTruthUsedToSelectPredictionClasses": False,
                }
            )
        elif prediction_scope == "mesh_present":
            self.metadata_payload.update(
                {
                    "predictionClassScope": "gt_mesh_present_oracle_classes",
                    "groundTruthUsedToSelectPredictionClasses": True,
                    "predictionClassSource": os.environ.get(
                        "UNIFIED_PREDICTION_CLASS_SOURCE", ""
                    ),
                    "predictionClassSourceSha256": os.environ.get(
                        "UNIFIED_PREDICTION_CLASS_SOURCE_SHA256", ""
                    ),
                }
            )
        else:
            self.metadata_payload.update(
                {
                    "predictionClassScope": "gt_present_oracle_classes",
                    "groundTruthUsedToSelectPredictionClasses": True,
                }
            )
        self.overwrite = bool(overwrite)
        self.written: list[str] = []
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

    def write(self, view_name: str, prediction: np.ndarray) -> Path:
        if view_name not in self.gt.view_names:
            raise RuntimeError(f"Prediction view is outside the GT contract: {view_name}")
        if view_name in self.written:
            raise RuntimeError(f"Prediction view was already written in this run: {view_name}")
        prediction = np.asarray(prediction)
        self.gt.validate_map(prediction, f"Prediction {view_name}", allow_invalid=True)
        path = self.prediction_dir / f"{view_name}.npy"
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"Prediction already exists; pass overwrite to replace it: {path}")
        np.save(path, prediction.astype(np.int32, copy=False), allow_pickle=False)
        self.written.append(view_name)
        return path

    def finalize(self) -> Path:
        expected = set(self.gt.view_names)
        if set(self.written) != expected:
            raise RuntimeError("All GT views must be written in this run; stale predictions cannot complete it")
        actual = {path.stem for path in self.prediction_dir.glob("*.npy")}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise RuntimeError(
                f"Prediction contract is incomplete: missing={missing[:8]} extra={extra[:8]}"
            )
        manifest = {
            "protocol": PROTOCOL_NAME,
            "createdAt": datetime.now(timezone.utc).astimezone().isoformat(),
            "method": self.method,
            "scene": self.gt.scene,
            "split": self.gt.split,
            "numViews": len(self.gt.view_names),
            "numClasses": self.gt.num_classes,
            "width": self.gt.width,
            "height": self.gt.height,
            "gtFingerprint": self.gt.fingerprint,
            "predictionEncoding": "int32 class_index; -1 means invalid prediction",
            "metadata": self.metadata_payload,
        }
        path = self.output_dir / "prediction_manifest.json"
        _json_dump(path, manifest)
        return path


def _accumulate(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    num_classes: int,
    exclude_indices: Iterable[int] = (),
) -> tuple[np.ndarray, np.ndarray]:
    gt_flat = gt_map.reshape(-1).astype(np.int64, copy=False)
    pred_flat = pred_map.reshape(-1).astype(np.int64, copy=False)
    valid_gt = (gt_flat >= 0) & (gt_flat < num_classes)
    excluded = tuple(int(index) for index in exclude_indices)
    if excluded:
        valid_gt &= ~np.isin(gt_flat, excluded)
    valid_pred = (pred_flat >= 0) & (pred_flat < num_classes)
    valid_pairs = valid_gt & valid_pred
    encoded = gt_flat[valid_pairs] * num_classes + pred_flat[valid_pairs]
    confusion = np.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    missing = np.bincount(gt_flat[valid_gt & ~valid_pred], minlength=num_classes)
    return confusion.astype(np.int64, copy=False), missing.astype(np.int64, copy=False)


def _compute_metrics(
    confusion: np.ndarray,
    missing: np.ndarray,
    classes: Sequence[SemanticClass],
    metric_class_scope: str,
    exclude_indices: Iterable[int],
) -> dict:
    confusion_f = confusion.astype(np.float64, copy=False)
    missing_f = missing.astype(np.float64, copy=False)
    true_positive = np.diag(confusion_f)
    gt_count = confusion_f.sum(axis=1) + missing_f
    pred_count = confusion_f.sum(axis=0)
    union = gt_count + pred_count - true_positive

    if metric_class_scope == "gt_present":
        eval_mask = gt_count > 0
    elif metric_class_scope == "all":
        eval_mask = union > 0
    else:
        raise RuntimeError(f"Unsupported metric class scope: {metric_class_scope}")
    for index in exclude_indices:
        eval_mask[int(index)] = False

    valid_iou = (union > 0) & eval_mask
    valid_acc = (gt_count > 0) & eval_mask
    iou = np.zeros_like(true_positive)
    acc = np.zeros_like(true_positive)
    iou[union > 0] = true_positive[union > 0] / union[union > 0]
    acc[gt_count > 0] = true_positive[gt_count > 0] / gt_count[gt_count > 0]
    total = float(gt_count.sum())

    per_class = []
    for item in classes:
        index = item.index
        per_class.append(
            {
                "classIndex": index,
                "classId": item.class_id,
                "className": item.name,
                "IoU": float(iou[index]) if valid_iou[index] else None,
                "Acc": float(acc[index]) if valid_acc[index] else None,
                "GT_COUNT": int(gt_count[index]),
                "PRED_COUNT": int(pred_count[index]),
                "MISSING_PRED_COUNT": int(missing[index]),
                "EVALUATED": bool(eval_mask[index]),
            }
        )

    return {
        "mIoU": float(iou[valid_iou].mean()) if np.any(valid_iou) else 0.0,
        "mAcc": float(acc[valid_acc].mean()) if np.any(valid_acc) else 0.0,
        "overallAcc": float(true_positive.sum() / total) if total > 0 else 0.0,
        "numClasses": len(classes),
        "numEvaluatedClasses": int(eval_mask.sum()),
        "numLabeled": int(total),
        "numMissingPredictions": int(missing.sum()),
        "predictionCoverage": float((total - missing.sum()) / total) if total > 0 else 0.0,
        "perClass": per_class,
    }


def _labels_to_rgb(label_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    valid = (label_map >= 0) & (label_map < len(palette))
    rgb[valid] = palette[label_map[valid]]
    return rgb


def _save_visualization(
    output_dir: Path,
    view_name: str,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    palette: np.ndarray,
) -> None:
    gt_rgb = _labels_to_rgb(gt_map, palette)
    pred_rgb = _labels_to_rgb(pred_map, palette)
    comparison = np.concatenate((gt_rgb, pred_rgb), axis=1)
    for dirname, image in (
        ("gt_semantic_rgb", gt_rgb),
        ("pred_semantic_rgb", pred_rgb),
        ("compare_gt_left_pred_right", comparison),
    ):
        target = output_dir / dirname
        target.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(target / f"{view_name}.png")


def _write_per_class_csv(path: Path, per_class: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "classIndex",
        "classId",
        "className",
        "IoU",
        "Acc",
        "GT_COUNT",
        "PRED_COUNT",
        "MISSING_PRED_COUNT",
        "EVALUATED",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class)


def score_prediction_directory(
    gt: UnifiedSemanticGT,
    prediction_dir: str | Path,
    output_dir: str | Path,
    metric_class_scope: str = "gt_present",
    exclude_classes: Sequence[str] | None = None,
    save_visualizations: bool = False,
    require_prediction_manifest: bool = True,
) -> dict:
    prediction_dir = Path(prediction_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_manifest_path = prediction_dir.parent / "prediction_manifest.json"
    prediction_manifest = None
    if prediction_manifest_path.is_file():
        prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
        if prediction_manifest.get("protocol") != PROTOCOL_NAME:
            raise RuntimeError(
                "Prediction manifest protocol mismatch: "
                f"expected={PROTOCOL_NAME!r} "
                f"actual={prediction_manifest.get('protocol')!r}"
            )
        if prediction_manifest.get("gtFingerprint") != gt.fingerprint:
            raise RuntimeError("Prediction manifest GT fingerprint does not match the selected unified GT")
    elif require_prediction_manifest:
        raise FileNotFoundError(f"Prediction manifest is missing: {prediction_manifest_path}")

    expected = {f"{name}.npy" for name in gt.view_names}
    actual = {path.name for path in prediction_dir.glob("*.npy")}
    missing_files = sorted(expected - actual)
    extra_files = sorted(actual - expected)
    if missing_files or extra_files:
        raise RuntimeError(
            f"Prediction/view mismatch: missing={missing_files[:8]} extra={extra_files[:8]}"
        )

    exclude_indices = gt.resolve_class_indices(exclude_classes, allow_missing=True)
    confusion = np.zeros((gt.num_classes, gt.num_classes), dtype=np.int64)
    missing = np.zeros((gt.num_classes,), dtype=np.int64)
    per_view = {}

    for view_name in gt.view_names:
        gt_map = gt.load_label(view_name)
        pred_map = np.load(prediction_dir / f"{view_name}.npy", allow_pickle=False)
        gt.validate_map(pred_map, f"Prediction {view_name}", allow_invalid=True)
        pred_map = pred_map.astype(np.int32, copy=False)
        view_confusion, view_missing = _accumulate(
            gt_map,
            pred_map,
            gt.num_classes,
            exclude_indices=exclude_indices,
        )
        confusion += view_confusion
        missing += view_missing
        view_metrics = _compute_metrics(
            view_confusion,
            view_missing,
            gt.classes,
            metric_class_scope,
            exclude_indices,
        )
        view_metrics.pop("perClass", None)
        per_view[view_name] = view_metrics
        if save_visualizations:
            visual_gt = gt_map.copy()
            if exclude_indices:
                visual_gt[np.isin(visual_gt, exclude_indices)] = -1
            _save_visualization(output_dir, view_name, visual_gt, pred_map, gt.palette)

    summary = _compute_metrics(
        confusion,
        missing,
        gt.classes,
        metric_class_scope,
        exclude_indices,
    )
    summary.update(
        {
            "protocol": PROTOCOL_NAME,
            "scene": gt.scene,
            "split": gt.split,
            "numViews": len(gt.view_names),
            "metricClassScope": metric_class_scope,
            "excludedClassIndices": exclude_indices,
            "excludedClasses": [gt.class_names[index] for index in exclude_indices],
            "requestedExcludedClasses": list(exclude_classes or ()),
            "gtFingerprint": gt.fingerprint,
            "gtSplitDir": str(gt.split_dir),
            "predictionDir": str(prediction_dir),
            "predictionManifest": prediction_manifest,
        }
    )
    if prediction_manifest is not None:
        prediction_metadata = prediction_manifest.get("metadata", {})
        for key in (
            "predictionVocabulary",
            "predictionClassScope",
            "predictionClassCount",
            "metricClassScope",
            "vocabularySource",
            "vocabularySha256",
        ):
            if key in prediction_metadata:
                summary[key] = prediction_metadata[key]

    serializable = dict(summary)
    per_class = serializable.pop("perClass")
    _json_dump(output_dir / "summary.json", serializable)
    _json_dump(output_dir / "per_view_metrics.json", per_view)
    np.save(output_dir / "confusion_matrix.npy", confusion, allow_pickle=False)
    np.save(output_dir / "missing_predictions_by_gt_class.npy", missing, allow_pickle=False)
    _write_per_class_csv(output_dir / "per_class_metrics.csv", per_class)
    (output_dir / "semantic_palette.csv").write_bytes(gt.palette_path.read_bytes())
    return summary
