from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from eval_direct_3d_19 import text_prompt_metadata


def validate_prompt_metadata(metadata: dict, mode: str) -> None:
    if metadata.get("promptMode") != mode:
        raise RuntimeError("Text prompt mode mismatch")
    # Historical single-template outputs predate the expanded metadata.
    if mode == "maskclustering":
        for key, value in text_prompt_metadata(mode).items():
            if metadata.get(key) != value:
                raise RuntimeError(f"Four-template protocol mismatch: {key}")


DEFAULT_METHODS = ("ours",)


def expected_transfer(method: str, requested: str) -> str:
    if requested != "method_native":
        return requested
    return "knn_latent"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def confusion_metrics(confusion: np.ndarray, vocabulary: list[str], gt_count=None) -> tuple[dict, list[dict]]:
    paired_count = confusion.sum(axis=1)
    if gt_count is None:
        raise ValueError("Full GT counts are required: rejected predictions must not disappear")
    gt_count = np.asarray(gt_count, dtype=np.int64)
    if gt_count.shape != paired_count.shape or np.any(gt_count < paired_count):
        raise ValueError("Full GT counts do not match the confusion matrix")
    pred_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    union = gt_count + pred_count - true_positive
    present = gt_count > 0
    iou = np.divide(
        true_positive,
        union,
        out=np.full(len(vocabulary), np.nan, dtype=np.float64),
        where=union > 0,
    )
    accuracy = np.divide(
        true_positive,
        gt_count,
        out=np.full(len(vocabulary), np.nan, dtype=np.float64),
        where=gt_count > 0,
    )
    total = int(gt_count.sum())
    summary = {
        "mIoU": float(np.nanmean(iou[present])),
        "mAcc": float(np.nanmean(accuracy[present])),
        "overallAcc": float(true_positive.sum() / max(total, 1)),
        "evaluatedVertexCount": total,
        "presentClassCount": int(present.sum()),
        "presentClassNames": [vocabulary[index] for index in np.flatnonzero(present)],
    }
    per_class = []
    for index, class_name in enumerate(vocabulary):
        per_class.append(
            {
                "class_index": index,
                "class_name": class_name,
                "present_in_aggregate_gt": bool(present[index]),
                "gt_count": int(gt_count[index]),
                "pred_count": int(pred_count[index]),
                "true_positive": int(true_positive[index]),
                "iou": None if math.isnan(float(iou[index])) else float(iou[index]),
                "accuracy": None if math.isnan(float(accuracy[index])) else float(accuracy[index]),
            }
        )
    return summary, per_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate InstGuide-GS ScanNet++ Top-100 direct-3D results")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--summary_subdir", default="summary")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--geometry_transfer",
        choices=("method_native", "nearest_label", "knn_score", "knn_latent"),
        default="method_native",
    )
    parser.add_argument("--knn_k", type=int, default=1)
    parser.add_argument(
        "--knn_weight",
        choices=("inverse_distance", "gaussian_rbf", "uniform"),
        default="inverse_distance",
    )
    parser.add_argument("--prompt_mode", choices=("single", "maskclustering"), default="single")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    summary_dir = (output_root / args.summary_subdir).resolve()
    if summary_dir == output_root or not summary_dir.is_relative_to(output_root):
        raise ValueError("Summary directory must be a child of output_root")
    summary_dir.mkdir(parents=True, exist_ok=True)

    reference_vocabulary: list[str] | None = None
    scene_rows: list[dict] = []
    method_confusions: dict[str, np.ndarray] = {}
    method_gt_counts: dict[str, np.ndarray] = {}
    method_scene_metrics: dict[str, list[dict]] = {method: [] for method in args.methods}
    audit_tasks: list[dict] = []
    scene_gt_signatures: dict[str, tuple] = {}
    scene_gt_summaries: dict[str, dict] = {}

    for scene in args.scenes:
        for method in args.methods:
            required_transfer = expected_transfer(method, args.geometry_transfer)
            task_dir = output_root / scene / method
            metrics_path = task_dir / "metrics.json"
            confusion_path = task_dir / "confusion_matrix.csv"
            if not metrics_path.is_file() or not confusion_path.is_file():
                raise FileNotFoundError(f"Incomplete evaluation output: {scene}/{method}")
            payload = read_json(metrics_path)
            vocabulary = payload.get("predictionVocabulary")
            if payload.get("classProtocol") != "scannetpp100" or payload.get("classCount") != 100:
                raise RuntimeError(f"Wrong class protocol in {metrics_path}")
            if payload.get("allProtocolClassesCompeteInArgmax") is not True:
                raise RuntimeError(f"Closed-set competition is not recorded in {metrics_path}")
            if payload.get("uses2DSemanticPrediction") is not False:
                raise RuntimeError(f"Unexpected 2D semantic prediction in {metrics_path}")
            transfer_config = payload.get("geometryTransferConfig", {})
            if not transfer_config and payload.get("geometryTransfer") in {
                "one nearest native semantic carrier in Euclidean 3D; no distance rejection",
                "one nearest native semantic carrier hard label in Euclidean 3D; no distance rejection",
            }:
                transfer_config = {
                    "mode": "nearest_label",
                    "neighborCount": 1,
                    "weight": None,
                    "scoreThenArgmax": False,
                    "cameraProjectionUsed": False,
                    "twoDimensionalPredictionUsed": False,
                }
            if transfer_config.get("mode") != required_transfer:
                raise RuntimeError(f"Unexpected geometry transfer mode in {metrics_path}")
            if transfer_config.get("cameraProjectionUsed") is not False:
                raise RuntimeError(f"Camera projection was used in {metrics_path}")
            if transfer_config.get("twoDimensionalPredictionUsed") is not False:
                raise RuntimeError(f"2D semantic prediction was used in {metrics_path}")
            if required_transfer in {"knn_score", "knn_latent"} and (
                transfer_config.get("neighborCount") != args.knn_k
                or transfer_config.get("weight") != args.knn_weight
            ):
                raise RuntimeError(f"Unexpected kNN transfer settings in {metrics_path}")
            if required_transfer == "knn_score" and transfer_config.get("scoreThenArgmax") is not True:
                raise RuntimeError(f"Expected score-then-argmax transfer in {metrics_path}")
            if required_transfer == "knn_latent" and transfer_config.get("latentThenDecodeThenArgmax") is not True:
                raise RuntimeError(f"Expected latent-then-decode transfer in {metrics_path}")
            text_encoder = payload.get("carrier", {}).get("textEncoder", {})
            validate_prompt_metadata(text_encoder, args.prompt_mode)
            if payload.get("groundTruthMultiplicity") != "single_label_vertex_field" or payload.get("topK") != 1:
                raise RuntimeError(f"Unexpected local GT/Top-K protocol in {metrics_path}")
            if not isinstance(vocabulary, list) or len(vocabulary) != 100:
                raise RuntimeError(f"Invalid Top-100 vocabulary in {metrics_path}")
            if reference_vocabulary is None:
                reference_vocabulary = vocabulary
            elif vocabulary != reference_vocabulary:
                raise RuntimeError(f"Vocabulary order mismatch in {metrics_path}")

            gt = payload["gt"]
            gt_signature = (
                gt.get("mesh"),
                gt.get("rawClasses"),
                gt.get("benchmarkMapSha256"),
                gt.get("top100ClassesSha256"),
                gt.get("evaluatedVertexCount"),
                tuple(gt.get("presentClassNames", [])),
            )
            previous_signature = scene_gt_signatures.setdefault(scene, gt_signature)
            if gt_signature != previous_signature:
                raise RuntimeError(f"GT protocol mismatch between methods for scene {scene}")
            scene_gt_summaries.setdefault(
                scene,
                {
                    "meshVertexCount": gt.get("meshVertexCount"),
                    "evaluatedVertexCount": gt.get("evaluatedVertexCount"),
                    "mappedFractionOfRawNonVoid": gt.get("mappedFractionOfRawNonVoid"),
                    "presentClassCount": len(gt.get("presentClassNames", [])),
                    "presentClassNames": gt.get("presentClassNames", []),
                },
            )

            confusion = np.loadtxt(confusion_path, delimiter=",", dtype=np.int64)
            if confusion.shape != (100, 100):
                raise RuntimeError(f"Expected a 100x100 confusion matrix: {confusion_path}")
            method_confusions[method] = method_confusions.get(method, np.zeros((100, 100), dtype=np.int64)) + confusion
            metrics = payload["metrics"]
            per_class = metrics["perClass"]
            if [row["class_name"] for row in per_class] != vocabulary:
                raise RuntimeError(f"Per-class metric order mismatch in {metrics_path}")
            gt_counts = np.asarray([row["gt_count"] for row in per_class], dtype=np.int64)
            if gt_counts.sum() != metrics["evaluatedVertexCount"]:
                raise RuntimeError(f"Incomplete GT counts in {metrics_path}")
            # The square confusion matrix excludes rejected predictions, but
            # those vertices still contribute false negatives and the OA denominator.
            recomputed, _ = confusion_metrics(confusion, vocabulary, gt_counts)
            for key in ("mIoU", "mAcc", "overallAcc"):
                if not np.isclose(recomputed[key], metrics[key], atol=1e-10, rtol=0):
                    raise RuntimeError(f"Stored {key} disagrees with counts in {metrics_path}")
            method_gt_counts[method] = method_gt_counts.get(method, np.zeros(100, dtype=np.int64)) + gt_counts
            metric_row = {
                "method": method,
                "scene": scene,
                "mIoU": metrics["mIoU"],
                "mAcc": metrics["mAcc"],
                "overallAcc": metrics["overallAcc"],
                "evaluatedVertexCount": metrics["evaluatedVertexCount"],
                "presentClassCount": metrics["presentClassCount"],
                "predictionCoverage": metrics["predictionCoverage"],
            }
            scene_rows.append(metric_row)
            method_scene_metrics[method].append(metric_row)
            audit_tasks.append(
                {
                    "scene": scene,
                    "method": method,
                    "metrics": str(metrics_path),
                    "gtSignature": list(gt_signature[:-1]) + [list(gt_signature[-1])],
                    "carrier": payload.get("carrier", {}),
                    "nearestDistanceMeters": payload.get("nearestDistanceMeters", {}),
                    "knnSupportDistanceMeters": payload.get("knnSupportDistanceMeters", {}),
                    "geometryTransferConfig": transfer_config,
                }
            )

    if reference_vocabulary is None:
        raise RuntimeError("No evaluation results found")

    scene_mean_rows = []
    aggregate_rows = []
    all_per_class = []
    for method in args.methods:
        rows = method_scene_metrics[method]
        if len(rows) != len(args.scenes):
            raise RuntimeError(f"Method {method} has {len(rows)} scenes; expected {len(args.scenes)}")
        scene_mean = {
            "method": method,
            "scene": "unweighted_scene_mean",
            "mIoU": float(np.mean([row["mIoU"] for row in rows])),
            "mAcc": float(np.mean([row["mAcc"] for row in rows])),
            "overallAcc": float(np.mean([row["overallAcc"] for row in rows])),
            "evaluatedVertexCount": int(sum(row["evaluatedVertexCount"] for row in rows)),
            "presentClassCount": "",
            "predictionCoverage": float(np.mean([row["predictionCoverage"] for row in rows])),
        }
        scene_mean_rows.append(scene_mean)
        aggregate, per_class = confusion_metrics(method_confusions[method], reference_vocabulary, method_gt_counts[method])
        aggregate_rows.append(
            {
                "method": method,
                "aggregate_mIoU": aggregate["mIoU"],
                "aggregate_mAcc": aggregate["mAcc"],
                "aggregate_overallAcc": aggregate["overallAcc"],
                "unweighted_scene_mean_mIoU": scene_mean["mIoU"],
                "unweighted_scene_mean_mAcc": scene_mean["mAcc"],
                "unweighted_scene_mean_overallAcc": scene_mean["overallAcc"],
                "evaluatedVertexCount": aggregate["evaluatedVertexCount"],
                "aggregatePresentClassCount": aggregate["presentClassCount"],
            }
        )
        for row in per_class:
            all_per_class.append({"method": method, **row})
        np.savetxt(summary_dir / f"{method}_pooled_confusion.csv", method_confusions[method], delimiter=",", fmt="%d")

    write_csv(
        summary_dir / "per_scene_metrics.csv",
        list(scene_rows[0].keys()),
        scene_rows + scene_mean_rows,
    )
    write_csv(
        summary_dir / "aggregate_metrics.csv",
        list(aggregate_rows[0].keys()),
        aggregate_rows,
    )
    write_csv(
        summary_dir / "aggregate_per_class.csv",
        list(all_per_class[0].keys()),
        all_per_class,
    )
    audit = {
        "auditStatus": "PASS",
        "protocol": "ScanNet++ official Top-100 vocabulary direct 3D single-label per-vertex evaluation v2",
        "scope": "selected ScanNet++ scenes; not an official validation-server submission",
        "scenes": args.scenes,
        "methods": args.methods,
        "predictionVocabulary": reference_vocabulary,
        "all100ClassesCompeteInArgmax": True,
        "metricMacroAverage": "GT-present Top-100 classes",
        "overallAccuracyScope": "all vertices whose raw GT maps to Top-100",
        "voidAndUnmappedGtIgnored": True,
        "groundTruthMultiplicity": "single label from vertex.label",
        "topK": 1,
        "geometryTransfer": {
            "mode": args.geometry_transfer,
            "methodModes": {
                method: expected_transfer(method, args.geometry_transfer) for method in args.methods
            },
            "neighborCount": args.knn_k,
            "weight": args.knn_weight,
            "cameraProjectionUsed": False,
            "twoDimensionalPredictionUsed": False,
        },
        "promptMode": args.prompt_mode,
        "textPromptProtocol": text_prompt_metadata(args.prompt_mode),
        "promptProtocolScope": (
            "controlled common positive-template adaptation, not original-paper prompt reproduction"
            if args.prompt_mode == "maskclustering" else "single class-name queries"
        ),
        "aggregatePolicy": "sum selected scene confusion matrices before computing aggregate metrics",
        "alsoReports": "unweighted arithmetic mean of the per-scene metrics",
        "sceneGtSummaries": scene_gt_summaries,
        "tasks": audit_tasks,
    }
    (summary_dir / "protocol_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({"output": str(summary_dir), "aggregate": aggregate_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
