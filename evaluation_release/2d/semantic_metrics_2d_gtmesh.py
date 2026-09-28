import csv
import hashlib
import json
import os
import sys
import traceback
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.semantic_benchmark_utils import (
    _apply_semantic_eval_exclude_indices,
    _build_gt_label_mapper,
    _encode_open_vocab_text_features,
    _load_replica_object_to_class,
    _map_gt_labels_to_indices,
    _render_semantic_feature_map_for_view,
    _resolve_semantic_eval_exclude_indices,
    _semantic_eval_present_class_indices,
    _to_abs_path,
    load_semantic_prompt_bank,
)
from utils.semantic_mesh_rasterizer import rasterize_semantic_mesh
from utils.system_utils import configure_visible_devices


SELECTED_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES_SOURCE = configure_visible_devices(sys.argv[1:])


def _resolve_teacher_clip_model(args):
    explicit = str(getattr(args, "teacher_clip_model", "") or "").strip()
    if not explicit:
        raise RuntimeError(
            "--teacher_clip_model is required for semantic evaluation. "
            "Use the same text encoder architecture and checkpoint as the teacher."
        )
    return explicit

def _select_views(scene, split, max_views=0, stride=1):
    split = str(split or "test").lower()
    views = []
    if split in {"train", "all"}:
        views.extend(scene.getTrainCameras())
    if split in {"test", "all"}:
        views.extend(scene.getTestCameras())
    if split not in {"train", "test", "all"}:
        raise RuntimeError("--semantic_2d_split must be one of train, test, all.")
    views = views[:: max(1, int(stride))]
    if int(max_views) > 0:
        views = views[: int(max_views)]
    return views


def _triangulate_faces(face_vertex_lists):
    lengths = np.fromiter((len(face) for face in face_vertex_lists), dtype=np.int64)
    if lengths.size == 0:
        raise RuntimeError("GT semantic mesh has no faces.")
    if np.all(lengths == 3):
        triangles = np.vstack(face_vertex_lists).astype(np.int64, copy=False)
        source_face_ids = np.arange(triangles.shape[0], dtype=np.int64)
        return triangles, source_face_ids

    triangles = []
    source_face_ids = []
    for face_id, polygon in enumerate(face_vertex_lists):
        polygon = np.asarray(polygon, dtype=np.int64)
        for offset in range(1, len(polygon) - 1):
            triangles.append((polygon[0], polygon[offset], polygon[offset + 1]))
            source_face_ids.append(face_id)
    if not triangles:
        raise RuntimeError("GT semantic mesh contains no triangulatable faces.")
    return np.asarray(triangles, dtype=np.int64), np.asarray(source_face_ids, dtype=np.int64)


def _load_gt_semantic_mesh(
    mesh_path,
    gt_info_path,
    gt_to_index,
    ignore_values,
    exclude_indices,
    mixed_face_policy,
):
    mesh_path = Path(mesh_path)
    if mesh_path.suffix.lower() != ".ply":
        raise RuntimeError("GT-mesh 2D rasterization currently requires a face-bearing .ply file.")
    ply = PlyData.read(str(mesh_path))
    if "vertex" not in ply or "face" not in ply:
        raise RuntimeError(f"GT semantic PLY must contain vertex and face elements: {mesh_path}")

    vertex_data = ply["vertex"].data
    vertices = np.column_stack(
        (vertex_data["x"], vertex_data["y"], vertex_data["z"])
    ).astype(np.float32, copy=False)
    face_data = ply["face"].data
    face_names = set(face_data.dtype.names or [])
    face_index_key = "vertex_indices" if "vertex_indices" in face_names else "vertex_index"
    if face_index_key not in face_names:
        raise RuntimeError(f"GT semantic PLY face element has no vertex index list: {mesh_path}")
    triangles, source_face_ids = _triangulate_faces(face_data[face_index_key])

    direct_face_label_key = next(
        (key for key in ("label", "semantic_gt", "class_id", "category_id") if key in face_names),
        None,
    )
    label_source = ""
    mixed_faces = 0
    if direct_face_label_key is not None:
        raw_face_labels = np.asarray(face_data[direct_face_label_key], dtype=np.int64)
        mapped_source_labels = _map_gt_labels_to_indices(
            torch.from_numpy(raw_face_labels),
            gt_to_index,
            ignore_values,
        )
        mapped_source_labels = _apply_semantic_eval_exclude_indices(mapped_source_labels, exclude_indices)
        triangle_labels = mapped_source_labels.numpy()[source_face_ids]
        label_source = f"face.{direct_face_label_key}"
    elif "object_id" in face_names:
        if gt_info_path is None:
            raise RuntimeError(
                "GT mesh stores face.object_id. Pass --semantic_gt_info with the Habitat/Replica mapping JSON."
            )
        object_to_class = _load_replica_object_to_class(gt_info_path)
        object_ids = np.asarray(face_data["object_id"], dtype=np.int64)
        raw_face_labels = np.asarray(
            [object_to_class.get(int(object_id), -1) for object_id in object_ids],
            dtype=np.int64,
        )
        mapped_source_labels = _map_gt_labels_to_indices(
            torch.from_numpy(raw_face_labels),
            gt_to_index,
            ignore_values | {-1},
        )
        mapped_source_labels = _apply_semantic_eval_exclude_indices(mapped_source_labels, exclude_indices)
        triangle_labels = mapped_source_labels.numpy()[source_face_ids]
        label_source = "face.object_id->class_id"
    else:
        vertex_names = set(vertex_data.dtype.names or [])
        vertex_label_key = next(
            (key for key in ("label", "semantic_gt", "class_id", "category_id") if key in vertex_names),
            None,
        )
        if vertex_label_key is None:
            raise RuntimeError(f"GT semantic PLY has no supported vertex or face semantic label: {mesh_path}")
        raw_vertex_labels = np.asarray(vertex_data[vertex_label_key], dtype=np.int64)
        mapped_vertex_labels = _map_gt_labels_to_indices(
            torch.from_numpy(raw_vertex_labels),
            gt_to_index,
            ignore_values,
        )
        mapped_vertex_labels = _apply_semantic_eval_exclude_indices(mapped_vertex_labels, exclude_indices)
        triangle_vertex_labels = mapped_vertex_labels.numpy()[triangles]
        all_equal = (
            (triangle_vertex_labels[:, 0] == triangle_vertex_labels[:, 1])
            & (triangle_vertex_labels[:, 0] == triangle_vertex_labels[:, 2])
        )
        mixed_faces = int((~all_equal).sum())
        if mixed_face_policy == "majority":
            sorted_labels = np.sort(triangle_vertex_labels, axis=1)
            majority = np.where(
                sorted_labels[:, 0] == sorted_labels[:, 1],
                sorted_labels[:, 0],
                sorted_labels[:, 2],
            )
            all_distinct = (
                (sorted_labels[:, 0] != sorted_labels[:, 1])
                & (sorted_labels[:, 1] != sorted_labels[:, 2])
            )
            majority[all_distinct] = -1
            triangle_labels = majority
        else:
            triangle_labels = triangle_vertex_labels[:, 0].copy()
            triangle_labels[~all_equal] = -1
        label_source = f"vertex.{vertex_label_key}"

    return vertices, triangles, triangle_labels.astype(np.int64, copy=False), {
        "path": str(mesh_path),
        "numVertices": int(vertices.shape[0]),
        "numTriangles": int(triangles.shape[0]),
        "labelSource": label_source,
        "mixedFacePolicy": mixed_face_policy,
        "numMixedTriangles": mixed_faces,
        "numValidLabelTriangles": int((triangle_labels >= 0).sum()),
    }


def _predict_semantic_map(feature_map, text_features, chunk_size, class_indices=None):
    channels, height, width = feature_map.shape
    flat_features = feature_map.permute(1, 2, 0).reshape(-1, channels)
    flat_features = F.normalize(flat_features, dim=-1, eps=1e-6)
    text_features = F.normalize(text_features.to(flat_features.device), dim=-1, eps=1e-6)
    class_index_tensor = None
    if class_indices is not None:
        class_index_tensor = torch.as_tensor(class_indices, dtype=torch.long, device=flat_features.device)
        if class_index_tensor.numel() == 0:
            raise RuntimeError("Prediction vocabulary restriction is empty.")
        text_features = text_features[class_index_tensor]

    predictions = torch.empty(flat_features.shape[0], dtype=torch.long, device="cpu")
    with torch.no_grad():
        for start in range(0, flat_features.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), flat_features.shape[0])
            local_predictions = (flat_features[start:end] @ text_features.T).argmax(dim=1)
            if class_index_tensor is not None:
                local_predictions = class_index_tensor[local_predictions]
            predictions[start:end] = local_predictions.cpu()
    return predictions.view(height, width)


def _update_confusion(confusion, missing_by_gt, gt_indices, pred_indices, pred_valid, num_classes):
    gt_indices = torch.as_tensor(gt_indices, dtype=torch.long).view(-1)
    pred_indices = torch.as_tensor(pred_indices, dtype=torch.long).view(-1)
    pred_valid = torch.as_tensor(pred_valid, dtype=torch.bool).view(-1)
    if not (gt_indices.numel() == pred_indices.numel() == pred_valid.numel()):
        raise RuntimeError("GT, prediction and validity maps have inconsistent sizes.")
    valid_gt = (gt_indices >= 0) & (gt_indices < int(num_classes))
    valid_pairs = valid_gt & pred_valid & (pred_indices >= 0) & (pred_indices < int(num_classes))
    if bool(valid_pairs.any().item()):
        encoded = gt_indices[valid_pairs] * int(num_classes) + pred_indices[valid_pairs]
        confusion += torch.bincount(
            encoded,
            minlength=int(num_classes) * int(num_classes),
        ).view(num_classes, num_classes)
    missing_gt = gt_indices[valid_gt & ~valid_pairs]
    if missing_gt.numel():
        missing_by_gt += torch.bincount(missing_gt, minlength=int(num_classes))


def _compute_metrics_with_missing(
    confusion,
    missing_by_gt,
    class_names,
    metric_class_scope,
    exclude_indices=None,
):
    confusion = confusion.to(torch.float64)
    missing_by_gt = missing_by_gt.to(torch.float64)
    true_positive = confusion.diag()
    gt_count = confusion.sum(dim=1) + missing_by_gt
    pred_count = confusion.sum(dim=0)
    union = gt_count + pred_count - true_positive

    if metric_class_scope == "gt_present":
        eval_mask = gt_count > 0
    else:
        eval_mask = union > 0
    if exclude_indices:
        eval_mask[torch.as_tensor(sorted(exclude_indices), dtype=torch.long)] = False
    valid_iou = (union > 0) & eval_mask
    valid_acc = (gt_count > 0) & eval_mask
    iou = torch.zeros_like(true_positive)
    acc = torch.zeros_like(true_positive)
    iou[union > 0] = true_positive[union > 0] / union[union > 0]
    acc[gt_count > 0] = true_positive[gt_count > 0] / gt_count[gt_count > 0]

    total = float(gt_count.sum().item())
    per_class = {}
    for class_index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "IoU": float(iou[class_index].item()) if bool(valid_iou[class_index].item()) else None,
            "Acc": float(acc[class_index].item()) if bool(valid_acc[class_index].item()) else None,
            "GT_COUNT": int(gt_count[class_index].item()),
            "PRED_COUNT": int(pred_count[class_index].item()),
            "MISSING_PRED_COUNT": int(missing_by_gt[class_index].item()),
            "EVALUATED": bool(eval_mask[class_index].item()),
        }
    return {
        "mIoU": float(iou[valid_iou].mean().item()) if bool(valid_iou.any().item()) else 0.0,
        "mAcc": float(acc[valid_acc].mean().item()) if bool(valid_acc.any().item()) else 0.0,
        "overallAcc": float(true_positive.sum().item() / total) if total > 0 else 0.0,
        "numClasses": len(class_names),
        "numEvaluatedClasses": int(eval_mask.sum().item()),
        "numLabeled": int(total),
        "numMissingPredictions": int(missing_by_gt.sum().item()),
        "predictionCoverage": float((total - missing_by_gt.sum().item()) / total) if total > 0 else 0.0,
        "perClass": per_class,
    }


def _file_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtimeNs": int(stat.st_mtime_ns),
    }


def _optional_fingerprint(path, source_path):
    resolved = _to_abs_path(path, source_path)
    return _file_fingerprint(resolved) if resolved is not None else None


def _camera_signature(views):
    signature = []
    for view in views:
        signature.append(
            {
                "imageName": str(view.image_name),
                "width": int(view.image_width),
                "height": int(view.image_height),
                "worldView": view.world_view_transform.detach().cpu().float().reshape(-1).tolist(),
                "fullProjection": view.full_proj_transform.detach().cpu().float().reshape(-1).tolist(),
            }
        )
    return signature


def _cache_signature(mesh_path, prompt_path, gt_info_path, dataset, views, args):
    return {
        "formatVersion": 1,
        "mesh": _file_fingerprint(mesh_path),
        "prompt": _file_fingerprint(prompt_path),
        "gtInfo": _file_fingerprint(gt_info_path) if gt_info_path is not None else None,
        "rawClassPath": _optional_fingerprint(
            getattr(dataset, "semantic_raw_class_path", ""),
            dataset.source_path,
        ),
        "benchmarkMapPath": _optional_fingerprint(
            getattr(dataset, "semantic_benchmark_map_path", ""),
            dataset.source_path,
        ),
        "ignoreLabel": int(args.semantic_eval_ignore_label),
        "excludeClasses": str(args.semantic_eval_exclude_classes or ""),
        "mixedFacePolicy": args.gt_mesh_mixed_face_policy,
        "tileSize": int(args.gt_raster_tile_size),
        "barycentricEpsilon": float(args.gt_raster_barycentric_epsilon),
        "cameras": _camera_signature(views),
    }


def _load_cache_manifest(cache_dir):
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_per_class_csv(path, metrics):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "IoU", "Acc", "GT_COUNT", "PRED_COUNT", "MISSING_PRED_COUNT", "EVALUATED"))
        for class_name, values in metrics["perClass"].items():
            writer.writerow(
                (
                    class_name,
                    values["IoU"],
                    values["Acc"],
                    values["GT_COUNT"],
                    values["PRED_COUNT"],
                    values["MISSING_PRED_COUNT"],
                    values["EVALUATED"],
                )
            )


def _build_gaussian_model(dataset):
    return GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.ratio,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
        dataset.use_semantic_features,
        dataset.semantic_latent_dim,
        dataset.semantic_feature_dim,
        semantic_teacher_dim=getattr(dataset, "semantic_teacher_dim", 0),
        semantic_codec_path=getattr(dataset, "semantic_codec_path", ""),
        sh_degree=dataset.sh_degree,
        use_anchor_offset_semantic=getattr(dataset, "use_anchor_offset_semantic", False),
        semantic_residual_dim=getattr(dataset, "semantic_residual_dim", 4),
        semantic_gate_init=getattr(dataset, "semantic_gate_init", -4.0),
    )


def _run_evaluation(dataset, pipeline, iteration, args):
    if not getattr(dataset, "semantic_eval_prompt_path", ""):
        raise RuntimeError("2D GT-mesh evaluation requires --semantic_eval_prompt_path.")
    if not getattr(dataset, "semantic_gt_3d", ""):
        raise RuntimeError("2D GT-mesh evaluation requires --semantic_gt_3d.")

    dataset.depths = ""
    dataset.use_semantic_features = True
    gaussians = _build_gaussian_model(dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.eval()

    prompt_path = _to_abs_path(dataset.semantic_eval_prompt_path, dataset.source_path)
    gt_path = _to_abs_path(dataset.semantic_gt_3d, dataset.source_path)
    gt_info_path = _to_abs_path(getattr(dataset, "semantic_gt_info", ""), dataset.source_path)
    prompt_bank = load_semantic_prompt_bank(prompt_path)
    gt_to_index, ignore_values, gt_mapping_info = _build_gt_label_mapper(dataset, prompt_bank, logger=None)
    exclude_indices, exclude_names = _resolve_semantic_eval_exclude_indices(dataset, prompt_bank, logger=None)
    vertices, triangles, triangle_labels, mesh_info = _load_gt_semantic_mesh(
        gt_path,
        gt_info_path,
        gt_to_index,
        ignore_values,
        exclude_indices,
        args.gt_mesh_mixed_face_policy,
    )

    teacher_clip_model = _resolve_teacher_clip_model(args)
    text_features = _encode_open_vocab_text_features(
        prompt_bank["prompts"],
        teacher_clip_model,
        getattr(dataset, "semantic_feature_dim", 512),
        device=str(gaussians.get_anchor.device),
        prompt_template_mode=args.semantic_text_prompt_mode,
    )
    present_class_indices = _semantic_eval_present_class_indices(
        torch.from_numpy(triangle_labels),
        len(prompt_bank["class_ids"]),
    )
    prediction_class_indices = present_class_indices if args.oracle_gt_present_vocabulary else None
    if bool(getattr(dataset, "semantic_eval_mesh_classes_only", False)) and not args.oracle_gt_present_vocabulary:
        print(
            "[Semantic-2D-GTMesh] Ignoring --semantic_eval_mesh_classes_only for prediction. "
            "Use --oracle_gt_present_vocabulary only for an explicitly labeled oracle protocol."
        )

    views = _select_views(
        scene,
        args.semantic_2d_split,
        max_views=args.semantic_2d_max_views,
        stride=args.semantic_2d_stride,
    )
    if not views:
        raise RuntimeError("No cameras were selected for GT-mesh 2D semantic evaluation.")

    output_dir = Path(
        args.semantic_2d_output_dir
        or Path(dataset.model_path) / f"semantic_2d_gtmesh_eval_{scene.loaded_iter}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _cache_signature(
        gt_path,
        prompt_path,
        gt_info_path,
        dataset,
        views,
        args,
    )
    signature_hash = hashlib.sha256(
        json.dumps(signature, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = Path(args.gt_label_cache_dir or output_dir / "gt_labels")
    cache_dir = cache_root / signature_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_compatible = (
        not args.force_gt_rerender
        and _load_cache_manifest(cache_dir) == signature
    )
    (cache_dir / "manifest.json").write_text(
        json.dumps(signature, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    device = gaussians.get_anchor.device
    mesh_vertices = torch.from_numpy(vertices).to(device=device, dtype=torch.float32)
    mesh_triangles = torch.from_numpy(triangles).to(device=device, dtype=torch.long)
    mesh_triangle_labels = torch.from_numpy(triangle_labels).to(device=device, dtype=torch.long)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )

    num_classes = len(prompt_bank["class_ids"])
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    missing_by_gt = torch.zeros((num_classes,), dtype=torch.long)
    per_view_metrics = {}
    per_view_raster_stats = {}
    pred_dir = output_dir / "pred_labels"
    if args.save_pred_labels:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for view in tqdm(views, desc="GT-mesh 2D semantic evaluation", dynamic_ncols=True):
        cache_path = cache_dir / f"{view.image_name}.npy"
        gt_indices = None
        if cache_compatible and cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (int(view.image_height), int(view.image_width)):
                gt_indices = torch.from_numpy(cached.astype(np.int64, copy=False))
                per_view_raster_stats[view.image_name] = {
                    "cacheHit": True,
                    "numValidLabelPixels": int((cached >= 0).sum()),
                }
        if gt_indices is None:
            gt_indices_gpu, raster_stats = rasterize_semantic_mesh(
                mesh_vertices,
                mesh_triangles,
                mesh_triangle_labels,
                view,
                tile_size=args.gt_raster_tile_size,
                face_chunk_size=args.gt_raster_face_chunk_size,
                pair_chunk_size=args.gt_raster_pair_chunk_size,
                max_face_tile_pairs=args.gt_raster_max_face_tile_pairs,
                barycentric_epsilon=args.gt_raster_barycentric_epsilon,
            )
            gt_indices = gt_indices_gpu.cpu()
            np.save(cache_path, gt_indices.numpy().astype(np.int32, copy=False))
            raster_stats["cacheHit"] = False
            per_view_raster_stats[view.image_name] = raster_stats

        valid_gt = gt_indices >= 0
        if not bool(valid_gt.any().item()):
            tqdm.write(f"{view.image_name}: no valid GT pixels; skipped")
            continue

        decoded_feature_map, alpha = _render_semantic_feature_map_for_view(
            view,
            gaussians,
            pipeline,
            background,
        )
        pred_indices = _predict_semantic_map(
            decoded_feature_map,
            text_features,
            args.semantic_classify_chunk_size,
            class_indices=prediction_class_indices,
        )
        pred_valid = torch.ones_like(pred_indices, dtype=torch.bool)
        if float(args.pred_min_alpha) > 0:
            if alpha is None or alpha.numel() == 0:
                raise RuntimeError("--pred_min_alpha > 0 requires semantic render alpha output.")
            alpha_map = alpha[0] if alpha.ndim == 3 else alpha
            pred_valid = alpha_map.detach().cpu() > float(args.pred_min_alpha)

        view_confusion = torch.zeros_like(confusion)
        view_missing = torch.zeros_like(missing_by_gt)
        _update_confusion(
            view_confusion,
            view_missing,
            gt_indices,
            pred_indices,
            pred_valid,
            num_classes,
        )
        confusion += view_confusion
        missing_by_gt += view_missing
        per_view_metrics[view.image_name] = _compute_metrics_with_missing(
            view_confusion,
            view_missing,
            prompt_bank["class_names"],
            args.metric_class_scope,
            exclude_indices=exclude_indices,
        )
        if args.save_pred_labels:
            pred_to_save = pred_indices.numpy().astype(np.int32, copy=False)
            pred_to_save[~pred_valid.numpy()] = -1
            np.save(pred_dir / f"{view.image_name}.npy", pred_to_save)

    if not per_view_metrics:
        raise RuntimeError("No selected camera produced valid GT semantic pixels.")

    summary = _compute_metrics_with_missing(
        confusion,
        missing_by_gt,
        prompt_bank["class_names"],
        args.metric_class_scope,
        exclude_indices=exclude_indices,
    )
    summary.update(
        {
            "protocol": "rasterized_gt_mesh_2d",
            "split": args.semantic_2d_split,
            "numViews": len(per_view_metrics),
            "iteration": int(scene.loaded_iter),
            "teacherClipModel": teacher_clip_model,
            "semanticTextPromptMode": args.semantic_text_prompt_mode,
            "predictionVocabulary": (
                "gt_present_oracle" if args.oracle_gt_present_vocabulary else "full_prompt_bank"
            ),
            "metricClassScope": args.metric_class_scope,
            "predMinAlpha": float(args.pred_min_alpha),
            "excludedClasses": exclude_names,
            "gtMapping": gt_mapping_info,
            "mesh": mesh_info,
            "gtLabelCacheDir": str(cache_dir),
        }
    )

    serializable_summary = dict(summary)
    serializable_summary.pop("confusion", None)
    (output_dir / "summary.json").write_text(
        json.dumps(serializable_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "per_view_metrics.json").write_text(
        json.dumps(per_view_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "per_view_raster_stats.json").write_text(
        json.dumps(per_view_raster_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (cache_dir / "manifest.json").write_text(
        json.dumps(signature, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(output_dir / "confusion_matrix.npy", confusion.numpy())
    np.save(output_dir / "missing_predictions_by_gt_class.npy", missing_by_gt.numpy())
    _save_per_class_csv(output_dir / "per_class_metrics.csv", summary)

    print(
        f"[Semantic-2D-GTMesh] split={summary['split']} views={summary['numViews']} "
        f"pixels={summary['numLabeled']} coverage={summary['predictionCoverage']:.4f} "
        f"mIoU={summary['mIoU']:.4f} mAcc={summary['mAcc']:.4f} "
        f"overallAcc={summary['overallAcc']:.4f}"
    )
    print(f"[Semantic-2D-GTMesh] Outputs: {output_dir}")
    return summary


def main():
    parser = ArgumentParser(description="Rasterized GT-mesh 2D semantic evaluation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--semantic_2d_split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--semantic_2d_max_views", type=int, default=0)
    parser.add_argument("--semantic_2d_stride", type=int, default=1)
    parser.add_argument("--semantic_2d_output_dir", type=str, default="")
    parser.add_argument("--gt_label_cache_dir", type=str, default="")
    parser.add_argument("--force_gt_rerender", action="store_true", default=False)
    parser.add_argument("--gt_mesh_mixed_face_policy", choices=("ignore", "majority"), default="ignore")
    parser.add_argument("--gt_raster_tile_size", type=int, default=16)
    parser.add_argument("--gt_raster_face_chunk_size", type=int, default=200000)
    parser.add_argument("--gt_raster_pair_chunk_size", type=int, default=2048)
    parser.add_argument("--gt_raster_max_face_tile_pairs", type=int, default=2000000)
    parser.add_argument("--gt_raster_barycentric_epsilon", type=float, default=1e-5)
    parser.add_argument("--semantic_classify_chunk_size", type=int, default=16384)
    parser.add_argument("--pred_min_alpha", type=float, default=0.01)
    parser.add_argument("--metric_class_scope", choices=("gt_present", "all"), default="gt_present")
    parser.add_argument("--oracle_gt_present_vocabulary", action="store_true", default=False)
    parser.add_argument("--save_pred_labels", action="store_true", default=False)
    parser.add_argument(
        "--semantic_text_prompt_mode",
        choices=("single", "scaffold", "maskclustering"),
        default="maskclustering",
    )
    args = get_combined_args(parser)

    print("Rasterized GT-mesh 2D semantic evaluation for", args.model_path)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        "Using GPU visibility:",
        visible_devices if visible_devices else "<unset>",
        f"(source={CUDA_VISIBLE_DEVICES_SOURCE})",
    )
    safe_state(args.quiet)
    dataset = model.extract(args, for_inference=True)
    dataset.use_semantic_features = True
    _run_evaluation(dataset, pipeline.extract(args), args.iteration, args)
    print("Rasterized GT-mesh 2D semantic evaluation complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Rasterized GT-mesh 2D semantic evaluation failed: {exc}")
        traceback.print_exc()
        raise SystemExit(1)
