from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from semantic_3dgs_teacher.clustering.construction import mask_graph_construction
from semantic_3dgs_teacher.clustering.iterative_clustering import iterative_clustering
from semantic_3dgs_teacher.clustering.post_process import post_process
from semantic_3dgs_teacher.config import build_dataset_from_cli


def default_object_dict_path(dataset: Any, config: str) -> Path:
    return Path(dataset.object_dict_dir) / config / "object_dict.npy"


def run_instance_clustering(args: argparse.Namespace, config_dir: str = "") -> Dict[str, Any]:
    dataset, ds_args = build_dataset_from_cli(args, config_dir=config_dir)
    frame_list = dataset.get_frame_list(int(ds_args.step))
    scene_points = dataset.get_scene_points()
    print(
        "[teacher-pipeline] instance clustering input: "
        f"{len(frame_list)} frames, {len(scene_points)} scene points",
        flush=True,
    )

    with torch.no_grad():
        nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix = mask_graph_construction(
            ds_args,
            scene_points,
            frame_list,
            dataset,
        )
        object_list = iterative_clustering(
            nodes,
            observer_num_thresholds,
            float(ds_args.view_consensus_threshold),
            bool(ds_args.debug),
        )
        post_process(dataset, object_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, ds_args)

    object_path = default_object_dict_path(dataset, str(ds_args.config))
    if not object_path.is_file():
        raise FileNotFoundError(f"Instance clustering finished but object_dict.npy was not found: {object_path}")
    object_dict = np.load(object_path, allow_pickle=True).item()
    print(
        "[teacher-pipeline] instance clustering output: "
        f"{len(object_dict)} instances -> {object_path}",
        flush=True,
    )
    return {
        "object_dict_path": str(object_path),
        "num_instances": int(len(object_dict)),
        "num_frames": int(len(frame_list)),
        "scene_root": str(dataset.root),
        "config": str(ds_args.config),
        "dataset": str(ds_args.dataset),
    }
