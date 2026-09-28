#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
import subprocess


def _parse_requested_gpu(argv):
    for index, arg in enumerate(argv):
        if arg == "--gpu" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
    return None


def _auto_select_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        used_memory = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if used_memory:
            best_gpu = min(range(len(used_memory)), key=used_memory.__getitem__)
            return str(best_gpu)
    except Exception:
        return None
    return None


def configure_visible_devices(argv, env=None):
    if env is None:
        env = os.environ

    requested_gpu = _parse_requested_gpu(argv)
    if requested_gpu is not None and requested_gpu != "-1":
        env["CUDA_VISIBLE_DEVICES"] = requested_gpu
        return requested_gpu, "arg"

    existing = env.get("CUDA_VISIBLE_DEVICES")
    if existing:
        return existing, "env"

    auto_gpu = _auto_select_gpu()
    if auto_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = auto_gpu
        return auto_gpu, "auto"

    return env.get("CUDA_VISIBLE_DEVICES", ""), "unset"

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
