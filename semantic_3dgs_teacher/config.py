from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from semantic_3dgs_teacher.datasets.scannetpp_dslr import ScanNetPPDslrDataset


PACKAGE_ROOT = Path(__file__).resolve().parent


def _candidate_config_paths(config: str, config_dir: str = "") -> Iterable[Path]:
    value = Path(str(config)).expanduser()
    if value.suffix == ".json":
        yield value
    name = value.stem if value.suffix else str(config)
    if config_dir:
        yield Path(config_dir).expanduser() / f"{name}.json"
    yield PACKAGE_ROOT / "configs" / f"{name}.json"
    yield Path.cwd() / "configs" / f"{name}.json"


def resolve_config_path(config: str, config_dir: str = "") -> Path:
    for path in _candidate_config_paths(config, config_dir):
        if path.is_file():
            return path.resolve()
    searched = [str(path) for path in _candidate_config_paths(config, config_dir)]
    raise FileNotFoundError(f"Config not found for `{config}`. Searched: {searched}")


def load_config(config: str, config_dir: str = "") -> Dict[str, Any]:
    path = resolve_config_path(config, config_dir)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON config in {path}.")
    return payload


def update_args(args: argparse.Namespace, config_dir: str = "") -> argparse.Namespace:
    config = load_config(str(args.config), config_dir=config_dir)
    for key, value in config.items():
        setattr(args, key, value)
    return args


def dataset_args_from_cli(args: argparse.Namespace, config_dir: str = "") -> argparse.Namespace:
    ds_args = argparse.Namespace(
        seq_name=args.seq_name,
        seq_name_list=getattr(args, "seq_name_list", args.seq_name),
        config=args.config,
        debug=bool(getattr(args, "debug", False)),
    )
    ds_args = update_args(ds_args, config_dir=config_dir)
    if getattr(args, "dataset", ""):
        setattr(ds_args, "dataset", args.dataset)
    if getattr(args, "scannetpp_dslr_root", ""):
        setattr(ds_args, "scannetpp_dslr_root", args.scannetpp_dslr_root)
    if getattr(args, "scannetpp_dslr_mesh_name", ""):
        setattr(ds_args, "scannetpp_dslr_mesh_name", args.scannetpp_dslr_mesh_name)
    if getattr(args, "scannetpp_dslr_depth_scale", None) is not None:
        setattr(ds_args, "scannetpp_dslr_depth_scale", args.scannetpp_dslr_depth_scale)
    return ds_args


def get_dataset(args: argparse.Namespace) -> Any:
    dataset_name = str(getattr(args, "dataset", ""))
    if dataset_name == "scannetpp_dslr":
        return ScanNetPPDslrDataset(args)
    raise NotImplementedError(
        f"semantic_3dgs_teacher currently ships a self-contained reader for scannetpp_dslr only, got `{dataset_name}`."
    )


def build_dataset_from_cli(args: argparse.Namespace, config_dir: str = "") -> Tuple[Any, argparse.Namespace]:
    ds_args = dataset_args_from_cli(args, config_dir=config_dir)
    return get_dataset(ds_args), ds_args
