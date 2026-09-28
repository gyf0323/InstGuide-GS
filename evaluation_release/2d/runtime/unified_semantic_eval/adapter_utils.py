from __future__ import annotations

import importlib.util
import importlib.machinery
import os
import platform
import sys
import types
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .protocol import PredictionWriter, UnifiedSemanticGT, score_prediction_directory


def bootstrap_repository(repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository root does not exist: {root}")
    import_roots = [
        root,
        root / "submodules" / "diff-gaussian-rasterization",
        root / "submodules" / "simple-knn",
    ]
    for import_root in reversed(import_roots):
        if not import_root.is_dir():
            continue
        token = str(import_root)
        while token in sys.path:
            sys.path.remove(token)
        sys.path.insert(0, token)
    _bind_repository_namespace(root, "utils")
    os.chdir(root)
    return root


def _bind_repository_namespace(root: Path, module_name: str) -> None:
    """Prefer a repository namespace directory over an installed namesake."""
    package_dir = root / module_name
    if not package_dir.is_dir() or (package_dir / "__init__.py").is_file():
        return

    for loaded_name in list(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
            del sys.modules[loaded_name]

    module = types.ModuleType(module_name)
    module.__file__ = None
    module.__package__ = module_name
    module.__path__ = [str(package_dir)]
    spec = importlib.machinery.ModuleSpec(module_name, loader=None, is_package=True)
    spec.submodule_search_locations = module.__path__
    module.__spec__ = spec
    sys.modules[module_name] = module


def ensure_eval_import_shims() -> None:
    """Provide training-only optional imports without changing inference behavior."""
    if importlib.util.find_spec("torch_scatter") is None:
        module = types.ModuleType("torch_scatter")

        def scatter_max(*_args, **_kwargs):
            raise RuntimeError(
                "torch_scatter.scatter_max was called during evaluation. "
                "The lightweight evaluation shim is valid only when training/densification is disabled."
            )

        module.scatter_max = scatter_max
        sys.modules["torch_scatter"] = module

def load_repo_module(module_name: str, script_path: str | Path):
    path = Path(script_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module specification: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prompt_bank_from_gt(gt: UnifiedSemanticGT) -> dict:
    class_ids = []
    for value in gt.class_ids:
        try:
            class_ids.append(int(value))
        except ValueError:
            class_ids.append(value)
    return {
        "class_ids": class_ids,
        "class_names": list(gt.class_names),
        "prompts": list(gt.class_names),
        "path": str(gt.palette_path),
    }


def select_contract_views(scene, gt: UnifiedSemanticGT):
    test_views = list(scene.getTestCameras())
    by_name = {}
    for view in test_views:
        name = str(view.image_name)
        if name in by_name:
            raise RuntimeError(f"Duplicate test camera name: {name}")
        by_name[name] = view
    expected = set(gt.view_names)
    actual = set(by_name)
    if expected != actual:
        raise RuntimeError(
            "Model test cameras do not match unified GT: "
            f"missing={sorted(expected - actual)[:8]} extra={sorted(actual - expected)[:8]}"
        )
    ordered = [by_name[name] for name in gt.view_names]
    for view in ordered:
        size = (int(view.image_height), int(view.image_width))
        if size != (gt.height, gt.width):
            raise RuntimeError(
                f"Camera {view.image_name} has HxW={size}; unified GT expects {(gt.height, gt.width)}"
            )
    return ordered


def tensor_map_to_numpy(value) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    return value.astype(np.int32, copy=False)


def finalize_predictions(
    writer: PredictionWriter,
    gt: UnifiedSemanticGT,
    output_dir: str | Path,
    save_visualizations: bool,
    exclude_classes: Sequence[str] | None = None,
) -> dict:
    writer.finalize()
    output_dir = Path(output_dir).expanduser().resolve()
    metrics_dir = output_dir / "metrics"
    summary = score_prediction_directory(
        gt,
        writer.prediction_dir,
        metrics_dir,
        metric_class_scope="gt_present",
        exclude_classes=exclude_classes,
        save_visualizations=save_visualizations,
        require_prediction_manifest=True,
    )
    print(
        f"[UNIFIED] scene={gt.scene} views={summary['numViews']} "
        f"coverage={summary['predictionCoverage']:.4f} "
        f"mIoU={summary['mIoU']:.4f} mAcc={summary['mAcc']:.4f} "
        f"overallAcc={summary['overallAcc']:.4f}"
    )
    print(f"[UNIFIED] outputs={output_dir}")
    return summary


def writer_for_method(
    output_dir: str | Path,
    gt: UnifiedSemanticGT,
    method: str,
    metadata: Mapping[str, object],
    overwrite: bool,
    exclude_classes: Sequence[str] | None = None,
) -> PredictionWriter:
    return PredictionWriter(
        output_dir=output_dir,
        gt=gt,
        method=method,
        metadata=metadata,
        overwrite=overwrite,
        exclude_classes=exclude_classes,
    )


def runtime_metadata(torch_module=None) -> dict:
    payload = {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "ompNumThreads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
        "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    if torch_module is not None:
        payload.update(
            {
                "torchVersion": str(torch_module.__version__),
                "torchCudaBuild": str(torch_module.version.cuda),
                "cudaAvailable": bool(torch_module.cuda.is_available()),
                "gpuName": (
                    torch_module.cuda.get_device_name(0)
                    if torch_module.cuda.is_available()
                    else None
                ),
            }
        )
    return payload
