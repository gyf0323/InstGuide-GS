from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.instance_geometry_semantic import (  # noqa: E402
    build_point_instance_owners,
    export_instance_geometry_binding,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export the MaskClustering point-instance membership needed by geometry-gated "
            "Anchor-Offset semantic training."
        )
    )
    parser.add_argument("--teacher_dir", required=True, help="Compressed instance-level teacher directory.")
    parser.add_argument("--object_dict", required=True, help="MaskClustering object_dict.npy.")
    parser.add_argument("--point_cloud", required=True, help="Exact PLY whose vertex indices object_dict.npy uses.")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Defaults to <teacher_dir>/geometry_binding.",
    )
    parser.add_argument(
        "--coordinate_system",
        default="geometry_3dgs_world",
        help="Human-readable coordinate-frame description saved in binding_meta.json.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    metadata = export_instance_geometry_binding(
        teacher_dir=args.teacher_dir,
        object_dict_path=args.object_dict,
        point_cloud_path=args.point_cloud,
        output_dir=args.output_dir,
        coordinate_system=args.coordinate_system,
        overwrite=bool(args.force),
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else Path(args.teacher_dir).expanduser().resolve() / "geometry_binding"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved geometry binding to: {output_dir}")


if __name__ == "__main__":
    main()
