from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


DERIVED_PROTOCOL = "unified_precomputed_gt_2d_fixed_vocab_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gt_fingerprint(split_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("manifest.json", "semantic_palette.csv", "view_names.txt"):
        digest.update(sha256_file(split_dir / name).encode("ascii"))
    return digest.hexdigest()


def read_lines(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise RuntimeError(f"Vocabulary contains duplicate class names: {path}")
    return values


def read_palette(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["class_index"]))
    indices = [int(row["class_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError(f"Palette indices are not contiguous: {path}")
    return rows


def scannetpp_mapping(
    source_palette: list[dict],
    metadata_root: Path,
) -> tuple[list[dict], np.ndarray, dict]:
    top100_path = metadata_root / "semantic_benchmark" / "top100.txt"
    mapping_path = metadata_root / "semantic_benchmark" / "map_benchmark.csv"
    semantic_classes_path = metadata_root / "semantic_classes.txt"
    for required in (top100_path, mapping_path, semantic_classes_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    top100 = read_lines(top100_path)
    if len(top100) != 100:
        raise RuntimeError(f"Expected 100 ScanNet++ benchmark classes, got {len(top100)}")
    if "object" in {value.lower() for value in top100}:
        raise RuntimeError("The official Top-100 unexpectedly contains the excluded 'object' class")

    metadata_classes = read_lines(semantic_classes_path)
    source_names = [row["class_name"].strip() for row in source_palette]
    if source_names != metadata_classes:
        raise RuntimeError(
            "Source ScanNet++ GT palette does not exactly match metadata/semantic_classes.txt"
        )

    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        mapping_rows = list(csv.DictReader(handle))
    by_raw_name = {row["class"].strip(): row for row in mapping_rows}
    if len(by_raw_name) != len(mapping_rows):
        raise RuntimeError(f"Duplicate raw labels in {mapping_path}")
    missing_rows = [name for name in source_names if name not in by_raw_name]
    if missing_rows:
        raise RuntimeError(f"Raw labels are missing from map_benchmark.csv: {missing_rows[:12]}")

    target_index = {name: index for index, name in enumerate(top100)}
    source_to_target = np.full((len(source_names),), -1, dtype=np.int32)
    mapping_counts: dict[str, int] = {}
    for source_index, source_name in enumerate(source_names):
        mapped_name = by_raw_name[source_name]["semantic_map_to"].strip()
        if not mapped_name:
            mapped_name = source_name
        elif mapped_name == "None":
            continue
        if mapped_name in target_index:
            source_to_target[source_index] = target_index[mapped_name]
            mapping_counts[mapped_name] = mapping_counts.get(mapped_name, 0) + 1

    source_color = {
        row["class_name"].strip(): (int(row["r"]), int(row["g"]), int(row["b"]))
        for row in source_palette
    }
    missing_colors = [name for name in top100 if name not in source_color]
    if missing_colors:
        raise RuntimeError(f"Top-100 classes are absent from the source palette: {missing_colors}")
    target_palette = [
        {
            "class_index": index,
            "class_id": index,
            "class_name": name,
            "r": source_color[name][0],
            "g": source_color[name][1],
            "b": source_color[name][2],
        }
        for index, name in enumerate(top100)
    ]
    metadata = {
        "name": "scannetpp_official_top100",
        "source": str(top100_path.resolve()),
        "sourceSha256": sha256_file(top100_path),
        "mappingSource": str(mapping_path.resolve()),
        "mappingSha256": sha256_file(mapping_path),
        "semanticClassesSource": str(semantic_classes_path.resolve()),
        "semanticClassesSha256": sha256_file(semantic_classes_path),
        "numSourceClasses": len(source_palette),
        "numPredictionClasses": len(target_palette),
        "numRawClassesMappedToVocabulary": int(np.count_nonzero(source_to_target >= 0)),
        "rawClassMappingsPerTarget": mapping_counts,
    }
    return target_palette, source_to_target, metadata


def replica_mapping(source_palette: list[dict], source_manifest: dict) -> tuple[list[dict], np.ndarray, dict]:
    if len(source_palette) != 101:
        raise RuntimeError(f"Expected the full Replica 101-class vocabulary, got {len(source_palette)}")
    names = [row["class_name"].strip() for row in source_palette]
    if "object" in {name.lower() for name in names}:
        raise RuntimeError("Replica 101 vocabulary unexpectedly contains the excluded 'object' class")
    target_palette = []
    for index, row in enumerate(source_palette):
        target_palette.append(
            {
                "class_index": index,
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "r": int(row["r"]),
                "g": int(row["g"]),
                "b": int(row["b"]),
            }
        )
    source_to_target = np.arange(len(source_palette), dtype=np.int32)
    mapping_source = Path(str(source_manifest["classMappingPath"])).expanduser().resolve()
    metadata = {
        "name": "replica_full_101",
        "source": str(mapping_source),
        "sourceSha256": sha256_file(mapping_source),
        "numSourceClasses": len(source_palette),
        "numPredictionClasses": len(target_palette),
        "numRawClassesMappedToVocabulary": len(source_palette),
    }
    return target_palette, source_to_target, metadata


def prepare_scene(
    source_gt_root: Path,
    output_gt_root: Path,
    metadata_root: Path,
    scene: str,
    split: str,
    overwrite: bool,
) -> dict:
    source_dir = (source_gt_root / scene / split).resolve()
    output_dir = (output_gt_root / scene / split).resolve()
    if source_gt_root.resolve() not in source_dir.parents or output_gt_root.resolve() not in output_dir.parents:
        raise ValueError("Scene/split paths must stay inside their GT roots")
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("Source and output GT directories must not overlap")
    for required_name in ("manifest.json", "semantic_palette.csv", "view_names.txt", "labels"):
        required = source_dir / required_name
        if not required.exists():
            raise FileNotFoundError(required)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_dir}")
        if output_gt_root.resolve() not in output_dir.parents:
            raise RuntimeError(f"Refusing to replace output outside output root: {output_dir}")
        shutil.rmtree(output_dir)
    labels_out = output_dir / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    source_palette = read_palette(source_dir / "semantic_palette.csv")
    dataset_type = str(source_manifest.get("datasetType", "")).lower()
    if dataset_type == "scannetpp":
        target_palette, source_to_target, vocabulary = scannetpp_mapping(
            source_palette, metadata_root
        )
    elif dataset_type == "replica":
        target_palette, source_to_target, vocabulary = replica_mapping(
            source_palette, source_manifest
        )
    else:
        raise RuntimeError(f"Unsupported dataset type for {scene}: {dataset_type!r}")

    view_names = read_lines(source_dir / "view_names.txt")
    expected_files = {f"{name}.npy" for name in view_names}
    actual_files = {path.name for path in (source_dir / "labels").glob("*.npy")}
    if expected_files != actual_files:
        raise RuntimeError(
            f"Source label/view mismatch for {scene}: "
            f"missing={sorted(expected_files - actual_files)[:8]} "
            f"extra={sorted(actual_files - expected_files)[:8]}"
        )

    total_pixels = 0
    source_void_pixels = 0
    source_valid_pixels = 0
    mapped_valid_pixels = 0
    target_counts = np.zeros((len(target_palette),), dtype=np.int64)
    for view_name in view_names:
        source = np.load(source_dir / "labels" / f"{view_name}.npy", allow_pickle=False)
        if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
            raise RuntimeError(f"Invalid source label map {scene}/{view_name}: {source.shape} {source.dtype}")
        if source.shape != (int(source_manifest["outputHeight"]), int(source_manifest["outputWidth"])):
            raise RuntimeError(f"Source label shape differs from manifest: {scene}/{view_name}")
        if np.any(source >= len(source_palette)):
            raise RuntimeError(f"Source label indices exceed the palette: {scene}/{view_name}")
        valid_source = (source >= 0) & (source < len(source_palette))
        mapped = np.full(source.shape, -1, dtype=np.int32)
        mapped[valid_source] = source_to_target[source[valid_source].astype(np.int64)]
        valid_target = mapped >= 0
        np.save(labels_out / f"{view_name}.npy", mapped, allow_pickle=False)

        total_pixels += int(source.size)
        source_void_pixels += int(np.count_nonzero(source < 0))
        source_valid_pixels += int(np.count_nonzero(valid_source))
        mapped_valid_pixels += int(np.count_nonzero(valid_target))
        if np.any(valid_target):
            target_counts += np.bincount(
                mapped[valid_target], minlength=len(target_palette)
            ).astype(np.int64, copy=False)

    with (output_dir / "semantic_palette.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("class_index", "class_id", "class_name", "r", "g", "b"),
        )
        writer.writeheader()
        writer.writerows(target_palette)
    (output_dir / "view_names.txt").write_text(
        "\n".join(view_names) + "\n", encoding="utf-8"
    )

    present_indices = np.flatnonzero(target_counts > 0).astype(int).tolist()
    present_names = [target_palette[index]["class_name"] for index in present_indices]
    derived_manifest = dict(source_manifest)
    derived_manifest.update(
        {
            "protocol": "rasterized_gt_mesh_2d_gt_only",
            "derivedProtocol": DERIVED_PROTOCOL,
            "derivedAt": datetime.now().astimezone().isoformat(),
            "derivedFrom": str(source_dir),
            "sourceGtFingerprint": gt_fingerprint(source_dir),
            "classMappingPath": vocabulary["source"],
            "numClasses": len(target_palette),
            "fixedVocabulary": {
                **vocabulary,
                "predictionClassScope": "all_fixed_vocabulary_classes",
                "metricClassScope": "gt_present_within_fixed_vocabulary",
                "requestedExcludedClasses": ["object"],
                "objectPresentInVocabulary": any(
                    row["class_name"].strip().lower() == "object"
                    for row in target_palette
                ),
                "numGtPresentClasses": len(present_indices),
                "gtPresentClassIndices": present_indices,
                "gtPresentClasses": present_names,
            },
            "mapping": {
                "totalPixels": total_pixels,
                "sourceVoidPixels": source_void_pixels,
                "sourceValidPixels": source_valid_pixels,
                "mappedValidPixels": mapped_valid_pixels,
                "ignoredOutsideVocabularyPixels": source_valid_pixels - mapped_valid_pixels,
                "mappedCoverageOfSourceValid": (
                    mapped_valid_pixels / source_valid_pixels if source_valid_pixels else 0.0
                ),
            },
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(derived_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "scene": scene,
        "dataset_type": dataset_type,
        "views": len(view_names),
        "width": int(source_manifest["outputWidth"]),
        "height": int(source_manifest["outputHeight"]),
        "prediction_vocabulary": vocabulary["name"],
        "prediction_classes": len(target_palette),
        "gt_present_classes": len(present_indices),
        "total_pixels": total_pixels,
        "source_valid_pixels": source_valid_pixels,
        "mapped_valid_pixels": mapped_valid_pixels,
        "mapped_coverage": mapped_valid_pixels / source_valid_pixels if source_valid_pixels else 0.0,
        "ignored_outside_vocabulary_pixels": source_valid_pixels - mapped_valid_pixels,
        "object_in_vocabulary": False,
        "source_gt_fingerprint": derived_manifest["sourceGtFingerprint"],
        "derived_gt_fingerprint": gt_fingerprint(output_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "fixed_vocabulary_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive fixed-vocabulary held-out 2D GT from the existing unified GT cache."
    )
    parser.add_argument("--source_gt_root", required=True)
    parser.add_argument("--output_gt_root", required=True)
    parser.add_argument("--scannetpp_metadata_root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_gt_root).expanduser().resolve()
    output_root = Path(args.output_gt_root).expanduser().resolve()
    metadata_root = Path(args.scannetpp_metadata_root).expanduser().resolve()
    rows = []
    for index, scene in enumerate(args.scenes, 1):
        print(f"[{index}/{len(args.scenes)}] preparing {scene}", flush=True)
        row = prepare_scene(
            source_root,
            output_root,
            metadata_root,
            scene,
            args.split,
            args.overwrite,
        )
        rows.append(row)
        print(
            f"  vocabulary={row['prediction_vocabulary']} "
            f"classes={row['prediction_classes']} present={row['gt_present_classes']} "
            f"mapped_coverage={row['mapped_coverage']:.4f}",
            flush=True,
        )
    write_summary(output_root / "固定词表GT汇总.csv", rows)
    print(f"summary={output_root / '固定词表GT汇总.csv'}", flush=True)


if __name__ == "__main__":
    main()
