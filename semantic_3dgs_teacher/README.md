# Geometry-indexed instance teacher

This package implements the teacher used by InstGuide-GS:

1. CropFormer category-agnostic 2D masks;
2. multi-view mask clustering on indexed 3D geometry;
3. Alpha-CLIP ViT-B/16 region features;
4. quality-aware instance consensus and one semantic prototype per 3D instance;
5. optional scene-level MLP compression to a low-dimensional latent.

## External dependencies

The repository includes source for:

```text
third_party/Entity/Entityv2/CropFormer/
third_party/AlphaCLIP-main/
```

Place the required weights in `third_party/checkpoints/`. Teacher generation also
requires compatible installations of PyTorch, Detectron2, PyTorch3D, Open3D,
OpenCV, and the dependencies declared by CropFormer and Alpha-CLIP.

## Full pipeline

```bash
python -m semantic_3dgs_teacher.pipeline \
  --repo_root "$PWD" \
  --seq_name SCENE_ID \
  --scannetpp_dslr_root /path/to/data
```

Defaults are `--stages all`, `--config scannetpp_dslr_small`,
`--dataset scannetpp_dslr`, `--compressed_dim 12`, `--latent_dtype float16`,
`--frame_data_mode copy`, semantic batch 32, codec epochs 100, codec batch 512,
codec evaluation batch 4096 and codec learning rate `7e-4`. Explicit overrides
remain available. Server setup is in [environment.yml](../environment.yml).

The reader defaults to `mesh_aligned_0.05.ply` and divides stored depth by `1000`
to obtain meters. For another mesh filename or depth encoding, pass
`--scannetpp_dslr_mesh_name` and `--scannetpp_dslr_depth_scale`. For example,
Replica data exported as `uint16` over a 0-10 meter range needs a divisor of
`6553.5`, not `1000`; confirm the dataset's encoding before changing this value.
The selected mesh must be the exact indexed instance source, not an arbitrary
sparse initialization cloud.

Stages can be selected with `--stages masks`, `instances`, `teacher`, or
`compress`. Use `--force` only when intentionally replacing an existing stage;
otherwise existing outputs are reused subject to the stage-specific checks below.

The compressed output directory includes `--compressed_dim` (for example,
`semantic_3dgs_teacher_16d` for 16D). Mask reuse checks all registered RGB images
matched by `--image_path_pattern`, including output readability and dimensions.
Teacher reuse checks the feature/ID tables, listed frame maps and weights,
geometry sidecar, and (for compressed outputs) a strictly loadable codec.
Partial outputs stop with an explanatory error; the pipeline does not silently
overwrite them. These checks do not prove identical parameter provenance. Use a
new output or explicitly rebuild affected stages after changing teacher settings.
The `instances` stage still skips when `object_dict.npy` exists. Its cache is not
validated against source geometry, masks or clustering settings; rebuild it when
those inputs change. Exported teachers are checked for instance membership, but
that does not prove original point-cloud order or generation provenance.

## Portable compressed teacher contract

A self-contained compressed teacher contains:

```text
instance_features.npy             [num_instances, d] latent table
instance_ids.npy                  raw instance ID for each table row
semantic_codec.pth                d <-> 512D Alpha-CLIP codec
semantic_codec.json               codec metadata when available
object_dict.npy                   indexed 3D instance memberships
frame_table.json                  exported frame IDs and image shapes
frame_instance_maps/<frame>.npy   pixel -> instance index
frame_weights/<frame>.npy         per-pixel observation reliability
summary.json
teacher_config.json
```

`decoded_instance_features.npy` and `decoded_for_visualization/` are optional
inspection artifacts and are not used for training. They are disabled by default;
request `--write_decoded_features` and/or `--write_decoded_visualization_teacher`
when needed.

Use `--frame_data_mode copy` for a portable teacher. `reference` writes absolute
source paths to `summary.json`; those paths must remain valid on the training
machine. `scripts/train.sh` checks every registered training frame's
feature dimension, actual map resolution, index range, and weight values before
launching optimization, using the same referenced directories as the adapter.
The runtime adapter requires NPY maps; leave `--compressed_maps` disabled when
generating teachers for training. NPZ output remains available for other tools.

`object_dict.npy` and the exact source PLY form an indexed pair. Never replace the
PLY with a reordered or independently downsampled cloud when preparing geometry
binding.

## Visualization

```bash
python visualize_instance_teacher_semantic.py \
  --source_path /path/to/scene \
  --teacher_path /path/to/compressed_teacher \
  --output_path /path/to/new_teacher_visualization \
  --prompt_path /path/to/classes.txt \
  --all_prompt_classes \
  --clip_model "alphaclip:ViT-B/16:$PWD/third_party/checkpoints/clip_b16_grit20m_fultune_2xe.pth:$PWD/third_party/AlphaCLIP-main"
```

This tool reads the compressed teacher and its codec directly. The package's
`semantic_3dgs_teacher.visualize_teacher` entry remains available for an explicitly
exported decoded directory. Qualitative inspection is not metric evaluation.
