# Third-party acknowledgements

InstGuide-GS builds on the following projects. Original copyright notices and
licenses remain applicable to their respective components.

| Component | Source and use | License / notice |
| --- | --- | --- |
| Scaffold-GS / Gaussian Splatting | [Scaffold-GS](https://github.com/city-super/Scaffold-GS), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting): geometry representation, training and rendering foundation | [Gaussian Splatting license](LICENSE.md); non-commercial research/evaluation terms |
| Differentiable Gaussian rasterizer | [GRAPHDECO rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization): locally extended for semantic features and auxiliary outputs | [License](submodules/diff-gaussian-rasterization/LICENSE.md); existing source notices retained |
| simple-knn | [Original project](https://gitlab.inria.fr/bkerbl/simple-knn), distributed through Scaffold-GS: CUDA nearest-neighbor initialization | [License](submodules/simple-knn/LICENSE.md), [source note](submodules/simple-knn/UPSTREAM_NOTICE.md) |
| GLM | Bundled rasterizer math library | [Copyright and license](submodules/diff-gaussian-rasterization/third_party/glm/copying.txt) |
| Alpha-CLIP | [SunzeY/AlphaCLIP](https://github.com/SunzeY/AlphaCLIP): masked region and text features | [Apache-2.0](third_party/AlphaCLIP-main/LICENSE) |
| CropFormer / Entity | [qqlu/Entity](https://github.com/qqlu/Entity): class-agnostic image masks | [License](third_party/Entity/Entityv2/CropFormer/LICENSE); additional component notices apply |
| MaskClustering | [PKU-EPIC/MaskClustering](https://github.com/PKU-EPIC/MaskClustering): adapted mask graph, backprojection, clustering and post-processing | [Source note](semantic_3dgs_teacher/clustering/UPSTREAM_NOTICE.md); no explicit upstream license found in the checked revision, redistribution permission unconfirmed |

## Adaptations

- MaskClustering modules are integrated into the teacher package with adjusted
  imports, progress reporting, output paths, and DSLR dataset/configuration
  support. The instance-teacher extensions are not an unchanged upstream release.
- The rasterizer is modified for this project and must be compiled locally.
- Alpha-CLIP packaging/dependencies and its version-helper import are adjusted;
  tokenizer data and inference source are retained.
- CropFormer includes a sequential mask-export adapter and CUDA API compatibility
  adjustments. GLM retains its headers/notices; optional documentation and tests
  are omitted, with the associated CMake target guarded.

Checkpoints and datasets are not bundled and keep their original terms.
Attribution does not grant additional rights or replace a missing license.
