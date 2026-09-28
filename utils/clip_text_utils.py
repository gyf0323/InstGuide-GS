"""Shared CLIP text-loading helpers used by batch semantic tools."""

import importlib.util
import os
import sys
from pathlib import Path
from typing import List

import torch


_CLIP_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_clip_package_dir() -> Path:
    try:
        import clip  # type: ignore

        package_dir = Path(clip.__file__).resolve().parent
        if (package_dir / "model.py").exists() and (package_dir / "simple_tokenizer.py").exists():
            return package_dir
    except Exception:
        pass

    for raw_path in sys.path:
        if not raw_path:
            continue
        candidate = Path(raw_path)
        direct_dir = candidate / "clip"
        if (direct_dir / "model.py").exists() and (direct_dir / "simple_tokenizer.py").exists():
            return direct_dir

        egg_link = candidate / "clip.egg-link"
        if egg_link.exists():
            lines = [line.strip() for line in egg_link.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                continue
            linked_dir = Path(lines[0]) / "clip"
            if (linked_dir / "model.py").exists() and (linked_dir / "simple_tokenizer.py").exists():
                return linked_dir

    raise RuntimeError("Unable to locate the CLIP package files.")


def _resolve_model_path(model_name_or_path: str) -> Path:
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return candidate

    if model_name_or_path in _CLIP_MODELS:
        filename = os.path.basename(_CLIP_MODELS[model_name_or_path])
        cached_path = Path.home() / ".cache" / "clip" / filename
        if cached_path.exists():
            return cached_path
        raise RuntimeError(
            f"CLIP model cache is missing: {cached_path}. "
            "Please download it first or pass a checkpoint path directly."
        )

    raise RuntimeError(f"Unknown CLIP model: {model_name_or_path}")


def _load_text_model(clip_pkg_dir: Path, model_path: Path, device: str):
    model_module = _load_module("_semantic_clip_model", clip_pkg_dir / "model.py")

    with open(model_path, "rb") as opened_file:
        try:
            loaded = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = loaded.state_dict()
        except RuntimeError:
            opened_file.seek(0)
            state_dict = torch.load(opened_file, map_location="cpu")

    model = model_module.build_model(state_dict).to(device)
    if device == "cpu":
        model.float()
    model.eval()
    return model


def _tokenize_queries(clip_pkg_dir: Path, prompts: List[str]) -> torch.Tensor:
    tokenizer_module = _load_module("_semantic_clip_tokenizer", clip_pkg_dir / "simple_tokenizer.py")
    tokenizer = tokenizer_module.SimpleTokenizer(str(clip_pkg_dir / "bpe_simple_vocab_16e6.txt.gz"))
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    context_length = 77
    result = torch.zeros(len(prompts), context_length, dtype=torch.long)

    for row_index, prompt in enumerate(prompts):
        tokens = [sot_token] + tokenizer.encode(prompt) + [eot_token]
        if len(tokens) > context_length:
            raise RuntimeError(f"Input '{prompt}' is too long for CLIP context length {context_length}.")
        result[row_index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    return result
