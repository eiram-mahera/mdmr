from __future__ import annotations
from typing import Dict, Type, Iterable, List, Any

_DATASET_REGISTRY: Dict[str, Type["BaseDataset"]] = {}

def register_dataset(name: str):
    def deco(cls: Type["BaseDataset"]):
        _DATASET_REGISTRY[name] = cls; return cls
    return deco

def build_dataset(name: str, *args, **kwargs) -> "BaseDataset":
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(_DATASET_REGISTRY)}")
    return _DATASET_REGISTRY[name](*args, **kwargs)

class BaseDataset:
    def load_images(self, indices: Iterable[int], video: str) -> List[Any]: raise NotImplementedError
    def load_masks(self, indices: Iterable[int], video: str) -> List[Any]: raise NotImplementedError
    def annotate(self, indices: Iterable[int]) -> None: raise NotImplementedError
    def unannotate(self, indices: Iterable[int]) -> None: raise NotImplementedError
    def uncrop_like_original(self, arr, video: str, idx: int): return arr
