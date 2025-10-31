from __future__ import annotations
import importlib, inspect
from typing import List, Tuple, Type, Any
import numpy as np
from importlib import import_module

_SHORTCUTS = {
    "MDMR": ("mdmr_paper.selectors.mdmr_selector", "MDMRSelector"),
    "KMC": ("mdmr_paper.selectors.kmeans_clustering_selector", "KMeansClusteringSelector"),
    "DPP": ("mdmr_paper.selectors.dpp_selector", "DPPSelector"),
    "CORESET": ("mdmr_paper.selectors.core_set_selector", "CoreSetSelector"),
    "FADS": ("mdmr_paper.selectors.fads_selector", "FADSSelector"),
    "FACLOC": ("mdmr_paper.selectors.facility_location_selector", "FacilityLocationSelector"),
    "TYPICLUST": ("mdmr_paper.selectors.typiclust_selector", "TypiClustSelector"),
    "RANDOM": ("mdmr_paper.selectors.random_selector", "RandomSelector"),
    # add more here if you want short aliases
}

def _resolve(selector: str) -> Tuple[str, str]:
    sel = selector.strip()
    if ":" in sel:
        mod, cls = sel.split(":", 1)
        return mod.strip(), cls.strip()
    key = sel.upper()
    if key not in _SHORTCUTS:
        raise ValueError(
            f"Unknown selector '{selector}'. Use one of {list(_SHORTCUTS.keys())} "
            f"or 'module.path:ClassName'."
        )
    return _SHORTCUTS[key]

def load_selector_class(selector: str) -> Type:
    mod_name, cls_name = _resolve(selector)
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, cls_name):
        raise ImportError(f"Class '{cls_name}' not found in module '{mod_name}'")
    return getattr(mod, cls_name)

def instantiate_selector(
    selector: str,
    features: np.ndarray,
    **selector_kwargs,
):
    cls = load_selector_class(selector)
    sig = inspect.signature(cls)
    # (Optional) Warn on unexpected kwargs to catch typos early
    accepted = {p.name for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        extras = set(selector_kwargs) - (accepted - {"self"})
        if extras:
            raise TypeError(
                f"{cls.__name__} got unexpected keyword(s): {sorted(extras)}. "
                f"Accepted: {sorted(accepted - {'self'})} or add **kwargs in the constructor."
            )
    return cls(features, **selector_kwargs)
