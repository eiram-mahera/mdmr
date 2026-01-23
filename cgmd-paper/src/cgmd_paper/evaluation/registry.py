from typing import Callable, Dict, Any

_METRICS: Dict[str, Callable[..., float]] = {}

def register_metric(name: str):
    def deco(fn): _METRICS[name] = fn; return fn
    return deco

def list_metrics(): return sorted(_METRICS.keys())

def compute(name: str, **kwargs) -> float:
    if name not in _METRICS:
        raise KeyError(f"Unknown metric '{name}'. Available: {list_metrics()}")
    return _METRICS[name](**kwargs)
