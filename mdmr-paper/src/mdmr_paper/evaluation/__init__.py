"""
Evaluation metrics registry and built-ins.
Importing this package auto-registers bundled metrics.
"""
from .registry import register_metric, compute, list_metrics  # public API

# Import built-in metric modules so their @register_metric decorators run
from . import ctc_seg  # noqa: F401

__all__ = ["register_metric", "compute", "list_metrics"]
