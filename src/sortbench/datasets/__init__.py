"""
Datasets package public API.

Re-export the dataset generator so callers can write:
    from sortbench.datasets import make_dataset, SUPPORTED_DISTS
"""

from .generators import make_dataset, SUPPORTED_DISTS

__all__ = ["make_dataset", "SUPPORTED_DISTS"]
