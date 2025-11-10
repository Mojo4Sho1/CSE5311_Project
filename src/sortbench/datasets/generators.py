"""
Dataset generators for sorting benchmarks.

Currently implemented:
- dist == "random": integer arrays drawn uniformly from an inclusive range.

Public API (stable):
    make_dataset(n: int, spec: dict, rng: numpy.random.Generator) -> list[int]

Conventions:
- The integer range in `spec["params"]["range"]` is **inclusive** on both ends.
- Returns a Python `list[int]` (algorithms stay NumPy-agnostic).
- The caller supplies the RNG (for reproducibility across runs).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

SUPPORTED_DISTS = {"random"}
__all__ = ["SUPPORTED_DISTS", "make_dataset"]


def make_dataset(n: int, spec: Dict[str, Any], rng: np.random.Generator) -> List[int]:
    """
    Generate an integer dataset according to `spec`, using the provided RNG.

    Parameters
    ----------
    n : int
        Number of elements to generate. Must be >= 0.
    spec : dict
        Distribution specification. For the random case:
            {
                "dist": "random",
                "params": {
                    "range": [min_int, max_int]  # inclusive bounds
                }
            }
    rng : numpy.random.Generator
        Random number generator owned by the caller (seeded upstream).

    Returns
    -------
    list[int]
        A list of length `n` containing integers consistent with `spec`.

    Raises
    ------
    ValueError
        If inputs are invalid or if the distribution is unsupported.
    """
    _validate_n(n)

    if not isinstance(spec, dict):
        raise ValueError("spec must be a dict")

    dist = spec.get("dist", None)
    if dist not in SUPPORTED_DISTS:
        raise ValueError(f"Unsupported dataset dist: {dist!r}. Supported: {sorted(SUPPORTED_DISTS)}")

    if dist == "random":
        params = spec.get("params", {})
        lo, hi = _parse_inclusive_range(params)
        # Note: np.random.Generator.integers uses half-open [low, high) by default.
        # We add +1 to make the upper bound inclusive.
        if n == 0:
            return []
        arr = rng.integers(lo, hi + 1, size=n, dtype=np.int64)  # type: ignore[arg-type]
        return arr.tolist()

    # Should be unreachable because of the check above; keep explicit for clarity.
    raise ValueError(f"Unhandled dataset dist: {dist!r}")


# ------------------------- helpers ------------------------- #

def _validate_n(n: int) -> None:
    if not isinstance(n, int):
        raise ValueError("n must be an int")
    if n < 0:
        raise ValueError("n must be nonnegative")


def _parse_inclusive_range(params: Dict[str, Any]) -> Tuple[int, int]:
    """
    Validate and parse the inclusive integer range from params.

    Expected:
        params["range"] == [min_int, max_int]  (both inclusive)

    Returns
    -------
    (lo, hi) : tuple[int, int]
    """
    if "range" not in params:
        raise ValueError('random.params.range must be provided as [min, max] (inclusive)')

    rng_spec = params["range"]
    if (
        not isinstance(rng_spec, (list, tuple))
        or len(rng_spec) != 2
    ):
        raise ValueError("random.params.range must be a 2-element list/tuple [min, max]")

    lo_raw, hi_raw = rng_spec[0], rng_spec[1]
    if not _is_int_like(lo_raw) or not _is_int_like(hi_raw):
        raise ValueError("random.params.range values must be integers")

    lo, hi = int(lo_raw), int(hi_raw)
    if lo > hi:
        raise ValueError(f"random.params.range invalid: min > max ({lo} > {hi})")

    return lo, hi


def _is_int_like(x: Any) -> bool:
    # Accept Python ints and NumPy integer types
    return isinstance(x, (int, np.integer))
