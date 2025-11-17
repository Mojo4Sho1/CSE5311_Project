"""
Dataset generators for sorting benchmarks.

Currently implemented:
- dist == "random":
    Integer arrays drawn uniformly from an inclusive range.

- dist == "nearly_sorted":
    Start from strictly increasing [0, 1, ..., n-1] then perform
    ceil(swap_frac * n) random index swaps using the provided RNG.

- dist == "few_uniques":
    Choose up to k distinct integer values (uniform over an inclusive range),
    then fill the array by sampling indices in [0, k) uniformly.

- dist == "small_range":
    Like "random" but intended for small value ranges (e.g., bytes).
    By default draws uniformly from [0, 255] inclusive; can be configured.

- dist == "reversed":
    Deterministic reversed order: [n-1, n-2, ..., 0].

Public API (stable):
    make_dataset(n: int, spec: dict, rng: numpy.random.Generator) -> list[int]

Conventions:
- For "random", the integer range in params["range"] is **inclusive** on both ends.
- For "nearly_sorted":
    * No explicit value range; starts from [0..n-1] (unique, strictly increasing),
      then performs controlled swaps to degrade sortedness.
- For "few_uniques":
    * Optional inclusive "range" (defaults to [0, 4294967295]); pick at most
      min(k, n, span) distinct values from that range, then populate length-n
      array by sampling indices into that value list.
- For "small_range":
    * Intended for small integer domains (e.g., bytes). By default uses [0, 255]
      inclusive. You can override via:
        - params["max_val"] (and optional params["min_val"]), or
        - params["range"] == [min_int, max_int] (inclusive), like "random".
- For "reversed":
    * Ignores params and RNG; returns [n-1, ..., 0] deterministically.
- Returns a Python `list[int]` (algorithms stay NumPy-agnostic).
- The caller supplies the RNG (for reproducibility across runs where applicable).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

SUPPORTED_DISTS = {
    "random",
    "nearly_sorted",
    "few_uniques",
    "small_range",
    "reversed",
}
__all__ = ["SUPPORTED_DISTS", "make_dataset"]


def make_dataset(n: int, spec: Dict[str, Any], rng: np.random.Generator) -> List[int]:
    """
    Generate an integer dataset according to `spec`, using the provided RNG.

    Parameters
    ----------
    n : int
        Number of elements to generate. Must be >= 0.
    spec : dict
        Distribution specification.

        Random:
            {
                "dist": "random",
                "params": { "range": [min_int, max_int] }  # inclusive
            }

        Nearly-sorted:
            {
                "dist": "nearly_sorted",
                "params": { "swap_frac": 0.05 }            # in [0.0, 1.0]
            }

        Few-uniques:
            {
                "dist": "few_uniques",
                "params": {
                    "k": 100,                              # desired #unique values (>=1)
                    "range": [min_int, max_int]            # optional; inclusive; default [0, 4294967295]
                }
            }

        Small-range:
            {
                "dist": "small_range",
                "params": {
                    "max_val": 255,                        # optional; default 255
                    "min_val": 0                           # optional; default 0
                    # OR, alternatively:
                    # "range": [min_int, max_int]          # inclusive, like "random"
                }
            }

        Reversed:
            {
                "dist": "reversed",
                "params": {}                               # params unused
            }

    rng : numpy.random.Generator
        Random number generator owned by the caller (seeded upstream).
        Note: for deterministic distributions like "reversed", `rng` is unused.

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
        raise ValueError(
            f"Unsupported dataset dist: {dist!r}. Supported: {sorted(SUPPORTED_DISTS)}"
        )

    params = spec.get("params", {})

    if dist == "random":
        lo, hi = _parse_inclusive_range(params)
        if n == 0:
            return []
        # np.random.Generator.integers is half-open [low, high) by default.
        # Add +1 to make the upper bound inclusive.
        arr = rng.integers(lo, hi + 1, size=n, dtype=np.int64)  # type: ignore[arg-type]
        return arr.tolist()

    if dist == "nearly_sorted":
        swap_frac = _parse_swap_frac(params)
        if n == 0:
            return []
        # Start from already-sorted unique integers [0..n-1]
        arr = list(range(n))
        # Number of swaps to apply (ceil so small nonzero frac makes at least one swap)
        num_swaps = int(np.ceil(swap_frac * n))
        if num_swaps <= 0:
            return arr
        # Draw 2 * num_swaps indices in [0, n) and swap in pairs
        idxs = rng.integers(0, n, size=2 * num_swaps)
        for k in range(num_swaps):
            i = int(idxs[2 * k])
            j = int(idxs[2 * k + 1])
            if i != j:
                arr[i], arr[j] = arr[j], arr[i]
            # If i == j, the swap is a no-op; effective swaps may be fewer than requested.
        return arr

    if dist == "few_uniques":
        k = _parse_k(params)
        lo, hi = _parse_optional_inclusive_range(params, default=(0, 4294967295))
        if n == 0:
            return []
        span = hi - lo + 1
        if span <= 0:
            raise ValueError(
                f"few_uniques.params.range invalid (empty span): [{lo}, {hi}]"
            )

        # We cannot use more unique values than available in the span or positions in the array.
        actual_k = int(min(k, n, span))

        # Sample `actual_k` unique integers in [lo, hi] using the provided RNG.
        # We avoid Python's random.sample (different RNG) to keep determinism tied to `rng`.
        # Strategy: draw until we collect `actual_k` unique values; for typical few-uniques, k << span.
        chosen_vals_list: List[int] = []
        chosen_set = set()
        while len(chosen_vals_list) < actual_k:
            # Draw a small batch to reduce RNG calls in Python loop
            need = actual_k - len(chosen_vals_list)
            batch = rng.integers(lo, hi + 1, size=need * 2)  # oversample to reduce collisions
            for v in map(int, batch):
                if v not in chosen_set:
                    chosen_set.add(v)
                    chosen_vals_list.append(v)
                    if len(chosen_vals_list) == actual_k:
                        break

        # Now fill the array by sampling indices into [0, actual_k)
        idxs = rng.integers(0, actual_k, size=n)
        out = [chosen_vals_list[int(t)] for t in idxs]
        return out

    if dist == "small_range":
        lo, hi = _parse_small_range(params)
        if n == 0:
            return []
        if lo > hi:
            raise ValueError(f"small_range invalid: min > max ({lo} > {hi})")
        arr = rng.integers(lo, hi + 1, size=n, dtype=np.int64)  # type: ignore[arg-type]
        return arr.tolist()

    if dist == "reversed":
        # Deterministic reversed order [n-1, ..., 0]; `rng` is unused.
        if n == 0:
            return []
        return list(range(n - 1, -1, -1))

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
    Validate and parse the inclusive integer range from params (REQUIRED).

    Expected:
        params["range"] == [min_int, max_int]  (both inclusive)

    Returns
    -------
    (lo, hi) : tuple[int, int]
    """
    if "range" not in params:
        raise ValueError(
            "random.params.range must be provided as [min, max] (inclusive)"
        )

    rng_spec = params["range"]
    if not isinstance(rng_spec, (list, tuple)) or len(rng_spec) != 2:
        raise ValueError("random.params.range must be a 2-element list/tuple [min, max]")

    lo_raw, hi_raw = rng_spec[0], rng_spec[1]
    if not _is_int_like(lo_raw) or not _is_int_like(hi_raw):
        raise ValueError("random.params.range values must be integers")

    lo, hi = int(lo_raw), int(hi_raw)
    if lo > hi:
        raise ValueError(f"random.params.range invalid: min > max ({lo} > {hi})")

    return lo, hi


def _parse_optional_inclusive_range(
    params: Dict[str, Any], default: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Parse an optional inclusive integer range from params.
    If not present, return `default`.
    """
    if "range" not in params:
        return default
    spec = params["range"]
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValueError("params.range must be a 2-element list/tuple [min, max]")
    lo_raw, hi_raw = spec
    if not _is_int_like(lo_raw) or not _is_int_like(hi_raw):
        raise ValueError("params.range values must be integers")
    lo, hi = int(lo_raw), int(hi_raw)
    if lo > hi:
        raise ValueError(f"params.range invalid: min > max ({lo} > {hi})")
    return lo, hi


def _parse_swap_frac(params: Dict[str, Any]) -> float:
    """
    Parse and validate swap_frac in [0.0, 1.0] for nearly_sorted.
    Default to 0.05 if not provided.
    """
    val = params.get("swap_frac", 0.05)
    try:
        x = float(val)
    except Exception as e:
        raise ValueError(
            f"nearly_sorted.params.swap_frac must be a float in [0.0, 1.0]; got {val!r}"
        ) from e
    if not (0.0 <= x <= 1.0):
        raise ValueError(
            f"nearly_sorted.params.swap_frac must be in [0.0, 1.0]; got {x}"
        )
    return x


def _parse_k(params: Dict[str, Any]) -> int:
    """
    Parse and validate k (desired #unique values) for few_uniques.
    Must be an integer >= 1.
    """
    if "k" not in params:
        raise ValueError("few_uniques.params.k must be provided (int >= 1)")
    k = params["k"]
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"few_uniques.params.k must be an integer >= 1; got {k!r}")
    return k


def _parse_small_range(params: Dict[str, Any]) -> Tuple[int, int]:
    """
    Parse the inclusive integer range for the "small_range" distribution.

    Supports two equivalent forms:

    1) Explicit range (like "random"):
        params["range"] == [min_int, max_int]

    2) Named bounds (more semantic for small domains):
        params["min_val"] (optional, default 0)
        params["max_val"] (optional, default 255)

    Returns
    -------
    (lo, hi) : tuple[int, int]
        Inclusive bounds for the small-range distribution.
    """
    # If a full "range" is provided, respect it (including any custom defaults).
    if "range" in params:
        return _parse_optional_inclusive_range(params, default=(0, 255))

    # Otherwise fall back to min_val / max_val with sensible defaults.
    min_raw = params.get("min_val", 0)
    max_raw = params.get("max_val", 255)

    if not _is_int_like(min_raw) or not _is_int_like(max_raw):
        raise ValueError("small_range params.min_val/max_val must be integers")

    lo, hi = int(min_raw), int(max_raw)
    return lo, hi


def _is_int_like(x: Any) -> bool:
    # Accept Python ints and NumPy integer types
    return isinstance(x, (int, np.integer))
