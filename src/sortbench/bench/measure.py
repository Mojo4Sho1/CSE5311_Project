"""
Timing harness for sorting algorithms.

We measure exactly one call to an algorithm's `sort(a, config=...)` per sample,
using a monotonic high-resolution clock. All non-essential work (copying, GC,
warmup) happens outside the timed block to keep measurements clean.

Public API (stable):
    time_sort_call(... ) -> dict

Returned dict schema:
    {
        "algo": str,
        "repeats": int,
        "samples_ns": list[int],            # elapsed ns for each successful sample
        "status": "ok" | "timeout" | "error",
        "error": str | None,                # populated if status == "error"
        "timed_out_on_repeat": int | None,  # 0-based repeat index if timeout occurred
    }
"""

from __future__ import annotations

import gc
import time
from typing import Any, Callable, Dict, List, Optional

__all__ = ["time_sort_call"]


def time_sort_call(
    *,
    algo_name: str,
    algo_fn: Callable[..., List[int]],
    a: List[int],
    config: Optional[Dict[str, Any]],
    repeats: int,
    warmup: bool,
    disable_gc: bool,
    timeout_seconds: float,
    defensive_copy: bool,
) -> Dict[str, Any]:
    """
    Time repeated calls to `algo_fn(a, config=config)`.

    Parameters
    ----------
    algo_name : str
        Logical name of the algorithm (for logs/records).
    algo_fn : Callable[..., list[int]]
        Callable implementing the signature sort(a: list[int], *, config: dict | None).
    a : list[int]
        Input array. The algorithm must not mutate this input.
    config : dict | None
        Algorithm configuration passed through unchanged.
    repeats : int
        Number of timed samples to collect (best practice: >=5).
    warmup : bool
        If True, make one untimed call before timing to prime caches/JITs.
    disable_gc : bool
        If True, collect and disable Python GC during the timed loop; restore afterward.
    timeout_seconds : float
        Per-sample timeout threshold. If a single call exceeds this threshold,
        we mark status="timeout" and stop further sampling.
    defensive_copy : bool
        If True, copy the input outside each timed call and pass the copy to the algo.

    Returns
    -------
    dict
        See module docstring for exact schema.
    """
    if repeats < 0:
        raise ValueError("repeats must be nonnegative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    result: Dict[str, Any] = {
        "algo": algo_name,
        "repeats": repeats,
        "samples_ns": [],  # type: List[int]
        "status": "ok",
        "error": None,
        "timed_out_on_repeat": None,
    }

    # ---- Warmup (outside GC disable & outside timed block) ----
    if warmup and repeats > 0:
        try:
            warm_input = list(a) if defensive_copy else a
            _ = algo_fn(warm_input, config=config)
        except Exception as e:  # pragma: no cover
            result["status"] = "error"
            result["error"] = f"warmup failed: {e!r}"
            return result

    # ---- GC control ----
    prev_gc_enabled = gc.isenabled()
    try:
        if disable_gc:
            try:
                gc.collect()
            except Exception:
                # Even if gc.collect fails for some reason, continue; we still disable.
                pass
            gc.disable()

        # ---- Timed loop ----
        threshold_ns = int(timeout_seconds * 1e9)
        for r in range(repeats):
            try:
                # Prepare input OUTSIDE the timed block
                arg = list(a) if defensive_copy else a

                t0 = time.perf_counter_ns()
                _out = algo_fn(arg, config=config)
                t1 = time.perf_counter_ns()

                elapsed = t1 - t0
                result["samples_ns"].append(int(elapsed))

                # Per-sample timeout check
                if elapsed > threshold_ns:
                    result["status"] = "timeout"
                    result["timed_out_on_repeat"] = r
                    break

            except Exception as e:  # pragma: no cover
                result["status"] = "error"
                result["error"] = f"run failed at repeat {r}: {e!r}"
                break

    finally:
        # Restore original GC state
        if disable_gc and prev_gc_enabled:
            gc.enable()
        # If GC was previously disabled, leave it disabled (respect caller's global state).

    return result
