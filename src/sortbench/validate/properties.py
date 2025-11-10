"""
Property helpers for validating sorting results.

These functions provide lightweight checks you can use in tests and (optionally)
inside the benchmark runner for sanity validation.

Public API (stable):
    is_nondecreasing(xs: Sequence[int]) -> bool
    first_nondecreasing_violation_index(xs: Sequence[int]) -> int | None
    is_permutation(a: Sequence[int], b: Sequence[int]) -> bool
    permutation_counter_diff(a: Sequence[int], b: Sequence[int]) -> dict[int, int]
    assert_no_mutation(before: Sequence[int], after: Sequence[int]) -> None

Notes
-----
- We intentionally keep the API integer-focused since the project benchmarks
  integer sorting. If you later expand to generic orderable types, these still
  work for any comparable elements.
- Stability is *not* checked here because you cannot infer stability from values
  alone when equal keys are indistinguishable. A stability test would require
  tagging items with tie-breaker IDs (e.g., pairs (key, id)) and checking
  relative order of equal keys.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence


__all__ = [
    "is_nondecreasing",
    "first_nondecreasing_violation_index",
    "is_permutation",
    "permutation_counter_diff",
    "assert_no_mutation",
]


def is_nondecreasing(xs: Sequence[int]) -> bool:
    """Return True iff xs[i] <= xs[i+1] for all i."""
    # Fast paths
    n = len(xs)
    if n < 2:
        return True
    # Pairwise check
    return all(xs[i] <= xs[i + 1] for i in range(n - 1))


def first_nondecreasing_violation_index(xs: Sequence[int]) -> int | None:
    """
    Return the first index i where xs[i] > xs[i+1], or None if nondecreasing.

    Useful for precise error messages:
        i = first_nondecreasing_violation_index(out)
        assert i is None, f"not nondecreasing at i={i}: {out[i]} > {out[i+1]}"
    """
    n = len(xs)
    for i in range(n - 1):
        if xs[i] > xs[i + 1]:
            return i
    return None


def is_permutation(a: Sequence[int], b: Sequence[int]) -> bool:
    """
    Return True iff `a` and `b` contain exactly the same multiset of values.
    """
    if len(a) != len(b):
        return False
    return Counter(a) == Counter(b)


def permutation_counter_diff(a: Sequence[int], b: Sequence[int]) -> Dict[int, int]:
    """
    Return a dict of value -> count difference (count_a - count_b).

    Empty dict means `a` and `b` have identical multiplicities.
    Positive values indicate extra occurrences in `a`, negative in `b`.
    """
    ca = Counter(a)
    cb = Counter(b)
    # Build full diff across the union of keys
    diff: Dict[int, int] = {}
    for k in set(ca.keys()) | set(cb.keys()):
        d = ca.get(k, 0) - cb.get(k, 0)
        if d != 0:
            diff[k] = d
    return diff


def assert_no_mutation(before: Sequence[int], after: Sequence[int]) -> None:
    """
    Assert that two sequences are exactly equal (element-wise), used to ensure
    an algorithm did not mutate its input in-place.

    Raises AssertionError with a concise message if they differ.
    """
    if len(before) != len(after):
        raise AssertionError(
            f"Input mutated: length changed from {len(before)} to {len(after)}"
        )
    # Cheap early-out: equality check
    if before == after:
        return
    # Find the first differing index for a clear error
    for i, (x, y) in enumerate(zip(before, after)):
        if x != y:
            raise AssertionError(
                f"Input mutated at index {i}: before={x}, after={y}"
            )
    # If not returned yet, lengths are equal but a trailing mismatch exists (shouldn't happen)
    raise AssertionError("Input mutated (values differ)")
