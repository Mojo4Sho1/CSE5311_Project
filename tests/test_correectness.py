"""
Correctness tests for sorting algorithms against the oracle (Python's built-in sorted).

These tests currently target the builtin reference implementation:
    sortbench.algorithms.builtin_timsort.sort

What we check:
- Output exactly matches the oracle (strongest guarantee)
- Nondecreasing order (diagnostic)
- Permutation preservation (no lost/duplicated elements)
- No input mutation (API contract)
- Determinism for a given config (same input -> same output)

Note:
- This file inserts the project `src/` onto sys.path so tests run without installing the package.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter
from typing import List

import pytest
from hypothesis import given, settings, strategies as st

# Ensure `src/` is importable when running `pytest` from the repo root
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# --- Imports from the project ---
from sortbench.validate.oracle import oracle_sort

# Try to import the builtin sorter; skip cleanly if it's not implemented yet.
try:
    from sortbench.algorithms import builtin_timsort as algo_module  # type: ignore
except Exception as e:  # pragma: no cover
    pytest.skip(f"builtin_timsort module not ready yet: {e}", allow_module_level=True)

assert hasattr(algo_module, "sort"), "Implement sort(a: list[int], *, config: dict | None) in builtin_timsort.py"
sort = algo_module.sort


# ------------------------- helpers ------------------------- #

def is_nondecreasing(xs: List[int]) -> bool:
    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))


def is_permutation(a: List[int], b: List[int]) -> bool:
    return Counter(a) == Counter(b)


def _check_one(a: List[int], *, config: dict | None = None) -> None:
    """Common assertion bundle for one input."""
    if config is None:
        config = {}

    a_before = list(a)  # ensure no mutation
    out = sort(a, config=config)

    # API: input must not be mutated
    assert a == a_before, "Algorithm must not mutate its input"

    # Primary: exact equality to oracle
    oracle_out = oracle_sort(a)
    assert out == oracle_out, "Output must exactly match the oracle"

    # Diagnostics: nondecreasing & permutation
    assert is_nondecreasing(out), "Output is not nondecreasing"
    assert is_permutation(a, out), "Output is not a permutation of input"

    # Determinism (for fixed config)
    out2 = sort(a, config=config)
    assert out2 == out, "Algorithm must be deterministic for a given config"


# ------------------------- unit tests (deterministic) ------------------------- #

@pytest.mark.parametrize(
    "a",
    [
        [],
        [5],
        [2, 1],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [7, 7, 7, 7],
        [1, 3, 2, 3, 1, 2],
        list(range(20)),
        list(range(20))[::-1],
        [0, -1, 5, -10, 3, 3, 2],  # include negatives; builtin_timsort handles these
    ],
)
def test_unit_cases_builtin_timsort(a: List[int]) -> None:
    _check_one(a)


# ------------------------- property-based tests (randomized) ------------------------- #

# Keep these quick; they test a variety of sizes/duplicates.
small_ints = st.integers(min_value=-10_000, max_value=10_000)

@settings(deadline=None, max_examples=100)
@given(st.lists(small_ints, min_size=0, max_size=400))
def test_property_random_small_range(a: List[int]) -> None:
    _check_one(a)


@settings(deadline=None, max_examples=60)
@given(st.lists(st.integers(min_value=0, max_value=2**31 - 1), min_size=0, max_size=200))
def test_property_random_full_range(a: List[int]) -> None:
    _check_one(a)


@settings(deadline=None, max_examples=60)
@given(
    st.lists(
        st.integers(min_value=0, max_value=255),  # few-uniques / small-range encourages duplicates
        min_size=0,
        max_size=600,
    )
)
def test_property_many_duplicates(a: List[int]) -> None:
    _check_one(a)
