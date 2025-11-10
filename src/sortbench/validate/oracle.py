"""
Oracle for sorting correctness.

We use Python's built-in `sorted()` as the ground-truth oracle:
- Correct total order for integers
- Deterministic and portable
- Stable (useful if you later test stability separately)

Public API (stable):
    oracle_sort(a: list[int]) -> list[int]
    equals_oracle(a: list[int], out: list[int]) -> bool

Conventions:
- The oracle never mutates its input and always returns a **new** list.
- All algorithms in this repo should match the oracle output exactly.
"""

from __future__ import annotations

from typing import List

ORACLE_NAME: str = "python_sorted_timsort"

__all__ = ["ORACLE_NAME", "oracle_sort", "equals_oracle"]


def oracle_sort(a: List[int]) -> List[int]:
    """
    Return the ground-truth sorted output for the given integer list.

    Parameters
    ----------
    a : list[int]
        Input sequence of integers. The oracle does not mutate `a`.

    Returns
    -------
    list[int]
        A new list with the same elements as `a`, sorted in nondecreasing order.
    """
    # Built-in `sorted` is stable and deterministic; it does not mutate `a`.
    return sorted(a)


def equals_oracle(a: List[int], out: List[int]) -> bool:
    """
    Check whether an algorithm's output matches the oracle exactly.

    Parameters
    ----------
    a : list[int]
        The original input list.
    out : list[int]
        The algorithm's output to check.

    Returns
    -------
    bool
        True iff `out` is exactly equal to `oracle_sort(a)`.
    """
    return out == oracle_sort(a)
