"""
Validation utilities public API.

Re-exports:
    - Oracle:
        ORACLE_NAME
        oracle_sort
        equals_oracle

    - Property checks:
        is_nondecreasing
        first_nondecreasing_violation_index
        is_permutation
        permutation_counter_diff
        assert_no_mutation
"""

from .oracle import ORACLE_NAME, equals_oracle, oracle_sort
from .properties import (
    assert_no_mutation,
    first_nondecreasing_violation_index,
    is_nondecreasing,
    is_permutation,
    permutation_counter_diff,
)

__all__ = [
    "ORACLE_NAME",
    "oracle_sort",
    "equals_oracle",
    "is_nondecreasing",
    "first_nondecreasing_violation_index",
    "is_permutation",
    "permutation_counter_diff",
    "assert_no_mutation",
]
