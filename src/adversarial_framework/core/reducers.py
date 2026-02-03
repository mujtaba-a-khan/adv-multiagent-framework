"""Re-exports of reducer functions for convenience.

All reducers are defined in `state.py` alongside the types they operate on.
This module provides a single import point.
"""

from adversarial_framework.core.state import (
    append_defenses,
    append_turns,
    latest,
    merge_budget,
)

__all__ = [
    "append_defenses",
    "append_turns",
    "latest",
    "merge_budget",
]
