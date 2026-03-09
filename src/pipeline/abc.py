"""Abstract Base Classes"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


class Stage(Generic[A, B]):
    """One stage of pipeline

    Can be piped with others using `|` operator.
    """

    def __init__(self, func: Callable[[A], B]) -> None:
        self.func = func

    def __call__(self, data: A) -> B:
        return self.func(data)

    def __or__(self, other: Stage[B, C]) -> Stage[A, C]:
        """Implements function composition"""
        return Stage(lambda x: other.func(self.func(x)))
