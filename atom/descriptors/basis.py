"""Angular basis abstractions for multipole descriptors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class AngularBasis(ABC):
    """Angular basis contract for multipole descriptor construction."""

    name: str

    @abstractmethod
    def component_specs(self, l: int) -> Sequence[tuple[str, float]]:
        """Return component labels and invariant weights for order ``l``."""

    @abstractmethod
    def evaluate_component(
        self,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        l: int,
        label: str,
    ) -> np.ndarray:
        """Evaluate one angular component on displacement arrays."""

    @abstractmethod
    def combine_invariant(
        self,
        l: int,
        components: Sequence[tuple[float, float]],
    ) -> float:
        """Reduce component projections to a rotation-invariant descriptor."""


def resolve_angular_basis(basis: str | AngularBasis) -> AngularBasis:
    """Resolve an angular basis specifier into a concrete basis object."""
    if isinstance(basis, AngularBasis):
        return basis

    if not isinstance(basis, str):
        raise TypeError(
            f"angular_basis must be a string or AngularBasis, got {type(basis)!r}"
        )

    normalized = basis.lower()
    if normalized == "mcsh":
        from .mcsh import MCSHBasis

        return MCSHBasis()

    raise ValueError(f"Unknown angular_basis: {basis!r}")
