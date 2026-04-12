"""Generic descriptor interfaces for solver post-processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DescriptorContext:
    """Minimal solver output needed by descriptor calculators.

    Attributes
    ----------
    quadrature_nodes : np.ndarray
        Radial quadrature nodes from the atomic solver.
    density : np.ndarray
        Self-consistent radial electron density on the quadrature grid.
    """

    quadrature_nodes: np.ndarray
    density: np.ndarray


class DescriptorCalculator(ABC):
    """Abstract base class for descriptor post-processing calculators."""

    name: str

    @abstractmethod
    def compute(self, context: DescriptorContext) -> Any:
        """Compute descriptor results from solver context."""
