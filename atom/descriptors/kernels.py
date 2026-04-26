from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RadialKernel(Protocol):
    name: str

    def evaluate(self, q: np.ndarray) -> np.ndarray: ...

    def as_dict(self) -> dict[str, float | str]: ...


def build_radial_kernel(
    radial_basis: str | RadialKernel,
    radius: float,
    order: int = 0,
) -> RadialKernel:
    """Construct a radial kernel for one cutoff radius.

    Parameters
    ----------
    radial_basis : str or RadialKernel
        Either a kernel instance or one of the supported string specifiers.
    radius : float
        Cutoff radius used for kernels defined on a compact sphere.
    order : int
        Polynomial order for Legendre kernels.
    """
    if hasattr(radial_basis, "evaluate"):
        return radial_basis

    if not isinstance(radial_basis, str):
        raise TypeError(
            f"radial_basis must be a string or kernel-like object, got {type(radial_basis)!r}"
        )

    normalized = radial_basis.lower()
    if normalized == "heaviside":
        return HeavisideKernel(radius=radius)
    if normalized == "legendre":
        return LegendreKernel(radius=radius, order=order)

    raise ValueError(
        f"radial_basis must be 'heaviside' or 'legendre', got {radial_basis!r}"
    )


@dataclass(frozen=True)
class HeavisideKernel:
    radius: float
    name: str = "heaviside"

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return (q <= self.radius).astype(float)

    def as_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "radius": float(self.radius)}


@dataclass(frozen=True)
class GaussianKernel:
    sigma: float
    center: float = 0.0
    name: str = "gaussian"

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self.sigma <= 0.0:
            raise ValueError("Gaussian sigma must be positive.")
        arg = (q - self.center) / self.sigma
        return np.exp(-0.5 * arg * arg)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "sigma": float(self.sigma),
            "center": float(self.center),
        }


@dataclass(frozen=True)
class LegendreKernel:
    radius: float
    order: int
    name: str = "legendre"

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        from scipy.special import eval_legendre

        q = np.asarray(q, dtype=float)
        mask = q <= self.radius
        r_scaled = np.where(mask, (2.0 * q - self.radius) / self.radius, 0.0)
        return np.where(mask, eval_legendre(self.order, r_scaled), 0.0)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "radius": float(self.radius),
            "order": self.order,
        }


@dataclass(frozen=True)
class CosineCutoffKernel:
    radius: float
    name: str = "cosine_cutoff"

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self.radius <= 0.0:
            raise ValueError("Cutoff radius must be positive.")
        out = np.zeros_like(q)
        mask = q <= self.radius
        out[mask] = 0.5 * (1.0 + np.cos(np.pi * q[mask] / self.radius))
        return out

    def as_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "radius": float(self.radius)}
