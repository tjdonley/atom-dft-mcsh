"""MCSH descriptor computation for the atom-DFT solver.

Bridges the 1D radial atom-solver density with the 3D MCSH descriptor
computation provided by the standalone_rho_multipole package.

Usage
-----
    from atom.descriptors import MCSHCalculator, MCSHConfig

    config = MCSHConfig(rcuts=[1.0, 2.0, 3.0, 4.0])
    calc = MCSHCalculator(config)
    result = calc.compute_from_solver_result(solver.solve())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from standalone_rho_multipole.multipole import (
    MCSHResult,
    compute_descriptors_from_radial,
)


@dataclass(frozen=True)
class MCSHConfig:
    """Configuration for MCSH descriptor computation.

    Parameters
    ----------
    rcuts : sequence of float
        Cutoff radii in Bohr.
    l_max : int
        Maximum angular momentum order (0=monopole, 1=dipole, 2=quadrupole).
    box_size : float
        Side length of the cubic 3D box in Bohr for radial-to-3D projection.
    spacing : float
        Uniform grid spacing in Bohr for the 3D grid.
    radial_type : str
        Radial kernel: "heaviside" or "legendre".
    radial_order : int
        Legendre polynomial order (ignored for heaviside).
    periodic : bool
        Periodic boundary conditions on the 3D grid.
    """

    rcuts: Sequence[float]
    l_max: int = 2
    box_size: float = 20.0
    spacing: float = 0.3
    radial_type: str = "heaviside"
    radial_order: int = 0
    periodic: bool = True

    def __post_init__(self):
        # Freeze rcuts as tuple to match frozen dataclass intent
        object.__setattr__(self, "rcuts", tuple(self.rcuts))
        if not self.rcuts:
            raise ValueError("rcuts must be a non-empty sequence")
        if self.l_max < 0:
            raise ValueError(f"l_max must be non-negative, got {self.l_max}")
        if self.box_size <= 0:
            raise ValueError(f"box_size must be positive, got {self.box_size}")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")
        if self.radial_type not in ("heaviside", "legendre"):
            raise ValueError(
                f"radial_type must be 'heaviside' or 'legendre', got {self.radial_type!r}"
            )
        max_rcut = max(self.rcuts)
        if max_rcut > self.box_size / 2:
            raise ValueError(
                f"Largest rcut ({max_rcut}) exceeds half box_size ({self.box_size / 2}). "
                f"Increase box_size or reduce rcuts."
            )


class MCSHCalculator:
    """Compute MCSH descriptors from radial density.

    Bridges the atom-solver's 1D radial density with the
    standalone_rho_multipole package's 3D MCSH computation.
    """

    def __init__(self, config: MCSHConfig):
        self.config = config

    def compute_from_radial(
        self,
        r_quad: np.ndarray,
        rho: np.ndarray,
        eval_indices: Optional[np.ndarray] = None,
    ) -> MCSHResult:
        """Compute MCSH descriptors from radial density arrays.

        Parameters
        ----------
        r_quad : (N,) array
            Radial grid in Bohr.
        rho : (N,) array
            Electron density at grid points (1/Bohr^3).
        eval_indices : (M, 3) int array, optional
            3D grid indices to evaluate at. If None, evaluates along
            the x-axis through the atom center.

        Returns
        -------
        MCSHResult
        """
        c = self.config
        center = (c.box_size / 2, c.box_size / 2, c.box_size / 2)

        return compute_descriptors_from_radial(
            r_radial=r_quad,
            rho_radial=rho,
            box_size=c.box_size,
            spacing=c.spacing,
            atom_center=center,
            rcuts=c.rcuts,
            l_max=c.l_max,
            eval_indices=eval_indices,
            periodic=c.periodic,
            radial_type=c.radial_type,
            radial_order=c.radial_order,
        )

    def compute_from_solver_result(
        self,
        result: Dict,
        eval_indices: Optional[np.ndarray] = None,
    ) -> MCSHResult:
        """Compute MCSH descriptors from a solver.solve() result dict.

        Parameters
        ----------
        result : dict
            Must contain keys ``'quadrature_nodes'`` and ``'rho'``.
        eval_indices : (M, 3) int array, optional
            See :meth:`compute_from_radial`.
        """
        return self.compute_from_radial(
            r_quad=result["quadrature_nodes"],
            rho=result["rho"],
            eval_indices=eval_indices,
        )

    def extract_radial_profile(self, mcsh_result: MCSHResult) -> Dict:
        """Map 3D evaluation points back to radial distances from atom center.

        Returns
        -------
        dict
            Keys: ``'r'`` (radial distances), ``'descriptors'`` (values),
            ``'rcuts'``, ``'l_max'``.
        """
        center = np.array([self.config.box_size / 2] * 3)
        r = np.linalg.norm(mcsh_result.grid_positions - center, axis=1)
        return {
            "r": r,
            "descriptors": mcsh_result.descriptors,
            "rcuts": mcsh_result.rcuts,
            "l_max": mcsh_result.l_max,
        }
