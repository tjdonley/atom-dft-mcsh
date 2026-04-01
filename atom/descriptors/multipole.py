"""MCSH (Maxwell Cartesian Spherical Harmonic) descriptor computation.

Computes HSMP (Heaviside Step Multipole) descriptors matching the SPARC PBEq
C implementation (ssahoo41/dev_SPARC_PBEq).

References:
    Sahoo, J. (2024). PhD Thesis, Georgia Tech. Chapter 2.
    SPARC source: dev_SPARC_PBEq/src/multipole_features/MCSHHelper.c
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# MCSH component definitions: {l: [(index_label, power_spectrum_weight), ...]}
# l=0: 1 component (monopole)
# l=1: 3 components (dipole)
# l=2: 6 components (quadrupole) - note these are NOT traceless
MCSH_COMPONENTS = {
    0: [("000", 1.0)],
    1: [("100", 1.0), ("010", 1.0), ("001", 1.0)],
    2: [("200", 1.0), ("020", 1.0), ("002", 1.0),
        ("110", 2.0), ("101", 2.0), ("011", 2.0)],
}


@dataclass
class MCSHResult:
    """Result of MCSH descriptor computation.

    Attributes
    ----------
    grid_indices : (n_eval, 3) int array
        Grid indices (i, j, k) of evaluation points.
    grid_positions : (n_eval, 3) float array
        Cartesian positions (x, y, z) in Bohr.
    descriptors : (n_eval, n_rcuts, n_l) float array
        Power spectrum P_l(r, Rcut) at each evaluation point.
    rcuts : list of float
        Cutoff radii used (Bohr).
    l_max : int
        Maximum angular momentum.
    spacing : (3,) tuple
        Grid spacing (hx, hy, hz) in Bohr.
    """
    grid_indices: np.ndarray
    grid_positions: np.ndarray
    descriptors: np.ndarray
    rcuts: list[float]
    l_max: int
    spacing: tuple[float, float, float]

    def to_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            grid_indices=self.grid_indices,
            grid_positions=self.grid_positions,
            descriptors=self.descriptors,
            rcuts=np.array(self.rcuts),
            l_max=self.l_max,
            spacing=np.array(self.spacing),
        )


def mcsh_harmonic(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
                  l: int, n: str) -> np.ndarray:
    """Evaluate MCSH Cartesian harmonic S_n^l at displacement (dx, dy, dz).

    Matches SPARC C code (MCSHHelper.c) definitions exactly.
    Inputs can be arrays (broadcast).
    """
    if l == 0 and n == "000":
        return np.ones_like(dx)
    elif l == 1:
        if n == "100": return dx
        if n == "010": return dy
        if n == "001": return dz
    elif l == 2:
        if n == "200": return 3.0 * dx * dx - 1.0
        if n == "020": return 3.0 * dy * dy - 1.0
        if n == "002": return 3.0 * dz * dz - 1.0
        if n == "110": return 3.0 * dx * dy
        if n == "101": return 3.0 * dx * dz
        if n == "011": return 3.0 * dy * dz
    raise ValueError(f"Unknown MCSH component l={l}, n={n}")


def _build_stencil(rcut: float, hx: float, hy: float, hz: float,
                   radial_type: str = "heaviside", radial_order: int = 0):
    """Precompute stencil displacements and radial weights for a given rcut."""
    ri = int(np.ceil(rcut / hx))
    rj = int(np.ceil(rcut / hy))
    rk = int(np.ceil(rcut / hz))

    di = np.arange(-ri, ri + 1)
    dj = np.arange(-rj, rj + 1)
    dk = np.arange(-rk, rk + 1)
    DI, DJ, DK = np.meshgrid(di, dj, dk, indexing='ij')
    dx = DI * hx
    dy = DJ * hy
    dz = DK * hz
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    mask = r <= rcut

    if radial_type == "heaviside":
        weights = mask.astype(float)
    elif radial_type == "legendre":
        from scipy.special import eval_legendre
        r_scaled = np.where(mask, (2.0 * r - rcut) / rcut, 0.0)
        weights = np.where(mask, eval_legendre(radial_order, r_scaled), 0.0)
    else:
        raise ValueError(f"Unknown radial_type: {radial_type!r}")

    return DI, DJ, DK, dx, dy, dz, weights


def compute_mcsh_at_point(rho_3d: np.ndarray, spacing: tuple[float, float, float],
                          i0: int, j0: int, k0: int, rcut: float,
                          l_max: int = 2, periodic: bool = True,
                          radial_type: str = "heaviside",
                          radial_order: int = 0) -> dict[int, float]:
    """Compute MCSH power spectrum at a single grid point.

    Parameters
    ----------
    rho_3d : (nx, ny, nz) array
        3D electron density.
    spacing : (hx, hy, hz)
        Grid spacing in Bohr.
    i0, j0, k0 : int
        Grid indices of evaluation point.
    rcut : float
        Cutoff radius in Bohr.
    l_max : int
        Maximum angular momentum (default 2).
    periodic : bool
        Use periodic boundary conditions.

    Returns
    -------
    dict mapping l -> P_l (power spectrum value)
    """
    hx, hy, hz = spacing
    nx, ny, nz = rho_3d.shape
    dV = hx * hy * hz

    DI, DJ, DK, dx, dy, dz, weights = _build_stencil(rcut, hx, hy, hz,
                                                       radial_type, radial_order)

    if periodic:
        ii = (i0 + DI) % nx
        jj = (j0 + DJ) % ny
        kk = (k0 + DK) % nz
        boundary_weights = weights
    else:
        ii_raw = i0 + DI
        jj_raw = j0 + DJ
        kk_raw = k0 + DK
        valid = (
            (ii_raw >= 0) & (ii_raw < nx) &
            (jj_raw >= 0) & (jj_raw < ny) &
            (kk_raw >= 0) & (kk_raw < nz)
        )
        ii = np.where(valid, ii_raw, 0)
        jj = np.where(valid, jj_raw, 0)
        kk = np.where(valid, kk_raw, 0)
        boundary_weights = weights * valid

    rho_vals = rho_3d[ii, jj, kk]
    weighted_rho_dV = boundary_weights * rho_vals * dV

    result = {}
    for l in range(l_max + 1):
        if l == 0:
            # l=0 monopole: return raw (signed) zeta, matching SPARC C code.
            # The sign carries physical meaning for Legendre kernels.
            S = mcsh_harmonic(dx, dy, dz, 0, "000")
            result[0] = float(np.sum(S * weighted_rho_dV))
        else:
            # l>=1: rotationally invariant power spectrum (always non-negative)
            power = 0.0
            for n, coeff in MCSH_COMPONENTS[l]:
                S = mcsh_harmonic(dx, dy, dz, l, n)
                zeta_n = float(np.sum(S * weighted_rho_dV))
                power += coeff * zeta_n * zeta_n
            result[l] = np.sqrt(power)

    return result


def compute_descriptors(rho_3d: np.ndarray, spacing: tuple[float, float, float],
                        rcuts: Sequence[float], l_max: int = 2,
                        eval_indices: np.ndarray | None = None,
                        periodic: bool = True,
                        radial_type: str = "heaviside",
                        radial_order: int = 0) -> MCSHResult:
    """Compute MCSH descriptors at grid points.

    Parameters
    ----------
    rho_3d : (nx, ny, nz) array
        3D electron density (1/Bohr^3).
    spacing : (hx, hy, hz) tuple
        Grid spacing in Bohr.
    rcuts : sequence of float
        Cutoff radii in Bohr.
    l_max : int
        Maximum angular momentum (default 2).
    eval_indices : (n_eval, 3) int array, optional
        Grid indices (i, j, k) to evaluate at. If None, evaluates along
        the x-axis through the grid center (useful for radially symmetric
        densities).
    periodic : bool
        Use periodic boundary conditions (default True).

    Returns
    -------
    MCSHResult
    """
    hx, hy, hz = spacing
    nx, ny, nz = rho_3d.shape
    rcuts = list(rcuts)
    n_rcuts = len(rcuts)
    n_l = l_max + 1

    if eval_indices is None:
        center_j = ny // 2
        center_k = nz // 2
        eval_indices = np.column_stack([
            np.arange(nx),
            np.full(nx, center_j),
            np.full(nx, center_k),
        ])

    eval_indices = np.asarray(eval_indices, dtype=int)
    n_eval = eval_indices.shape[0]
    positions = eval_indices.astype(float) * np.array([hx, hy, hz])

    descriptors = np.zeros((n_eval, n_rcuts, n_l))

    # Precompute stencils for each rcut
    stencils = [_build_stencil(rcut, hx, hy, hz, radial_type, radial_order) for rcut in rcuts]

    dV = hx * hy * hz

    for idx in range(n_eval):
        i0, j0, k0 = int(eval_indices[idx, 0]), int(eval_indices[idx, 1]), int(eval_indices[idx, 2])

        for rc_idx, (DI, DJ, DK, dx, dy, dz, weights) in enumerate(stencils):
            if periodic:
                ii = (i0 + DI) % nx
                jj = (j0 + DJ) % ny
                kk = (k0 + DK) % nz
                local_weights = weights
            else:
                ii_raw = i0 + DI
                jj_raw = j0 + DJ
                kk_raw = k0 + DK
                valid = (
                    (ii_raw >= 0) & (ii_raw < nx) &
                    (jj_raw >= 0) & (jj_raw < ny) &
                    (kk_raw >= 0) & (kk_raw < nz)
                )
                ii = np.where(valid, ii_raw, 0)
                jj = np.where(valid, jj_raw, 0)
                kk = np.where(valid, kk_raw, 0)
                local_weights = weights * valid

            rho_vals = rho_3d[ii, jj, kk]
            weighted_rho_dV = local_weights * rho_vals * dV

            for l in range(n_l):
                if l == 0:
                    # l=0 monopole: raw signed zeta (matches SPARC C code)
                    S = mcsh_harmonic(dx, dy, dz, 0, "000")
                    descriptors[idx, rc_idx, 0] = float(np.sum(S * weighted_rho_dV))
                else:
                    # l>=1: rotationally invariant power spectrum
                    power = 0.0
                    for n, coeff in MCSH_COMPONENTS[l]:
                        S = mcsh_harmonic(dx, dy, dz, l, n)
                        zeta_n = float(np.sum(S * weighted_rho_dV))
                        power += coeff * zeta_n * zeta_n
                    descriptors[idx, rc_idx, l] = np.sqrt(power)

    return MCSHResult(
        grid_indices=eval_indices,
        grid_positions=positions,
        descriptors=descriptors,
        rcuts=rcuts,
        l_max=l_max,
        spacing=(hx, hy, hz),
    )


def compute_descriptors_from_radial(
    r_radial: np.ndarray,
    rho_radial: np.ndarray,
    box_size: float,
    spacing: float,
    atom_center: tuple[float, float, float],
    rcuts: Sequence[float],
    l_max: int = 2,
    eval_indices: np.ndarray | None = None,
    periodic: bool = True,
    radial_type: str = "heaviside",
    radial_order: int = 0,
) -> MCSHResult:
    """Project a radial density onto a 3D grid and compute MCSH descriptors.

    Convenience wrapper for the atom-solver workflow:
    radial rho(r) -> project to 3D grid -> compute MCSH.

    Parameters
    ----------
    r_radial : (M,) array
        Radial grid in Bohr (strictly increasing).
    rho_radial : (M,) array
        Radial density values (1/Bohr^3).
    box_size : float
        Cubic box side length in Bohr.
    spacing : float
        Grid spacing in Bohr (uniform in all directions).
    atom_center : (3,) tuple
        Atom position in Bohr.
    rcuts : sequence of float
        Cutoff radii in Bohr.
    l_max : int
        Maximum angular momentum (default 2).
    eval_indices : (n_eval, 3) int array, optional
        Grid indices to evaluate at.
    periodic : bool
        Use periodic boundary conditions (default True).

    Returns
    -------
    MCSHResult
    """
    from .grid3d import grid_radial_distances, make_cartesian_grid, project_radial_to_3d

    x_1d, X, Y, Z = make_cartesian_grid(box_size, spacing)
    h_actual = float(x_1d[1] - x_1d[0]) if len(x_1d) > 1 else spacing
    R_3d = grid_radial_distances(X, Y, Z, atom_center)
    rho_3d = project_radial_to_3d(r_radial, rho_radial, R_3d)

    return compute_descriptors(
        rho_3d, (h_actual, h_actual, h_actual), rcuts,
        l_max=l_max, eval_indices=eval_indices, periodic=periodic,
        radial_type=radial_type, radial_order=radial_order,
    )
