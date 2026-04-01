"""Utilities for projecting 1D radial functions onto 3D Cartesian grids and Gaussian cube file I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .density import make_linear_interpolator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOHR_PER_ANGSTROM: float = 1.8897259886


# ---------------------------------------------------------------------------
# 3D Cartesian grid construction
# ---------------------------------------------------------------------------


def make_cartesian_grid(
    box_side_bohr: float,
    spacing_bohr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a uniform 3D Cartesian grid centered in a cubic box.

    The grid spans from ``0`` to ``box_side_bohr`` along each axis with the
    given spacing.  An atom placed at the box center sits at
    ``(box_side_bohr / 2, box_side_bohr / 2, box_side_bohr / 2)``.

    Parameters
    ----------
    box_side_bohr : float
        Side length of the cubic box in Bohr.
    spacing_bohr : float
        Grid spacing in Bohr.

    Returns
    -------
    x_1d : np.ndarray, shape (N,)
        1-D axis coordinates (same for x, y, z).
    X, Y, Z : np.ndarray, shape (N, N, N)
        3-D meshgrid arrays (``indexing='ij'``).
    """
    if box_side_bohr <= 0.0:
        raise ValueError("box_side_bohr must be positive.")
    if spacing_bohr <= 0.0:
        raise ValueError("spacing_bohr must be positive.")

    n_pts = int(round(box_side_bohr / spacing_bohr)) + 1
    x_1d = np.linspace(0.0, box_side_bohr, n_pts)
    X, Y, Z = np.meshgrid(x_1d, x_1d, x_1d, indexing="ij")
    return x_1d, X, Y, Z


def grid_radial_distances(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center: tuple[float, float, float],
) -> np.ndarray:
    """Compute the radial distance from *center* for every 3-D grid point.

    Parameters
    ----------
    X, Y, Z : np.ndarray
        Meshgrid coordinate arrays (any compatible shape).
    center : tuple of float
        ``(cx, cy, cz)`` coordinates of the atom in Bohr.

    Returns
    -------
    R : np.ndarray, same shape as *X*
        Radial distance at each grid point.
    """
    cx, cy, cz = center
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)


# ---------------------------------------------------------------------------
# Radial → 3-D projection
# ---------------------------------------------------------------------------


def project_radial_to_3d(
    r_1d: np.ndarray,
    f_1d: np.ndarray,
    R_3d: np.ndarray,
) -> np.ndarray:
    """Project a 1-D radial function *f(r)* onto a 3-D grid.

    For each point in the 3-D grid the value is obtained by linear
    interpolation of *f_1d* at the corresponding radial distance.  Points
    outside the domain of *r_1d* are set to zero.

    Parameters
    ----------
    r_1d : np.ndarray, shape (M,)
        Strictly increasing radial grid (Bohr).
    f_1d : np.ndarray, shape (M,)
        Function values on the radial grid.
    R_3d : np.ndarray, shape (Nx, Ny, Nz)
        Radial distances at every 3-D grid point (from :func:`grid_radial_distances`).

    Returns
    -------
    f_3d : np.ndarray, shape (Nx, Ny, Nz)
        Interpolated values on the 3-D grid.
    """
    interp = make_linear_interpolator(r_1d, f_1d)
    flat = interp(R_3d.ravel())
    return flat.reshape(R_3d.shape)


# ---------------------------------------------------------------------------
# Electron-count check
# ---------------------------------------------------------------------------


def electron_count_3d(rho_3d: np.ndarray, spacing_bohr: float) -> float:
    """Integrate a 3-D density on a uniform grid to get the electron count.

    Parameters
    ----------
    rho_3d : np.ndarray, shape (Nx, Ny, Nz)
        Electron density (1/Bohr^3).
    spacing_bohr : float
        Uniform grid spacing in Bohr.

    Returns
    -------
    float
        Approximate total electron count.
    """
    dV = spacing_bohr ** 3
    return float(np.sum(rho_3d) * dV)


# ---------------------------------------------------------------------------
# Gaussian cube-file I/O
# ---------------------------------------------------------------------------


def write_cube_file(
    path: str | Path,
    data_3d: np.ndarray,
    box_side_bohr: float,
    spacing_bohr: float,
    atom_positions_bohr: list[tuple[int, float, float, float, float]],
    comment1: str = "",
    comment2: str = "",
) -> None:
    """Write volumetric data in Gaussian cube format.

    The cube file places the origin at ``(0, 0, 0)`` with orthogonal voxel
    vectors of length *spacing_bohr*.  Data is written in x-outer / z-inner
    loop order (standard cube convention).

    Parameters
    ----------
    path : str or Path
        Destination file path.
    data_3d : np.ndarray, shape (Nx, Ny, Nz)
        Volumetric data to write.
    box_side_bohr : float
        Box side length in Bohr (used only for informational comments).
    spacing_bohr : float
        Grid spacing in Bohr.
    atom_positions_bohr : list of (Z, charge, x, y, z)
        Each entry is ``(atomic_number, nuclear_charge, x, y, z)`` in Bohr.
    comment1, comment2 : str
        Two header comment lines.
    """
    path = Path(path)
    nx, ny, nz = data_3d.shape
    n_atoms = len(atom_positions_bohr)

    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{comment1}\n")
        fh.write(f"{comment2}\n")

        # Number of atoms and origin
        fh.write(f"{n_atoms:5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")

        # Voxel vectors (orthogonal)
        fh.write(f"{nx:5d} {spacing_bohr:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        fh.write(f"{ny:5d} {0.0:12.6f} {spacing_bohr:12.6f} {0.0:12.6f}\n")
        fh.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {spacing_bohr:12.6f}\n")

        # Atom records
        for Z, charge, ax, ay, az in atom_positions_bohr:
            fh.write(f"{Z:5d} {charge:12.6f} {ax:12.6f} {ay:12.6f} {az:12.6f}\n")

        # Volumetric data: x-outer, y-middle, z-inner, 6 values per line
        for ix in range(nx):
            for iy in range(ny):
                vals = data_3d[ix, iy, :]
                for iz in range(nz):
                    if iz % 6 == 0 and iz > 0:
                        fh.write("\n")
                    fh.write(f" {vals[iz]:12.5E}")
                fh.write("\n")


def read_cube_file(path: str | Path) -> dict:
    """Read a Gaussian cube file and return its contents.

    Parameters
    ----------
    path : str or Path
        Path to the ``.cube`` file.

    Returns
    -------
    dict
        Keys:
        - ``comments``: list of two comment strings
        - ``n_atoms``: int
        - ``origin``: np.ndarray, shape (3,)
        - ``n_grid``: tuple (Nx, Ny, Nz)
        - ``spacing``: np.ndarray, shape (3,) — diagonal voxel sizes
        - ``atoms``: list of (Z, charge, x, y, z) tuples
        - ``data``: np.ndarray, shape (Nx, Ny, Nz)
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as fh:
        comment1 = fh.readline().rstrip("\n")
        comment2 = fh.readline().rstrip("\n")

        # Atom count and origin
        parts = fh.readline().split()
        n_atoms = int(parts[0])
        origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

        # Voxel vectors (we only store the diagonal element)
        grid_counts = []
        spacing = np.zeros(3)
        for axis in range(3):
            parts = fh.readline().split()
            grid_counts.append(int(parts[0]))
            spacing[axis] = float(parts[1 + axis])

        nx, ny, nz = grid_counts

        # Atoms
        atoms = []
        for _ in range(n_atoms):
            parts = fh.readline().split()
            atoms.append((
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            ))

        # Volumetric data — all remaining tokens
        tokens: list[str] = []
        for line in fh:
            tokens.extend(line.split())

        data = np.array([float(t) for t in tokens]).reshape((nx, ny, nz))

    return {
        "comments": [comment1, comment2],
        "n_atoms": n_atoms,
        "origin": origin,
        "n_grid": (nx, ny, nz),
        "spacing": spacing,
        "atoms": atoms,
        "data": data,
    }
