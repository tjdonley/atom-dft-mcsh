"""Generate publication-quality multipole validation figures for 7 atoms (H through O).

Runs the atom-DFT solver, computes multipole descriptors with the MCSH angular
basis and Heaviside/Legendre radial bases, and produces figures demonstrating
descriptor behavior.

Usage:
    micromamba run -n base python atom-main/atom-main/generate_validation_figures.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "validation_figures"
OUT_DIR.mkdir(exist_ok=True)

# ── imports from atom package ────────────────────────────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator

# ── atom definitions ─────────────────────────────────────────────────────────
# (symbol, Z, n_valence for pseudopotential calc)
# H, He: all 1s electrons are valence (Z_val = Z)
# Li: 1s2 2s1 => Z_val = 3 (psp treats all as valence for small Z)
# Be: Z_val = 4
# C, N, O: frozen 1s2 core => Z_val = Z - 2
ATOMS = [
    ("H", 1, 1),
    ("He", 2, 2),
    ("Li", 3, 3),
    ("Be", 4, 4),
    ("C", 6, 4),
    ("N", 7, 5),
    ("O", 8, 6),
]

# colors for each atom (colorblind-friendly palette)
COLORS = {
    "H": "#1f77b4",
    "He": "#ff7f0e",
    "Li": "#2ca02c",
    "Be": "#d62728",
    "C": "#9467bd",
    "N": "#8c564b",
    "O": "#e377c2",
}

# ── solver / MCSH parameters ────────────────────────────────────────────────
SOLVER_KWARGS = dict(
    xc_functional="GGA_PBE",
    domain_size=20.0,
    finite_element_number=17,
    polynomial_order=31,
    quadrature_point_number=95,
    verbose=False,
)

RCUTS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
L_MAX = 2
BOX_SIZE = 16.0
SPACING = 0.4

# ── matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# ── helpers ──────────────────────────────────────────────────────────────────


def run_atom(symbol: str, Z: int) -> dict:
    """Run DFT solver and return result dict."""
    solver = AtomicDFTSolver(atomic_number=Z, **SOLVER_KWARGS)
    return solver.solve()


def compute_mcsh(
    result: dict, radial_basis: str = "heaviside", radial_order: int = 0
) -> dict:
    """Compute MCSH descriptors and return radial profile dict."""
    calc = MultipoleCalculator(
        angular_basis="mcsh",
        rcuts=RCUTS,
        l_max=L_MAX,
        box_size=BOX_SIZE,
        spacing=SPACING,
        radial_basis=radial_basis,
        radial_order=radial_order,
    )
    mcsh_result = calc.compute_from_solver_result(result)
    profile = calc.extract_radial_profile(mcsh_result)
    return profile


def center_index(profile: dict) -> int:
    """Index of the evaluation point closest to atom center (r ~ 0)."""
    return int(np.argmin(profile["r"]))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    # ── Step 1: run solver + compute descriptors for all atoms ───────────
    solver_results = {}
    energies = {}
    heaviside_profiles = {}
    legendre_profiles = {}  # keyed (symbol, order)

    for sym, Z, _ in ATOMS:
        print(f"[{time.time() - t0:6.1f}s] Running {sym} (Z={Z}) ...", flush=True)
        res = run_atom(sym, Z)
        solver_results[sym] = res
        energies[sym] = res["energy"]
        print(
            f"         Energy = {res['energy']:.6f} Ha, converged = {res['converged']}"
        )

        # Heaviside descriptors
        heaviside_profiles[sym] = compute_mcsh(res, "heaviside", 0)

        # Legendre descriptors (orders 0, 1, 2) -- for selected atoms only
        if sym in ("H", "He", "Be", "O"):
            for order in (0, 1, 2):
                legendre_profiles[(sym, order)] = compute_mcsh(res, "legendre", order)

        print(f"         Descriptors done.", flush=True)

    elapsed = time.time() - t0
    print(
        f"\n[{elapsed:.1f}s] All atoms computed. Generating figures ...\n", flush=True
    )

    # ── Figure 1: Radial densities ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for sym, Z, _ in ATOMS:
        r = solver_results[sym]["quadrature_nodes"]
        rho = solver_results[sym]["rho"]
        rho_r = 4.0 * np.pi * r**2 * rho
        mask = (r > 0) & (r < 8)
        label = f"{sym} (E={energies[sym]:.4f} Ha)"
        ax.semilogy(r[mask], rho_r[mask], color=COLORS[sym], lw=1.5, label=label)
    ax.set_xlim(0, 8)
    ax.set_ylim(bottom=1e-6)
    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel(r"$4\pi r^2 \rho(r)$")
    ax.set_title("Radial Electron Density: H through O")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "radial_densities.png")
    plt.close(fig)
    print("  Saved radial_densities.png")

    # ── Figure 2: Monopole (l=0) vs Rcut ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for sym, Z, Nval in ATOMS:
        prof = heaviside_profiles[sym]
        ci = center_index(prof)
        # descriptors shape: (n_eval, n_rcuts, n_l)
        l0_vals = prof["descriptors"][ci, :, 0]  # l=0 at center for each rcut
        ax.plot(RCUTS, l0_vals, "o-", color=COLORS[sym], lw=1.5, ms=5, label=sym)
        ax.axhline(Nval, color=COLORS[sym], ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Cutoff Radius (Bohr)")
    ax.set_ylabel("Monopole Descriptor (l=0)")
    ax.set_title("Monopole Descriptor (l=0) vs Cutoff Radius")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "l0_vs_rcut.png")
    plt.close(fig)
    print("  Saved l0_vs_rcut.png")

    # ── Figure 3: Dipole vanishing (l=1/l=0) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for sym, Z, _ in ATOMS:
        prof = heaviside_profiles[sym]
        ci = center_index(prof)
        l0 = prof["descriptors"][ci, :, 0]
        l1 = prof["descriptors"][ci, :, 1]
        ratio = np.where(np.abs(l0) > 1e-15, l1 / np.abs(l0), 0.0)
        ax.semilogy(
            RCUTS,
            np.abs(ratio) + 1e-16,
            "o-",
            color=COLORS[sym],
            lw=1.5,
            ms=5,
            label=sym,
        )
    ax.axhline(0.01, color="gray", ls=":", lw=1, alpha=0.6, label="1% threshold")
    ax.set_xlabel("Cutoff Radius (Bohr)")
    ax.set_ylabel("|l=1 / l=0| ratio")
    ax.set_title("Dipole Vanishing at Atom Center (l=1/l=0 ratio)")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "l1_vanishing.png")
    plt.close(fig)
    print("  Saved l1_vanishing.png")

    # ── Figure 4: Heaviside vs Legendre comparison ───────────────────────
    selected = ["H", "He", "Be", "O"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    rcut_idx = RCUTS.index(3.0)  # Rcut = 3.0 Bohr
    for ax, sym in zip(axes.flat, selected):
        # Heaviside l=0 profile
        prof_h = heaviside_profiles[sym]
        r_h = prof_h["r"]
        desc_h = prof_h["descriptors"][:, rcut_idx, 0]
        sort_h = np.argsort(r_h)
        ax.plot(r_h[sort_h], desc_h[sort_h], lw=1.8, label="Heaviside", color="#1f77b4")

        # Legendre orders 1 and 2
        for order, ls, clr in [(1, "--", "#ff7f0e"), (2, "-.", "#2ca02c")]:
            prof_lp = legendre_profiles[(sym, order)]
            r_lp = prof_lp["r"]
            desc_lp = prof_lp["descriptors"][:, rcut_idx, 0]
            sort_lp = np.argsort(r_lp)
            ax.plot(
                r_lp[sort_lp],
                desc_lp[sort_lp],
                ls=ls,
                lw=1.5,
                label=f"LP{order}",
                color=clr,
            )

        ax.set_title(f"{sym}")
        ax.set_xlabel("r (Bohr)")
        ax.set_ylabel("l=0 descriptor")
        ax.legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("Heaviside vs Legendre Radial Kernels (Rcut=3.0)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "legendre_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved legendre_comparison.png")

    # ── Figure 5: Descriptor heatmap ─────────────────────────────────────
    atom_labels = [sym for sym, _, _ in ATOMS]
    col_labels = [f"l={l}, Rc={rc}" for rc in RCUTS for l in range(L_MAX + 1)]
    n_atoms = len(ATOMS)
    n_cols = len(RCUTS) * (L_MAX + 1)
    matrix = np.zeros((n_atoms, n_cols))
    for i, (sym, _, _) in enumerate(ATOMS):
        prof = heaviside_profiles[sym]
        ci = center_index(prof)
        for j_rc, rc in enumerate(RCUTS):
            for l in range(L_MAX + 1):
                col = j_rc * (L_MAX + 1) + l
                matrix[i, col] = prof["descriptors"][ci, j_rc, l]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(n_atoms))
    ax.set_yticklabels(atom_labels)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=55, ha="right", fontsize=8)
    ax.set_title("MCSH Descriptor Values at Atom Center")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Descriptor value")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "descriptor_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved descriptor_heatmap.png")

    # ── Figure 6: LP0 == Heaviside identity ──────────────────────────────
    check_atoms = ["H", "Be", "O"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, sym in zip(axes, check_atoms):
        prof_h = heaviside_profiles[sym]
        prof_lp0 = legendre_profiles[(sym, 0)]
        # l=0 at all evaluation points, for all rcuts (flatten for scatter)
        h_vals = prof_h["descriptors"][:, :, 0].ravel()
        lp_vals = prof_lp0["descriptors"][:, :, 0].ravel()
        max_diff = np.max(np.abs(h_vals - lp_vals))
        ax.scatter(
            h_vals,
            lp_vals,
            s=6,
            alpha=0.5,
            color=COLORS[sym],
            label=f"max |diff| = {max_diff:.2e}",
        )
        vmin = min(h_vals.min(), lp_vals.min())
        vmax = max(h_vals.max(), lp_vals.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.8, label="y = x")
        ax.set_xlabel("Heaviside l=0")
        ax.set_ylabel("LP0 l=0")
        ax.set_title(f"{sym}")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("LP Order 0 = Heaviside Identity", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lp0_equals_heaviside.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved lp0_equals_heaviside.png")

    total = time.time() - t0
    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} figures in {OUT_DIR}")
    print(f"Total time: {total:.1f}s")


if __name__ == "__main__":
    main()
