#!/usr/bin/env python3
"""Generate validation figures for the atom-dft MCSH pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
OUTPUT_DIR = REPO_ROOT / "validation_figures" / "generated_mcsh_validation"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator


ATOMS = {
    "H": {"Z": 1, "Ne": 1},
    "He": {"Z": 2, "Ne": 2},
    "Li": {"Z": 3, "Ne": 3},
    "Be": {"Z": 4, "Ne": 4},
    "C": {"Z": 6, "Ne": 4},
    "N": {"Z": 7, "Ne": 5},
    "O": {"Z": 8, "Ne": 6},
}

SOLVER_KWARGS = dict(
    xc_functional="GGA_PBE",
    domain_size=20.0,
    finite_element_number=17,
    polynomial_order=31,
    quadrature_point_number=95,
    verbose=False,
)

RCUTS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
CONFIGS = {
    "heaviside": dict(
        angular_basis="mcsh",
        rcuts=RCUTS,
        l_max=2,
        box_size=16.0,
        spacing=0.4,
        radial_basis="heaviside",
    ),
    "legendre_0": dict(
        angular_basis="mcsh",
        rcuts=RCUTS,
        l_max=2,
        box_size=16.0,
        spacing=0.4,
        radial_basis="legendre",
        radial_order=0,
    ),
    "legendre_1": dict(
        angular_basis="mcsh",
        rcuts=RCUTS,
        l_max=2,
        box_size=16.0,
        spacing=0.4,
        radial_basis="legendre",
        radial_order=1,
    ),
    "legendre_2": dict(
        angular_basis="mcsh",
        rcuts=RCUTS,
        l_max=2,
        box_size=16.0,
        spacing=0.4,
        radial_basis="legendre",
        radial_order=2,
    ),
}

KERNEL_LABELS = {
    "heaviside": "Heaviside",
    "legendre_0": "Legendre P0",
    "legendre_1": "Legendre P1",
    "legendre_2": "Legendre P2",
}

KERNEL_COLORS = {
    "heaviside": "#1f77b4",
    "legendre_0": "#ff7f0e",
    "legendre_1": "#2ca02c",
    "legendre_2": "#d62728",
}


def center_idx(profile: dict[str, np.ndarray]) -> int:
    return int(np.argmin(profile["r"]))


def compute_atom_results() -> dict[str, dict[str, object]]:
    all_results: dict[str, dict[str, object]] = {}
    for name, info in ATOMS.items():
        print(f"Running {name} (Z={info['Z']})")
        solver = AtomicDFTSolver(atomic_number=info["Z"], **SOLVER_KWARGS)
        solver_result = solver.solve()

        r = solver_result["quadrature_nodes"]
        rho = solver_result["rho"]

        kernels: dict[str, object] = {}
        profiles: dict[str, dict[str, np.ndarray]] = {}
        for kernel_name, config in CONFIGS.items():
            calc = MultipoleCalculator(**config)
            mcsh_result = calc.compute_from_radial(r, rho)
            kernels[kernel_name] = mcsh_result
            profiles[kernel_name] = calc.extract_radial_profile(mcsh_result)

        all_results[name] = {
            "solver_result": solver_result,
            "kernels": kernels,
            "profiles": profiles,
        }
    return all_results


def build_summary(results: dict[str, dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"atoms": {}, "global": {}}
    flat_heaviside = []
    flat_legendre_0 = []

    for atom, data in results.items():
        solver_result = data["solver_result"]
        profiles = data["profiles"]
        quadrature_nodes = solver_result["quadrature_nodes"]
        quadrature_weights = solver_result["quadrature_weights"]
        rho = solver_result["rho"]
        electron_count = float(
            np.sum(4.0 * np.pi * quadrature_nodes**2 * rho * quadrature_weights)
        )

        atom_summary = {
            "energy_ha": float(solver_result["energy"]),
            "electron_count": electron_count,
            "expected_electrons": ATOMS[atom]["Ne"],
        }

        center_profile = profiles["heaviside"]
        ci = center_idx(center_profile)
        heaviside_l0 = np.abs(center_profile["descriptors"][ci, :, 0])
        atom_summary["heaviside_center_l0"] = heaviside_l0.tolist()
        atom_summary["heaviside_monotone"] = bool(
            np.all(np.diff(heaviside_l0) >= -1e-10)
        )
        atom_summary["heaviside_final_l0_fraction_of_Ne"] = float(
            heaviside_l0[-1] / ATOMS[atom]["Ne"]
        )

        for kernel_name, profile in profiles.items():
            ci = center_idx(profile)
            l0 = np.abs(profile["descriptors"][ci, :, 0])
            l1 = np.abs(profile["descriptors"][ci, :, 1])
            ratio = np.divide(
                l1,
                l0,
                out=np.zeros_like(l1),
                where=l0 > 1.0e-14,
            )
            atom_summary[f"{kernel_name}_max_center_l1_l0_ratio"] = float(np.max(ratio))

        flat_heaviside.append(np.ravel(data["kernels"]["heaviside"].descriptors))
        flat_legendre_0.append(np.ravel(data["kernels"]["legendre_0"].descriptors))
        summary["atoms"][atom] = atom_summary

    h = np.concatenate(flat_heaviside)
    l0 = np.concatenate(flat_legendre_0)
    summary["global"]["lp0_vs_heaviside_max_abs_diff"] = float(np.max(np.abs(h - l0)))
    summary["global"]["lp0_vs_heaviside_rmse"] = float(np.sqrt(np.mean((h - l0) ** 2)))
    return summary


def save_figure(fig: plt.Figure, name: str) -> None:
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_monopole_growth(results: dict[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    ordered_atoms = list(ATOMS.keys())

    for ax, atom in zip(axes, ordered_atoms):
        profile = results[atom]["profiles"]["heaviside"]
        ci = center_idx(profile)
        l0 = np.abs(profile["descriptors"][ci, :, 0])
        ne = ATOMS[atom]["Ne"]

        ax.plot(RCUTS, l0, marker="o", linewidth=2, color=KERNEL_COLORS["heaviside"])
        ax.axhline(ne, linestyle="--", linewidth=1.2, color="#444444")
        ax.set_title(atom)
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, 6.5)
        ax.text(
            0.97,
            0.10,
            f"final/Ne = {l0[-1] / ne:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    axes[-1].axis("off")
    fig.suptitle("MCSH Monopole Validation: Center |l=0| Grows with Rcut")
    fig.supxlabel("Cutoff Radius Rcut (Bohr)")
    fig.supylabel("Center |l=0| Descriptor")
    save_figure(fig, "mcsh_monopole_growth")


def plot_dipole_ratios(results: dict[str, dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ordered_atoms = list(ATOMS.keys())
    x = np.arange(len(ordered_atoms))
    width = 0.18
    kernel_names = list(CONFIGS.keys())

    for i, kernel_name in enumerate(kernel_names):
        values = []
        for atom in ordered_atoms:
            profile = results[atom]["profiles"][kernel_name]
            ci = center_idx(profile)
            l0 = np.abs(profile["descriptors"][ci, :, 0])
            l1 = np.abs(profile["descriptors"][ci, :, 1])
            ratio = np.divide(
                l1,
                l0,
                out=np.zeros_like(l1),
                where=l0 > 1.0e-14,
            )
            values.append(max(np.max(ratio), 1.0e-18))

        ax.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            color=KERNEL_COLORS[kernel_name],
            label=KERNEL_LABELS[kernel_name],
        )

    ax.axhline(
        1.0e-2, linestyle="--", color="#444444", linewidth=1.2, label="1% threshold"
    )
    ax.set_yscale("log")
    ax.set_ylim(1.0e-18, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_atoms)
    ax.set_ylabel("max_Rcut |l=1| / |l=0| at atom center")
    ax.set_title("Spherical Symmetry Check Across Kernels")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(ncol=3, fontsize=9)
    save_figure(fig, "mcsh_center_dipole_ratio")


def plot_lp0_parity(
    results: dict[str, dict[str, object]], summary: dict[str, object]
) -> None:
    h = np.concatenate(
        [np.ravel(results[atom]["kernels"]["heaviside"].descriptors) for atom in ATOMS]
    )
    l0 = np.concatenate(
        [np.ravel(results[atom]["kernels"]["legendre_0"].descriptors) for atom in ATOMS]
    )
    sample_size = min(12000, h.size)
    sample_idx = np.linspace(0, h.size - 1, sample_size, dtype=int)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(
        h[sample_idx],
        l0[sample_idx],
        s=6,
        alpha=0.35,
        color="#1f77b4",
        edgecolors="none",
    )
    bounds = [
        min(np.min(h[sample_idx]), np.min(l0[sample_idx])),
        max(np.max(h[sample_idx]), np.max(l0[sample_idx])),
    ]
    ax.plot(bounds, bounds, color="#d62728", linewidth=1.5)
    ax.set_xlabel("Heaviside descriptor value")
    ax.set_ylabel("Legendre P0 descriptor value")
    ax.set_title("Kernel Identity Validation: Legendre P0 = Heaviside")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.97,
        (
            f"max |Δ| = {summary['global']['lp0_vs_heaviside_max_abs_diff']:.2e}\n"
            f"RMSE = {summary['global']['lp0_vs_heaviside_rmse']:.2e}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    save_figure(fig, "legendre_p0_parity")


def plot_legendre_order_comparison(
    results: dict[str, dict[str, object]], atom: str = "O"
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    channel_info = [(0, "Center l=0"), (2, "Center l=2")]

    for ax, (channel, title) in zip(axes, channel_info):
        for kernel_name in CONFIGS:
            profile = results[atom]["profiles"][kernel_name]
            ci = center_idx(profile)
            values = profile["descriptors"][ci, :, channel]
            ax.plot(
                RCUTS,
                values,
                marker="o",
                linewidth=2,
                color=KERNEL_COLORS[kernel_name],
                label=KERNEL_LABELS[kernel_name],
            )
        ax.set_title(title)
        ax.set_xlabel("Cutoff Radius Rcut (Bohr)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Descriptor value")
    fig.suptitle(f"Representative Legendre Behavior for {atom}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, "legendre_order_comparison")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )

    results = compute_atom_results()
    summary = build_summary(results)

    plot_monopole_growth(results)
    plot_dipole_ratios(results)
    plot_lp0_parity(results, summary)
    plot_legendre_order_comparison(results)

    summary_path = OUTPUT_DIR / "mcsh_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote figures and summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
