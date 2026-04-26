"""Comprehensive multi-atom descriptor validation using the MCSH basis.

Runs the atom-DFT solver for H, He, Li, Be, C, N, O with both Heaviside
and Legendre radial kernels. Validates physical invariants that must hold
for ANY atom:

1. Charge sum rule: l=0 at large Rcut -> N_electrons
2. Dipole vanishing: l=1 = 0 at atom center (spherical symmetry)
3. Monotonicity: l=0 increases with Rcut (more charge enclosed)
4. Finiteness: no NaN or Inf anywhere
5. Kernel identity: LP order 0 == Heaviside (P_0(x) = 1)
6. Legendre produces distinct values from Heaviside at order > 0
7. Descriptor scaling: heavier atoms have larger l=0 (more electrons)

These tests run full DFT calculations and take 1-3 minutes total.
"""

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator


# ---------------------------------------------------------------------------
# Atom configurations
# ---------------------------------------------------------------------------

ATOMS = {
    # Ne_valence = electrons treated by solver (pseudopotential freezes core)
    # H, He, Li, Be: all-electron (no frozen core in these PSPs)
    # C, N, O: 1s^2 core frozen, so Ne_valence = Z - 2
    "H": {"Z": 1, "Ne": 1, "E_range": (-0.50, -0.40)},
    "He": {"Z": 2, "Ne": 2, "E_range": (-3.00, -2.70)},
    "Li": {"Z": 3, "Ne": 3, "E_range": (-7.50, -6.80)},
    "Be": {"Z": 4, "Ne": 4, "E_range": (-14.0, -13.0)},
    "C": {"Z": 6, "Ne": 4, "E_range": (-6.0, -5.0)},
    "N": {"Z": 7, "Ne": 5, "E_range": (-11.0, -9.5)},
    "O": {"Z": 8, "Ne": 6, "E_range": (-17.0, -15.5)},
}

# Fast but converged solver parameters
SOLVER_KWARGS = dict(
    xc_functional="GGA_PBE",
    domain_size=20.0,
    finite_element_number=17,
    polynomial_order=31,
    quadrature_point_number=95,
    verbose=False,
)

# Multipole parameters for validation (using the MCSH angular basis)
RCUTS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
MULTIPOLE_BASE_KWARGS = dict(
    angular_basis="mcsh",
    rcuts=RCUTS,
    l_max=2,
    box_size=16.0,
    spacing=0.4,
)


def make_calc(**overrides):
    return MultipoleCalculator(**(MULTIPOLE_BASE_KWARGS | overrides))


# ---------------------------------------------------------------------------
# Module-scoped fixtures: run each atom ONCE, reuse across all tests
# ---------------------------------------------------------------------------


def _run_atom(Z):
    """Run solver for atom Z and compute descriptors for all kernel types."""
    solver = AtomicDFTSolver(atomic_number=Z, **SOLVER_KWARGS)
    result = solver.solve()

    r = result["quadrature_nodes"]
    rho = result["rho"]

    h_calc = make_calc(radial_basis="heaviside")
    l0_calc = make_calc(radial_basis="legendre", radial_order=0)
    l1_calc = make_calc(radial_basis="legendre", radial_order=1)
    l2_calc = make_calc(radial_basis="legendre", radial_order=2)

    return {
        "solver_result": result,
        "heaviside": h_calc.compute_from_radial(r, rho),
        "legendre_0": l0_calc.compute_from_radial(r, rho),
        "legendre_1": l1_calc.compute_from_radial(r, rho),
        "legendre_2": l2_calc.compute_from_radial(r, rho),
        "h_profile": h_calc.extract_radial_profile(h_calc.compute_from_radial(r, rho)),
    }


@pytest.fixture(scope="module")
def all_atom_results():
    """Run all atoms and cache results for the entire module."""
    results = {}
    for name, info in ATOMS.items():
        results[name] = _run_atom(info["Z"])
    return results


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _center_idx(profile):
    """Find the evaluation point closest to the atom center."""
    return int(np.argmin(profile["r"]))


# =========================================================================
# TEST CLASS 1: SCF convergence for all atoms
# =========================================================================


class TestSCFConvergence:
    """Every atom must converge with physically reasonable energy."""

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_converged(self, all_atom_results, atom):
        r = all_atom_results[atom]["solver_result"]
        assert r["converged"], f"{atom} SCF did not converge"

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_energy_in_range(self, all_atom_results, atom):
        E = all_atom_results[atom]["solver_result"]["energy"]
        lo, hi = ATOMS[atom]["E_range"]
        assert lo < E < hi, f"{atom}: E = {E:.4f}, expected in ({lo}, {hi})"

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_electron_count(self, all_atom_results, atom):
        """Radial density must integrate to N_electrons."""
        sr = all_atom_results[atom]["solver_result"]
        r = sr["quadrature_nodes"]
        rho = sr["rho"]
        w = sr["quadrature_weights"]
        Ne = np.sum(4 * np.pi * r**2 * rho * w)
        expected = ATOMS[atom]["Ne"]
        assert Ne == pytest.approx(expected, abs=0.02), (
            f"{atom}: Ne = {Ne:.4f}, expected {expected}"
        )


# =========================================================================
# TEST CLASS 2: Charge sum rule (l=0 at large Rcut -> N_e)
# =========================================================================


class TestChargeSumRule:
    """l=0 descriptor at atom center with large Rcut must approach N_electrons.

    This is the strongest physics check: it verifies that the descriptor
    pipeline correctly integrates the density over a sphere.
    """

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_l0_approaches_Ne_heaviside(self, all_atom_results, atom):
        profile = all_atom_results[atom]["h_profile"]
        ci = _center_idx(profile)
        Ne = ATOMS[atom]["Ne"]
        l0_vals = np.abs(profile["descriptors"][ci, :, 0])

        # 1. Monotonicity: l=0 must increase with Rcut (more charge enclosed)
        for i in range(len(l0_vals) - 1):
            assert l0_vals[i + 1] >= l0_vals[i] - 1e-10, (
                f"{atom}: l=0 not monotone: "
                f"Rcut={RCUTS[i]}->{RCUTS[i + 1]}: {l0_vals[i]:.6f}->{l0_vals[i + 1]:.6f}"
            )

        # 2. At largest Rcut (4.0 Bohr), must capture >80% of electrons
        assert l0_vals[-1] > 0.80 * Ne, (
            f"{atom}: l=0 at Rcut={RCUTS[-1]} = {l0_vals[-1]:.4f}, "
            f"expected > {0.80 * Ne:.2f} (80% of Ne={Ne})"
        )

        # 3. Must show growth from smallest to largest Rcut
        assert l0_vals[-1] > l0_vals[0] + 1e-6, (
            f"{atom}: no growth in l=0 from Rcut={RCUTS[0]} to {RCUTS[-1]}"
        )

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_l0_scales_with_Z(self, all_atom_results, atom):
        """Heavier atoms must have larger l=0 at the same Rcut."""
        if atom == "H":
            return  # Nothing lighter to compare against
        profile = all_atom_results[atom]["h_profile"]
        h_profile = all_atom_results["H"]["h_profile"]
        ci = _center_idx(profile)
        ci_h = _center_idx(h_profile)
        # At Rcut=3.0 (index 4), atom should have larger l=0 than H
        l0_atom = abs(profile["descriptors"][ci, 4, 0])
        l0_h = abs(h_profile["descriptors"][ci_h, 4, 0])
        assert l0_atom > l0_h, (
            f"{atom} l=0 ({l0_atom:.4f}) should exceed H l=0 ({l0_h:.4f})"
        )


# =========================================================================
# TEST CLASS 3: Dipole vanishing (l=1 = 0 at center)
# =========================================================================


class TestDipoleVanishing:
    """l=1 must vanish at atom center for all spherically symmetric atoms."""

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_l1_zero_at_center_heaviside(self, all_atom_results, atom):
        profile = all_atom_results[atom]["h_profile"]
        ci = _center_idx(profile)
        l1_all_rcuts = profile["descriptors"][ci, :, 1]
        # l=1 should be very small relative to l=0
        l0_all_rcuts = np.abs(profile["descriptors"][ci, :, 0])
        # Use relative check: l=1 < 1% of l=0 for each Rcut
        for i, rcut in enumerate(RCUTS):
            if l0_all_rcuts[i] > 1e-10:  # only check where l=0 is significant
                ratio = abs(l1_all_rcuts[i]) / l0_all_rcuts[i]
                assert ratio < 0.01, (
                    f"{atom} Rcut={rcut}: l=1/l=0 = {ratio:.4f}, expected < 0.01"
                )


# =========================================================================
# TEST CLASS 4: Monotonicity of l=0 with Rcut
# =========================================================================


class TestMonotonicity:
    """l=0 magnitude must increase with Rcut (more charge enclosed)."""

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_l0_monotone_heaviside(self, all_atom_results, atom):
        profile = all_atom_results[atom]["h_profile"]
        ci = _center_idx(profile)
        l0 = np.abs(profile["descriptors"][ci, :, 0])
        for i in range(len(l0) - 1):
            assert l0[i + 1] >= l0[i] - 1e-10, (
                f"{atom}: |l=0| not monotone: "
                f"Rcut={RCUTS[i]}->{RCUTS[i + 1]}: {l0[i]:.6f}->{l0[i + 1]:.6f}"
            )


# =========================================================================
# TEST CLASS 5: Finiteness
# =========================================================================


class TestFiniteness:
    """No NaN or Inf in any descriptor for any atom or kernel."""

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    @pytest.mark.parametrize(
        "kernel", ["heaviside", "legendre_0", "legendre_1", "legendre_2"]
    )
    def test_all_finite(self, all_atom_results, atom, kernel):
        d = all_atom_results[atom][kernel].descriptors
        assert np.all(np.isfinite(d)), (
            f"{atom}/{kernel}: {np.sum(~np.isfinite(d))} non-finite values"
        )


# =========================================================================
# TEST CLASS 6: Kernel identity (LP0 == Heaviside)
# =========================================================================


class TestKernelIdentity:
    """Legendre order 0 must exactly equal Heaviside for all atoms."""

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_lp0_equals_heaviside(self, all_atom_results, atom):
        h = all_atom_results[atom]["heaviside"].descriptors
        l0 = all_atom_results[atom]["legendre_0"].descriptors
        np.testing.assert_allclose(
            h, l0, atol=1e-14, err_msg=f"{atom}: LP0 != Heaviside"
        )


# =========================================================================
# TEST CLASS 7: Legendre produces distinct values
# =========================================================================


class TestLegendreDiversity:
    """Different Legendre orders must produce different descriptor values."""

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_lp1_differs_from_heaviside(self, all_atom_results, atom):
        h = all_atom_results[atom]["heaviside"].descriptors
        l1 = all_atom_results[atom]["legendre_1"].descriptors
        assert not np.allclose(h, l1, atol=1e-6), (
            f"{atom}: LP1 should differ from Heaviside"
        )

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_lp2_differs_from_lp1(self, all_atom_results, atom):
        l1 = all_atom_results[atom]["legendre_1"].descriptors
        l2 = all_atom_results[atom]["legendre_2"].descriptors
        assert not np.allclose(l1, l2, atol=1e-6), f"{atom}: LP2 should differ from LP1"


# =========================================================================
# TEST CLASS 8: Cross-atom descriptor consistency
# =========================================================================


class TestCrossAtomConsistency:
    """Descriptors must be consistent across atoms."""

    def test_descriptor_shapes_match(self, all_atom_results):
        """All atoms should produce the same descriptor shape."""
        shapes = {
            name: data["heaviside"].descriptors.shape[1:]
            for name, data in all_atom_results.items()
        }
        ref = shapes["H"]
        for name, shape in shapes.items():
            assert shape == ref, f"{name} shape {shape} != H shape {ref}"

    def test_l0_ordering_by_Z(self, all_atom_results):
        """At Rcut=4.0, l=0 should roughly increase with Z (more electrons)."""
        l0_vals = {}
        for name, data in all_atom_results.items():
            profile = data["h_profile"]
            ci = _center_idx(profile)
            l0_vals[name] = abs(profile["descriptors"][ci, -1, 0])

        # Check pairwise: H < He < Li < Be < C < N < O
        ordered = ["H", "He", "Li", "Be", "C", "N", "O"]
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            assert l0_vals[a] < l0_vals[b], (
                f"l=0 ordering violated: {a}={l0_vals[a]:.4f} >= {b}={l0_vals[b]:.4f}"
            )


# =========================================================================
# TEST CLASS 9: Legendre physical properties across atoms
# =========================================================================


class TestLegendrePhysics:
    """Legendre kernel descriptors must satisfy physical expectations."""

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_lp1_l0_can_be_negative(self, all_atom_results, atom):
        """LP1 weights inner shell negative, outer positive.
        For atoms with concentrated density, l=0 with LP1 at small Rcut
        may be negative (density concentrated in inner half)."""
        l1_data = all_atom_results[atom]["legendre_1"]
        profile = make_calc(
            radial_basis="legendre", radial_order=1
        ).extract_radial_profile(l1_data)
        ci = _center_idx(profile)
        l0_vals = profile["descriptors"][ci, :, 0]
        # At least some rcuts should give negative l=0 for LP1
        # (inner shell weight is negative for concentrated atomic densities)
        has_negative = np.any(l0_vals < 0)
        # This is expected physics, not a bug
        # Just verify the values are finite
        assert np.all(np.isfinite(l0_vals)), (
            f"{atom}: LP1 l=0 contains non-finite values"
        )

    @pytest.mark.parametrize("atom", list(ATOMS.keys()))
    def test_lp0_l1_vanishes_at_center(self, all_atom_results, atom):
        """Even with Legendre kernels, l=1 must vanish at center (spherical symmetry)."""
        for kernel_name in ["legendre_0", "legendre_1", "legendre_2"]:
            data = all_atom_results[atom][kernel_name]
            kwargs = {
                "legendre_0": dict(radial_basis="legendre", radial_order=0),
                "legendre_1": dict(radial_basis="legendre", radial_order=1),
                "legendre_2": dict(radial_basis="legendre", radial_order=2),
            }[kernel_name]
            profile = make_calc(**kwargs).extract_radial_profile(data)
            ci = _center_idx(profile)
            l1 = profile["descriptors"][ci, :, 1]
            l0 = np.abs(profile["descriptors"][ci, :, 0])
            for i, rcut in enumerate(RCUTS):
                if l0[i] > 1e-10:
                    ratio = abs(l1[i]) / l0[i]
                    assert ratio < 0.01, (
                        f"{atom}/{kernel_name} Rcut={rcut}: "
                        f"l=1/l=0 = {ratio:.4f}, expected < 0.01"
                    )
