"""Integration tests: atom-solver MCSH descriptors vs standalone package.

These tests run the full solver, extract the density, compute descriptors
both through the solver and directly via the standalone package, and
verify they match. Also tests physical invariants of the descriptors.

Note: these tests run actual DFT calculations and may take 30-60 seconds.
"""

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MCSHCalculator, MCSHConfig
from atom.descriptors.multipole import compute_descriptors_from_radial, MCSHResult


# Solver configuration: fast but converged enough for descriptor comparison
SOLVER_KWARGS = dict(
    atomic_number=1,
    xc_functional="LDA_PZ",
    domain_size=15.0,
    finite_element_number=10,
    polynomial_order=20,
    quadrature_point_number=43,
    verbose=False,
)

MCSH_KWARGS = dict(
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
    box_size=12.0,
    spacing=0.4,
)


@pytest.fixture(scope="module")
def solver_result_no_mcsh():
    """Run solver once without MCSH (shared across tests in this module)."""
    solver = AtomicDFTSolver(**SOLVER_KWARGS)
    return solver.solve()


@pytest.fixture(scope="module")
def solver_result_with_mcsh():
    """Run solver once with MCSH enabled."""
    config = MCSHConfig(**MCSH_KWARGS)
    calc = MCSHCalculator(config)
    solver = AtomicDFTSolver(**SOLVER_KWARGS, mcsh_calculator=calc)
    return solver.solve()


class TestSolverVsStandalone:
    """Solver descriptors must exactly match standalone package."""

    def test_solver_mcsh_matches_standalone_heaviside(
        self, solver_result_no_mcsh
    ):
        """Compute descriptors from solver density via both paths.
        They call the same underlying function, so must be bitwise equal."""
        r = solver_result_no_mcsh["quadrature_nodes"]
        rho = solver_result_no_mcsh["rho"]

        config = MCSHConfig(**MCSH_KWARGS, radial_type="heaviside")
        center = (config.box_size / 2,) * 3

        # Path A: standalone package directly
        standalone = compute_descriptors_from_radial(
            r_radial=r, rho_radial=rho,
            box_size=config.box_size, spacing=config.spacing,
            atom_center=center, rcuts=config.rcuts, l_max=config.l_max,
        )

        # Path B: via MCSHCalculator
        calc = MCSHCalculator(config)
        via_calc = calc.compute_from_solver_result(solver_result_no_mcsh)

        np.testing.assert_array_equal(standalone.descriptors, via_calc.descriptors)

    def test_solver_mcsh_matches_standalone_legendre(
        self, solver_result_no_mcsh
    ):
        """Same as above but with Legendre order 2."""
        r = solver_result_no_mcsh["quadrature_nodes"]
        rho = solver_result_no_mcsh["rho"]

        config = MCSHConfig(
            **MCSH_KWARGS, radial_type="legendre", radial_order=2,
        )
        center = (config.box_size / 2,) * 3

        standalone = compute_descriptors_from_radial(
            r_radial=r, rho_radial=rho,
            box_size=config.box_size, spacing=config.spacing,
            atom_center=center, rcuts=config.rcuts, l_max=config.l_max,
            radial_type="legendre", radial_order=2,
        )

        calc = MCSHCalculator(config)
        via_calc = calc.compute_from_solver_result(solver_result_no_mcsh)

        np.testing.assert_array_equal(standalone.descriptors, via_calc.descriptors)

    def test_inline_mcsh_matches_post_hoc(
        self, solver_result_no_mcsh, solver_result_with_mcsh,
    ):
        """Descriptors from solver's built-in mcsh_calculator must match
        computing them post-hoc from the same density."""
        config = MCSHConfig(**MCSH_KWARGS)
        calc = MCSHCalculator(config)
        post_hoc = calc.compute_from_solver_result(solver_result_no_mcsh)

        inline = solver_result_with_mcsh["mcsh_result"]

        np.testing.assert_array_equal(inline.descriptors, post_hoc.descriptors)


class TestPhysicalInvariants:
    """MCSH descriptors of hydrogen must satisfy physical properties."""

    def test_l0_is_nonzero_near_center(self, solver_result_with_mcsh):
        """l=0 (monopole) integrates density in sphere, must be nonzero and finite."""
        mcsh = solver_result_with_mcsh["mcsh_result"]
        profile = MCSHCalculator(MCSHConfig(**MCSH_KWARGS)).extract_radial_profile(mcsh)
        near_center = profile["r"] < 2.0
        if np.any(near_center):
            l0_vals = profile["descriptors"][near_center, :, 0]  # l=0 column
            assert np.all(np.isfinite(l0_vals))
            # l=0 is signed total charge in sphere -- must be nonzero near atom
            assert np.all(l0_vals != 0.0)

    def test_l1_vanishes_at_center(self, solver_result_with_mcsh):
        """For spherical density, l=1 (dipole) should vanish at the atom center."""
        mcsh = solver_result_with_mcsh["mcsh_result"]
        config = MCSHConfig(**MCSH_KWARGS)
        profile = MCSHCalculator(config).extract_radial_profile(mcsh)

        # Find point closest to center
        center_idx = np.argmin(profile["r"])
        l1_at_center = profile["descriptors"][center_idx, :, 1]  # l=1

        # Should be very small (not exactly zero due to grid discretization)
        assert np.all(np.abs(l1_at_center) < 1e-6), (
            f"l=1 at center should vanish for spherical density, got {l1_at_center}"
        )

    def test_l0_monotone_with_rcut(self, solver_result_with_mcsh):
        """l=0 magnitude should increase with rcut (more charge enclosed)."""
        mcsh = solver_result_with_mcsh["mcsh_result"]
        config = MCSHConfig(**MCSH_KWARGS)
        profile = MCSHCalculator(config).extract_radial_profile(mcsh)

        # At the center point, |l=0| should grow with rcut
        center_idx = np.argmin(profile["r"])
        l0_vs_rcut = np.abs(profile["descriptors"][center_idx, :, 0])

        for i in range(len(l0_vs_rcut) - 1):
            assert l0_vs_rcut[i + 1] >= l0_vs_rcut[i] - 1e-10, (
                f"|l=0| should increase with rcut: {l0_vs_rcut}"
            )

    def test_descriptors_are_finite(self, solver_result_with_mcsh):
        """No NaN or Inf in descriptor output."""
        mcsh = solver_result_with_mcsh["mcsh_result"]
        assert np.all(np.isfinite(mcsh.descriptors))

    def test_lp0_equals_heaviside_through_solver(self, solver_result_no_mcsh):
        """LP order 0 must exactly match Heaviside when computed through solver."""
        r = solver_result_no_mcsh["quadrature_nodes"]
        rho = solver_result_no_mcsh["rho"]

        h_config = MCSHConfig(**MCSH_KWARGS, radial_type="heaviside")
        l_config = MCSHConfig(**MCSH_KWARGS, radial_type="legendre", radial_order=0)

        h_result = MCSHCalculator(h_config).compute_from_radial(r, rho)
        l_result = MCSHCalculator(l_config).compute_from_radial(r, rho)

        np.testing.assert_allclose(
            h_result.descriptors, l_result.descriptors, atol=1e-14,
        )
