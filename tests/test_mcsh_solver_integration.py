"""Integration tests: solver multipole descriptors vs direct low-level engine.

These tests run the full solver, extract the density, compute descriptors
both through the solver and directly via the low-level multipole engine, and
verify they match. The concrete angular basis used here is MCSH.
"""

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator
from atom.descriptors.multipole import compute_descriptors_from_radial, MultipoleResult


SOLVER_KWARGS = dict(
    atomic_number=1,
    xc_functional="LDA_PZ",
    domain_size=15.0,
    finite_element_number=10,
    polynomial_order=20,
    quadrature_point_number=43,
    verbose=False,
)

MULTIPOLE_KWARGS = dict(
    angular_basis="mcsh",
    rcuts=[1.0, 2.0, 3.0],
    l_max=2,
    box_size=12.0,
    spacing=0.4,
)


@pytest.fixture(scope="module")
def solver_result_no_multipole():
    solver = AtomicDFTSolver(**SOLVER_KWARGS)
    return solver.solve()


@pytest.fixture(scope="module")
def solver_result_with_multipole():
    calc = MultipoleCalculator(**MULTIPOLE_KWARGS)
    solver = AtomicDFTSolver(**SOLVER_KWARGS, descriptor_calculators=[calc])
    return solver.solve()


class TestSolverVsLowLevelEngine:
    """Solver descriptors must exactly match the low-level multipole engine."""

    def test_solver_multipole_matches_direct_heaviside(
        self, solver_result_no_multipole
    ):
        r = solver_result_no_multipole["quadrature_nodes"]
        rho = solver_result_no_multipole["rho"]
        center = (MULTIPOLE_KWARGS["box_size"] / 2,) * 3

        direct = compute_descriptors_from_radial(
            r_radial=r,
            rho_radial=rho,
            box_size=MULTIPOLE_KWARGS["box_size"],
            spacing=MULTIPOLE_KWARGS["spacing"],
            atom_center=center,
            rcuts=MULTIPOLE_KWARGS["rcuts"],
            angular_basis="mcsh",
            l_max=MULTIPOLE_KWARGS["l_max"],
            radial_basis="heaviside",
        )

        via_calc = MultipoleCalculator(**MULTIPOLE_KWARGS).compute_from_solver_result(
            solver_result_no_multipole
        )

        np.testing.assert_array_equal(direct.descriptors, via_calc.descriptors)

    def test_solver_multipole_matches_direct_legendre(self, solver_result_no_multipole):
        r = solver_result_no_multipole["quadrature_nodes"]
        rho = solver_result_no_multipole["rho"]
        center = (MULTIPOLE_KWARGS["box_size"] / 2,) * 3

        direct = compute_descriptors_from_radial(
            r_radial=r,
            rho_radial=rho,
            box_size=MULTIPOLE_KWARGS["box_size"],
            spacing=MULTIPOLE_KWARGS["spacing"],
            atom_center=center,
            rcuts=MULTIPOLE_KWARGS["rcuts"],
            angular_basis="mcsh",
            l_max=MULTIPOLE_KWARGS["l_max"],
            radial_basis="legendre",
            radial_order=2,
        )

        via_calc = MultipoleCalculator(
            **MULTIPOLE_KWARGS,
            radial_basis="legendre",
            radial_order=2,
        ).compute_from_solver_result(solver_result_no_multipole)

        np.testing.assert_array_equal(direct.descriptors, via_calc.descriptors)

    def test_inline_matches_post_hoc(
        self, solver_result_no_multipole, solver_result_with_multipole
    ):
        post_hoc = MultipoleCalculator(**MULTIPOLE_KWARGS).compute_from_solver_result(
            solver_result_no_multipole
        )
        inline = solver_result_with_multipole["descriptor_results"]["multipole"]
        np.testing.assert_array_equal(inline.descriptors, post_hoc.descriptors)


class TestPhysicalInvariants:
    """MCSH-based multipole descriptors of hydrogen must satisfy physical properties."""

    def test_l0_is_nonzero_near_center(self, solver_result_with_multipole):
        result = solver_result_with_multipole["descriptor_results"]["multipole"]
        profile = MultipoleCalculator(**MULTIPOLE_KWARGS).extract_radial_profile(result)
        near_center = profile["r"] < 2.0
        if np.any(near_center):
            l0_vals = profile["descriptors"][near_center, :, 0]
            assert np.all(np.isfinite(l0_vals))
            assert np.all(l0_vals != 0.0)

    def test_l1_vanishes_at_center(self, solver_result_with_multipole):
        result = solver_result_with_multipole["descriptor_results"]["multipole"]
        profile = MultipoleCalculator(**MULTIPOLE_KWARGS).extract_radial_profile(result)

        center_idx = np.argmin(profile["r"])
        l1_at_center = profile["descriptors"][center_idx, :, 1]
        assert np.all(np.abs(l1_at_center) < 1e-6), (
            f"l=1 at center should vanish for spherical density, got {l1_at_center}"
        )

    def test_l0_monotone_with_rcut(self, solver_result_with_multipole):
        result = solver_result_with_multipole["descriptor_results"]["multipole"]
        profile = MultipoleCalculator(**MULTIPOLE_KWARGS).extract_radial_profile(result)

        center_idx = np.argmin(profile["r"])
        l0_vs_rcut = np.abs(profile["descriptors"][center_idx, :, 0])
        for i in range(len(l0_vs_rcut) - 1):
            assert l0_vs_rcut[i + 1] >= l0_vs_rcut[i] - 1e-10, (
                f"|l=0| should increase with rcut: {l0_vs_rcut}"
            )

    def test_descriptors_are_finite(self, solver_result_with_multipole):
        result = solver_result_with_multipole["descriptor_results"]["multipole"]
        assert np.all(np.isfinite(result.descriptors))

    def test_lp0_equals_heaviside_through_solver(self, solver_result_no_multipole):
        r = solver_result_no_multipole["quadrature_nodes"]
        rho = solver_result_no_multipole["rho"]

        h_result = MultipoleCalculator(
            **MULTIPOLE_KWARGS,
            radial_basis="heaviside",
        ).compute_from_radial(r, rho)
        l_result = MultipoleCalculator(
            **MULTIPOLE_KWARGS,
            radial_basis="legendre",
            radial_order=0,
        ).compute_from_radial(r, rho)

        np.testing.assert_allclose(
            h_result.descriptors, l_result.descriptors, atol=1e-14
        )

    def test_result_type(self, solver_result_with_multipole):
        result = solver_result_with_multipole["descriptor_results"]["multipole"]
        assert isinstance(result, MultipoleResult)
        assert result.angular_basis == "mcsh"
