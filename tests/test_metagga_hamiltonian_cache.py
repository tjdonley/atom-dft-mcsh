"""Regression tests for meta-GGA Hamiltonian angular-term assembly."""

import numpy as np

from atom import AtomicDFTSolver
from atom.scf.driver import SwitchesFlags


def _build_scan_hamiltonian_fixture():
    solver = AtomicDFTSolver(
        atomic_number=6,
        all_electron_flag=True,
        xc_functional="SCAN",
        domain_size=6.0,
        finite_element_number=3,
        polynomial_order=3,
        quadrature_point_number=9,
        max_scf_iterations=1,
        use_pulay_mixing=False,
        verbose=False,
    )
    builder = solver.hamiltonian_builder
    n_quad = len(builder.ops_builder.quadrature_nodes)
    zero_potential = np.zeros(n_quad)
    de_xc_dtau = np.full(n_quad, 0.125)
    switches = SwitchesFlags("SCAN")
    return builder, zero_potential, de_xc_dtau, switches


def test_metagga_hamiltonian_build_does_not_mutate_cached_angular_operator():
    builder, zero_potential, de_xc_dtau, switches = _build_scan_hamiltonian_fixture()
    cached_angular_operator = builder.H_r_inv_sq.copy()

    builder.build_for_l_channel(
        l=1,
        v_hartree=zero_potential,
        v_x=zero_potential,
        v_c=zero_potential,
        switches=switches,
        de_xc_dtau=de_xc_dtau,
    )

    np.testing.assert_allclose(builder.H_r_inv_sq, cached_angular_operator)


def test_repeated_metagga_hamiltonian_builds_are_idempotent():
    builder, zero_potential, de_xc_dtau, switches = _build_scan_hamiltonian_fixture()

    first = builder.build_for_l_channel(
        l=1,
        v_hartree=zero_potential,
        v_x=zero_potential,
        v_c=zero_potential,
        switches=switches,
        de_xc_dtau=de_xc_dtau,
    )
    second = builder.build_for_l_channel(
        l=1,
        v_hartree=zero_potential,
        v_x=zero_potential,
        v_c=zero_potential,
        switches=switches,
        de_xc_dtau=de_xc_dtau,
    )

    np.testing.assert_allclose(second, first)


def test_scan_solve_preserves_cached_angular_operator():
    solver = AtomicDFTSolver(
        atomic_number=6,
        all_electron_flag=True,
        xc_functional="SCAN",
        domain_size=6.0,
        finite_element_number=3,
        polynomial_order=3,
        quadrature_point_number=9,
        max_scf_iterations=2,
        use_pulay_mixing=False,
        verbose=False,
    )
    cached_angular_operator = solver.hamiltonian_builder.H_r_inv_sq.copy()

    result = solver.solve()

    assert np.isfinite(result["energy"])
    np.testing.assert_allclose(
        solver.hamiltonian_builder.H_r_inv_sq,
        cached_angular_operator,
    )
