"""End-to-end solver regressions adapted from master branch workflows."""

from __future__ import annotations

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator


def test_basic_gga_pbe_end_to_end():
    solver = AtomicDFTSolver(
        atomic_number=13,
        verbose=False,
        xc_functional="GGA_PBE",
        all_electron_flag=False,
    )
    results = solver.solve()

    assert results["converged"]
    assert results["rho"] is not None
    assert results["orbitals"] is not None
    assert len(results["rho"]) > 0
    assert results["orbitals"].shape[0] > 0
    assert np.isfinite(results["energy"])


def test_save_intermediate_toggle_is_consistent():
    solver = AtomicDFTSolver(
        atomic_number=13,
        verbose=False,
        xc_functional="GGA_PBE",
        all_electron_flag=False,
    )

    without = solver.solve(save_intermediate=False)
    with_intermediate = solver.solve(save_intermediate=True)

    assert without["intermediate_info"] is None
    assert with_intermediate["intermediate_info"] is not None
    assert len(with_intermediate["intermediate_info"].inner_iterations) > 0
    assert without["energy"] == pytest.approx(with_intermediate["energy"], abs=1e-12)
    np.testing.assert_allclose(
        without["rho"], with_intermediate["rho"], atol=1e-12, rtol=0.0
    )


def test_pbe0_end_to_end():
    solver = AtomicDFTSolver(
        atomic_number=10,
        domain_size=13.0,
        finite_element_number=10,
        polynomial_order=20,
        quadrature_point_number=43,
        oep_basis_number=5,
        verbose=False,
        xc_functional="PBE0",
        all_electron_flag=True,
        use_oep=False,
        mesh_type="polynomial",
        mesh_concentration=2.0,
    )
    results = solver.solve()

    assert results["converged"]
    assert results["orbitals"] is not None
    assert results["v_x_local"] is not None
    assert results["v_c_local"] is not None
    assert len(results["v_x_local"]) > 0
    assert len(results["v_c_local"]) > 0
    assert np.isfinite(results["energy"])


def test_pbe0_with_multipole_postprocessing_is_noninvasive():
    solver_kwargs = dict(
        atomic_number=10,
        domain_size=13.0,
        finite_element_number=10,
        polynomial_order=20,
        quadrature_point_number=43,
        oep_basis_number=5,
        verbose=False,
        xc_functional="PBE0",
        all_electron_flag=True,
        use_oep=False,
        mesh_type="polynomial",
        mesh_concentration=2.0,
    )
    calc = MultipoleCalculator(
        angular_basis="mcsh",
        rcuts=[1.0, 2.0, 3.0],
        l_max=2,
        box_size=12.0,
        spacing=0.4,
    )

    plain = AtomicDFTSolver(**solver_kwargs).solve()
    with_mp = AtomicDFTSolver(
        **solver_kwargs,
        descriptor_calculators=[calc],
    ).solve()

    assert plain["converged"] and with_mp["converged"]
    assert plain["energy"] == pytest.approx(with_mp["energy"], abs=1e-12)
    np.testing.assert_allclose(plain["rho"], with_mp["rho"], atol=1e-12, rtol=0.0)
    assert "multipole" in with_mp["descriptor_results"]
    assert np.all(np.isfinite(with_mp["descriptor_results"]["multipole"].descriptors))
