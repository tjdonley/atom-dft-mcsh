"""Regression tests for SCFResult serialization helpers."""

from __future__ import annotations

import numpy as np

from atom.scf.density import DensityData
from atom.scf.driver import SCFResult


def _make_result() -> SCFResult:
    density_data = DensityData(
        rho=np.array([0.2, 0.3, 0.5]),
        grad_rho=np.array([0.01, 0.02, 0.03]),
        tau=np.array([0.1, 0.2, 0.3]),
    )
    return SCFResult(
        eigen_energies=np.array([-0.5, -0.125]),
        orbitals=np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        ),
        density_data=density_data,
        converged=True,
        iterations=4,
        rho_residual=1e-7,
        full_eigen_energies=np.array([-0.5, -0.125, 0.2]),
        full_orbitals=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        ),
        full_l_terms=np.array([0, 0, 1]),
        outer_iterations=2,
        outer_converged=True,
        total_energy=-0.75,
        energy_components={"kinetic": 0.25, "external": -1.0},
    )


def test_scf_result_to_dict_uses_current_field_names():
    result = _make_result()

    result_dict = result.to_dict()

    assert "eigen_energies" in result_dict
    assert "orbitals" in result_dict
    assert "density_data" in result_dict
    assert "rho_residual" in result_dict
    assert "eigenvalues" not in result_dict
    assert "eigenvectors" not in result_dict
    assert "residual" not in result_dict
    np.testing.assert_allclose(result_dict["rho"], result.density_data.rho)


def test_scf_result_from_dict_round_trips_current_format():
    original = _make_result()

    loaded = SCFResult.from_dict(original.to_dict())

    np.testing.assert_allclose(loaded.eigen_energies, original.eigen_energies)
    np.testing.assert_allclose(loaded.orbitals, original.orbitals)
    np.testing.assert_allclose(loaded.density_data.rho, original.density_data.rho)
    np.testing.assert_allclose(
        loaded.density_data.grad_rho, original.density_data.grad_rho
    )
    np.testing.assert_allclose(loaded.density_data.tau, original.density_data.tau)
    assert loaded.rho_residual == original.rho_residual
    assert loaded.converged is True


def test_scf_result_from_dict_accepts_legacy_key_names():
    legacy_result_dict = {
        "eigenvalues": np.array([-0.5, -0.125]),
        "eigenvectors": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "rho": np.array([0.2, 0.8]),
        "converged": False,
        "iterations": 3,
        "residual": 2e-6,
    }

    result = SCFResult.from_dict(legacy_result_dict)

    np.testing.assert_allclose(result.eigen_energies, legacy_result_dict["eigenvalues"])
    np.testing.assert_allclose(result.orbitals, legacy_result_dict["eigenvectors"])
    np.testing.assert_allclose(result.density_data.rho, legacy_result_dict["rho"])
    assert result.rho_residual == legacy_result_dict["residual"]
    assert result.converged is False


def test_scf_result_summary_uses_current_field_names():
    result = _make_result()

    summary = result.summary()

    assert "Final residual: 1.000000e-07" in summary
    assert "Number of states: 2" in summary
    assert "Lowest eigenvalue: -0.500000 Ha" in summary
