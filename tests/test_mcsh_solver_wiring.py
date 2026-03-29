"""Tests that AtomicDFTSolver correctly accepts and uses mcsh_config.

These tests verify the wiring only. They use a fast solver config
(low mesh resolution) since we only need convergence, not accuracy.
"""

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MCSHConfig


class TestSolverMCSHWiring:
    """Verify solver accepts mcsh_config and returns mcsh_result."""

    FAST_SOLVER_KWARGS = dict(
        atomic_number=1,
        xc_functional="LDA_PZ",
        domain_size=15.0,
        finite_element_number=8,
        polynomial_order=15,
        quadrature_point_number=33,
        verbose=False,
    )

    def test_no_mcsh_config_returns_none(self):
        """Without mcsh_config, result dict should have mcsh_result=None."""
        solver = AtomicDFTSolver(**self.FAST_SOLVER_KWARGS)
        result = solver.solve()
        assert "mcsh_result" in result
        assert result["mcsh_result"] is None

    def test_mcsh_config_returns_result(self):
        """With mcsh_config, result dict should contain an MCSHResult."""
        from standalone_rho_multipole import MCSHResult

        config = MCSHConfig(
            rcuts=[1.0, 2.0],
            l_max=1,
            box_size=12.0,
            spacing=0.5,
        )
        solver = AtomicDFTSolver(
            **self.FAST_SOLVER_KWARGS,
            mcsh_config=config,
        )
        result = solver.solve()

        assert result["mcsh_result"] is not None
        assert isinstance(result["mcsh_result"], MCSHResult)
        assert result["mcsh_result"].descriptors.shape[1] == 2  # 2 rcuts
        assert result["mcsh_result"].descriptors.shape[2] == 2  # l_max=1 -> 2 channels

    def test_mcsh_does_not_affect_energy(self):
        """MCSH is post-processing: enabling it must not change SCF results."""
        config = MCSHConfig(rcuts=[2.0], box_size=12.0, spacing=0.5)

        solver_without = AtomicDFTSolver(**self.FAST_SOLVER_KWARGS)
        solver_with = AtomicDFTSolver(**self.FAST_SOLVER_KWARGS, mcsh_config=config)

        result_without = solver_without.solve()
        result_with = solver_with.solve()

        assert result_without["energy"] == pytest.approx(result_with["energy"], abs=1e-12)
        np.testing.assert_array_equal(result_without["rho"], result_with["rho"])
