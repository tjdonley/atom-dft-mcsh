"""Tests that AtomicDFTSolver correctly accepts generic descriptor calculators.

These tests verify the wiring only. They use a fast solver config
(low mesh resolution) since we only need convergence, not accuracy.
"""

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MCSHCalculator, MCSHConfig


class TestSolverMCSHWiring:
    """Verify solver accepts descriptor calculators and returns results."""

    FAST_SOLVER_KWARGS = dict(
        atomic_number=1,
        xc_functional="LDA_PZ",
        domain_size=15.0,
        finite_element_number=8,
        polynomial_order=15,
        quadrature_point_number=33,
        verbose=False,
    )

    def test_no_descriptor_calculators_returns_empty_dict(self):
        """Without descriptor calculators, result dict should be empty."""
        solver = AtomicDFTSolver(**self.FAST_SOLVER_KWARGS)
        result = solver.solve()
        assert "descriptor_results" in result
        assert result["descriptor_results"] == {}

    def test_descriptor_calculator_returns_result(self):
        """With a descriptor calculator, result dict should contain an MCSHResult."""
        from atom.descriptors.multipole import MCSHResult

        config = MCSHConfig(
            rcuts=[1.0, 2.0],
            l_max=1,
            box_size=12.0,
            spacing=0.5,
        )
        calc = MCSHCalculator(config)
        solver = AtomicDFTSolver(
            **self.FAST_SOLVER_KWARGS,
            descriptor_calculators=[calc],
        )
        result = solver.solve()

        assert result["descriptor_results"]["mcsh"] is not None
        assert isinstance(result["descriptor_results"]["mcsh"], MCSHResult)
        assert result["descriptor_results"]["mcsh"].descriptors.shape[1] == 2
        assert result["descriptor_results"]["mcsh"].descriptors.shape[2] == 2

    def test_mcsh_does_not_affect_energy(self):
        """MCSH is post-processing: enabling it must not change SCF results."""
        config = MCSHConfig(rcuts=[2.0], box_size=12.0, spacing=0.5)
        calc = MCSHCalculator(config)

        solver_without = AtomicDFTSolver(**self.FAST_SOLVER_KWARGS)
        solver_with = AtomicDFTSolver(
            **self.FAST_SOLVER_KWARGS,
            descriptor_calculators=[calc],
        )

        result_without = solver_without.solve()
        result_with = solver_with.solve()

        assert result_without["energy"] == pytest.approx(
            result_with["energy"], abs=1e-12
        )
        np.testing.assert_array_equal(result_without["rho"], result_with["rho"])

    def test_invalid_descriptor_calculator_raises_at_init(self):
        """Passing a non-descriptor calculator should raise TypeError."""
        with pytest.raises(TypeError, match="DescriptorCalculator"):
            AtomicDFTSolver(
                **self.FAST_SOLVER_KWARGS,
                descriptor_calculators=["not a calculator"],
            )

    def test_duplicate_descriptor_names_raise_at_init(self):
        """Descriptor calculator names must be unique in solver results."""
        calc_a = MCSHCalculator(MCSHConfig(rcuts=[1.0], box_size=12.0, spacing=0.5))
        calc_b = MCSHCalculator(MCSHConfig(rcuts=[2.0], box_size=12.0, spacing=0.5))

        with pytest.raises(ValueError, match="duplicate"):
            AtomicDFTSolver(
                **self.FAST_SOLVER_KWARGS,
                descriptor_calculators=[calc_a, calc_b],
            )
