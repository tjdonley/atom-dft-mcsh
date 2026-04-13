"""Tests that AtomicDFTSolver correctly accepts generic multipole calculators."""

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator


class TestSolverMultipoleWiring:
    """Verify solver accepts descriptor calculators and returns results."""

    FAST_SOLVER_KWARGS = dict(
        atomic_number=1,
        xc_functional="LDA_PZ",
        domain_size=8.0,
        finite_element_number=6,
        polynomial_order=14,
        quadrature_point_number=31,
        verbose=False,
    )

    def test_no_descriptor_calculators_returns_empty_dict(self):
        solver = AtomicDFTSolver(**self.FAST_SOLVER_KWARGS)
        result = solver.solve()
        assert "descriptor_results" in result
        assert result["descriptor_results"] == {}

    def test_descriptor_calculator_returns_result(self):
        from atom.descriptors import MultipoleResult

        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0],
            l_max=1,
            box_size=12.0,
            spacing=0.5,
        )
        solver = AtomicDFTSolver(
            **self.FAST_SOLVER_KWARGS,
            descriptor_calculators=[calc],
        )
        result = solver.solve()

        assert result["descriptor_results"]["multipole"] is not None
        assert isinstance(result["descriptor_results"]["multipole"], MultipoleResult)
        assert result["descriptor_results"]["multipole"].descriptors.shape[1] == 2
        assert result["descriptor_results"]["multipole"].descriptors.shape[2] == 2

    def test_multipole_does_not_affect_energy(self):
        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[2.0],
            box_size=12.0,
            spacing=0.5,
        )

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
        with pytest.raises(TypeError, match="DescriptorCalculator"):
            AtomicDFTSolver(
                **self.FAST_SOLVER_KWARGS,
                descriptor_calculators=["not a calculator"],
            )

    def test_duplicate_descriptor_names_raise_at_init(self):
        calc_a = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0],
            box_size=12.0,
            spacing=0.5,
        )
        calc_b = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[2.0],
            box_size=12.0,
            spacing=0.5,
        )

        with pytest.raises(ValueError, match="duplicate"):
            AtomicDFTSolver(
                **self.FAST_SOLVER_KWARGS,
                descriptor_calculators=[calc_a, calc_b],
            )
