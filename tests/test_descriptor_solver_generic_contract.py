"""Generic solver-contract tests independent of the multipole implementation."""

from __future__ import annotations

from dataclasses import dataclass

from atom import AtomicDFTSolver
from atom.descriptors import (
    DescriptorCalculator,
    DescriptorContext,
    MultipoleCalculator,
)


@dataclass
class DummyDescriptorResult:
    quadrature_size: int
    density_size: int
    density_norm: float


class DummyDescriptorCalculator(DescriptorCalculator):
    name = "dummy"

    def compute(self, context: DescriptorContext) -> DummyDescriptorResult:
        return DummyDescriptorResult(
            quadrature_size=len(context.quadrature_nodes),
            density_size=len(context.density),
            density_norm=float(context.density.sum()),
        )


FAST_SOLVER_KWARGS = dict(
    atomic_number=1,
    xc_functional="LDA_PZ",
    domain_size=8.0,
    finite_element_number=6,
    polynomial_order=14,
    quadrature_point_number=31,
    verbose=False,
)


def test_solver_accepts_non_multipole_descriptor_calculator():
    result = AtomicDFTSolver(
        **FAST_SOLVER_KWARGS,
        descriptor_calculators=[DummyDescriptorCalculator()],
    ).solve()

    dummy = result["descriptor_results"]["dummy"]
    assert isinstance(dummy, DummyDescriptorResult)
    assert dummy.quadrature_size > 0
    assert dummy.density_size > 0
    assert dummy.quadrature_size == dummy.density_size
    assert dummy.density_norm > 0.0


def test_solver_stores_multiple_descriptor_results_by_name():
    multipole = MultipoleCalculator(
        angular_basis="mcsh",
        rcuts=[1.0, 2.0],
        l_max=1,
        box_size=12.0,
        spacing=0.5,
        name="mp_test",
    )

    result = AtomicDFTSolver(
        **FAST_SOLVER_KWARGS,
        descriptor_calculators=[DummyDescriptorCalculator(), multipole],
    ).solve()

    assert set(result["descriptor_results"].keys()) == {"dummy", "mp_test"}
    assert result["descriptor_results"]["mp_test"].angular_basis == "mcsh"
    assert result["descriptor_results"]["mp_test"].descriptors.shape[1] == 2
