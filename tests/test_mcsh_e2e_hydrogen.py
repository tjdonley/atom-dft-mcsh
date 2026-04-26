"""End-to-end tests for the generic multipole API using the MCSH basis."""

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MultipoleCalculator


@pytest.fixture(scope="module")
def hydrogen_full_result():
    calc = MultipoleCalculator(
        angular_basis="mcsh",
        rcuts=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        l_max=2,
        box_size=20.0,
        spacing=0.3,
    )
    solver = AtomicDFTSolver(
        atomic_number=1,
        xc_functional="GGA_PBE",
        domain_size=20.0,
        verbose=False,
        descriptor_calculators=[calc],
    )
    return solver.solve()


class TestHydrogenMultipoleDescriptors:
    """Verify MCSH-based multipole descriptors are physically reasonable."""

    def test_multipole_result_exists(self, hydrogen_full_result):
        assert hydrogen_full_result["descriptor_results"]["multipole"] is not None

    def test_descriptor_shape(self, hydrogen_full_result):
        d = hydrogen_full_result["descriptor_results"]["multipole"].descriptors
        assert d.shape[1] == 6
        assert d.shape[2] == 3

    def test_all_finite(self, hydrogen_full_result):
        d = hydrogen_full_result["descriptor_results"]["multipole"].descriptors
        assert np.all(np.isfinite(d))

    def test_solver_result_preserves_basis_metadata(self, hydrogen_full_result):
        result = hydrogen_full_result["descriptor_results"]["multipole"]
        assert result.angular_basis == "mcsh"
        assert result.radial_basis == "heaviside"

    def test_post_hoc_matches_inline(self, hydrogen_full_result):
        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            l_max=2,
            box_size=20.0,
            spacing=0.3,
        )
        post_hoc = calc.compute_from_solver_result(hydrogen_full_result)
        inline = hydrogen_full_result["descriptor_results"]["multipole"]

        np.testing.assert_array_equal(inline.descriptors, post_hoc.descriptors)

    def test_extract_radial_profile(self, hydrogen_full_result):
        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0, 3.0],
            l_max=2,
            box_size=12.0,
            spacing=0.4,
        )
        result = calc.compute_from_solver_result(hydrogen_full_result)
        profile = calc.extract_radial_profile(result)

        assert np.all(profile["r"] >= 0)
        assert len(profile["r"]) == result.descriptors.shape[0]
        assert profile["descriptors"].shape == result.descriptors.shape

    def test_legendre_order0_equals_heaviside(self, hydrogen_full_result):
        r = hydrogen_full_result["quadrature_nodes"]
        rho = hydrogen_full_result["rho"]

        heaviside = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0, 3.0],
            l_max=2,
            box_size=12.0,
            spacing=0.4,
            radial_basis="heaviside",
        )
        legendre0 = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0, 3.0],
            l_max=2,
            box_size=12.0,
            spacing=0.4,
            radial_basis="legendre",
            radial_order=0,
        )

        h_result = heaviside.compute_from_radial(r, rho)
        l_result = legendre0.compute_from_radial(r, rho)
        np.testing.assert_allclose(
            h_result.descriptors, l_result.descriptors, atol=1e-14
        )

    def test_legendre_order2_differs_from_heaviside(self, hydrogen_full_result):
        r = hydrogen_full_result["quadrature_nodes"]
        rho = hydrogen_full_result["rho"]

        heaviside = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0, 3.0],
            l_max=2,
            box_size=12.0,
            spacing=0.4,
            radial_basis="heaviside",
        )
        legendre2 = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0, 3.0],
            l_max=2,
            box_size=12.0,
            spacing=0.4,
            radial_basis="legendre",
            radial_order=2,
        )

        h_result = heaviside.compute_from_radial(r, rho)
        l_result = legendre2.compute_from_radial(r, rho)
        assert not np.allclose(h_result.descriptors, l_result.descriptors)
