"""End-to-end test: hydrogen atom SCF + MCSH descriptor computation.

Runs the full atom-solver for hydrogen at publication quality, computes
MCSH descriptors, and validates physical invariants.

These tests are slow (~60-120 seconds).
"""

import numpy as np
import pytest

from atom import AtomicDFTSolver
from atom.descriptors import MCSHCalculator, MCSHConfig


@pytest.fixture(scope="module")
def hydrogen_full_result():
    """Run a full hydrogen calculation with MCSH descriptors.

    Uses parameters close to the pipeline_B validation (matching the
    atom-solver output that was validated against SPARC).
    """
    config = MCSHConfig(
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
        mcsh_config=config,
    )
    return solver.solve()


class TestHydrogenSCFConvergence:
    """Verify the hydrogen SCF calculation converges correctly."""

    def test_converged(self, hydrogen_full_result):
        assert hydrogen_full_result["converged"]

    def test_energy_reasonable(self, hydrogen_full_result):
        """Hydrogen total energy should be around -0.459 Ha (GGA_PBE)."""
        E = hydrogen_full_result["energy"]
        assert -0.50 < E < -0.40, f"H energy = {E} Ha, expected ~ -0.459"

    def test_density_integrates_to_one(self, hydrogen_full_result):
        """Radial density should integrate to ~1 electron."""
        r = hydrogen_full_result["quadrature_nodes"]
        rho = hydrogen_full_result["rho"]
        w = hydrogen_full_result["quadrature_weights"]
        # Integral of 4*pi*r^2*rho(r) dr
        Ne = np.sum(4 * np.pi * r**2 * rho * w)
        assert Ne == pytest.approx(1.0, abs=0.01), f"Ne = {Ne}, expected 1.0"


class TestHydrogenMCSHDescriptors:
    """Verify MCSH descriptors are physically reasonable."""

    def test_mcsh_result_exists(self, hydrogen_full_result):
        assert hydrogen_full_result["mcsh_result"] is not None

    def test_descriptor_shape(self, hydrogen_full_result):
        """6 rcuts, l_max=2 -> 3 channels."""
        d = hydrogen_full_result["mcsh_result"].descriptors
        assert d.shape[1] == 6  # rcuts
        assert d.shape[2] == 3  # l=0,1,2

    def test_all_finite(self, hydrogen_full_result):
        d = hydrogen_full_result["mcsh_result"].descriptors
        assert np.all(np.isfinite(d))

    def test_l1_near_zero_at_center(self, hydrogen_full_result):
        """Spherical H density: l=1 should vanish at atom center.

        Uses a grid-aligned config (spacing=0.4, box=12 so center=6.0 is
        exactly on-grid) to get machine-precision zeros at the center point.
        The main fixture uses spacing=0.3 where 10.0 / 0.3 is not an integer,
        so the closest evaluation point is 0.3 Bohr off-center and l=1 is only
        approximately zero there.
        """
        r = hydrogen_full_result["quadrature_nodes"]
        rho = hydrogen_full_result["rho"]

        # Grid-aligned config: center=6.0, spacing=0.4 -> 6.0/0.4=15 (integer)
        config = MCSHConfig(rcuts=[1.0, 2.0, 3.0], l_max=2, box_size=12.0, spacing=0.4)
        calc = MCSHCalculator(config)
        mcsh = calc.compute_from_radial(r, rho)
        profile = calc.extract_radial_profile(mcsh)

        center_idx = np.argmin(profile["r"])
        assert profile["r"][center_idx] < 1e-10, (
            f"Expected center point at r=0, got r={profile['r'][center_idx]}"
        )
        l1 = profile["descriptors"][center_idx, :, 1]
        assert np.all(np.abs(l1) < 1e-6), f"l=1 at center = {l1}"

    def test_descriptors_match_standalone_on_same_density(
        self, hydrogen_full_result
    ):
        """Descriptors computed through solver must match standalone package
        when given the same density. This is the critical round-trip test."""
        r = hydrogen_full_result["quadrature_nodes"]
        rho = hydrogen_full_result["rho"]

        config = MCSHConfig(
            rcuts=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            l_max=2, box_size=20.0, spacing=0.3,
        )
        calc = MCSHCalculator(config)

        # Post-hoc computation from same density
        post_hoc = calc.compute_from_radial(r, rho)

        # Inline computation from solver
        inline = hydrogen_full_result["mcsh_result"]

        np.testing.assert_array_equal(inline.descriptors, post_hoc.descriptors)
