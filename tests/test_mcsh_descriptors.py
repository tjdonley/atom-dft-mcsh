"""Unit tests for atom.descriptors.mcsh module.

Tests the MCSHConfig dataclass validation and MCSHCalculator
computation against the underlying multipole module.
"""

import numpy as np
import pytest


# ---- MCSHConfig validation tests ----


class TestMCSHConfig:
    """Test that MCSHConfig validates inputs correctly."""

    def test_valid_config_heaviside(self):
        from atom.descriptors import MCSHConfig
        config = MCSHConfig(rcuts=[1.0, 2.0, 3.0])
        assert config.radial_type == "heaviside"
        assert config.l_max == 2
        assert config.box_size == 20.0
        assert config.spacing == 0.3

    def test_valid_config_legendre(self):
        from atom.descriptors import MCSHConfig
        config = MCSHConfig(
            rcuts=[1.0, 2.0],
            radial_type="legendre",
            radial_order=3,
        )
        assert config.radial_type == "legendre"
        assert config.radial_order == 3

    def test_empty_rcuts_raises(self):
        from atom.descriptors import MCSHConfig
        with pytest.raises(ValueError, match="rcuts"):
            MCSHConfig(rcuts=[])

    def test_negative_l_max_raises(self):
        from atom.descriptors import MCSHConfig
        with pytest.raises(ValueError, match="l_max"):
            MCSHConfig(rcuts=[1.0], l_max=-1)

    def test_invalid_radial_type_raises(self):
        from atom.descriptors import MCSHConfig
        with pytest.raises(ValueError, match="radial_type"):
            MCSHConfig(rcuts=[1.0], radial_type="cubic")

    def test_rcut_exceeds_half_box_raises(self):
        from atom.descriptors import MCSHConfig
        with pytest.raises(ValueError, match="box_size"):
            MCSHConfig(rcuts=[12.0], box_size=20.0)

    def test_zero_spacing_raises(self):
        from atom.descriptors import MCSHConfig
        with pytest.raises(ValueError, match="spacing"):
            MCSHConfig(rcuts=[1.0], spacing=0.0)


# ---- MCSHCalculator tests ----


class TestMCSHCalculator:
    """Test MCSHCalculator produces correct results."""

    @pytest.fixture
    def hydrogen_radial(self):
        """Synthetic hydrogen-like radial density: rho(r) = (1/pi)*exp(-2r)."""
        r = np.linspace(0, 10.0, 500)
        rho = (1.0 / np.pi) * np.exp(-2.0 * r)
        return r, rho

    def test_compute_from_radial_returns_mcsh_result(self, hydrogen_radial):
        from atom.descriptors import MCSHCalculator, MCSHConfig
        from atom.descriptors.multipole import MCSHResult

        r, rho = hydrogen_radial
        config = MCSHConfig(rcuts=[1.0, 2.0], l_max=2, box_size=10.0, spacing=0.5)
        calc = MCSHCalculator(config)
        result = calc.compute_from_radial(r, rho)

        assert isinstance(result, MCSHResult)
        assert result.descriptors.ndim == 3
        # shape: (n_eval, n_rcuts=2, n_l=3)
        assert result.descriptors.shape[1] == 2
        assert result.descriptors.shape[2] == 3

    def test_compute_from_radial_matches_standalone(self, hydrogen_radial):
        """MCSHCalculator must produce identical results to calling
        compute_descriptors_from_radial directly."""
        from atom.descriptors import MCSHCalculator, MCSHConfig
        from atom.descriptors.multipole import compute_descriptors_from_radial

        r, rho = hydrogen_radial
        rcuts = [1.0, 2.0, 3.0]
        box_size = 10.0
        spacing = 0.5
        center = (box_size / 2,) * 3

        # Direct call to standalone package
        standalone_result = compute_descriptors_from_radial(
            r_radial=r, rho_radial=rho,
            box_size=box_size, spacing=spacing, atom_center=center,
            rcuts=rcuts, l_max=2, periodic=True,
            radial_type="heaviside", radial_order=0,
        )

        # Via MCSHCalculator
        config = MCSHConfig(
            rcuts=rcuts, box_size=box_size, spacing=spacing,
        )
        calc = MCSHCalculator(config)
        calc_result = calc.compute_from_radial(r, rho)

        # Must be bitwise identical (same code path)
        np.testing.assert_array_equal(
            calc_result.descriptors, standalone_result.descriptors,
        )

    def test_legendre_produces_different_values(self, hydrogen_radial):
        """Legendre order>0 must give different descriptors than Heaviside."""
        from atom.descriptors import MCSHCalculator, MCSHConfig

        r, rho = hydrogen_radial
        kwargs = dict(rcuts=[2.0], box_size=10.0, spacing=0.5, l_max=2)

        heaviside = MCSHCalculator(MCSHConfig(**kwargs, radial_type="heaviside"))
        legendre1 = MCSHCalculator(MCSHConfig(**kwargs, radial_type="legendre", radial_order=1))

        h_result = heaviside.compute_from_radial(r, rho)
        l_result = legendre1.compute_from_radial(r, rho)

        # LP order 1 != Heaviside (LP order 0 == Heaviside, tested elsewhere)
        assert not np.allclose(h_result.descriptors, l_result.descriptors)

    def test_legendre_order0_equals_heaviside(self, hydrogen_radial):
        """LP order 0 is P_0(x)=1, which is identical to Heaviside."""
        from atom.descriptors import MCSHCalculator, MCSHConfig

        r, rho = hydrogen_radial
        kwargs = dict(rcuts=[2.0], box_size=10.0, spacing=0.5, l_max=2)

        heaviside = MCSHCalculator(MCSHConfig(**kwargs, radial_type="heaviside"))
        legendre0 = MCSHCalculator(MCSHConfig(**kwargs, radial_type="legendre", radial_order=0))

        h_result = heaviside.compute_from_radial(r, rho)
        l_result = legendre0.compute_from_radial(r, rho)

        np.testing.assert_allclose(
            h_result.descriptors, l_result.descriptors, atol=1e-14,
        )

    def test_compute_from_solver_result(self, hydrogen_radial):
        """compute_from_solver_result extracts r and rho from a dict."""
        from atom.descriptors import MCSHCalculator, MCSHConfig

        r, rho = hydrogen_radial
        # Simulate a solver result dict (only the keys we need)
        mock_result = {
            "quadrature_nodes": r,
            "rho": rho,
        }

        config = MCSHConfig(rcuts=[2.0], box_size=10.0, spacing=0.5)
        calc = MCSHCalculator(config)

        # Both entry points should give identical results
        from_radial = calc.compute_from_radial(r, rho)
        from_solver = calc.compute_from_solver_result(mock_result)

        np.testing.assert_array_equal(
            from_radial.descriptors, from_solver.descriptors,
        )

    def test_extract_radial_profile(self, hydrogen_radial):
        """extract_radial_profile returns radial distances and descriptors."""
        from atom.descriptors import MCSHCalculator, MCSHConfig

        r, rho = hydrogen_radial
        config = MCSHConfig(rcuts=[2.0], box_size=10.0, spacing=0.5)
        calc = MCSHCalculator(config)
        result = calc.compute_from_radial(r, rho)
        profile = calc.extract_radial_profile(result)

        assert "r" in profile
        assert "descriptors" in profile
        assert "rcuts" in profile
        assert "l_max" in profile
        assert len(profile["r"]) == result.descriptors.shape[0]
        assert np.all(profile["r"] >= 0)  # distances are non-negative
