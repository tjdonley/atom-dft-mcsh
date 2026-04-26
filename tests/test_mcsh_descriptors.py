"""Unit tests for the generic multipole descriptor API using the MCSH basis."""

import numpy as np
import pytest


class TestMultipoleCalculatorValidation:
    """Test that MultipoleCalculator validates inputs correctly."""

    def test_valid_config_heaviside(self):
        from atom.descriptors import MultipoleCalculator

        calc = MultipoleCalculator(angular_basis="mcsh", rcuts=[1.0, 2.0, 3.0])
        assert calc.angular_basis == "mcsh"
        assert calc.radial_basis == "heaviside"
        assert calc.l_max == 2
        assert calc.box_size == 20.0
        assert calc.spacing == (0.3, 0.3, 0.3)

    def test_valid_config_legendre(self):
        from atom.descriptors import MultipoleCalculator

        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0],
            radial_basis="legendre",
            radial_order=3,
        )
        assert calc.radial_basis == "legendre"
        assert calc.radial_order == 3

    def test_empty_rcuts_raises(self):
        from atom.descriptors import MultipoleCalculator

        with pytest.raises(ValueError, match="rcuts"):
            MultipoleCalculator(angular_basis="mcsh", rcuts=[])

    def test_negative_l_max_raises(self):
        from atom.descriptors import MultipoleCalculator

        with pytest.raises(ValueError, match="l_max"):
            MultipoleCalculator(angular_basis="mcsh", rcuts=[1.0], l_max=-1)

    def test_invalid_angular_basis_raises(self):
        from atom.descriptors import MultipoleCalculator

        with pytest.raises(ValueError, match="angular_basis"):
            MultipoleCalculator(angular_basis="orthonormal", rcuts=[1.0])

    def test_invalid_radial_basis_raises(self):
        from atom.descriptors import MultipoleCalculator

        with pytest.raises(ValueError, match="radial_basis"):
            MultipoleCalculator(
                angular_basis="mcsh",
                rcuts=[1.0],
                radial_basis="cubic",
            )

    def test_rcut_exceeds_half_box_raises(self):
        from atom.descriptors import MultipoleCalculator

        with pytest.raises(ValueError, match="box_size"):
            MultipoleCalculator(angular_basis="mcsh", rcuts=[12.0], box_size=20.0)

    def test_zero_spacing_raises(self):
        from atom.descriptors import MultipoleCalculator

        with pytest.raises(ValueError, match="spacing"):
            MultipoleCalculator(angular_basis="mcsh", rcuts=[1.0], spacing=0.0)


class TestMultipoleCalculator:
    """Test MultipoleCalculator produces correct results for the MCSH basis."""

    @pytest.fixture
    def hydrogen_radial(self):
        """Synthetic hydrogen-like radial density: rho(r) = (1/pi)*exp(-2r)."""
        r = np.linspace(0, 10.0, 500)
        rho = (1.0 / np.pi) * np.exp(-2.0 * r)
        return r, rho

    def test_compute_from_radial_returns_multipole_result(self, hydrogen_radial):
        from atom.descriptors import MultipoleCalculator, MultipoleResult

        r, rho = hydrogen_radial
        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0],
            l_max=2,
            box_size=10.0,
            spacing=0.5,
        )
        result = calc.compute_from_radial(r, rho)

        assert isinstance(result, MultipoleResult)
        assert result.angular_basis == "mcsh"
        assert result.radial_basis == "heaviside"
        assert result.descriptors.ndim == 3
        assert result.descriptors.shape[1] == 2
        assert result.descriptors.shape[2] == 3

    def test_compute_from_radial_matches_low_level_engine(self, hydrogen_radial):
        """MultipoleCalculator must match direct low-level engine calls."""
        from atom.descriptors import MultipoleCalculator
        from atom.descriptors.multipole import compute_descriptors_from_radial

        r, rho = hydrogen_radial
        rcuts = [1.0, 2.0, 3.0]
        box_size = 10.0
        spacing = 0.5
        center = (box_size / 2,) * 3

        direct_result = compute_descriptors_from_radial(
            r_radial=r,
            rho_radial=rho,
            box_size=box_size,
            spacing=spacing,
            atom_center=center,
            rcuts=rcuts,
            angular_basis="mcsh",
            l_max=2,
            periodic=True,
            radial_basis="heaviside",
            radial_order=0,
        )

        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=rcuts,
            box_size=box_size,
            spacing=spacing,
        )
        calc_result = calc.compute_from_radial(r, rho)

        np.testing.assert_array_equal(
            calc_result.descriptors, direct_result.descriptors
        )

    def test_legendre_produces_different_values(self, hydrogen_radial):
        from atom.descriptors import MultipoleCalculator

        r, rho = hydrogen_radial
        kwargs = dict(
            angular_basis="mcsh", rcuts=[2.0], box_size=10.0, spacing=0.5, l_max=2
        )

        heaviside = MultipoleCalculator(**kwargs, radial_basis="heaviside")
        legendre1 = MultipoleCalculator(
            **kwargs, radial_basis="legendre", radial_order=1
        )

        h_result = heaviside.compute_from_radial(r, rho)
        l_result = legendre1.compute_from_radial(r, rho)

        assert not np.allclose(h_result.descriptors, l_result.descriptors)

    def test_legendre_order0_equals_heaviside(self, hydrogen_radial):
        from atom.descriptors import MultipoleCalculator

        r, rho = hydrogen_radial
        kwargs = dict(
            angular_basis="mcsh", rcuts=[2.0], box_size=10.0, spacing=0.5, l_max=2
        )

        heaviside = MultipoleCalculator(**kwargs, radial_basis="heaviside")
        legendre0 = MultipoleCalculator(
            **kwargs, radial_basis="legendre", radial_order=0
        )

        h_result = heaviside.compute_from_radial(r, rho)
        l_result = legendre0.compute_from_radial(r, rho)

        np.testing.assert_allclose(
            h_result.descriptors, l_result.descriptors, atol=1e-14
        )

    def test_compute_from_solver_result(self, hydrogen_radial):
        from atom.descriptors import MultipoleCalculator

        r, rho = hydrogen_radial
        mock_result = {"quadrature_nodes": r, "rho": rho}
        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[2.0],
            box_size=10.0,
            spacing=0.5,
        )

        from_radial = calc.compute_from_radial(r, rho)
        from_solver = calc.compute_from_solver_result(mock_result)
        np.testing.assert_array_equal(from_radial.descriptors, from_solver.descriptors)

    def test_extract_radial_profile(self, hydrogen_radial):
        from atom.descriptors import MultipoleCalculator

        r, rho = hydrogen_radial
        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[2.0],
            box_size=10.0,
            spacing=0.5,
        )
        result = calc.compute_from_radial(r, rho)
        profile = calc.extract_radial_profile(result)

        assert "r" in profile
        assert "descriptors" in profile
        assert "rcuts" in profile
        assert "l_max" in profile
        assert len(profile["r"]) == result.descriptors.shape[0]
        assert np.all(profile["r"] >= 0)

    def test_extract_radial_profile_custom_center(self, hydrogen_radial):
        from atom.descriptors import MultipoleCalculator

        r, rho = hydrogen_radial
        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[2.0],
            box_size=10.0,
            spacing=0.5,
        )
        result = calc.compute_from_radial(r, rho)

        profile_default = calc.extract_radial_profile(result)
        profile_explicit = calc.extract_radial_profile(result, center=(5.0, 5.0, 5.0))
        np.testing.assert_array_equal(profile_default["r"], profile_explicit["r"])

        profile_shifted = calc.extract_radial_profile(result, center=(3.0, 3.0, 3.0))
        assert not np.array_equal(profile_default["r"], profile_shifted["r"])

    def test_compute_from_3d_returns_multipole_result(self):
        from atom.descriptors import MultipoleCalculator, MultipoleResult

        nx = ny = nz = 21
        h = 0.5
        rho_3d = np.ones((nx, ny, nz)) * 0.1

        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0],
            l_max=2,
            box_size=10.0,
            spacing=h,
        )
        result = calc.compute_from_3d(rho_3d, spacing=(h, h, h))

        assert isinstance(result, MultipoleResult)
        assert result.descriptors.ndim == 3
        assert result.descriptors.shape[1] == 2
        assert result.descriptors.shape[2] == 3

    def test_compute_from_3d_matches_low_level_engine(self, hydrogen_radial):
        from atom.descriptors import MultipoleCalculator
        from atom.descriptors.grid3d import (
            grid_radial_distances,
            make_cartesian_grid,
            project_radial_to_3d,
        )
        from atom.descriptors.multipole import compute_descriptors

        r, rho = hydrogen_radial
        box_size = 10.0
        spacing = 0.5
        center = (box_size / 2,) * 3

        x_1d, X, Y, Z = make_cartesian_grid(box_size, spacing)
        R_3d = grid_radial_distances(X, Y, Z, center)
        rho_3d = project_radial_to_3d(r, rho, R_3d)
        h = float(x_1d[1] - x_1d[0])

        direct = compute_descriptors(
            rho_3d=rho_3d,
            spacing=(h, h, h),
            rcuts=[1.0, 2.0],
            angular_basis="mcsh",
            l_max=2,
            periodic=True,
        )

        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[1.0, 2.0],
            box_size=box_size,
            spacing=spacing,
        )
        via_calc = calc.compute_from_3d(rho_3d, spacing=(h, h, h))

        np.testing.assert_array_equal(direct.descriptors, via_calc.descriptors)

    def test_compute_from_3d_default_sampling_honors_center(self):
        from atom.descriptors import MultipoleCalculator

        n = 21
        h = 1.0
        center = (10.0, 3.0, 7.0)
        coords = np.arange(n) * h
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        rho_3d = np.exp(
            -((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
        )

        calc = MultipoleCalculator(
            angular_basis="mcsh",
            rcuts=[2.0],
            l_max=1,
            box_size=20.0,
            spacing=h,
        )

        default_result = calc.compute_from_3d(rho_3d, spacing=(h, h, h), center=center)
        default_profile = calc.extract_radial_profile(default_result)
        default_center_index = int(np.argmin(default_profile["r"]))

        explicit_indices = np.column_stack(
            [
                np.arange(n),
                np.full(n, int(round(center[1] / h))),
                np.full(n, int(round(center[2] / h))),
            ]
        )
        explicit_result = calc.compute_from_3d(
            rho_3d,
            spacing=(h, h, h),
            center=center,
            eval_indices=explicit_indices,
        )
        explicit_profile = calc.extract_radial_profile(explicit_result)
        explicit_center_index = int(np.argmin(explicit_profile["r"]))

        assert default_profile["r"][default_center_index] == pytest.approx(
            0.0, abs=1e-12
        )
        assert explicit_profile["r"][explicit_center_index] == pytest.approx(
            0.0, abs=1e-12
        )
        assert abs(default_profile["descriptors"][default_center_index, 0, 1]) < 1e-12
        np.testing.assert_array_equal(
            default_result.grid_indices, explicit_result.grid_indices
        )
        np.testing.assert_allclose(
            default_result.descriptors,
            explicit_result.descriptors,
            atol=1e-14,
            rtol=0.0,
        )
