from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from atom import AtomicDFTSolver
from atom.mesh.operators import RadialOperatorsBuilder
from atom.scf.driver import SwitchesFlags
from atom.scf.hamiltonian import HamiltonianBuilder


def _build_hydrogen_hamiltonian_components():
    solver = AtomicDFTSolver(
        atomic_number=1,
        all_electron_flag=True,
        xc_functional="None",
        domain_size=8.0,
        finite_element_number=4,
        polynomial_order=6,
        quadrature_point_number=15,
        verbose=False,
    )
    grid_data, _, _ = solver._initialize_grids()
    ops_builder = RadialOperatorsBuilder.from_grid_data(grid_data)
    hamiltonian_builder = HamiltonianBuilder(
        ops_builder=ops_builder,
        pseudo=solver.pseudo,
        occupation_info=solver.occupation_info,
        all_electron=True,
    )
    return ops_builder, hamiltonian_builder


def test_interior_inverse_square_root_whitens_interior_overlap():
    ops_builder, _ = _build_hydrogen_hamiltonian_components()

    S_interior = ops_builder.get_S(exclude_boundary=True)
    S_inv_sqrt_interior = ops_builder.get_S_inv_sqrt(exclude_boundary=True)
    identity = np.eye(S_interior.shape[0])

    np.testing.assert_allclose(
        S_inv_sqrt_interior @ S_interior @ S_inv_sqrt_interior,
        identity,
        atol=1e-10,
        rtol=0.0,
    )

    cropped_full_inv_sqrt = ops_builder.get_S_inv_sqrt()[1:-1, 1:-1]
    assert np.linalg.norm(cropped_full_inv_sqrt @ S_interior @ cropped_full_inv_sqrt - identity) > 1e-4


def test_symmetrized_interior_hamiltonian_matches_generalized_eigenproblem():
    ops_builder, hamiltonian_builder = _build_hydrogen_hamiltonian_components()
    zeros = np.zeros_like(ops_builder.quadrature_nodes)
    switches = SwitchesFlags("None")

    H_generalized = hamiltonian_builder.build_for_l_channel(
        l=0,
        v_hartree=zeros,
        v_x=zeros,
        v_c=zeros,
        switches=switches,
        symmetrize=False,
        exclude_boundary=True,
    )
    H_symmetrized = hamiltonian_builder.build_for_l_channel(
        l=0,
        v_hartree=zeros,
        v_x=zeros,
        v_c=zeros,
        switches=switches,
        symmetrize=True,
        exclude_boundary=True,
    )
    S_interior = ops_builder.get_S(exclude_boundary=True)

    generalized_eigenvalues, _ = eigh(
        H_generalized,
        S_interior,
        check_finite=False,
        driver="gv",
    )
    symmetrized_eigenvalues = np.linalg.eigvalsh(H_symmetrized)

    np.testing.assert_allclose(
        symmetrized_eigenvalues,
        generalized_eigenvalues,
        atol=1e-9,
        rtol=1e-10,
    )


def test_symmetrized_interpolation_uses_interior_inverse_square_root():
    ops_builder, hamiltonian_builder = _build_hydrogen_hamiltonian_components()
    rng = np.random.default_rng(12345)
    n_interior_nodes = ops_builder.get_S(exclude_boundary=True).shape[0]
    eigenvectors = rng.normal(size=(n_interior_nodes, 3))

    actual = hamiltonian_builder.interpolate_eigenvectors_to_quadrature(
        eigenvectors,
        symmetrize=True,
        pad_width=1,
    )
    transformed_interior = ops_builder.get_S_inv_sqrt(exclude_boundary=True) @ eigenvectors
    expected = ops_builder.get_global_interpolation_matrix() @ np.pad(
        transformed_interior,
        ((1, 1), (0, 0)),
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    old_cropped_transform = ops_builder.get_S_inv_sqrt()[1:-1, 1:-1] @ eigenvectors
    old_expected = ops_builder.get_global_interpolation_matrix() @ np.pad(
        old_cropped_transform,
        ((1, 1), (0, 0)),
    )
    assert np.linalg.norm(actual - old_expected) > 1e-3
