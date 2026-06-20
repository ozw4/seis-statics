from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from seis_statics.refraction.cell_regularization import (
    augment_design_matrix_with_cell_smoothing,
    build_cell_slowness_smoothing_rows,
)


def test_cell_slowness_smoothing_rows_use_active_neighbor_edges_and_mapping() -> None:
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=np.array([0, 1, 3, 4]),
        velocity_smoothing_weight=0.5,
        smoothing_reference_distance_m=20.0,
        n_total_cells=6,
        number_of_cell_x=3,
        number_of_cell_y=2,
        bedrock_slowness_cell_col_start=2,
        n_parameters=7,
    )

    assert rows.n_edges == 4
    assert rows.n_rows == 4
    assert rows.reference_distance_m == 20.0
    assert rows.row_scale == 10.0
    np.testing.assert_array_equal(
        rows.edge_cell_id,
        np.array([[0, 1], [0, 3], [1, 4], [3, 4]]),
    )
    np.testing.assert_array_equal(rows.active_cell_neighbor_count, np.array([2, 2, 2, 2]))
    np.testing.assert_allclose(
        rows.matrix.toarray(),
        np.array(
            [
                [0.0, 0.0, 10.0, -10.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, -10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 10.0, 0.0, -10.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 10.0, -10.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(rows.rhs_s, np.zeros(4))
    assert rows.qc == {
        'velocity_smoothing_weight': 0.5,
        'smoothing_reference_distance_m': 20.0,
        'n_cell_smoothing_edges': 4,
        'n_cell_smoothing_rows': 4,
        'smoothing_row_scale': 10.0,
        'active_cell_neighbor_count_min': 2,
        'active_cell_neighbor_count_median': 2.0,
        'active_cell_neighbor_count_max': 2,
    }


def test_cell_slowness_smoothing_reference_distance_defaults_to_row_distance_median() -> None:
    rows = build_cell_slowness_smoothing_rows(
        active_cell_id=np.array([0, 1]),
        velocity_smoothing_weight=2.0,
        row_distance_m=np.array([10.0, 30.0, 50.0]),
        n_total_cells=2,
        number_of_cell_x=2,
    )

    assert rows.reference_distance_m == 30.0
    assert rows.row_scale == 60.0
    np.testing.assert_allclose(rows.matrix.toarray(), [[60.0, -60.0]])


def test_zero_weight_and_disconnected_cells_return_empty_smoothing_rows() -> None:
    zero_weight = build_cell_slowness_smoothing_rows(
        active_cell_id=np.array([0, 2]),
        velocity_smoothing_weight=0.0,
        n_total_cells=3,
        number_of_cell_x=3,
        n_parameters=3,
    )
    disconnected = build_cell_slowness_smoothing_rows(
        active_cell_id=np.array([0, 2]),
        velocity_smoothing_weight=1.0,
        smoothing_reference_distance_m=10.0,
        n_total_cells=3,
        number_of_cell_x=3,
        n_parameters=3,
    )

    assert zero_weight.matrix.shape == (0, 3)
    assert zero_weight.n_rows == 0
    assert zero_weight.reference_distance_m is None
    np.testing.assert_array_equal(zero_weight.active_cell_neighbor_count, np.array([]))
    assert disconnected.matrix.shape == (0, 3)
    assert disconnected.n_rows == 0
    assert disconnected.reference_distance_m == 10.0
    np.testing.assert_array_equal(disconnected.active_cell_neighbor_count, np.array([0, 0]))


def test_augment_design_matrix_with_cell_smoothing_appends_rows() -> None:
    smoothing_rows = build_cell_slowness_smoothing_rows(
        active_cell_id=np.array([0, 1]),
        velocity_smoothing_weight=1.0,
        smoothing_reference_distance_m=5.0,
        n_total_cells=2,
        number_of_cell_x=2,
    )
    matrix = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    rhs = np.array([1.5, 2.5])

    matrix_aug, rhs_aug, n_added = augment_design_matrix_with_cell_smoothing(
        matrix,
        rhs,
        smoothing_rows,
    )

    assert n_added == 1
    np.testing.assert_allclose(matrix_aug.toarray(), [[1.0, 0.0], [0.0, 1.0], [5.0, -5.0]])
    np.testing.assert_allclose(rhs_aug, [1.5, 2.5, 0.0])


def test_cell_slowness_smoothing_rejects_invalid_active_cells_and_mappings() -> None:
    with pytest.raises(ValueError, match='unique'):
        build_cell_slowness_smoothing_rows(
            active_cell_id=np.array([0, 0]),
            velocity_smoothing_weight=1.0,
            smoothing_reference_distance_m=10.0,
            n_total_cells=2,
            number_of_cell_x=2,
        )

    with pytest.raises(ValueError, match='outside the grid'):
        build_cell_slowness_smoothing_rows(
            active_cell_id=np.array([2]),
            velocity_smoothing_weight=1.0,
            smoothing_reference_distance_m=10.0,
            n_total_cells=2,
            number_of_cell_x=2,
        )

    with pytest.raises(ValueError, match='missing an active cell ID'):
        build_cell_slowness_smoothing_rows(
            active_cell_id=np.array([0, 1]),
            velocity_smoothing_weight=1.0,
            smoothing_reference_distance_m=10.0,
            n_total_cells=2,
            number_of_cell_x=2,
            cell_id_to_col={0: 4},
            n_parameters=6,
        )
