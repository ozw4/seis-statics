from __future__ import annotations

import numpy as np

from seis_statics.refraction import RefractionStaticRefractorCellOptions
from seis_statics.refraction.cell_grid import (
    assign_observation_midpoint_cells,
    assign_points_to_refraction_cells,
    build_refraction_cell_grid,
)


def test_refraction_cell_grid_is_row_major_with_expected_boundaries() -> None:
    grid = build_refraction_cell_grid(
        RefractionStaticRefractorCellOptions(
            number_of_cell_x=3,
            size_of_cell_x_m=10.0,
            x_coordinate_origin_m=100.0,
            number_of_cell_y=2,
            size_of_cell_y_m=5.0,
            y_coordinate_origin_m=20.0,
        )
    )

    np.testing.assert_array_equal(grid.cell_id, np.array([0, 1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(grid.ix, np.array([0, 1, 2, 0, 1, 2]))
    np.testing.assert_array_equal(grid.iy, np.array([0, 0, 0, 1, 1, 1]))
    np.testing.assert_allclose(grid.x_min_m, [100.0, 110.0, 120.0, 100.0, 110.0, 120.0])
    np.testing.assert_allclose(grid.x_max_m, [110.0, 120.0, 130.0, 110.0, 120.0, 130.0])
    np.testing.assert_allclose(grid.y_min_m, [20.0, 20.0, 20.0, 25.0, 25.0, 25.0])
    np.testing.assert_allclose(grid.y_max_m, [25.0, 25.0, 25.0, 30.0, 30.0, 30.0])
    np.testing.assert_allclose(grid.x_center_m, [105.0, 115.0, 125.0, 105.0, 115.0, 125.0])
    np.testing.assert_allclose(grid.y_center_m, [22.5, 22.5, 22.5, 27.5, 27.5, 27.5])


def test_assign_points_preserves_boundary_and_out_of_grid_conventions() -> None:
    grid = build_refraction_cell_grid(
        RefractionStaticRefractorCellOptions(
            number_of_cell_x=3,
            size_of_cell_x_m=10.0,
            x_coordinate_origin_m=100.0,
            number_of_cell_y=2,
            size_of_cell_y_m=5.0,
            y_coordinate_origin_m=20.0,
        )
    )
    tolerance = 1.0e-9 * 130.0
    assignment = assign_points_to_refraction_cells(
        grid,
        x_m=np.array([100.0, 110.0, 129.0, 130.0, 130.0 + 0.5 * tolerance, 130.0 + 2.0 * tolerance, 99.0, np.nan]),
        y_m=np.array([20.0, 20.0, 29.0, 30.0, 30.0, 30.0, 20.0, 20.0]),
    )

    np.testing.assert_array_equal(assignment.cell_id, np.array([0, 1, 5, 5, 5, -1, -1, -1]))
    np.testing.assert_array_equal(
        assignment.inside_grid_mask,
        np.array([True, True, True, True, True, False, False, False]),
    )
    np.testing.assert_array_equal(assignment.ix, np.array([0, 1, 2, 2, 2, -1, -1, -1]))
    np.testing.assert_array_equal(assignment.iy, np.array([0, 0, 1, 1, 1, 1, 0, 0]))
    assert assignment.qc == {
        'n_points': 8,
        'n_inside_grid': 5,
        'n_outside_grid': 3,
        'inside_grid_fraction': 0.625,
        'x_min_m': 100.0,
        'x_max_m': 130.0 + 0.5 * tolerance,
        'y_min_m': 20.0,
        'y_max_m': 30.0,
        'active_cell_count': 3,
        'inactive_cell_count': 3,
        'min_points_per_active_cell': 1,
        'median_points_per_active_cell': 1.0,
        'max_points_per_active_cell': 3,
    }


def test_assign_observation_midpoint_cells_uses_source_receiver_midpoint() -> None:
    grid = build_refraction_cell_grid(
        RefractionStaticRefractorCellOptions(
            number_of_cell_x=4,
            size_of_cell_x_m=25.0,
            x_coordinate_origin_m=0.0,
        )
    )

    assignment = assign_observation_midpoint_cells(
        grid,
        source_x_m=np.array([0.0, 10.0, 100.0]),
        source_y_m=np.array([1.0, 2.0, 3.0]),
        receiver_x_m=np.array([40.0, 90.0, 130.0]),
        receiver_y_m=np.array([5.0, 6.0, 7.0]),
    )

    np.testing.assert_allclose(assignment.x_m, [20.0, 50.0, 115.0])
    np.testing.assert_allclose(assignment.y_m, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(assignment.cell_id, np.array([0, 2, -1]))
    np.testing.assert_array_equal(assignment.inside_grid_mask, np.array([True, True, False]))
