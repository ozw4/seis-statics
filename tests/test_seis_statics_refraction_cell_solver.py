from __future__ import annotations

import numpy as np

from seis_statics.refraction import (
    LOW_FOLD_CELL_VELOCITY_STATUS,
    RefractionStaticModelOptions,
    RefractionStaticRefractorCellOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
    build_refraction_static_design_matrix_from_arrays,
    solve_refraction_static_design_least_squares,
)


def _solver_options(**overrides: object) -> RefractionStaticSolverOptions:
    values: dict[str, object] = {
        'damping': 0.0,
        'max_abs_half_intercept_time_ms': 100.0,
        'robust': RefractionStaticRobustOptions(enabled=False),
    }
    values.update(overrides)
    return RefractionStaticSolverOptions(**values)


def _cell_model(
    *,
    n_cells: int = 2,
    smoothing_weight: float = 0.0,
    smoothing_reference_distance_m: float | None = None,
    initial_velocity: float = 2600.0,
) -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode='solve_cell',
        initial_bedrock_velocity_m_s=initial_velocity,
        min_bedrock_velocity_m_s=1200.0,
        max_bedrock_velocity_m_s=6000.0,
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=n_cells,
            size_of_cell_x_m=100.0,
            x_coordinate_origin_m=0.0,
            min_observations_per_cell=1,
            velocity_smoothing_weight=smoothing_weight,
            smoothing_reference_distance_m=smoothing_reference_distance_m,
        ),
    )


def _cell_synthetic_arrays() -> tuple[np.ndarray, ...]:
    source_node_id = np.asarray([10, 10, 20, 20, 10, 20], dtype=np.int64)
    receiver_node_id = np.asarray([30, 40, 30, 40, 30, 40], dtype=np.int64)
    distance_m = np.asarray([500.0, 700.0, 600.0, 850.0, 900.0, 950.0])
    midpoint_cell_id = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    t1_by_node = {
        10: 0.03,
        20: 0.05,
        30: 0.035,
        40: 0.045,
    }
    slowness_by_cell = {
        0: 1.0 / 2400.0,
        1: 1.0 / 3000.0,
    }
    pick_time = np.asarray(
        [
            t1_by_node[int(src)]
            + t1_by_node[int(rec)]
            + dist * slowness_by_cell[int(cell)]
            for src, rec, dist, cell in zip(
                source_node_id,
                receiver_node_id,
                distance_m,
                midpoint_cell_id,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    valid_mask = np.ones(source_node_id.shape, dtype=bool)
    return source_node_id, receiver_node_id, distance_m, midpoint_cell_id, pick_time, valid_mask


def _cell_design(
    *,
    midpoint_cell_id: np.ndarray | None = None,
    min_observations_per_cell: int = 1,
    n_total_cells: int = 2,
) -> object:
    source_node_id, receiver_node_id, distance_m, default_cell_id, pick_time, valid_mask = (
        _cell_synthetic_arrays()
    )
    return build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_cell',
        midpoint_cell_id_sorted=(
            default_cell_id if midpoint_cell_id is None else midpoint_cell_id
        ),
        n_total_cells=n_total_cells,
        number_of_cell_x=n_total_cells,
        number_of_cell_y=1,
        cell_assignment_mode='midpoint',
        min_observations_per_cell=min_observations_per_cell,
        n_traces=int(pick_time.shape[0]),
    )


def test_refraction_cell_solver_recovers_cell_v2_and_row_midpoint_v2() -> None:
    design = _cell_design()

    result = solve_refraction_static_design_least_squares(
        design,
        model=_cell_model(),
        solver_options=_solver_options(),
    )

    assert result.bedrock_velocity_mode == 'solve_cell'
    assert result.bedrock_velocity_status == 'cell'
    np.testing.assert_array_equal(result.cell_id, [0, 1])
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        [2400.0, 3000.0],
        atol=1.0e-3,
    )
    np.testing.assert_allclose(result.cell_v2_m_s, result.cell_bedrock_velocity_m_s)
    np.testing.assert_allclose(
        result.row_midpoint_bedrock_velocity_m_s,
        [2400.0, 2400.0, 2400.0, 3000.0, 3000.0, 3000.0],
        atol=1.0e-3,
    )
    np.testing.assert_allclose(
        result.row_midpoint_v2_m_s,
        result.row_midpoint_bedrock_velocity_m_s,
    )
    np.testing.assert_allclose(result.row_residual_s, 0.0, atol=1.0e-8)
    assert set(result.cell_velocity_status.tolist()) == {'solved'}
    assert result.system.n_smoothing_rows == 0
    assert result.qc['cell_velocity_status_counts'] == {'solved': 2}


def test_refraction_cell_solver_smoothing_pulls_neighbor_slowness_together() -> None:
    design = _cell_design()

    unsmoothed = solve_refraction_static_design_least_squares(
        design,
        model=_cell_model(smoothing_weight=0.0),
        solver_options=_solver_options(),
    )
    smoothed = solve_refraction_static_design_least_squares(
        design,
        model=_cell_model(
            smoothing_weight=1.0,
            smoothing_reference_distance_m=700.0,
        ),
        solver_options=_solver_options(),
    )

    unsmoothed_delta = abs(
        unsmoothed.cell_bedrock_slowness_s_per_m[0]
        - unsmoothed.cell_bedrock_slowness_s_per_m[1]
    )
    smoothed_delta = abs(
        smoothed.cell_bedrock_slowness_s_per_m[0]
        - smoothed.cell_bedrock_slowness_s_per_m[1]
    )
    assert smoothed_delta < unsmoothed_delta
    assert smoothed.system.n_smoothing_rows == 1
    assert smoothed.system.smoothing_rows is not None
    assert smoothed.system.smoothing_rows.reference_distance_m == 700.0
    assert smoothed.qc['cell_smoothing']['n_cell_smoothing_edges'] == 1


def test_refraction_cell_solver_marks_low_fold_and_inactive_cells() -> None:
    design = _cell_design(
        midpoint_cell_id=np.asarray([0, 0, 0, 1, 2, 2], dtype=np.int64),
        min_observations_per_cell=2,
        n_total_cells=4,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_cell_model(n_cells=4),
        solver_options=_solver_options(damping=1.0e-8),
    )

    np.testing.assert_array_equal(result.cell_id, [0, 1, 2, 3])
    assert result.cell_velocity_status[0] == 'solved'
    assert result.cell_velocity_status[1] == LOW_FOLD_CELL_VELOCITY_STATUS
    assert result.cell_velocity_status[2] == 'solved'
    assert result.cell_velocity_status[3] == 'inactive'
    assert np.isnan(result.cell_bedrock_velocity_m_s[1])
    assert np.isnan(result.cell_bedrock_velocity_m_s[3])
    np.testing.assert_array_equal(result.cell_observation_count, [3, 1, 2, 0])
    np.testing.assert_array_equal(result.row_midpoint_cell_id, [0, 0, 0, 2, 2])
