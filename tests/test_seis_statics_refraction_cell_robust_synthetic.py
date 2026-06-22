from __future__ import annotations

import numpy as np
import pytest

from seis_statics.refraction import (
    RefractionStaticModelOptions,
    RefractionStaticRefractorCellOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
    build_refraction_static_design_matrix_from_arrays,
    build_refraction_static_solver_system,
    solve_refraction_static_design_least_squares,
)
import seis_statics.refraction.solver as solver_module


def _solver_options(**overrides: object) -> RefractionStaticSolverOptions:
    values: dict[str, object] = {
        'half_intercept_damping_lambda': 0.0,
        'max_abs_half_intercept_time_ms': 100.0,
        'robust': RefractionStaticRobustOptions(enabled=True),
    }
    values.update(overrides)
    return RefractionStaticSolverOptions(**values)


def _cell_model() -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode='solve_cell',
        initial_bedrock_velocity_m_s=2600.0,
        min_bedrock_velocity_m_s=1200.0,
        max_bedrock_velocity_m_s=6000.0,
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=2,
            size_of_cell_x_m=100.0,
            x_coordinate_origin_m=0.0,
            min_observations_per_cell=1,
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


def test_refraction_cell_robust_synthetic_rejects_outlier_and_preserves_cells() -> None:
    (
        source_node_id,
        receiver_node_id,
        distance_m,
        midpoint_cell_id,
        pick_time,
        valid_mask,
    ) = _cell_synthetic_arrays()
    pick_time = pick_time.copy()
    pick_time[5] += 0.12
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_cell',
        midpoint_cell_id_sorted=midpoint_cell_id,
        n_total_cells=2,
        number_of_cell_x=2,
        number_of_cell_y=1,
        cell_assignment_mode='midpoint',
        min_observations_per_cell=1,
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_cell_model(),
        solver_options=_solver_options(
            robust=RefractionStaticRobustOptions(
                enabled=True,
                method='mad',
                threshold=2.0,
                scale_floor_ms=0.0,
                max_iterations=5,
                min_used_fraction=0.5,
                min_used_observations=1,
            )
        ),
    )

    assert result.robust_stop_reason == 'safe_rejection'
    assert result.n_rejected_observations == 1
    np.testing.assert_array_equal(
        result.rejected_observation_mask_sorted,
        [False, False, False, False, False, True],
    )
    assert result.robust_iteration_summaries[0].n_rejected_this_iteration == 1
    assert result.robust_iteration_summaries[-1].stop_reason == 'safe_rejection'
    np.testing.assert_allclose(
        result.cell_bedrock_velocity_m_s,
        [2400.0, 3000.0],
        atol=1.0e-3,
    )
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.03, 0.05, 0.035, 0.045],
        atol=1.0e-7,
    )
    assert result.qc['bedrock_velocity_status'] == 'cell'
    assert result.qc['n_final_used_observations'] == 5
    assert result.qc['robust_iterations'][0]['n_rejected_this_iteration'] == 1
    np.testing.assert_array_equal(result.cell_observation_count, [3, 2])
    assert result.qc['design_matrix']['cell_observation_count'] == [3, 2]
    assert result.qc['design_matrix']['min_observations_per_active_cell'] == 2
    assert result.qc['design_matrix']['n_observations_used'] == 5
    assert result.rms_residual_s == pytest.approx(0.0, abs=1.0e-8)


def test_refraction_cell_robust_safety_uses_design_min_cell_fold() -> None:
    source_node_id = np.asarray([10, 10, 20, 20, 10, 20], dtype=np.int64)
    receiver_node_id = np.asarray([30, 40, 30, 40, 30, 40], dtype=np.int64)
    distance_m = np.asarray([500.0, 700.0, 600.0, 850.0, 900.0, 950.0])
    midpoint_cell_id = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
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
    pick_time[5] += 1.0
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(source_node_id.shape, dtype=bool),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_cell',
        midpoint_cell_id_sorted=midpoint_cell_id,
        n_total_cells=2,
        number_of_cell_x=2,
        number_of_cell_y=1,
        cell_assignment_mode='midpoint',
        min_observations_per_cell=2,
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_cell_model(),
        solver_options=_solver_options(
            robust=RefractionStaticRobustOptions(
                enabled=True,
                method='mad',
                threshold=1.0,
                scale_floor_ms=0.0,
                max_iterations=5,
                min_used_fraction=0.5,
                min_used_observations=1,
            )
        ),
    )

    assert result.robust_stop_reason == 'safe_rejection'
    assert result.n_rejected_observations == 0
    assert result.robust_iteration_summaries[0].n_rejected_this_iteration == 0
    np.testing.assert_array_equal(result.cell_observation_count, [4, 2])
    assert result.qc['design_matrix']['min_observations_per_active_cell'] == 2


def test_refraction_cell_robust_rejection_safety_refuses_slowness_rank_loss() -> None:
    source_node_id = np.asarray([10, 10, 20, 20], dtype=np.int64)
    receiver_node_id = np.asarray([30, 30, 40, 40], dtype=np.int64)
    distance_m = np.asarray([500.0, 600.0, 500.0, 600.0], dtype=np.float64)
    midpoint_cell_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
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
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(4, dtype=bool),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_cell',
        midpoint_cell_id_sorted=midpoint_cell_id,
        n_total_cells=2,
        number_of_cell_x=2,
        number_of_cell_y=1,
        cell_assignment_mode='midpoint',
        min_observations_per_cell=1,
        min_observations_per_node=1,
        n_traces=4,
    )
    row_used_mask = np.asarray([True, False, True, True])
    system = build_refraction_static_solver_system(
        design,
        model=_cell_model(),
        solver_options=_solver_options(min_picks_per_node=1),
    )

    assert solver_module._node_coverage_is_safe(
        design,
        row_used_mask=row_used_mask,
    )
    assert solver_module._cell_coverage_is_safe(
        design,
        row_used_mask=row_used_mask,
    )
    assert not solver_module._robust_row_mask_is_safe(
        design=design,
        system=system,
        initial_row_mask=np.ones(4, dtype=bool),
        row_used_mask=row_used_mask,
        min_used_fraction=0.5,
        min_used_observations=1,
    )


def test_refraction_cell_robust_rejection_uses_verified_rebuild_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        source_node_id,
        receiver_node_id,
        distance_m,
        midpoint_cell_id,
        pick_time,
        valid_mask,
    ) = _cell_synthetic_arrays()
    pick_time = pick_time.copy()
    pick_time[5] += 0.12
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_cell',
        midpoint_cell_id_sorted=midpoint_cell_id,
        n_total_cells=2,
        number_of_cell_x=2,
        number_of_cell_y=1,
        cell_assignment_mode='midpoint',
        min_observations_per_cell=1,
        n_traces=6,
    )
    original_run_lsq_linear = solver_module._run_lsq_linear
    solver_quality: list[dict[str, object]] = []
    observation_rows: list[int] = []

    def wrapped_run_lsq_linear(
        system: solver_module.RefractionStaticSolveSystem,
    ) -> object:
        result = original_run_lsq_linear(system)
        solver_quality.append(dict(result.refraction_solver_quality))
        observation_rows.append(system.n_observation_rows)
        return result

    monkeypatch.setattr(solver_module, '_run_lsq_linear', wrapped_run_lsq_linear)

    result = solve_refraction_static_design_least_squares(
        design,
        model=_cell_model(),
        solver_options=_solver_options(
            robust=RefractionStaticRobustOptions(
                enabled=True,
                method='mad',
                threshold=2.0,
                scale_floor_ms=0.0,
                max_iterations=5,
                min_used_fraction=0.5,
                min_used_observations=1,
            )
        ),
    )

    assert result.robust_stop_reason == 'safe_rejection'
    assert result.n_rejected_observations == 1
    assert len(solver_quality) >= 2
    assert all(quality['verified'] is True for quality in solver_quality)
    assert observation_rows[-1] == result.system.n_observation_rows
    assert result.qc['n_final_observation_rows'] == result.system.n_observation_rows
    assert result.qc['solver_quality']['verified'] is True
    assert result.qc['solver_quality'] == solver_quality[-1]
