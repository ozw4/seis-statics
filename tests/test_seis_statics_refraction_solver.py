from __future__ import annotations

import numpy as np
import pytest

from seis_statics.refraction import (
    RefractionStaticModelOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverError,
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


def _model(
    *,
    mode: str,
    fixed_velocity: float | None = None,
    min_velocity: float = 1200.0,
    max_velocity: float = 6000.0,
    initial_velocity: float = 3000.0,
) -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode=mode,  # type: ignore[arg-type]
        bedrock_velocity_m_s=fixed_velocity,
        initial_bedrock_velocity_m_s=(
            None if mode == 'fixed_global' else initial_velocity
        ),
        min_bedrock_velocity_m_s=min_velocity,
        max_bedrock_velocity_m_s=max_velocity,
    )


def _known_global_arrays() -> tuple[np.ndarray, ...]:
    source_node_id = np.asarray([10, 10, 20, 20, 10, 20], dtype=np.int64)
    receiver_node_id = np.asarray([30, 40, 30, 40, 30, 40], dtype=np.int64)
    distance_m = np.asarray([500.0, 700.0, 600.0, 850.0, 900.0, 950.0])
    true_t1_by_node = {
        10: 0.03,
        20: 0.05,
        30: 0.035,
        40: 0.045,
    }
    true_slowness = 1.0 / 2500.0
    pick_time = np.asarray(
        [
            true_t1_by_node[int(src)]
            + true_t1_by_node[int(rec)]
            + dist * true_slowness
            for src, rec, dist in zip(
                source_node_id,
                receiver_node_id,
                distance_m,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    valid_mask = np.asarray([True, True, True, True, True, False])
    return source_node_id, receiver_node_id, distance_m, pick_time, valid_mask


def test_refraction_solver_solve_global_matches_known_parameters_and_residual() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_global',
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='solve_global'),
        solver_options=_solver_options(),
    )

    assert result.bedrock_velocity_mode == 'solve_global'
    assert result.bedrock_velocity_m_s == pytest.approx(2500.0, abs=1.0e-6)
    assert result.bedrock_slowness_s_per_m == pytest.approx(1.0 / 2500.0)
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.03, 0.05, 0.035, 0.045],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(result.row_modeled_pick_time_s, pick_time[:5])
    np.testing.assert_allclose(result.row_residual_s, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(result.modeled_pick_time_s_sorted[:5], pick_time[:5])
    assert np.isnan(result.modeled_pick_time_s_sorted[5])
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, True, True, True, True, False],
    )
    assert set(result.node_solution_status.tolist()) == {'solved'}
    assert result.system.n_gauge_rows == 1
    assert result.qc['solver_name'] == 'lsq_linear'


def test_refraction_solver_fixed_global_uses_fixed_velocity_distance_term() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=2500.0),
        solver_options=_solver_options(),
    )

    assert result.bedrock_velocity_mode == 'fixed_global'
    assert result.bedrock_velocity_status == 'fixed'
    assert result.bedrock_velocity_m_s == 2500.0
    assert result.bedrock_slowness_s_per_m == 1.0 / 2500.0
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.03, 0.05, 0.035, 0.045],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(result.row_modeled_pick_time_s, pick_time[:5])
    np.testing.assert_allclose(result.row_residual_s, 0.0, atol=1.0e-10)


def test_refraction_solver_marks_global_velocity_bound_status() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_global',
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(
            mode='solve_global',
            min_velocity=1200.0,
            max_velocity=2400.0,
            initial_velocity=2000.0,
        ),
        solver_options=_solver_options(),
    )

    assert result.bedrock_velocity_m_s == pytest.approx(2400.0)
    assert result.bedrock_velocity_status == 'clipped_upper'


def test_refraction_solver_rejects_unsupported_cell_and_robust_paths() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_global',
        n_traces=6,
    )

    with pytest.raises(RefractionStaticSolverError, match='robust'):
        solve_refraction_static_design_least_squares(
            design,
            model=_model(mode='solve_global'),
            solver_options=RefractionStaticSolverOptions(),
        )

    cell_design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_cell',
        midpoint_cell_id_sorted=np.asarray([0, 0, 0, 0, 0, 0]),
        n_total_cells=1,
        number_of_cell_x=1,
        number_of_cell_y=1,
        cell_assignment_mode='midpoint',
        n_traces=6,
    )
    with pytest.raises(RefractionStaticSolverError, match='solve_cell'):
        solve_refraction_static_design_least_squares(
            cell_design,
            model=_model(mode='solve_global'),
            solver_options=_solver_options(),
        )
