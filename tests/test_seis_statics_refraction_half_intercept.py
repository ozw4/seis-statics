from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_statics.refraction import (
    RefractionEndpointTable,
    RefractionStaticInputModel,
    RefractionStaticModelOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
    build_refraction_half_intercept_result_from_bedrock_result,
    estimate_global_bedrock_slowness_from_input_model,
    estimate_refraction_half_intercept_from_input_model,
)


def _solver_options(
    *,
    robust: RefractionStaticRobustOptions | None = None,
    min_picks_per_node: int = 1,
) -> RefractionStaticSolverOptions:
    return RefractionStaticSolverOptions(
        min_picks_per_node=min_picks_per_node,
        half_intercept_damping_lambda=0.0,
        max_abs_half_intercept_time_ms=100.0,
        robust=robust or RefractionStaticRobustOptions(enabled=False),
    )


def _global_model(
    *,
    mode: str = 'solve_global',
    fixed_velocity: float | None = None,
) -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode=mode,  # type: ignore[arg-type]
        bedrock_velocity_m_s=fixed_velocity,
        initial_bedrock_velocity_m_s=3000.0
        if mode == 'solve_global'
        else None,
        min_bedrock_velocity_m_s=1200.0,
        max_bedrock_velocity_m_s=6000.0,
    )


def _global_input_model(
    *,
    all_picks_valid: bool = False,
    outlier_last_pick_s: float = 0.0,
) -> RefractionStaticInputModel:
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
    pick_time[-1] += outlier_last_pick_s
    valid_mask = (
        np.ones(6, dtype=bool)
        if all_picks_valid
        else np.asarray([True, True, True, True, True, False])
    )
    endpoint_table = RefractionEndpointTable(
        node_id=np.asarray([10, 20, 30, 40], dtype=np.int64),
        endpoint_id=np.asarray([100, 200, 300, 400], dtype=np.int64),
        x_m=np.asarray([0.0, 100.0, 500.0, 700.0], dtype=np.float64),
        y_m=np.zeros(4, dtype=np.float64),
        elevation_m=np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
        kind=np.asarray(['source', 'source', 'receiver', 'receiver']),
        pick_count=np.asarray([3, 3, 3, 3], dtype=np.int64),
    )
    return RefractionStaticInputModel(
        file_id='half-intercept-unit',
        n_traces=6,
        sorted_trace_index=np.arange(6, dtype=np.int64),
        pick_time_s_sorted=pick_time,
        valid_pick_mask_sorted=valid_mask,
        valid_observation_mask_sorted=valid_mask,
        source_id_sorted=source_node_id,
        receiver_id_sorted=receiver_node_id,
        source_x_m_sorted=np.asarray([0.0, 0.0, 100.0, 100.0, 0.0, 100.0]),
        source_y_m_sorted=np.zeros(6, dtype=np.float64),
        receiver_x_m_sorted=np.asarray([500.0, 700.0, 500.0, 700.0, 500.0, 700.0]),
        receiver_y_m_sorted=np.zeros(6, dtype=np.float64),
        source_elevation_m_sorted=np.full(6, 10.0, dtype=np.float64),
        receiver_elevation_m_sorted=np.full(6, 12.0, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=distance_m,
        offset_m_sorted=None,
        distance_m_sorted=distance_m,
        source_endpoint_key_sorted=np.asarray(
            [f'source:{value}' for value in source_node_id]
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{value}' for value in receiver_node_id]
        ),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        node_x_m=endpoint_table.x_m,
        node_y_m=endpoint_table.y_m,
        node_elevation_m=endpoint_table.elevation_m,
        node_kind=endpoint_table.kind,
        rejection_reason_sorted=np.where(valid_mask, 'ok', 'invalid_pick'),
        qc={},
        endpoint_table=endpoint_table,
        metadata={},
    )


def test_half_intercept_facade_assembles_node_endpoint_and_trace_arrays() -> None:
    input_model = _global_input_model()

    result = estimate_refraction_half_intercept_from_input_model(
        input_model=input_model,
        model=_global_model(),
        solver_options=_solver_options(),
        include_debug_objects=True,
    )

    assert result.debug_design is not None
    assert result.debug_solve_result is not None
    assert result.bedrock_velocity_mode == 'solve_global'
    assert result.bedrock_velocity_m_s == pytest.approx(2500.0, abs=1.0e-6)
    assert result.bedrock_slowness_s_per_m == pytest.approx(1.0 / 2500.0)
    assert result.v2_m_s == result.bedrock_velocity_m_s
    np.testing.assert_array_equal(result.node_id, [10, 20, 30, 40])
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.03, 0.05, 0.035, 0.045],
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(result.node_pick_count, [3, 2, 3, 2])
    np.testing.assert_array_equal(result.node_used_observation_count, [3, 2, 3, 2])
    np.testing.assert_array_equal(result.node_rejected_observation_count, [0, 0, 0, 0])

    np.testing.assert_array_equal(
        result.source_endpoint.endpoint_key,
        ['source:10', 'source:20'],
    )
    np.testing.assert_array_equal(result.source_endpoint.endpoint_id, [100, 200])
    np.testing.assert_array_equal(result.source_endpoint.node_id, [10, 20])
    np.testing.assert_array_equal(result.source_endpoint.pick_count, [3, 2])
    np.testing.assert_array_equal(
        result.source_endpoint.used_observation_count,
        [3, 2],
    )
    np.testing.assert_array_equal(
        result.receiver_endpoint.endpoint_key,
        ['receiver:30', 'receiver:40'],
    )
    np.testing.assert_array_equal(result.receiver_endpoint.endpoint_id, [300, 400])
    np.testing.assert_array_equal(result.receiver_endpoint.pick_count, [3, 2])
    np.testing.assert_array_equal(
        result.receiver_endpoint.used_observation_count,
        [3, 2],
    )

    np.testing.assert_allclose(
        result.source_half_intercept_time_s_sorted,
        [0.03, 0.03, 0.05, 0.05, 0.03, 0.05],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        result.receiver_half_intercept_time_s_sorted,
        [0.035, 0.045, 0.035, 0.045, 0.035, 0.045],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        result.trace_half_intercept_time_s_sorted,
        [0.065, 0.075, 0.085, 0.095, 0.065, 0.095],
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, True, True, True, True, False],
    )
    assert np.isnan(result.modeled_pick_time_s_sorted[-1])
    assert result.residual_max_abs_s < 1.0e-10
    assert result.qc['half_intercept']['source_endpoint_key'] == [
        'source:10',
        'source:20',
    ]
    assert result.qc['half_intercept']['node_pick_count'] == [3, 2, 3, 2]


def test_half_intercept_trace_indexed_solver_arrays_return_sorted_rows() -> None:
    input_model = replace(
        _global_input_model(),
        sorted_trace_index=np.asarray([2, 0, 5, 1, 3, 4], dtype=np.int64),
    )

    result = estimate_refraction_half_intercept_from_input_model(
        input_model=input_model,
        model=_global_model(),
        solver_options=_solver_options(),
        include_debug_objects=True,
    )

    assert result.debug_solve_result is not None
    np.testing.assert_array_equal(
        np.flatnonzero(result.debug_solve_result.used_observation_mask_sorted),
        [0, 1, 2, 3, 5],
    )
    np.testing.assert_array_equal(result.trace_index_sorted, [2, 0, 5, 1, 3, 4])
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, True, True, True, True, False],
    )
    np.testing.assert_array_equal(
        result.rejected_observation_mask_sorted,
        [False, False, False, False, False, False],
    )
    np.testing.assert_allclose(
        result.modeled_pick_time_s_sorted[:5],
        input_model.pick_time_s_sorted[:5],
        atol=1.0e-10,
    )
    assert np.isnan(result.modeled_pick_time_s_sorted[-1])
    np.testing.assert_allclose(result.residual_s_sorted[:5], 0.0, atol=1.0e-10)
    assert np.isnan(result.residual_s_sorted[-1])
    np.testing.assert_array_equal(result.node_used_observation_count, [3, 2, 3, 2])
    np.testing.assert_array_equal(
        result.source_endpoint.used_observation_count,
        [3, 2],
    )
    np.testing.assert_array_equal(
        result.receiver_endpoint.used_observation_count,
        [3, 2],
    )


def test_half_intercept_bedrock_convenience_uses_public_bedrock_result() -> None:
    input_model = _global_input_model()
    bedrock_result = estimate_global_bedrock_slowness_from_input_model(
        input_model=input_model,
        model=_global_model(),
        solver_options=_solver_options(),
    )

    result = build_refraction_half_intercept_result_from_bedrock_result(
        input_model=input_model,
        bedrock_result=bedrock_result,
    )

    assert result.bedrock_velocity_m_s == pytest.approx(
        bedrock_result.bedrock_velocity_m_s
    )
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        bedrock_result.node_half_intercept_time_s,
    )


def test_half_intercept_bedrock_convenience_matches_direct_non_identity_trace_order() -> None:
    input_model = replace(
        _global_input_model(),
        sorted_trace_index=np.asarray([2, 0, 5, 1, 3, 4], dtype=np.int64),
    )
    direct_result = estimate_refraction_half_intercept_from_input_model(
        input_model=input_model,
        model=_global_model(),
        solver_options=_solver_options(),
    )
    bedrock_result = estimate_global_bedrock_slowness_from_input_model(
        input_model=input_model,
        model=_global_model(),
        solver_options=_solver_options(),
    )

    result = build_refraction_half_intercept_result_from_bedrock_result(
        input_model=input_model,
        bedrock_result=bedrock_result,
    )

    np.testing.assert_allclose(
        result.modeled_pick_time_s_sorted,
        direct_result.modeled_pick_time_s_sorted,
        atol=1.0e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.residual_s_sorted,
        direct_result.residual_s_sorted,
        atol=1.0e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.residual_ms_sorted,
        direct_result.residual_ms_sorted,
        atol=1.0e-7,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        direct_result.used_observation_mask_sorted,
    )
    np.testing.assert_array_equal(
        result.rejected_observation_mask_sorted,
        direct_result.rejected_observation_mask_sorted,
    )
    np.testing.assert_array_equal(
        result.rejected_iteration_sorted,
        direct_result.rejected_iteration_sorted,
    )
    np.testing.assert_array_equal(
        result.node_rejected_observation_count,
        direct_result.node_rejected_observation_count,
    )
    np.testing.assert_array_equal(
        result.source_endpoint.used_observation_count,
        direct_result.source_endpoint.used_observation_count,
    )
    np.testing.assert_array_equal(
        result.receiver_endpoint.used_observation_count,
        direct_result.receiver_endpoint.used_observation_count,
    )
    assert result.qc['residual_statistics'] == pytest.approx(
        direct_result.qc['residual_statistics']
    )


def test_half_intercept_bedrock_convenience_can_include_debug_objects() -> None:
    bedrock_result = estimate_global_bedrock_slowness_from_input_model(
        input_model=_global_input_model(),
        model=_global_model(),
        solver_options=_solver_options(),
        include_debug_objects=True,
    )

    result = build_refraction_half_intercept_result_from_bedrock_result(
        input_model=_global_input_model(),
        bedrock_result=bedrock_result,
        include_debug_objects=True,
    )

    assert result.debug_solve_result is bedrock_result.debug_solve_result
    assert result.debug_design is bedrock_result.debug_solve_result.design


def test_half_intercept_facade_preserves_bedrock_velocity_validation() -> None:
    with pytest.raises(
        ValueError,
        match='bedrock_velocity_m_s must be greater',
    ):
        estimate_refraction_half_intercept_from_input_model(
            input_model=_global_input_model(),
            model=_global_model(mode='fixed_global', fixed_velocity=400.0),
            solver_options=_solver_options(),
        )


def test_half_intercept_robust_rejected_counts_match_solver_behavior() -> None:
    input_model = _global_input_model(
        all_picks_valid=True,
        outlier_last_pick_s=0.12,
    )

    result = estimate_refraction_half_intercept_from_input_model(
        input_model=input_model,
        model=_global_model(),
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

    assert result.robust_enabled is True
    assert result.robust_stop_reason == 'converged'
    assert result.n_rejected_observations == 1
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, True, True, True, True, False],
    )
    np.testing.assert_array_equal(
        result.rejected_observation_mask_sorted,
        [False, False, False, False, False, True],
    )
    np.testing.assert_array_equal(result.rejected_iteration_sorted, [-1, -1, -1, -1, -1, 0])
    np.testing.assert_array_equal(result.node_pick_count, [3, 3, 3, 3])
    np.testing.assert_array_equal(result.node_used_observation_count, [3, 2, 3, 2])
    np.testing.assert_array_equal(result.node_rejected_observation_count, [0, 1, 0, 1])
    np.testing.assert_array_equal(result.source_endpoint.pick_count, [3, 3])
    np.testing.assert_array_equal(
        result.source_endpoint.used_observation_count,
        [3, 2],
    )
    np.testing.assert_array_equal(
        result.source_endpoint.rejected_observation_count,
        [0, 1],
    )
    np.testing.assert_array_equal(result.receiver_endpoint.pick_count, [3, 3])
    np.testing.assert_array_equal(
        result.receiver_endpoint.used_observation_count,
        [3, 2],
    )
    np.testing.assert_array_equal(
        result.receiver_endpoint.rejected_observation_count,
        [0, 1],
    )
    assert result.qc['half_intercept']['node_rejected_observation_count'] == [
        0,
        1,
        0,
        1,
    ]


def test_half_intercept_low_fold_node_status_is_preserved() -> None:
    input_model = _global_input_model()
    input_model = replace(
        input_model,
        n_traces=3,
        sorted_trace_index=np.arange(3, dtype=np.int64),
        pick_time_s_sorted=np.asarray([0.20, 0.25, 0.35]),
        valid_pick_mask_sorted=np.asarray([True, True, True]),
        valid_observation_mask_sorted=np.asarray([True, True, True]),
        source_id_sorted=np.asarray([10, 10, 30]),
        receiver_id_sorted=np.asarray([20, 20, 20]),
        source_node_id_sorted=np.asarray([10, 10, 30]),
        receiver_node_id_sorted=np.asarray([20, 20, 20]),
        source_endpoint_key_sorted=np.asarray(['source:10', 'source:10', 'source:30']),
        receiver_endpoint_key_sorted=np.asarray(
            ['receiver:20', 'receiver:20', 'receiver:20']
        ),
        source_x_m_sorted=np.asarray([0.0, 0.0, 300.0]),
        source_y_m_sorted=np.zeros(3),
        receiver_x_m_sorted=np.asarray([200.0, 200.0, 200.0]),
        receiver_y_m_sorted=np.zeros(3),
        source_elevation_m_sorted=np.zeros(3),
        receiver_elevation_m_sorted=np.zeros(3),
        geometry_distance_m_sorted=np.asarray([500.0, 600.0, 700.0]),
        distance_m_sorted=np.asarray([500.0, 600.0, 700.0]),
        rejection_reason_sorted=np.asarray(['ok', 'ok', 'ok']),
        endpoint_table=RefractionEndpointTable(
            node_id=np.asarray([10, 20, 30], dtype=np.int64),
            endpoint_id=np.asarray([100, 200, 300], dtype=np.int64),
            x_m=np.asarray([0.0, 200.0, 300.0]),
            y_m=np.zeros(3),
            elevation_m=np.zeros(3),
            kind=np.asarray(['source', 'receiver', 'source']),
            pick_count=np.asarray([2, 3, 1], dtype=np.int64),
        ),
    )

    result = estimate_refraction_half_intercept_from_input_model(
        input_model=input_model,
        model=_global_model(),
        solver_options=_solver_options(min_picks_per_node=2),
        include_diagnostics=True,
    )

    np.testing.assert_array_equal(result.node_id, [10, 20, 30])
    assert result.node_solution_status.tolist()[2] == 'low_fold'
    assert np.isnan(result.node_half_intercept_time_s[2])
    assert result.source_endpoint.solution_status.tolist()[-1] == 'low_fold'
    assert result.trace_half_intercept_status_sorted.tolist()[2] == 'low_fold'
