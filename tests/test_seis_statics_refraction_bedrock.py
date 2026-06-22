from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_statics.refraction import (
    RefractionBedrockEstimationError,
    RefractionEndpointTable,
    RefractionStaticDesignMatrixError,
    RefractionStaticInputModel,
    RefractionStaticModelOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverError,
    RefractionStaticSolverOptions,
    estimate_global_bedrock_slowness_from_input_model,
    solve_refraction_static_least_squares,
)


def _solver_options() -> RefractionStaticSolverOptions:
    return RefractionStaticSolverOptions(
        half_intercept_damping_lambda=0.0,
        max_abs_half_intercept_time_ms=100.0,
        robust=RefractionStaticRobustOptions(enabled=False),
    )


def _global_model(
    *,
    mode: str = 'solve_global',
) -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode=mode,  # type: ignore[arg-type]
        bedrock_velocity_m_s=2500.0 if mode == 'fixed_global' else None,
        initial_bedrock_velocity_m_s=3000.0
        if mode == 'solve_global'
        else None,
        min_bedrock_velocity_m_s=1200.0,
        max_bedrock_velocity_m_s=6000.0,
    )


def _global_input_model() -> RefractionStaticInputModel:
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
        file_id='bedrock-unit',
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
        rejection_reason_sorted=np.asarray(
            ['ok', 'ok', 'ok', 'ok', 'ok', 'invalid_pick']
        ),
        qc={},
        endpoint_table=endpoint_table,
        metadata={},
    )


def test_global_bedrock_facade_matches_solver_result() -> None:
    input_model = _global_input_model()
    model = _global_model()
    solver_options = _solver_options()

    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=input_model,
        model=model,
        solver_options=solver_options,
        include_debug_objects=True,
    )
    solver_result = solve_refraction_static_least_squares(
        input_model=input_model,
        model=model,
        solver_options=solver_options,
    )

    assert result.bedrock_velocity_mode == 'solve_global'
    assert result.bedrock_velocity_m_s == pytest.approx(2500.0, abs=1.0e-6)
    assert result.bedrock_slowness_s_per_m == pytest.approx(1.0 / 2500.0)
    assert result.v2_m_s == result.bedrock_velocity_m_s
    assert result.bedrock_velocity_status == 'solved'
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.03, 0.05, 0.035, 0.045],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        result.residual_s_sorted,
        solver_result.residual_s_sorted,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, True, True, True, True, False],
    )
    np.testing.assert_array_equal(
        result.rejected_observation_mask_sorted,
        [False, False, False, False, False, False],
    )
    assert result.rms_residual_s == pytest.approx(solver_result.rms_residual_s)
    assert result.residual_max_abs_s < 1.0e-10
    assert result.qc['bedrock_velocity_m_s'] == pytest.approx(2500.0)
    assert result.qc['residual_statistics']['max_abs_s'] < 1.0e-10
    assert result.debug_solve_result is not None
    np.testing.assert_allclose(
        result.debug_solve_result.residual_s_sorted,
        solver_result.residual_s_sorted,
        atol=1.0e-12,
    )


def test_global_bedrock_facade_omits_debug_objects_by_default() -> None:
    result = estimate_global_bedrock_slowness_from_input_model(
        input_model=_global_input_model(),
        model=_global_model(),
        solver_options=_solver_options(),
    )

    assert result.debug_solve_result is None


def test_global_bedrock_facade_rejects_non_solve_global_model() -> None:
    with pytest.raises(
        RefractionBedrockEstimationError,
        match='bedrock_velocity_mode must be solve_global',
    ):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=_global_input_model(),
            model=_global_model(mode='fixed_global'),
            solver_options=_solver_options(),
        )


def test_global_bedrock_facade_preserves_invalid_data_domain_error() -> None:
    input_model = replace(
        _global_input_model(),
        pick_time_s_sorted=np.asarray(
            [0.265, np.nan, 0.325, 0.435, 0.425, 0.475],
            dtype=np.float64,
        ),
    )

    with pytest.raises(RefractionStaticDesignMatrixError, match='pick_time'):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=input_model,
            model=_global_model(),
            solver_options=_solver_options(),
        )


def test_global_bedrock_facade_preserves_solver_failure_domain_error() -> None:
    input_model = replace(
        _global_input_model(),
        n_traces=1,
        sorted_trace_index=np.asarray([0], dtype=np.int64),
        pick_time_s_sorted=np.asarray([0.25], dtype=np.float64),
        valid_pick_mask_sorted=np.asarray([True]),
        valid_observation_mask_sorted=np.asarray([True]),
        source_id_sorted=np.asarray([10], dtype=np.int64),
        receiver_id_sorted=np.asarray([20], dtype=np.int64),
        source_x_m_sorted=np.asarray([0.0]),
        source_y_m_sorted=np.asarray([0.0]),
        receiver_x_m_sorted=np.asarray([500.0]),
        receiver_y_m_sorted=np.asarray([0.0]),
        source_elevation_m_sorted=np.asarray([10.0]),
        receiver_elevation_m_sorted=np.asarray([12.0]),
        geometry_distance_m_sorted=np.asarray([500.0]),
        distance_m_sorted=np.asarray([500.0]),
        source_endpoint_key_sorted=np.asarray(['source:10']),
        receiver_endpoint_key_sorted=np.asarray(['receiver:20']),
        source_node_id_sorted=np.asarray([10], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20], dtype=np.int64),
        rejection_reason_sorted=np.asarray(['ok']),
        endpoint_table=RefractionEndpointTable(
            node_id=np.asarray([10, 20], dtype=np.int64),
            endpoint_id=np.asarray([100, 200], dtype=np.int64),
            x_m=np.asarray([0.0, 500.0], dtype=np.float64),
            y_m=np.zeros(2, dtype=np.float64),
            elevation_m=np.asarray([10.0, 12.0], dtype=np.float64),
            kind=np.asarray(['source', 'receiver']),
            pick_count=np.asarray([1, 1], dtype=np.int64),
        ),
    )

    with pytest.raises(RefractionStaticSolverError, match='solve_global'):
        estimate_global_bedrock_slowness_from_input_model(
            input_model=input_model,
            model=_global_model(),
            solver_options=_solver_options(),
        )
