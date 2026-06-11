from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seis_statics.residual import ResidualStaticSolverInputs
from seis_statics.residual.solver import (
    ResidualStaticLsmrOptions,
    ResidualStaticStabilizationOptions,
    build_residual_static_observation_graph_summary,
    solve_residual_static_stabilized_least_squares,
    validate_minimum_residual_static_data,
)


INTERCEPT_S = 0.075
SLOWNESS_S_PER_OFFSET_UNIT = 2.0e-5
SOURCE_DELAY_S = np.asarray([-0.009, 0.003, 0.006], dtype=np.float64)
RECEIVER_DELAY_S = np.asarray([-0.004, -0.001, 0.002, 0.003], dtype=np.float64)


def _lsmr_options() -> ResidualStaticLsmrOptions:
    return ResidualStaticLsmrOptions(
        atol=1.0e-12,
        btol=1.0e-12,
        conlim=1.0e12,
        maxiter=10000,
    )


def _stabilization_options(
    **overrides: Any,
) -> ResidualStaticStabilizationOptions:
    payload: dict[str, Any] = {
        'gauge': 'zero_mean_source_receiver',
        'damping_lambda': 0.0,
        'min_valid_picks': 10,
        'min_picks_per_source': 1,
        'min_picks_per_receiver': 1,
        'max_abs_estimated_delay_ms': 250.0,
    }
    payload.update(overrides)
    return ResidualStaticStabilizationOptions(**payload)


def _grid_inputs(
    *,
    source_delay_s: np.ndarray = SOURCE_DELAY_S,
    receiver_delay_s: np.ndarray = RECEIVER_DELAY_S,
    intercept_s: float = INTERCEPT_S,
    slowness_s_per_offset_unit: float = SLOWNESS_S_PER_OFFSET_UNIT,
    moveout_model: str = 'none',
    **overrides: Any,
) -> ResidualStaticSolverInputs:
    source_unique_ids = np.arange(101, 101 + source_delay_s.size, dtype=np.int32)
    receiver_unique_ids = np.arange(201, 201 + receiver_delay_s.size, dtype=np.int32)
    n_sources = int(source_unique_ids.size)
    n_receivers = int(receiver_unique_ids.size)
    source_index = np.repeat(np.arange(n_sources, dtype=np.int64), n_receivers)
    receiver_index = np.tile(np.arange(n_receivers, dtype=np.int64), n_sources)
    n_traces = int(source_index.size)

    if n_sources == 3 and n_receivers == 4:
        abs_offset = np.asarray(
            [
                [100.0, 185.0, 260.0, 415.0],
                [150.0, 245.0, 395.0, 525.0],
                [230.0, 315.0, 475.0, 655.0],
            ],
            dtype=np.float64,
        ).reshape(-1)
    else:
        abs_offset = (
            100.0
            + 37.0 * source_index.astype(np.float64)
            + 83.0 * receiver_index.astype(np.float64)
        )
    offset = abs_offset * np.where(np.arange(n_traces) % 2 == 0, -1.0, 1.0)
    if moveout_model == 'linear_abs_offset':
        moveout_s = intercept_s + slowness_s_per_offset_unit * abs_offset
        offset_sorted: np.ndarray | None = offset
        abs_offset_sorted: np.ndarray | None = abs_offset
        offset_byte: int | None = 37
    else:
        moveout_s = np.full(n_traces, intercept_s, dtype=np.float64)
        offset_sorted = None
        abs_offset_sorted = None
        offset_byte = None

    pick_time_after_datum = (
        moveout_s + source_delay_s[source_index] + receiver_delay_s[receiver_index]
    )
    payload: dict[str, Any] = {
        'picks_time_s_sorted': pick_time_after_datum.copy(),
        'valid_pick_mask_sorted': np.ones(n_traces, dtype=bool),
        'pick_time_after_datum_s_sorted': pick_time_after_datum.copy(),
        'datum_trace_shift_s_sorted': np.zeros(n_traces, dtype=np.float64),
        'source_id_sorted': source_unique_ids[source_index],
        'receiver_id_sorted': receiver_unique_ids[receiver_index],
        'source_unique_ids': source_unique_ids,
        'receiver_unique_ids': receiver_unique_ids,
        'source_index_sorted': source_index,
        'receiver_index_sorted': receiver_index,
        'source_valid_pick_counts': np.bincount(
            source_index,
            minlength=n_sources,
        ).astype(np.int64),
        'receiver_valid_pick_counts': np.bincount(
            receiver_index,
            minlength=n_receivers,
        ).astype(np.int64),
        'offset_sorted': offset_sorted,
        'abs_offset_sorted': abs_offset_sorted,
        'key1_sorted': source_unique_ids[source_index].astype(np.int64),
        'key2_sorted': receiver_unique_ids[receiver_index].astype(np.int64),
        'source_elevation_m_sorted': np.zeros(n_traces, dtype=np.float64),
        'receiver_elevation_m_sorted': np.zeros(n_traces, dtype=np.float64),
        'dt': 0.004,
        'n_traces': n_traces,
        'n_samples': 96,
        'key1_byte': 189,
        'key2_byte': 193,
        'source_id_byte': 17,
        'receiver_id_byte': 13,
        'offset_byte': offset_byte,
        'moveout_model': moveout_model,
        'input_file_id': 'input-file',
        'datum_source_file_id': 'datum-source-file',
        'datum_job_id': 'datum-job',
        'pick_source_kind': 'batch_npz',
        'metadata': {'source': 'seis-statics-solver-test'},
    }
    payload.update(overrides)
    return ResidualStaticSolverInputs(**payload)


def test_stabilized_solver_recovers_trace_delays_for_connected_grid() -> None:
    inputs = _grid_inputs()
    result = solve_residual_static_stabilized_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        lsmr_options=_lsmr_options(),
    )

    expected_trace_delay = (
        SOURCE_DELAY_S[inputs.source_index_sorted]
        + RECEIVER_DELAY_S[inputs.receiver_index_sorted]
    )
    np.testing.assert_allclose(
        result.model_evaluation.estimated_trace_delay_s_sorted,
        expected_trace_delay,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.model_evaluation.modeled_pick_time_s_sorted,
        inputs.pick_time_after_datum_s_sorted,
        atol=1.0e-9,
    )
    assert result.rank_deficient_possible is False


def test_gauge_handling_verifies_predicted_trace_delays_not_raw_parts() -> None:
    source_delay_s = np.asarray([0.011, 0.018, 0.026], dtype=np.float64)
    receiver_delay_s = np.asarray([0.031, 0.036, 0.041, 0.047], dtype=np.float64)
    inputs = _grid_inputs(
        source_delay_s=source_delay_s,
        receiver_delay_s=receiver_delay_s,
    )

    result = solve_residual_static_stabilized_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        lsmr_options=_lsmr_options(),
    )

    assert not np.allclose(result.parameter_parts.source_delay_s, source_delay_s)
    assert not np.allclose(result.parameter_parts.receiver_delay_s, receiver_delay_s)
    np.testing.assert_allclose(
        result.model_evaluation.modeled_pick_time_s_sorted,
        inputs.pick_time_after_datum_s_sorted,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        np.mean(result.parameter_parts.source_delay_s),
        0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.mean(result.parameter_parts.receiver_delay_s),
        0.0,
        atol=1.0e-10,
    )


def test_underconstrained_source_and_receiver_checks_report_threshold() -> None:
    inputs = _grid_inputs()
    only_first_two_sources = inputs.source_index_sorted < 2
    with pytest.raises(ValueError, match='min_picks_per_source'):
        validate_minimum_residual_static_data(
            inputs,
            used_mask_sorted=only_first_two_sources,
            options=_stabilization_options(min_valid_picks=1),
        )

    only_first_receiver = inputs.receiver_index_sorted < 2
    with pytest.raises(ValueError, match='min_picks_per_receiver'):
        validate_minimum_residual_static_data(
            inputs,
            used_mask_sorted=only_first_receiver,
            options=_stabilization_options(min_valid_picks=1),
        )


def test_observation_graph_summary_reports_two_disconnected_components() -> None:
    inputs = _grid_inputs(
        source_delay_s=np.asarray([0.001, -0.001], dtype=np.float64),
        receiver_delay_s=np.asarray([0.002, -0.002], dtype=np.float64),
        valid_pick_mask_sorted=np.ones(4, dtype=bool),
    )
    used_mask = np.asarray([True, False, False, True], dtype=bool)

    graph = build_residual_static_observation_graph_summary(
        inputs,
        used_mask_sorted=used_mask,
    )

    assert graph.n_components == 2
    np.testing.assert_array_equal(graph.source_component_index, [0, 1])
    np.testing.assert_array_equal(graph.receiver_component_index, [0, 1])
    np.testing.assert_array_equal(graph.component_observation_counts, [1, 1])
    np.testing.assert_array_equal(graph.component_source_counts, [1, 1])
    np.testing.assert_array_equal(graph.component_receiver_counts, [1, 1])


def test_linear_abs_offset_solves_known_intercept_and_slowness() -> None:
    inputs = _grid_inputs(
        source_delay_s=np.zeros(3, dtype=np.float64),
        receiver_delay_s=np.zeros(4, dtype=np.float64),
        moveout_model='linear_abs_offset',
    )

    result = solve_residual_static_stabilized_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        lsmr_options=_lsmr_options(),
    )

    assert result.parameter_parts.intercept_s == pytest.approx(INTERCEPT_S, abs=1e-9)
    assert result.parameter_parts.slowness_s_per_offset_unit == pytest.approx(
        SLOWNESS_S_PER_OFFSET_UNIT,
        abs=1e-11,
    )
    np.testing.assert_allclose(
        result.model_evaluation.estimated_trace_delay_s_sorted,
        np.zeros(inputs.n_traces, dtype=np.float64),
        atol=1.0e-9,
    )
