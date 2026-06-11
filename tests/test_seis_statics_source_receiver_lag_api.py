from __future__ import annotations

import numpy as np
import pytest

from seis_statics.residual import (
    ResidualStaticLsmrOptions,
    ResidualStaticRobustOptions,
    delay_to_applied_shift,
    solve_source_receiver_statics,
)


def _lsmr_options() -> ResidualStaticLsmrOptions:
    return ResidualStaticLsmrOptions(
        atol=1.0e-12,
        btol=1.0e-12,
        conlim=1.0e12,
        maxiter=10000,
    )


def _grid(
    *,
    source_delay_s: np.ndarray | None = None,
    receiver_delay_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = (
        np.asarray([0.012, -0.006, 0.018, 0.004], dtype=np.float64)
        if source_delay_s is None
        else np.asarray(source_delay_s, dtype=np.float64)
    )
    receiver = (
        np.asarray([-0.008, 0.003, 0.017, -0.002, 0.009], dtype=np.float64)
        if receiver_delay_s is None
        else np.asarray(receiver_delay_s, dtype=np.float64)
    )
    source_ids = np.arange(101, 101 + source.size, dtype=np.int64)
    receiver_ids = np.arange(201, 201 + receiver.size, dtype=np.int64)
    source_index = np.repeat(np.arange(source.size, dtype=np.int64), receiver.size)
    receiver_index = np.tile(np.arange(receiver.size, dtype=np.int64), source.size)
    lag_s = source[source_index] + receiver[receiver_index]
    return (
        lag_s,
        source_ids[source_index],
        receiver_ids[receiver_index],
        source,
        receiver,
    )


def test_public_import_exports_source_receiver_solver() -> None:
    from seis_statics.residual import solve_source_receiver_statics as imported

    assert imported is solve_source_receiver_statics


def test_exact_recovery_of_synthetic_trace_delays() -> None:
    lag_s, source_id, receiver_id, source_delay, receiver_delay = _grid()

    result = solve_source_receiver_statics(
        lag_s=lag_s,
        source_id=source_id,
        receiver_id=receiver_id,
        robust=False,
        lsmr_options=_lsmr_options(),
    )

    expected_trace_delay = source_delay[source_id - 101] + receiver_delay[
        receiver_id - 201
    ]
    np.testing.assert_allclose(result.trace_delay_s, expected_trace_delay, atol=1e-10)
    np.testing.assert_allclose(result.residual_s, np.zeros_like(lag_s), atol=1e-10)
    assert result.robust_stop_reason == 'disabled'


def test_valid_mask_excludes_invalid_endpoint_ids_from_solver_universe() -> None:
    lag_s, source_id, receiver_id, _, _ = _grid()
    invalid_source_id = np.asarray([9001], dtype=np.int64)
    invalid_receiver_id = np.asarray([8001], dtype=np.int64)
    valid_mask = np.concatenate([np.ones_like(lag_s, dtype=bool), [False]])

    result = solve_source_receiver_statics(
        lag_s=np.concatenate([lag_s, np.asarray([np.nan], dtype=np.float64)]),
        source_id=np.concatenate([source_id, invalid_source_id]),
        receiver_id=np.concatenate([receiver_id, invalid_receiver_id]),
        valid_mask=valid_mask,
        robust=False,
        lsmr_options=_lsmr_options(),
    )

    assert invalid_source_id[0].item() not in result.source_unique_ids
    assert invalid_receiver_id[0].item() not in result.receiver_unique_ids
    np.testing.assert_array_equal(result.source_unique_ids, np.arange(101, 105))
    np.testing.assert_array_equal(result.receiver_unique_ids, np.arange(201, 206))
    assert result.minimum_data.n_sources == 4
    assert result.minimum_data.n_receivers == 5
    assert result.graph.n_components == 1
    assert result.used_mask[-1].item() is False
    assert np.isnan(result.residual_s[-1])


def test_gauge_freedom_recovers_source_receiver_after_alignment() -> None:
    source_delay = np.asarray([0.020, 0.014, 0.031, 0.026], dtype=np.float64)
    receiver_delay = np.asarray([0.011, 0.019, 0.024, 0.029, 0.033], dtype=np.float64)
    lag_s, source_id, receiver_id, _, _ = _grid(
        source_delay_s=source_delay,
        receiver_delay_s=receiver_delay,
    )

    result = solve_source_receiver_statics(
        lag_s=lag_s,
        source_id=source_id,
        receiver_id=receiver_id,
        robust=False,
        lsmr_options=_lsmr_options(),
    )

    source_gauge = float(np.mean(source_delay))
    np.testing.assert_allclose(
        result.source_delay_s,
        source_delay - source_gauge,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.receiver_delay_s,
        receiver_delay + source_gauge,
        atol=1e-10,
    )
    np.testing.assert_allclose(result.trace_delay_s, lag_s, atol=1e-10)


def test_weighted_least_squares_suppresses_low_weight_outlier() -> None:
    lag_s, source_id, receiver_id, _, _ = _grid()
    outlier_index = 7
    contaminated = lag_s.copy()
    contaminated[outlier_index] += 0.18
    weight = np.ones_like(lag_s)
    weight[outlier_index] = 1.0e-8
    weight[-1] = 0.0

    result = solve_source_receiver_statics(
        lag_s=contaminated,
        source_id=source_id,
        receiver_id=receiver_id,
        weight=weight,
        robust=False,
        lsmr_options=_lsmr_options(),
    )

    checked = np.ones_like(lag_s, dtype=bool)
    checked[[outlier_index, lag_s.size - 1]] = False
    np.testing.assert_allclose(result.trace_delay_s[checked], lag_s[checked], atol=1e-5)
    assert result.used_mask[outlier_index].item() is True
    assert result.used_mask[-1].item() is False
    assert result.minimum_data.n_zero_weight_observations == 1


def test_invalid_weight_raises_explicit_error() -> None:
    lag_s, source_id, receiver_id, _, _ = _grid()
    weight = np.ones_like(lag_s)
    weight[3] = -1.0

    with pytest.raises(ValueError, match='weight must be non-negative'):
        solve_source_receiver_statics(
            lag_s=lag_s,
            source_id=source_id,
            receiver_id=receiver_id,
            weight=weight,
            robust=False,
        )


def test_robust_solver_rejects_high_residual_outlier() -> None:
    lag_s, source_id, receiver_id, _, _ = _grid()
    sample_index = np.arange(lag_s.size, dtype=np.float64)
    contaminated = lag_s + 0.001 * np.sin(sample_index * 1.7)
    outlier_index = 11
    contaminated[outlier_index] += 0.07

    result = solve_source_receiver_statics(
        lag_s=contaminated,
        source_id=source_id,
        receiver_id=receiver_id,
        robust_options=ResidualStaticRobustOptions(method='mad', threshold=3.0),
        lsmr_options=_lsmr_options(),
    )

    assert result.n_rejected_total >= 1
    assert result.rejected_mask[outlier_index].item() is True
    assert result.used_mask[outlier_index].item() is False


def test_applied_shift_is_negative_trace_delay() -> None:
    lag_s, source_id, receiver_id, _, _ = _grid()

    result = solve_source_receiver_statics(
        lag_s=lag_s,
        source_id=source_id,
        receiver_id=receiver_id,
        robust=False,
        lsmr_options=_lsmr_options(),
    )

    np.testing.assert_allclose(result.applied_shift_s, -result.trace_delay_s)
    np.testing.assert_allclose(
        delay_to_applied_shift(result.trace_delay_s),
        result.applied_shift_s,
    )


def test_disconnected_source_receiver_graph_diagnostics_are_returned() -> None:
    result = solve_source_receiver_statics(
        lag_s=np.asarray([0.010, 0.020], dtype=np.float64),
        source_id=np.asarray([1, 2], dtype=np.int64),
        receiver_id=np.asarray([10, 20], dtype=np.int64),
        robust=False,
        lsmr_options=_lsmr_options(),
    )

    assert result.graph.n_components == 2
    np.testing.assert_array_equal(result.graph.source_component_index, [0, 1])
    np.testing.assert_array_equal(result.graph.receiver_component_index, [0, 1])
    np.testing.assert_array_equal(result.graph.component_observation_counts, [1, 1])
    assert result.minimum_data.rank_deficient_possible is True
