from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse

from seis_statics.time_term import (
    TimeTermDesignMatrix,
    TimeTermRobustOptions,
    TimeTermSparseSolverOptions,
    compose_time_term_applied_shifts,
    solve_time_term_robust_least_squares,
    solve_time_term_sparse_least_squares,
)


def _grid_design(
    *,
    outlier_index: int | None = None,
    outlier_s: float = 0.0,
    **overrides: Any,
) -> TimeTermDesignMatrix:
    n_nodes = 5
    source_node = np.repeat(np.arange(n_nodes, dtype=np.int64), n_nodes)
    receiver_node = np.tile(np.arange(n_nodes, dtype=np.int64), n_nodes)
    n_traces = int(source_node.size)
    true_node_delay = np.asarray([-0.006, -0.002, 0.001, 0.003, 0.004], dtype=np.float64)
    row_data = true_node_delay[source_node] + true_node_delay[receiver_node]
    if outlier_index is not None:
        row_data = row_data.copy()
        row_data[outlier_index] += outlier_s

    matrix = _matrix(source_node, receiver_node, n_observations=n_traces, n_nodes=n_nodes)
    payload: dict[str, Any] = {
        'matrix': matrix,
        'data_s': row_data,
        'n_traces': n_traces,
        'n_observations': n_traces,
        'n_nodes': n_nodes,
        'used_trace_mask_sorted': np.ones(n_traces, dtype=bool),
        'row_trace_index_sorted': np.arange(n_traces, dtype=np.int64),
        'trace_to_row_index_sorted': np.arange(n_traces, dtype=np.int64),
        'source_node_id_sorted': source_node.copy(),
        'receiver_node_id_sorted': receiver_node.copy(),
        'row_source_node_id': source_node.copy(),
        'row_receiver_node_id': receiver_node.copy(),
        'row_pick_time_after_static_s': row_data.copy(),
        'row_moveout_time_s': np.zeros(n_traces, dtype=np.float64),
        'row_data_s': row_data.copy(),
        'source_observation_count_by_node': np.bincount(
            source_node,
            minlength=n_nodes,
        ).astype(np.int64),
        'receiver_observation_count_by_node': np.bincount(
            receiver_node,
            minlength=n_nodes,
        ).astype(np.int64),
        'total_observation_count_by_node': (
            np.bincount(source_node, minlength=n_nodes)
            + np.bincount(receiver_node, minlength=n_nodes)
        ).astype(np.int64),
    }
    payload.update(overrides)
    return TimeTermDesignMatrix(**payload)


def _matrix(
    source_node: np.ndarray,
    receiver_node: np.ndarray,
    *,
    n_observations: int,
    n_nodes: int,
) -> sparse.csr_matrix:
    row_indices = np.repeat(np.arange(n_observations, dtype=np.int64), 2)
    col_indices = np.empty(n_observations * 2, dtype=np.int64)
    col_indices[0::2] = source_node
    col_indices[1::2] = receiver_node
    data = np.ones(n_observations * 2, dtype=np.float64)
    out = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_observations, n_nodes),
        dtype=np.float64,
    ).tocsr()
    out.sum_duplicates()
    out.sort_indices()
    return out


def _options(**overrides: Any) -> TimeTermSparseSolverOptions:
    payload: dict[str, Any] = {
        'damping_lambda': 0.0,
        'gauge': 'mean_zero',
        'solver': 'lsmr',
        'atol': 1.0e-12,
        'btol': 1.0e-12,
        'conlim': 1.0e12,
        'maxiter': 1000,
        'max_abs_node_time_term_ms': 1000.0,
        'max_abs_estimated_trace_delay_ms': 1000.0,
    }
    payload.update(overrides)
    return TimeTermSparseSolverOptions(**payload)


def test_robust_disabled_path_equals_one_sparse_solve() -> None:
    design = _grid_design(outlier_index=13, outlier_s=0.04)
    sparse_result = solve_time_term_sparse_least_squares(design, options=_options())
    robust_result = solve_time_term_robust_least_squares(
        design,
        sparse_solver_options=_options(),
        robust_options=TimeTermRobustOptions(enabled=False),
    )

    assert robust_result.stop_reason == 'disabled'
    assert robust_result.n_rejected_total == 0
    assert robust_result.iteration_summaries == ()
    assert robust_result.initial_solver_result is robust_result.final_solver_result
    np.testing.assert_array_equal(
        robust_result.final_used_trace_mask_sorted,
        sparse_result.used_trace_mask_sorted,
    )
    np.testing.assert_allclose(
        robust_result.final_solver_result.row_residual_after_s,
        sparse_result.row_residual_after_s,
        atol=1.0e-12,
    )


@pytest.mark.parametrize('method', ['mad', 'sigma'])
def test_robust_rejection_removes_synthetic_large_outlier(method: str) -> None:
    result = solve_time_term_robust_least_squares(
        _grid_design(outlier_index=13, outlier_s=0.04),
        sparse_solver_options=_options(),
        robust_options=TimeTermRobustOptions(method=method, threshold=3.0),
    )

    assert result.stop_reason == 'zero_scale'
    assert result.n_rejected_total == 1
    assert result.rejected_trace_mask_sorted[13].item() is True
    assert result.final_used_trace_mask_sorted[13].item() is False
    assert result.rejected_iteration_sorted[13] == 0
    assert result.iteration_summaries[0].n_rejected_this_iteration == 1


def test_min_used_fraction_prevents_rejecting_too_many_observations() -> None:
    with pytest.raises(ValueError, match='min_used_fraction'):
        solve_time_term_robust_least_squares(
            _grid_design(outlier_index=13, outlier_s=0.04),
            sparse_solver_options=_options(),
            robust_options=TimeTermRobustOptions(
                method='mad',
                threshold=3.0,
                min_used_fraction=1.0,
            ),
        )


def test_zero_scale_stop_reason_reported_for_perfectly_fitted_data() -> None:
    result = solve_time_term_robust_least_squares(
        _grid_design(),
        sparse_solver_options=_options(),
        robust_options=TimeTermRobustOptions(method='mad'),
    )

    assert result.stop_reason == 'zero_scale'
    assert result.n_rejected_total == 0
    assert len(result.iteration_summaries) == 1
    assert result.iteration_summaries[0].stop_reason == 'zero_scale'
    assert result.iteration_summaries[0].residual_scale_s <= 1.0e-12


def test_rejected_trace_keeps_nan_applied_shift_policy() -> None:
    result = solve_time_term_robust_least_squares(
        _grid_design(outlier_index=13, outlier_s=0.04),
        sparse_solver_options=_options(),
        robust_options=TimeTermRobustOptions(method='mad', threshold=3.0),
    )

    delay = result.final_solver_result.estimated_trace_time_term_delay_s_sorted
    shifts = compose_time_term_applied_shifts(
        trace_time_term_delay_s_sorted=delay,
        datum_applied_shift_s_sorted=np.zeros(delay.shape[0], dtype=np.float64),
        residual_applied_shift_s_sorted=np.zeros(delay.shape[0], dtype=np.float64),
    )

    assert np.isnan(delay[13])
    assert np.all(np.isfinite(delay[result.final_used_trace_mask_sorted]))
    assert shifts.valid_shift_mask_sorted[13].item() is False
    assert np.isnan(shifts.final_applied_shift_s_sorted[13])
