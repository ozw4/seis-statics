from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from seis_statics.time_term import (
    TimeTermDesignMatrix,
    TimeTermSparseSolverOptions,
    build_time_term_solver_system,
    solve_time_term_sparse_least_squares,
    summarize_time_term_sparse_solver_result,
)

TRUE_NODE_TIME_TERM_S = np.asarray([0.010, -0.002, 0.006], dtype=np.float64)
SOURCE_NODE_ID_SORTED = np.asarray([0, 0, 1, 2, 1], dtype=np.int64)
RECEIVER_NODE_ID_SORTED = np.asarray([1, 2, 2, 2, 1], dtype=np.int64)
ROW_SOURCE_NODE_ID = SOURCE_NODE_ID_SORTED[:4].copy()
ROW_RECEIVER_NODE_ID = RECEIVER_NODE_ID_SORTED[:4].copy()
DATA_S = np.asarray([0.008, 0.016, 0.004, 0.012], dtype=np.float64)
OBSERVATION_MATRIX = sparse.csr_matrix(
    [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0],
    ],
    dtype=np.float64,
)


def _design(**overrides: Any) -> TimeTermDesignMatrix:
    payload: dict[str, Any] = {
        'matrix': OBSERVATION_MATRIX.copy(),
        'data_s': DATA_S.copy(),
        'n_traces': 5,
        'n_observations': 4,
        'n_nodes': 3,
        'used_trace_mask_sorted': np.asarray([True, True, True, True, False]),
        'row_trace_index_sorted': np.arange(4, dtype=np.int64),
        'trace_to_row_index_sorted': np.asarray([0, 1, 2, 3, -1], dtype=np.int64),
        'source_node_id_sorted': SOURCE_NODE_ID_SORTED.copy(),
        'receiver_node_id_sorted': RECEIVER_NODE_ID_SORTED.copy(),
        'row_source_node_id': ROW_SOURCE_NODE_ID.copy(),
        'row_receiver_node_id': ROW_RECEIVER_NODE_ID.copy(),
        'row_pick_time_after_static_s': DATA_S.copy(),
        'row_moveout_time_s': np.zeros(4, dtype=np.float64),
        'row_data_s': DATA_S.copy(),
        'source_observation_count_by_node': np.asarray([2, 1, 1], dtype=np.int64),
        'receiver_observation_count_by_node': np.asarray([0, 1, 3], dtype=np.int64),
        'total_observation_count_by_node': np.asarray([2, 2, 4], dtype=np.int64),
    }
    payload.update(overrides)
    return TimeTermDesignMatrix(**payload)


def _disconnected_design() -> TimeTermDesignMatrix:
    matrix = sparse.csr_matrix(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return _design(
        matrix=matrix,
        data_s=np.asarray([0.1, 0.2], dtype=np.float64),
        n_traces=2,
        n_observations=2,
        n_nodes=4,
        used_trace_mask_sorted=np.asarray([True, True]),
        row_trace_index_sorted=np.asarray([0, 1], dtype=np.int64),
        trace_to_row_index_sorted=np.asarray([0, 1], dtype=np.int64),
        source_node_id_sorted=np.asarray([0, 2], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([1, 3], dtype=np.int64),
        row_source_node_id=np.asarray([0, 2], dtype=np.int64),
        row_receiver_node_id=np.asarray([1, 3], dtype=np.int64),
        row_pick_time_after_static_s=np.asarray([0.1, 0.2], dtype=np.float64),
        row_moveout_time_s=np.zeros(2, dtype=np.float64),
        row_data_s=np.asarray([0.1, 0.2], dtype=np.float64),
        source_observation_count_by_node=np.asarray([1, 0, 1, 0], dtype=np.int64),
        receiver_observation_count_by_node=np.asarray([0, 1, 0, 1], dtype=np.int64),
        total_observation_count_by_node=np.asarray([1, 1, 1, 1], dtype=np.int64),
    )


def _unobserved_node_design() -> TimeTermDesignMatrix:
    matrix = sparse.hstack(
        [OBSERVATION_MATRIX, sparse.csr_matrix((4, 1), dtype=np.float64)],
        format='csr',
    )
    return _design(
        matrix=matrix,
        n_nodes=4,
        source_node_id_sorted=np.asarray([0, 0, 1, 2, 3], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([1, 2, 2, 2, 3], dtype=np.int64),
        source_observation_count_by_node=np.asarray([2, 1, 1, 0], dtype=np.int64),
        receiver_observation_count_by_node=np.asarray([0, 1, 3, 0], dtype=np.int64),
        total_observation_count_by_node=np.asarray([2, 2, 4, 0], dtype=np.int64),
    )


def _accurate_options(**overrides: Any) -> TimeTermSparseSolverOptions:
    payload: dict[str, Any] = {
        'damping_lambda': 1.0e-10,
        'gauge': 'none',
        'solver': 'lsmr',
        'atol': 1.0e-12,
        'btol': 1.0e-12,
        'conlim': 1.0e12,
        'maxiter': 1000,
    }
    payload.update(overrides)
    return TimeTermSparseSolverOptions(**payload)


def test_time_term_solver_system_contains_observation_damping_and_gauge_rows() -> None:
    system = build_time_term_solver_system(
        _design(),
        options=TimeTermSparseSolverOptions(
            damping_lambda=0.5,
            gauge='auto_component',
            gauge_weight=2.0,
        ),
    )

    assert sparse.isspmatrix_csr(system.augmented_matrix)
    assert system.n_observation_rows == 4
    assert system.n_damping_rows == 3
    assert system.n_gauge_rows == 0
    assert system.n_augmented_rows == 7
    assert system.augmented_matrix.shape == (7, 3)
    np.testing.assert_allclose(
        system.augmented_matrix[:4].toarray(),
        OBSERVATION_MATRIX.toarray(),
    )
    np.testing.assert_allclose(system.augmented_data_s[:4], DATA_S)


def test_time_term_solver_system_adds_damping_identity_rows() -> None:
    system = build_time_term_solver_system(
        _design(),
        options=TimeTermSparseSolverOptions(damping_lambda=0.25, gauge='none'),
    )

    np.testing.assert_allclose(
        system.augmented_matrix[4:7].toarray(),
        np.eye(3, dtype=np.float64) * 0.25,
    )
    np.testing.assert_allclose(system.augmented_data_s[4:7], np.zeros(3))


def test_time_term_solver_system_uses_scalar_damping_prior() -> None:
    system = build_time_term_solver_system(
        _design(),
        options=TimeTermSparseSolverOptions(
            damping_lambda=0.5,
            damping_prior_s=0.02,
            gauge='none',
        ),
    )

    np.testing.assert_allclose(system.damping_prior_s, np.full(3, 0.02))
    np.testing.assert_allclose(system.augmented_data_s[4:7], np.full(3, 0.01))


def test_time_term_solver_system_uses_array_damping_prior() -> None:
    prior = np.asarray([0.01, 0.02, 0.03], dtype=np.float64)

    system = build_time_term_solver_system(
        _design(),
        options=TimeTermSparseSolverOptions(
            damping_lambda=0.5,
            damping_prior_s=prior,
            gauge='none',
        ),
    )

    np.testing.assert_allclose(system.damping_prior_s, prior)
    np.testing.assert_allclose(system.augmented_data_s[4:7], 0.5 * prior)


def test_time_term_solver_system_adds_auto_component_signed_gauge_rows() -> None:
    system = build_time_term_solver_system(
        _disconnected_design(),
        options=TimeTermSparseSolverOptions(
            damping_lambda=0.0,
            gauge='auto_component',
            gauge_weight=2.0,
        ),
    )

    assert system.n_damping_rows == 0
    assert system.n_gauge_rows == 2
    np.testing.assert_allclose(
        system.augmented_matrix[-2:].toarray(),
        [
            [2.0 / np.sqrt(2.0), -2.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 0.0, 2.0 / np.sqrt(2.0), -2.0 / np.sqrt(2.0)],
        ],
    )
    np.testing.assert_allclose(system.augmented_data_s[-2:], 0.0)


def test_time_term_solver_system_skips_auto_component_gauge_for_nonbipartite_graph() -> None:
    system = build_time_term_solver_system(
        _design(),
        options=TimeTermSparseSolverOptions(
            damping_lambda=0.0,
            gauge='auto_component',
        ),
    )

    assert system.n_gauge_rows == 0
    assert system.n_bipartite_components == 0


def test_time_term_solver_system_none_zero_damping_allows_nonbipartite_graph() -> None:
    system = build_time_term_solver_system(
        _design(),
        options=TimeTermSparseSolverOptions(
            damping_lambda=0.0,
            gauge='none',
        ),
    )

    assert system.n_gauge_rows == 0
    assert system.n_bipartite_components == 0


def test_time_term_solver_system_rejects_none_zero_damping_for_bipartite_graph() -> None:
    with pytest.raises(ValueError, match='zero damping'):
        build_time_term_solver_system(
            _disconnected_design(),
            options=TimeTermSparseSolverOptions(damping_lambda=0.0, gauge='none'),
        )


@pytest.mark.parametrize('gauge', ['mean_zero', 'component_mean_zero', 'reference_node'])
def test_time_term_solver_system_rejects_legacy_gauge_modes(gauge: str) -> None:
    with pytest.raises(ValueError, match='unsupported gauge'):
        build_time_term_solver_system(
            _design(),
            options=TimeTermSparseSolverOptions(gauge=gauge),  # type: ignore[arg-type]
        )


def test_time_term_solver_system_rejects_unobserved_node_when_required() -> None:
    with pytest.raises(ValueError, match='all nodes'):
        build_time_term_solver_system(_unobserved_node_design())


def test_time_term_solver_system_allows_unobserved_node_when_configured() -> None:
    system = build_time_term_solver_system(
        _unobserved_node_design(),
        options=TimeTermSparseSolverOptions(
            damping_lambda=0.01,
            gauge='auto_component',
            require_all_nodes_observed=False,
            min_total_observations_per_node=0,
        ),
    )

    assert system.n_components == 2
    np.testing.assert_array_equal(system.component_id_by_node, [0, 0, 0, 1])


@pytest.mark.parametrize('solver', ['lsmr', 'lsqr'])
def test_time_term_sparse_solver_recovers_known_node_time_terms(
    solver: str,
) -> None:
    result = solve_time_term_sparse_least_squares(
        _design(),
        options=_accurate_options(solver=solver),
    )

    np.testing.assert_allclose(
        result.node_time_term_s,
        TRUE_NODE_TIME_TERM_S,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        OBSERVATION_MATRIX @ result.node_time_term_s,
        DATA_S,
        atol=1.0e-9,
    )


def test_time_term_sparse_solver_reduces_rms_residual() -> None:
    result = solve_time_term_sparse_least_squares(
        _design(),
        options=_accurate_options(),
    )

    assert result.rms_residual_after_s < result.rms_residual_before_s
    assert result.rms_residual_after_s < 1.0e-9


def test_time_term_sparse_solver_returns_row_residuals_in_observation_order() -> None:
    result = solve_time_term_sparse_least_squares(
        _design(),
        options=_accurate_options(),
    )

    np.testing.assert_allclose(result.row_residual_before_s, DATA_S)
    np.testing.assert_allclose(result.row_residual_after_s, np.zeros(4), atol=1.0e-9)
    np.testing.assert_array_equal(result.row_trace_index_sorted, [0, 1, 2, 3])


def test_time_term_sparse_solver_returns_estimated_trace_delay_in_sorted_order() -> None:
    result = solve_time_term_sparse_least_squares(
        _design(),
        options=_accurate_options(),
    )

    expected = (
        TRUE_NODE_TIME_TERM_S[SOURCE_NODE_ID_SORTED]
        + TRUE_NODE_TIME_TERM_S[RECEIVER_NODE_ID_SORTED]
    )
    np.testing.assert_allclose(
        result.estimated_trace_time_term_delay_s_sorted,
        expected,
        atol=1.0e-9,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        result.prediction_valid_trace_mask_sorted,
        [True, True, True, True, True],
    )
    assert result.used_trace_mask_sorted[4].item() is False
    assert np.isfinite(result.estimated_trace_time_term_delay_s_sorted[4])


def test_time_term_sparse_solver_fit_used_only_prediction_policy_limits_delay() -> None:
    result = solve_time_term_sparse_least_squares(
        _design(),
        options=_accurate_options(trace_prediction_policy='fit_used_only'),
    )
    summary = summarize_time_term_sparse_solver_result(result)

    np.testing.assert_array_equal(
        result.prediction_valid_trace_mask_sorted,
        result.used_trace_mask_sorted,
    )
    assert np.isnan(result.estimated_trace_time_term_delay_s_sorted[4])
    assert summary['n_prediction_valid_traces'] == 4
    assert summary['n_fit_unused_prediction_valid_traces'] == 0
    assert summary['n_unsupported_endpoint_traces'] == 0


def test_time_term_sparse_solver_unsupported_endpoint_prediction_stays_nan() -> None:
    result = solve_time_term_sparse_least_squares(
        _unobserved_node_design(),
        options=_accurate_options(
            damping_lambda=0.01,
            gauge='auto_component',
            require_all_nodes_observed=False,
            min_total_observations_per_node=0,
        ),
    )

    np.testing.assert_array_equal(
        result.prediction_valid_trace_mask_sorted,
        [True, True, True, True, False],
    )
    assert np.isnan(result.estimated_trace_time_term_delay_s_sorted[4])


def test_time_term_sparse_solver_same_source_receiver_node_uses_two_times_node_term() -> None:
    result = solve_time_term_sparse_least_squares(
        _design(),
        options=_accurate_options(),
    )

    assert result.row_estimated_time_term_delay_s[3] == pytest.approx(
        2.0 * result.node_time_term_s[2],
        abs=1.0e-12,
    )


def test_time_term_sparse_solver_rejects_non_finite_data_vector() -> None:
    data = DATA_S.copy()
    data[0] = np.nan

    with pytest.raises(ValueError, match='data_s'):
        build_time_term_solver_system(_design(data_s=data))


def test_time_term_sparse_solver_rejects_node_time_term_above_limit() -> None:
    with pytest.raises(ValueError, match='max_abs_node_time_term_ms'):
        solve_time_term_sparse_least_squares(
            _design(),
            options=_accurate_options(max_abs_node_time_term_ms=5.0),
        )


def test_time_term_sparse_solver_rejects_estimated_trace_delay_above_limit() -> None:
    with pytest.raises(ValueError, match='max_abs_estimated_trace_delay_ms'):
        solve_time_term_sparse_least_squares(
            _design(),
            options=_accurate_options(
                max_abs_node_time_term_ms=None,
                max_abs_estimated_trace_delay_ms=5.0,
            ),
        )


def test_time_term_sparse_solver_result_does_not_expose_applied_shift() -> None:
    result = solve_time_term_sparse_least_squares(
        _design(),
        options=_accurate_options(),
    )

    assert not hasattr(result, 'applied_weathering_shift_s')
    assert not hasattr(result, 'applied_trace_time_term_shift_s_sorted')


def test_summarize_time_term_sparse_solver_result_is_json_safe() -> None:
    result = solve_time_term_sparse_least_squares(
        _design(),
        options=_accurate_options(),
    )

    summary = summarize_time_term_sparse_solver_result(result)

    json.dumps(summary, allow_nan=False)
    assert summary['n_nodes'] == 3
    assert summary['n_observations'] == 4
    assert summary['n_damping_rows'] == 3
    assert summary['n_gauge_rows'] == 0
    assert summary['gauge_mode'] == 'none'
    assert summary['solver_name'] == 'lsmr'
    assert summary['n_unobserved_nodes'] == 0
    assert summary['n_fit_used_traces'] == 4
    assert summary['n_robust_rejected_traces'] == 0
    assert summary['n_prediction_valid_traces'] == 5
    assert summary['n_fit_unused_prediction_valid_traces'] == 1
    assert summary['n_unsupported_endpoint_traces'] == 0
    assert summary['node_time_term_ms']['count'] == 3


def test_time_term_sparse_solver_rejects_nonfinite_solver_output(monkeypatch) -> None:
    class BadLsmr:
        def __call__(self, *args: Any, **kwargs: Any):
            return (
                np.asarray([np.nan, 0.0, 0.0], dtype=np.float64),
                1,
                1,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
            )

    import seis_statics.time_term.sparse_solver as solver_module

    monkeypatch.setattr(solver_module.sparse_linalg, 'lsmr', BadLsmr())

    with pytest.raises(ValueError, match='solver x'):
        solve_time_term_sparse_least_squares(
            _design(),
            options=_accurate_options(),
        )


def test_time_term_sparse_solver_rejects_damping_prior_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='damping_prior_s'):
        build_time_term_solver_system(
            _design(),
            options=TimeTermSparseSolverOptions(
                damping_prior_s=np.asarray([0.0, 0.0]),
            ),
        )


def test_time_term_sparse_solver_rejects_min_observation_violation() -> None:
    with pytest.raises(ValueError, match='not enough'):
        build_time_term_solver_system(
            _design(),
            options=TimeTermSparseSolverOptions(min_observations=5),
        )


def test_time_term_sparse_solver_rejects_node_below_min_total_observations() -> None:
    with pytest.raises(ValueError, match='min_total_observations_per_node'):
        build_time_term_solver_system(
            _design(),
            options=TimeTermSparseSolverOptions(min_total_observations_per_node=3),
        )


def test_time_term_sparse_solver_validates_design_total_count_consistency() -> None:
    bad_design = replace(
        _design(),
        total_observation_count_by_node=np.asarray([2, 2, 3], dtype=np.int64),
    )

    with pytest.raises(ValueError, match='total_observation_count_by_node'):
        build_time_term_solver_system(bad_design)
