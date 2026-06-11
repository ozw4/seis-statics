from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from seis_statics.residual import ResidualStaticSolverInputs
from seis_statics.residual.robust import (
    ResidualStaticRobustOptions,
    solve_residual_static_robust_least_squares,
)
from seis_statics.residual.solver import (
    ResidualStaticLsmrOptions,
    ResidualStaticStabilizationOptions,
    solve_residual_static_stabilized_least_squares,
)


def _lsmr_options() -> ResidualStaticLsmrOptions:
    return ResidualStaticLsmrOptions(
        atol=1.0e-12,
        btol=1.0e-12,
        conlim=1.0e12,
        maxiter=10000,
    )


def _stabilization_options() -> ResidualStaticStabilizationOptions:
    return ResidualStaticStabilizationOptions(
        gauge='zero_mean_source_receiver',
        damping_lambda=0.0,
        min_valid_picks=10,
        min_picks_per_source=1,
        min_picks_per_receiver=1,
        max_abs_estimated_delay_ms=1000.0,
    )


def _grid_inputs(
    *,
    noise_scale_s: float = 0.0,
    outlier_index: int | None = None,
    outlier_s: float = 0.0,
    **overrides: Any,
) -> ResidualStaticSolverInputs:
    n_sources = 5
    n_receivers = 6
    source_unique_ids = np.arange(101, 101 + n_sources, dtype=np.int32)
    receiver_unique_ids = np.arange(201, 201 + n_receivers, dtype=np.int32)
    source_index = np.repeat(np.arange(n_sources, dtype=np.int64), n_receivers)
    receiver_index = np.tile(np.arange(n_receivers, dtype=np.int64), n_sources)
    n_traces = int(source_index.size)

    source_delay_s = np.linspace(-0.009, 0.007, n_sources, dtype=np.float64)
    receiver_delay_s = np.linspace(-0.004, 0.005, n_receivers, dtype=np.float64)
    pick_time_after_datum = (
        0.075 + source_delay_s[source_index] + receiver_delay_s[receiver_index]
    )
    if noise_scale_s:
        sample_index = np.arange(n_traces, dtype=np.float64)
        noise = np.sin(sample_index * 1.7) + 0.5 * np.cos(sample_index * 2.3)
        noise -= np.mean(noise)
        pick_time_after_datum = pick_time_after_datum + noise_scale_s * noise
    if outlier_index is not None:
        pick_time_after_datum = pick_time_after_datum.copy()
        pick_time_after_datum[outlier_index] += outlier_s

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
        'offset_sorted': None,
        'abs_offset_sorted': None,
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
        'offset_byte': None,
        'moveout_model': 'none',
        'input_file_id': 'input-file',
        'datum_source_file_id': 'datum-source-file',
        'datum_job_id': 'datum-job',
        'pick_source_kind': 'batch_npz',
        'metadata': {'source': 'seis-statics-robust-solver-test'},
    }
    payload.update(overrides)
    return ResidualStaticSolverInputs(**payload)


def _solve_robust(
    inputs: ResidualStaticSolverInputs,
    robust_options: ResidualStaticRobustOptions,
):
    return solve_residual_static_robust_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        robust_options=robust_options,
        lsmr_options=_lsmr_options(),
    )


def test_robust_disabled_path_equals_one_stabilized_sparse_solve() -> None:
    inputs = _grid_inputs(noise_scale_s=0.002, outlier_index=13, outlier_s=0.05)
    sparse_result = solve_residual_static_stabilized_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        lsmr_options=_lsmr_options(),
    )
    robust_result = _solve_robust(
        inputs,
        ResidualStaticRobustOptions(enabled=False),
    )

    assert robust_result.stop_reason == 'disabled'
    assert robust_result.n_rejected_total == 0
    assert robust_result.iteration_summaries == ()
    assert robust_result.initial_solver_result is robust_result.final_solver_result
    np.testing.assert_array_equal(
        robust_result.final_used_mask_sorted,
        sparse_result.used_mask_sorted,
    )
    np.testing.assert_allclose(
        robust_result.final_solver_result.model_evaluation.residual_s_sorted,
        sparse_result.model_evaluation.residual_s_sorted,
        atol=1.0e-12,
    )


@pytest.mark.parametrize('method', ['mad', 'sigma'])
def test_robust_rejection_removes_synthetic_large_outlier(method: str) -> None:
    inputs = _grid_inputs(noise_scale_s=0.002, outlier_index=13, outlier_s=0.03)

    result = _solve_robust(
        inputs,
        ResidualStaticRobustOptions(method=method, threshold=3.0),
    )

    assert result.stop_reason == 'converged'
    assert result.n_rejected_total == 1
    assert result.rejected_mask_sorted[13].item() is True
    assert result.final_used_mask_sorted[13].item() is False
    assert result.rejected_iteration_sorted[13] == 0
    assert result.iteration_summaries[0].n_rejected_this_iteration == 1


def test_min_used_fraction_prevents_rejecting_too_many_observations() -> None:
    inputs = _grid_inputs(noise_scale_s=0.002, outlier_index=13, outlier_s=0.05)

    with pytest.raises(ValueError, match='min_used_fraction'):
        _solve_robust(
            inputs,
            ResidualStaticRobustOptions(
                method='mad',
                threshold=3.0,
                min_used_fraction=0.9,
            ),
        )


def test_zero_scale_stop_reason_reported_for_perfectly_fitted_data() -> None:
    result = _solve_robust(
        _grid_inputs(),
        ResidualStaticRobustOptions(method='mad'),
    )

    assert result.stop_reason == 'zero_scale'
    assert result.n_rejected_total == 0
    assert len(result.iteration_summaries) == 1
    assert result.iteration_summaries[0].stop_reason == 'zero_scale'
    assert result.iteration_summaries[0].residual_scale_s <= 1.0e-12
