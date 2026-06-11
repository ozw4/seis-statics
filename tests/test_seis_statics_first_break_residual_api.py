from __future__ import annotations

from typing import Any

import numpy as np

from seis_statics.residual import (
    FirstBreakResidualStaticsResult,
    ResidualStaticRobustOptions,
    ResidualStaticSolverInputs,
    solve_first_break_residual_statics,
)
from seis_statics.residual.robust import solve_residual_static_robust_least_squares
from seis_statics.residual.solver import (
    ResidualStaticLsmrOptions,
    ResidualStaticStabilizationOptions,
)

_ATOL = 1.0e-9


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
    moveout_model: str = 'none',
    outlier_index: int | None = None,
    outlier_s: float = 0.0,
    noise_scale_s: float = 0.0,
) -> ResidualStaticSolverInputs:
    n_sources = 5
    n_receivers = 6
    source_unique_ids = np.arange(101, 101 + n_sources, dtype=np.int64)
    receiver_unique_ids = np.arange(201, 201 + n_receivers, dtype=np.int64)
    source_index = np.repeat(np.arange(n_sources, dtype=np.int64), n_receivers)
    receiver_index = np.tile(np.arange(n_receivers, dtype=np.int64), n_sources)
    n_traces = int(source_index.size)

    source_delay_s = np.linspace(-0.009, 0.007, n_sources, dtype=np.float64)
    source_delay_s -= float(np.mean(source_delay_s))
    receiver_delay_s = np.linspace(-0.004, 0.005, n_receivers, dtype=np.float64)
    receiver_delay_s -= float(np.mean(receiver_delay_s))
    abs_offset = (
        120.0
        + 57.0 * source_index.astype(np.float64)
        + 91.0 * receiver_index.astype(np.float64)
        + 13.0 * source_index.astype(np.float64) * receiver_index.astype(np.float64)
    )
    offset = abs_offset * np.where(np.arange(n_traces) % 2 == 0, -1.0, 1.0)
    intercept_s = 0.075
    slowness_s_per_offset_unit = 2.0e-5
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
    if noise_scale_s:
        sample_index = np.arange(n_traces, dtype=np.float64)
        noise = np.sin(sample_index * 1.7) + 0.5 * np.cos(sample_index * 2.3)
        noise -= float(np.mean(noise))
        pick_time_after_datum = pick_time_after_datum + noise_scale_s * noise
    if outlier_index is not None:
        pick_time_after_datum = pick_time_after_datum.copy()
        pick_time_after_datum[outlier_index] += outlier_s

    datum_trace_shift = 0.001 * np.sin(np.arange(n_traces, dtype=np.float64))
    picks_time = pick_time_after_datum - datum_trace_shift
    valid_mask = np.ones(n_traces, dtype=bool)
    return ResidualStaticSolverInputs(
        picks_time_s_sorted=picks_time,
        valid_pick_mask_sorted=valid_mask,
        pick_time_after_datum_s_sorted=pick_time_after_datum,
        datum_trace_shift_s_sorted=datum_trace_shift,
        source_id_sorted=source_unique_ids[source_index],
        receiver_id_sorted=receiver_unique_ids[receiver_index],
        source_unique_ids=source_unique_ids,
        receiver_unique_ids=receiver_unique_ids,
        source_index_sorted=source_index,
        receiver_index_sorted=receiver_index,
        source_valid_pick_counts=np.bincount(
            source_index,
            minlength=n_sources,
        ).astype(np.int64),
        receiver_valid_pick_counts=np.bincount(
            receiver_index,
            minlength=n_receivers,
        ).astype(np.int64),
        offset_sorted=offset_sorted,
        abs_offset_sorted=abs_offset_sorted,
        key1_sorted=source_unique_ids[source_index],
        key2_sorted=receiver_unique_ids[receiver_index],
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        dt=0.004,
        n_traces=n_traces,
        n_samples=96,
        key1_byte=189,
        key2_byte=193,
        source_id_byte=17,
        receiver_id_byte=13,
        offset_byte=offset_byte,
        moveout_model=moveout_model,  # type: ignore[arg-type]
        input_file_id='input-file',
        datum_source_file_id='datum-source-file',
        datum_job_id='datum-job',
        pick_source_kind='batch_npz',
        metadata={'source': 'first-break-residual-api-test'},
    )


def _solve_lower(
    inputs: ResidualStaticSolverInputs,
    *,
    robust_options: ResidualStaticRobustOptions,
):
    return solve_residual_static_robust_least_squares(
        inputs,
        stabilization_options=_stabilization_options(),
        robust_options=robust_options,
        lsmr_options=_lsmr_options(),
    )


def _solve_public(
    inputs: ResidualStaticSolverInputs,
    *,
    robust_options: ResidualStaticRobustOptions | dict[str, Any],
    use_indices: bool = False,
) -> FirstBreakResidualStaticsResult:
    source_kwargs: dict[str, object]
    receiver_kwargs: dict[str, object]
    if use_indices:
        source_kwargs = {
            'source_index': inputs.source_index_sorted,
            'source_unique_ids': inputs.source_unique_ids,
        }
        receiver_kwargs = {
            'receiver_index': inputs.receiver_index_sorted,
            'receiver_unique_ids': inputs.receiver_unique_ids,
        }
    else:
        source_kwargs = {'source_id': inputs.source_id_sorted}
        receiver_kwargs = {'receiver_id': inputs.receiver_id_sorted}

    return solve_first_break_residual_statics(
        pick_time_s=inputs.picks_time_s_sorted,
        datum_trace_shift_s=inputs.datum_trace_shift_s_sorted,
        valid_pick_mask=inputs.valid_pick_mask_sorted,
        offset=inputs.offset_sorted,
        abs_offset=inputs.abs_offset_sorted,
        moveout_model=inputs.moveout_model,
        stabilization_options=_stabilization_options(),
        robust_options=robust_options,
        lsmr_options=_lsmr_options(),
        key1=inputs.key1_sorted,
        key2=inputs.key2_sorted,
        source_elevation_m=inputs.source_elevation_m_sorted,
        receiver_elevation_m=inputs.receiver_elevation_m_sorted,
        dt=inputs.dt,
        n_samples=inputs.n_samples,
        key1_byte=inputs.key1_byte,
        key2_byte=inputs.key2_byte,
        source_id_byte=inputs.source_id_byte,
        receiver_id_byte=inputs.receiver_id_byte,
        offset_byte=inputs.offset_byte,
        input_file_id=inputs.input_file_id,
        datum_source_file_id=inputs.datum_source_file_id,
        datum_job_id=inputs.datum_job_id,
        pick_source_kind=inputs.pick_source_kind,
        metadata=inputs.metadata,
        **source_kwargs,
        **receiver_kwargs,
    )


def _assert_public_matches_lower(
    public: FirstBreakResidualStaticsResult,
    lower,
    inputs: ResidualStaticSolverInputs,
) -> None:
    final = lower.final_solver_result
    np.testing.assert_array_equal(public.source_id, inputs.source_unique_ids)
    np.testing.assert_array_equal(public.receiver_id, inputs.receiver_unique_ids)
    np.testing.assert_allclose(
        public.source_delay_s,
        final.parameter_parts.source_delay_s,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        public.receiver_delay_s,
        final.parameter_parts.receiver_delay_s,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        public.intercept_s,
        final.parameter_parts.intercept_s,
        atol=_ATOL,
    )
    if final.parameter_parts.slowness_s_per_offset_unit is None:
        assert public.slowness_s_per_offset_unit is None
    else:
        np.testing.assert_allclose(
            public.slowness_s_per_offset_unit,
            final.parameter_parts.slowness_s_per_offset_unit,
            atol=1.0e-12,
        )
    np.testing.assert_allclose(
        public.moveout_model_time_s,
        final.model_evaluation.moveout_model_time_s_sorted,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        public.estimated_trace_delay_s,
        final.model_evaluation.estimated_trace_delay_s_sorted,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        public.modeled_pick_time_s,
        final.model_evaluation.modeled_pick_time_s_sorted,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        public.residual_s,
        final.model_evaluation.residual_s_sorted,
        atol=_ATOL,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        public.residual_valid_mask,
        final.model_evaluation.residual_valid_mask_sorted,
    )
    np.testing.assert_array_equal(
        public.initial_used_pick_mask,
        lower.initial_used_mask_sorted,
    )
    np.testing.assert_array_equal(public.used_pick_mask, lower.final_used_mask_sorted)
    np.testing.assert_array_equal(public.rejected_pick_mask, lower.rejected_mask_sorted)
    np.testing.assert_array_equal(
        public.rejected_iteration,
        lower.rejected_iteration_sorted,
    )
    assert public.diagnostics.istop == final.diagnostics.istop
    assert public.diagnostics.itn == final.diagnostics.itn
    for value in (
        public.diagnostics.normr,
        public.diagnostics.normar,
        public.diagnostics.norma,
        public.diagnostics.conda,
        public.diagnostics.normx,
    ):
        assert np.isfinite(value)
        assert value >= 0.0
    assert public.minimum_data.n_used_picks == final.minimum_data.n_used_picks
    assert public.minimum_data.n_sources == final.minimum_data.n_sources
    assert public.minimum_data.n_receivers == final.minimum_data.n_receivers
    assert (
        public.minimum_data.n_effective_parameters
        == final.minimum_data.n_effective_parameters
    )
    np.testing.assert_array_equal(
        public.minimum_data.source_used_pick_counts,
        final.minimum_data.source_used_pick_counts,
    )
    np.testing.assert_array_equal(
        public.minimum_data.receiver_used_pick_counts,
        final.minimum_data.receiver_used_pick_counts,
    )
    assert public.graph.n_components == final.minimum_data.graph.n_components
    np.testing.assert_array_equal(
        public.graph.component_observation_counts,
        final.minimum_data.graph.component_observation_counts,
    )
    assert public.robust_stop_reason == lower.stop_reason
    assert public.n_initial_used_picks == lower.n_initial_used_picks
    assert public.n_final_used_picks == lower.n_final_used_picks
    assert public.n_rejected_total == lower.n_rejected_total


def test_public_import_exports_first_break_solver() -> None:
    from seis_statics.residual import solve_first_break_residual_statics as imported

    assert imported is solve_first_break_residual_statics


def test_public_api_matches_lower_solver_for_none_moveout_robust_disabled() -> None:
    inputs = _grid_inputs(moveout_model='none')
    robust_options = ResidualStaticRobustOptions(enabled=False)

    lower = _solve_lower(inputs, robust_options=robust_options)
    public = _solve_public(inputs, robust_options={'enabled': False})

    assert public.moveout_model == 'none'
    _assert_public_matches_lower(public, lower, inputs)


def test_public_api_matches_lower_solver_for_linear_abs_offset() -> None:
    inputs = _grid_inputs(moveout_model='linear_abs_offset')
    robust_options = ResidualStaticRobustOptions(enabled=False)

    lower = _solve_lower(inputs, robust_options=robust_options)
    public = _solve_public(inputs, robust_options=robust_options, use_indices=True)

    assert public.moveout_model == 'linear_abs_offset'
    _assert_public_matches_lower(public, lower, inputs)


def test_public_api_matches_lower_solver_with_robust_rejection_enabled() -> None:
    inputs = _grid_inputs(
        moveout_model='none',
        noise_scale_s=0.002,
        outlier_index=13,
        outlier_s=0.03,
    )
    robust_options = ResidualStaticRobustOptions(method='mad', threshold=3.0)

    lower = _solve_lower(inputs, robust_options=robust_options)
    public = _solve_public(inputs, robust_options=robust_options)

    assert public.n_rejected_total == 1
    assert public.rejected_pick_mask[13].item() is True
    _assert_public_matches_lower(public, lower, inputs)


def test_public_api_ignores_invalid_pick_endpoint_ids() -> None:
    inputs = _grid_inputs(moveout_model='none')
    invalid_source_id = np.asarray([9001], dtype=np.int64)
    invalid_receiver_id = np.asarray([8001], dtype=np.int64)
    valid_mask = np.concatenate(
        [inputs.valid_pick_mask_sorted, np.asarray([False], dtype=bool)]
    )

    result = solve_first_break_residual_statics(
        pick_time_s=np.concatenate(
            [inputs.picks_time_s_sorted, np.asarray([np.nan], dtype=np.float64)]
        ),
        datum_trace_shift_s=np.concatenate(
            [
                inputs.datum_trace_shift_s_sorted,
                np.asarray([0.0], dtype=np.float64),
            ]
        ),
        valid_pick_mask=valid_mask,
        source_id=np.concatenate([inputs.source_id_sorted, invalid_source_id]),
        receiver_id=np.concatenate([inputs.receiver_id_sorted, invalid_receiver_id]),
        moveout_model=inputs.moveout_model,
        stabilization_options=_stabilization_options(),
        robust_options={'enabled': False},
        lsmr_options=_lsmr_options(),
    )

    assert invalid_source_id[0].item() not in result.source_id
    assert invalid_receiver_id[0].item() not in result.receiver_id
    np.testing.assert_array_equal(result.source_id, inputs.source_unique_ids)
    np.testing.assert_array_equal(result.receiver_id, inputs.receiver_unique_ids)
    assert result.minimum_data.n_sources == inputs.source_unique_ids.size
    assert result.minimum_data.n_receivers == inputs.receiver_unique_ids.size
    assert result.graph.n_components == 1
    assert result.used_pick_mask[-1].item() is False
    assert result.residual_valid_mask[-1].item() is False
