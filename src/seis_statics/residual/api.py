"""Public first-break residual statics API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from scipy import sparse

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics.residual.result import (
    FirstBreakResidualStaticsResult,
    SourceReceiverStaticsMinimumDataSummary,
    SourceReceiverStaticsResult,
)
from seis_statics.residual.robust import (
    ResidualStaticRobustOptions,
    ResidualStaticRobustIterationSummary,
    ResidualStaticRobustStopReason,
    ROBUST_SCALE_FLOOR_S,
    build_residual_static_outlier_mask,
    solve_residual_static_robust_least_squares,
    validate_residual_static_robust_options,
)
from seis_statics.residual.solver import (
    ResidualStaticLsmrOptions,
    ResidualStaticObservationGraphSummary,
    ResidualStaticStabilizationOptions,
    run_sparse_lsmr,
    validate_lsmr_options,
    validate_residual_static_stabilization_options,
)
from seis_statics.residual.types import MoveoutModel, ResidualStaticSolverInputs


def solve_first_break_residual_statics(
    *,
    solver_inputs: ResidualStaticSolverInputs | None = None,
    pick_time_s: object | None = None,
    pick_time_after_datum_s: object | None = None,
    valid_pick_mask: object | None = None,
    source_id: object | None = None,
    receiver_id: object | None = None,
    source_index: object | None = None,
    receiver_index: object | None = None,
    source_unique_ids: object | None = None,
    receiver_unique_ids: object | None = None,
    offset: object | None = None,
    abs_offset: object | None = None,
    datum_trace_shift_s: object | None = None,
    used_pick_mask: object | None = None,
    moveout_model: MoveoutModel = 'none',
    stabilization_options: ResidualStaticStabilizationOptions
    | Mapping[str, object]
    | None = None,
    robust_options: ResidualStaticRobustOptions | Mapping[str, object] | None = None,
    lsmr_options: ResidualStaticLsmrOptions | Mapping[str, object] | None = None,
    key1: object | None = None,
    key2: object | None = None,
    source_elevation_m: object | None = None,
    receiver_elevation_m: object | None = None,
    dt: float = 1.0,
    n_samples: int = 1,
    key1_byte: int = 0,
    key2_byte: int = 0,
    source_id_byte: int = 0,
    receiver_id_byte: int = 0,
    offset_byte: int | None = None,
    input_file_id: str = '',
    datum_source_file_id: str = '',
    datum_job_id: str = '',
    pick_source_kind: str = '',
    metadata: Mapping[str, Any] | None = None,
) -> FirstBreakResidualStaticsResult:
    """Solve the package-level first-break/datum residual statics workflow."""
    if solver_inputs is not None:
        inputs = _coerce_solver_inputs(solver_inputs)
        robust_result = solve_residual_static_robust_least_squares(
            inputs,
            used_mask_sorted=_optional_used_mask(
                used_pick_mask,
                n_traces=int(inputs.n_traces),
            ),
            stabilization_options=_stabilization_options(stabilization_options),
            robust_options=_robust_options(robust_options),
            lsmr_options=_lsmr_options(lsmr_options),
        )
        source, receiver = _endpoint_indices_from_inputs(inputs)
        return _public_result(robust_result, source=source, receiver=receiver)

    if valid_pick_mask is None:
        raise ValueError('valid_pick_mask is required')
    if (pick_time_s is None) == (pick_time_after_datum_s is None):
        raise ValueError(
            'provide exactly one of pick_time_s or pick_time_after_datum_s'
        )
    valid_mask = _coerce_1d_bool_array(valid_pick_mask, name='valid_pick_mask')
    n_traces = int(valid_mask.shape[0])
    if n_traces == 0:
        raise ValueError('valid_pick_mask must be non-empty')

    datum_shift = _datum_trace_shift(datum_trace_shift_s, n_traces=n_traces)
    picks_time = _pick_time(
        pick_time_s,
        pick_time_after_datum_s=pick_time_after_datum_s,
        datum_trace_shift_s=datum_shift,
        valid_mask=valid_mask,
    )
    pick_after_datum = _pick_time_after_datum(
        pick_time_s,
        pick_time_after_datum_s=pick_time_after_datum_s,
        datum_trace_shift_s=datum_shift,
        valid_mask=valid_mask,
    )

    source = _endpoint_index(
        id_values=source_id,
        index_values=source_index,
        unique_id_values=source_unique_ids,
        valid_mask=valid_mask,
        n_traces=n_traces,
        role='source',
    )
    receiver = _endpoint_index(
        id_values=receiver_id,
        index_values=receiver_index,
        unique_id_values=receiver_unique_ids,
        valid_mask=valid_mask,
        n_traces=n_traces,
        role='receiver',
    )
    offset_array, abs_offset_array = _offset_arrays(
        offset=offset,
        abs_offset=abs_offset,
        moveout_model=moveout_model,
        n_traces=n_traces,
    )

    inputs = ResidualStaticSolverInputs(
        picks_time_s_sorted=picks_time,
        valid_pick_mask_sorted=valid_mask,
        pick_time_after_datum_s_sorted=pick_after_datum,
        datum_trace_shift_s_sorted=datum_shift,
        source_id_sorted=source.id_sorted,
        receiver_id_sorted=receiver.id_sorted,
        source_unique_ids=source.unique_ids,
        receiver_unique_ids=receiver.unique_ids,
        source_index_sorted=source.index_sorted,
        receiver_index_sorted=receiver.index_sorted,
        source_valid_pick_counts=source.valid_pick_counts,
        receiver_valid_pick_counts=receiver.valid_pick_counts,
        offset_sorted=offset_array,
        abs_offset_sorted=abs_offset_array,
        key1_sorted=_optional_integer_array(key1, n_traces=n_traces, name='key1'),
        key2_sorted=_optional_integer_array(key2, n_traces=n_traces, name='key2'),
        source_elevation_m_sorted=_optional_float_array(
            source_elevation_m,
            n_traces=n_traces,
            name='source_elevation_m',
        ),
        receiver_elevation_m_sorted=_optional_float_array(
            receiver_elevation_m,
            n_traces=n_traces,
            name='receiver_elevation_m',
        ),
        dt=_coerce_positive_finite_float(dt, name='dt'),
        n_traces=n_traces,
        n_samples=_coerce_positive_int(n_samples, name='n_samples'),
        key1_byte=int(key1_byte),
        key2_byte=int(key2_byte),
        source_id_byte=int(source_id_byte),
        receiver_id_byte=int(receiver_id_byte),
        offset_byte=offset_byte,
        moveout_model=moveout_model,
        input_file_id=str(input_file_id),
        datum_source_file_id=str(datum_source_file_id),
        datum_job_id=str(datum_job_id),
        pick_source_kind=str(pick_source_kind),
        metadata=dict(metadata or {}),
    )
    robust_result = solve_residual_static_robust_least_squares(
        inputs,
        used_mask_sorted=_optional_used_mask(used_pick_mask, n_traces=n_traces),
        stabilization_options=_stabilization_options(stabilization_options),
        robust_options=_robust_options(robust_options),
        lsmr_options=_lsmr_options(lsmr_options),
    )
    return _public_result(robust_result, source=source, receiver=receiver)


def solve_source_receiver_statics(
    *,
    lag_s,
    source_id,
    receiver_id,
    valid_mask=None,
    weight=None,
    robust=True,
    stabilization_options: ResidualStaticStabilizationOptions
    | Mapping[str, object]
    | None = None,
    robust_options: ResidualStaticRobustOptions | Mapping[str, object] | None = None,
    lsmr_options: ResidualStaticLsmrOptions | Mapping[str, object] | None = None,
) -> SourceReceiverStaticsResult:
    """Decompose lag observations into source and receiver delays.

    ``lag_s > 0`` means the observed event is later than the reference/pilot.
    The model is ``trace_delay_s = S[source_id] + R[receiver_id]`` and the
    correction to apply to traces is ``-trace_delay_s``. If ``weight`` is
    provided, least-squares observation rows and right-hand-side values are
    multiplied by ``sqrt(weight)``. Weight values must be finite and
    non-negative; zero-weight observations are reported in diagnostics and are
    excluded from the solve.
    """
    lag = _coerce_1d_real_numeric_float64(lag_s, name='lag_s')
    n_traces = int(lag.shape[0])
    if n_traces == 0:
        raise ValueError('lag_s must be non-empty')
    finite_lag_mask = np.isfinite(lag)
    if valid_mask is None:
        base_valid_mask = np.ascontiguousarray(finite_lag_mask, dtype=bool)
    else:
        base_valid_mask = _coerce_1d_bool_array(
            valid_mask,
            name='valid_mask',
            expected_shape=(n_traces,),
        )
        if np.any(~finite_lag_mask[base_valid_mask]):
            raise ValueError('lag_s must be finite for valid observations')

    source = _endpoint_index(
        id_values=source_id,
        index_values=None,
        unique_id_values=None,
        valid_mask=base_valid_mask,
        n_traces=n_traces,
        role='source',
    )
    receiver = _endpoint_index(
        id_values=receiver_id,
        index_values=None,
        unique_id_values=None,
        valid_mask=base_valid_mask,
        n_traces=n_traces,
        role='receiver',
    )
    observation_weight = _source_receiver_weight(
        weight,
        n_traces=n_traces,
        base_valid_mask=base_valid_mask,
    )
    initial_used_mask = np.ascontiguousarray(
        base_valid_mask & (observation_weight > 0.0),
        dtype=bool,
    )
    if not np.any(initial_used_mask):
        raise ValueError('at least one valid positive-weight observation is required')

    validated_stabilization_options = _stabilization_options(stabilization_options)
    validated_robust_options = _source_receiver_robust_options(
        robust=robust,
        robust_options=robust_options,
    )
    validated_lsmr_options = _lsmr_options(lsmr_options)

    robust_result = _solve_source_receiver_robust(
        lag=lag,
        source=source,
        receiver=receiver,
        base_valid_mask=base_valid_mask,
        initial_used_mask=initial_used_mask,
        weight=observation_weight,
        stabilization_options=validated_stabilization_options,
        robust_options=validated_robust_options,
        lsmr_options=validated_lsmr_options,
    )
    return _source_receiver_public_result(
        robust_result,
        source=source,
        receiver=receiver,
        weight=observation_weight,
    )


def delay_to_applied_shift(delay_s):
    """Return the trace shift that applies the negative of a delay estimate."""
    delay = np.asarray(delay_s, dtype=np.float64)
    shift = -delay
    if shift.ndim == 0:
        return float(shift)
    return np.ascontiguousarray(shift, dtype=np.float64)


class _EndpointIndex:
    def __init__(
        self,
        *,
        id_sorted: np.ndarray,
        unique_ids: np.ndarray,
        index_sorted: np.ndarray,
        valid_pick_counts: np.ndarray,
    ) -> None:
        self.id_sorted = id_sorted
        self.unique_ids = unique_ids
        self.index_sorted = index_sorted
        self.valid_pick_counts = valid_pick_counts


def _coerce_solver_inputs(
    inputs: ResidualStaticSolverInputs,
) -> ResidualStaticSolverInputs:
    if not isinstance(inputs, ResidualStaticSolverInputs):
        raise ValueError('solver_inputs must be a ResidualStaticSolverInputs instance')
    return inputs


def _endpoint_indices_from_inputs(
    inputs: ResidualStaticSolverInputs,
) -> tuple[_EndpointIndex, _EndpointIndex]:
    n_traces = int(inputs.n_traces)
    source_unique_ids = _coerce_1d_integer_int64(
        inputs.source_unique_ids,
        name='source_unique_ids',
    )
    receiver_unique_ids = _coerce_1d_integer_int64(
        inputs.receiver_unique_ids,
        name='receiver_unique_ids',
    )
    source = _EndpointIndex(
        id_sorted=_coerce_1d_integer_int64(
            inputs.source_id_sorted,
            name='source_id_sorted',
            expected_shape=(n_traces,),
        ),
        unique_ids=source_unique_ids,
        index_sorted=_coerce_1d_integer_int64(
            inputs.source_index_sorted,
            name='source_index_sorted',
            expected_shape=(n_traces,),
        ),
        valid_pick_counts=_coerce_1d_integer_int64(
            inputs.source_valid_pick_counts,
            name='source_valid_pick_counts',
            expected_shape=(int(source_unique_ids.shape[0]),),
        ),
    )
    receiver = _EndpointIndex(
        id_sorted=_coerce_1d_integer_int64(
            inputs.receiver_id_sorted,
            name='receiver_id_sorted',
            expected_shape=(n_traces,),
        ),
        unique_ids=receiver_unique_ids,
        index_sorted=_coerce_1d_integer_int64(
            inputs.receiver_index_sorted,
            name='receiver_index_sorted',
            expected_shape=(n_traces,),
        ),
        valid_pick_counts=_coerce_1d_integer_int64(
            inputs.receiver_valid_pick_counts,
            name='receiver_valid_pick_counts',
            expected_shape=(int(receiver_unique_ids.shape[0]),),
        ),
    )
    return source, receiver


def _endpoint_index(
    *,
    id_values: object | None,
    index_values: object | None,
    unique_id_values: object | None,
    valid_mask: np.ndarray,
    n_traces: int,
    role: str,
) -> _EndpointIndex:
    if (id_values is None) == (index_values is None):
        raise ValueError(f'provide exactly one of {role}_id or {role}_index')
    if id_values is not None:
        ids = _coerce_1d_integer_int64(
            id_values,
            name=f'{role}_id',
            expected_shape=(n_traces,),
        )
        unique_ids, valid_inverse = np.unique(ids[valid_mask], return_inverse=True)
        if unique_ids.size == 0:
            raise ValueError(f'{role}_id must contain at least one valid observation')
        index = np.zeros(n_traces, dtype=np.int64)
        index[valid_mask] = valid_inverse
        ids = np.ascontiguousarray(unique_ids[index], dtype=np.int64)
    else:
        input_index = _coerce_1d_integer_int64(
            index_values,
            name=f'{role}_index',
            expected_shape=(n_traces,),
        )
        if np.any(input_index < 0):
            raise ValueError(f'{role}_index must be greater than or equal to 0')
        valid_unique_index, valid_inverse = np.unique(
            input_index[valid_mask],
            return_inverse=True,
        )
        if valid_unique_index.size == 0:
            raise ValueError(f'{role}_index must contain at least one valid observation')
        if unique_id_values is None:
            unique_ids = np.ascontiguousarray(valid_unique_index, dtype=np.int64)
        else:
            all_unique_ids = _coerce_1d_integer_int64(
                unique_id_values,
                name=f'{role}_unique_ids',
            )
            if int(valid_unique_index[-1]) >= int(all_unique_ids.shape[0]):
                raise ValueError(
                    f'{role}_unique_ids must include every valid {role}_index'
                )
            unique_ids = np.ascontiguousarray(
                all_unique_ids[valid_unique_index],
                dtype=np.int64,
            )
        index = np.zeros(n_traces, dtype=np.int64)
        index[valid_mask] = valid_inverse
        ids = np.ascontiguousarray(unique_ids[index], dtype=np.int64)
    counts = np.bincount(
        index[valid_mask],
        minlength=int(unique_ids.shape[0]),
    ).astype(np.int64)
    return _EndpointIndex(
        id_sorted=np.ascontiguousarray(ids, dtype=np.int64),
        unique_ids=np.ascontiguousarray(unique_ids, dtype=np.int64),
        index_sorted=np.ascontiguousarray(index, dtype=np.int64),
        valid_pick_counts=np.ascontiguousarray(counts, dtype=np.int64),
    )


def _datum_trace_shift(value: object | None, *, n_traces: int) -> np.ndarray:
    if value is None:
        return np.zeros(n_traces, dtype=np.float64)
    return _coerce_1d_real_numeric_float64(
        value,
        name='datum_trace_shift_s',
        expected_shape=(n_traces,),
        require_finite=True,
    )


def _pick_time(
    value: object | None,
    *,
    pick_time_after_datum_s: object | None,
    datum_trace_shift_s: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if value is None:
        after_datum = _coerce_pick_array(
            pick_time_after_datum_s,
            valid_mask=valid_mask,
            name='pick_time_after_datum_s',
        )
        out = after_datum - datum_trace_shift_s
        out[~valid_mask] = np.nan
        return np.ascontiguousarray(out, dtype=np.float64)
    return _coerce_pick_array(value, valid_mask=valid_mask, name='pick_time_s')


def _pick_time_after_datum(
    value: object | None,
    *,
    pick_time_after_datum_s: object | None,
    datum_trace_shift_s: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if pick_time_after_datum_s is not None:
        return _coerce_pick_array(
            pick_time_after_datum_s,
            valid_mask=valid_mask,
            name='pick_time_after_datum_s',
        )
    pick_time = _coerce_pick_array(value, valid_mask=valid_mask, name='pick_time_s')
    out = pick_time + datum_trace_shift_s
    out[~valid_mask] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _coerce_pick_array(
    value: object,
    *,
    valid_mask: np.ndarray,
    name: str,
) -> np.ndarray:
    n_traces = int(valid_mask.shape[0])
    arr = _coerce_1d_real_numeric_float64(
        value,
        name=name,
        expected_shape=(n_traces,),
    )
    if np.any(~np.isfinite(arr[valid_mask])):
        raise ValueError(f'{name} must be finite for valid picks')
    out = np.ascontiguousarray(arr, dtype=np.float64)
    out[~valid_mask] = np.nan
    return out


def _offset_arrays(
    *,
    offset: object | None,
    abs_offset: object | None,
    moveout_model: MoveoutModel,
    n_traces: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if offset is None:
        offset_array = None
    else:
        offset_array = _coerce_1d_real_numeric_float64(
            offset,
            name='offset',
            expected_shape=(n_traces,),
            require_finite=True,
        )
    if abs_offset is None:
        abs_offset_array = (
            np.ascontiguousarray(np.abs(offset_array), dtype=np.float64)
            if offset_array is not None
            else None
        )
    else:
        abs_offset_array = _coerce_1d_real_numeric_float64(
            abs_offset,
            name='abs_offset',
            expected_shape=(n_traces,),
            require_finite=True,
        )
    if moveout_model == 'linear_abs_offset':
        if abs_offset_array is None:
            raise ValueError('offset or abs_offset is required for linear_abs_offset')
        if np.any(abs_offset_array < 0.0):
            raise ValueError('abs_offset must be non-negative')
    elif moveout_model != 'none':
        raise ValueError(f'unsupported moveout_model: {moveout_model!r}')
    return offset_array, abs_offset_array


def _optional_integer_array(
    value: object | None,
    *,
    n_traces: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.arange(n_traces, dtype=np.int64)
    return _coerce_1d_integer_int64(value, name=name, expected_shape=(n_traces,))


def _optional_float_array(
    value: object | None,
    *,
    n_traces: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.zeros(n_traces, dtype=np.float64)
    return _coerce_1d_real_numeric_float64(
        value,
        name=name,
        expected_shape=(n_traces,),
        require_finite=True,
    )


def _optional_used_mask(value: object | None, *, n_traces: int) -> np.ndarray | None:
    if value is None:
        return None
    return _coerce_1d_bool_array(
        value,
        name='used_pick_mask',
        expected_shape=(n_traces,),
    )


def _stabilization_options(
    options: ResidualStaticStabilizationOptions | Mapping[str, object] | None,
) -> ResidualStaticStabilizationOptions:
    if options is None:
        return ResidualStaticStabilizationOptions()
    if isinstance(options, Mapping):
        return validate_residual_static_stabilization_options(
            ResidualStaticStabilizationOptions(**options)
        )
    return validate_residual_static_stabilization_options(options)


def _robust_options(
    options: ResidualStaticRobustOptions | Mapping[str, object] | None,
) -> ResidualStaticRobustOptions:
    if options is None:
        return ResidualStaticRobustOptions()
    if isinstance(options, Mapping):
        return validate_residual_static_robust_options(
            ResidualStaticRobustOptions(**options)
        )
    return validate_residual_static_robust_options(options)


def _lsmr_options(
    options: ResidualStaticLsmrOptions | Mapping[str, object] | None,
) -> ResidualStaticLsmrOptions | None:
    if options is None:
        return None
    if isinstance(options, Mapping):
        return validate_lsmr_options(ResidualStaticLsmrOptions(**options))
    return validate_lsmr_options(options)


def _public_result(
    robust_result,
    *,
    source: _EndpointIndex,
    receiver: _EndpointIndex,
) -> FirstBreakResidualStaticsResult:
    final = robust_result.final_solver_result
    parts = final.parameter_parts
    evaluation = final.model_evaluation
    public_result = FirstBreakResidualStaticsResult(
        moveout_model=final.layout.moveout_model,
        source_id=source.unique_ids,
        receiver_id=receiver.unique_ids,
        intercept_s=parts.intercept_s,
        slowness_s_per_offset_unit=parts.slowness_s_per_offset_unit,
        source_delay_s=parts.source_delay_s,
        receiver_delay_s=parts.receiver_delay_s,
        moveout_model_time_s=evaluation.moveout_model_time_s_sorted,
        estimated_trace_delay_s=evaluation.estimated_trace_delay_s_sorted,
        modeled_pick_time_s=evaluation.modeled_pick_time_s_sorted,
        residual_s=evaluation.residual_s_sorted,
        residual_valid_mask=evaluation.residual_valid_mask_sorted,
        initial_used_pick_mask=robust_result.initial_used_mask_sorted,
        used_pick_mask=robust_result.final_used_mask_sorted,
        rejected_pick_mask=robust_result.rejected_mask_sorted,
        rejected_iteration=robust_result.rejected_iteration_sorted,
        diagnostics=final.diagnostics,
        minimum_data=final.minimum_data,
        graph=final.minimum_data.graph,
        stabilization_options=robust_result.stabilization_options,
        robust_options=robust_result.robust_options,
        robust_iteration_summaries=robust_result.iteration_summaries,
        robust_stop_reason=robust_result.stop_reason,
        n_initial_used_picks=robust_result.n_initial_used_picks,
        n_final_used_picks=robust_result.n_final_used_picks,
        n_rejected_total=robust_result.n_rejected_total,
        n_observations=final.n_observations,
        n_model_parameters=final.n_model_parameters,
        n_gauge_rows=final.n_gauge_rows,
        n_damping_rows=final.n_damping_rows,
        max_abs_estimated_delay_s=final.max_abs_estimated_delay_s,
    )
    object.__setattr__(public_result, '_robust_solve_result', robust_result)
    return public_result


class _SourceReceiverSolveResult:
    def __init__(
        self,
        *,
        source_delay_s: np.ndarray,
        receiver_delay_s: np.ndarray,
        trace_delay_s: np.ndarray,
        residual_s: np.ndarray,
        used_mask: np.ndarray,
        diagnostics,
        graph: ResidualStaticObservationGraphSummary,
        minimum_data: SourceReceiverStaticsMinimumDataSummary,
        n_observations: int,
        n_model_parameters: int,
        n_gauge_rows: int,
        n_damping_rows: int,
        max_abs_trace_delay_s: float,
    ) -> None:
        self.source_delay_s = source_delay_s
        self.receiver_delay_s = receiver_delay_s
        self.trace_delay_s = trace_delay_s
        self.residual_s = residual_s
        self.used_mask = used_mask
        self.diagnostics = diagnostics
        self.graph = graph
        self.minimum_data = minimum_data
        self.n_observations = n_observations
        self.n_model_parameters = n_model_parameters
        self.n_gauge_rows = n_gauge_rows
        self.n_damping_rows = n_damping_rows
        self.max_abs_trace_delay_s = max_abs_trace_delay_s


class _SourceReceiverRobustResult:
    def __init__(
        self,
        *,
        initial_solver_result: _SourceReceiverSolveResult,
        final_solver_result: _SourceReceiverSolveResult,
        stabilization_options: ResidualStaticStabilizationOptions,
        robust_options: ResidualStaticRobustOptions,
        initial_used_mask: np.ndarray,
        rejected_iteration: np.ndarray,
        iteration_summaries: tuple[ResidualStaticRobustIterationSummary, ...],
        stop_reason: ResidualStaticRobustStopReason,
    ) -> None:
        final_used_mask = np.ascontiguousarray(
            final_solver_result.used_mask,
            dtype=bool,
        )
        rejected_mask = rejected_iteration >= 0
        self.initial_solver_result = initial_solver_result
        self.final_solver_result = final_solver_result
        self.stabilization_options = stabilization_options
        self.robust_options = robust_options
        self.initial_used_mask = np.ascontiguousarray(initial_used_mask, dtype=bool)
        self.final_used_mask = final_used_mask
        self.rejected_mask = np.ascontiguousarray(rejected_mask, dtype=bool)
        self.rejected_iteration = np.ascontiguousarray(
            rejected_iteration,
            dtype=np.int64,
        )
        self.iteration_summaries = iteration_summaries
        self.stop_reason = stop_reason
        self.n_initial_used_observations = int(np.count_nonzero(initial_used_mask))
        self.n_final_used_observations = int(np.count_nonzero(final_used_mask))
        self.n_rejected_total = int(np.count_nonzero(rejected_mask))


def _source_receiver_weight(
    value: object | None,
    *,
    n_traces: int,
    base_valid_mask: np.ndarray,
) -> np.ndarray:
    if value is None:
        return np.ones(n_traces, dtype=np.float64)
    arr = _coerce_1d_real_numeric_float64(
        value,
        name='weight',
        expected_shape=(n_traces,),
        require_finite=True,
    )
    if np.any(arr < 0.0):
        raise ValueError('weight must be non-negative')
    out = np.ascontiguousarray(arr, dtype=np.float64)
    out[~base_valid_mask] = 0.0
    return out


def _source_receiver_robust_options(
    *,
    robust: object,
    robust_options: ResidualStaticRobustOptions | Mapping[str, object] | None,
) -> ResidualStaticRobustOptions:
    if not isinstance(robust, bool):
        raise ValueError('robust must be a bool')
    options = _robust_options(robust_options)
    if robust is False:
        return replace(options, enabled=False)
    return options


def _solve_source_receiver_robust(
    *,
    lag: np.ndarray,
    source: _EndpointIndex,
    receiver: _EndpointIndex,
    base_valid_mask: np.ndarray,
    initial_used_mask: np.ndarray,
    weight: np.ndarray,
    stabilization_options: ResidualStaticStabilizationOptions,
    robust_options: ResidualStaticRobustOptions,
    lsmr_options: ResidualStaticLsmrOptions | None,
) -> _SourceReceiverRobustResult:
    rejected_iteration = np.full(lag.shape, -1, dtype=np.int64)
    initial_solver_result = _solve_source_receiver_once(
        lag=lag,
        source=source,
        receiver=receiver,
        base_valid_mask=base_valid_mask,
        used_mask=initial_used_mask,
        weight=weight,
        stabilization_options=stabilization_options,
        lsmr_options=lsmr_options,
    )
    if not robust_options.enabled:
        return _SourceReceiverRobustResult(
            initial_solver_result=initial_solver_result,
            final_solver_result=initial_solver_result,
            stabilization_options=stabilization_options,
            robust_options=robust_options,
            initial_used_mask=initial_used_mask,
            rejected_iteration=rejected_iteration,
            iteration_summaries=(),
            stop_reason='disabled',
        )

    current_used_mask = np.ascontiguousarray(initial_used_mask, dtype=bool)
    final_solver_result = initial_solver_result
    stop_reason: ResidualStaticRobustStopReason | None = None
    iteration_summaries: list[ResidualStaticRobustIterationSummary] = []

    for iteration_index in range(robust_options.max_iterations):
        solver_result = (
            initial_solver_result
            if iteration_index == 0
            else _solve_source_receiver_once(
                lag=lag,
                source=source,
                receiver=receiver,
                base_valid_mask=base_valid_mask,
                used_mask=current_used_mask,
                weight=weight,
                stabilization_options=stabilization_options,
                lsmr_options=lsmr_options,
            )
        )
        final_solver_result = solver_result
        residual_s = solver_result.residual_s[current_used_mask]
        outlier_local, center_s, scale_s, cutoff_s = build_residual_static_outlier_mask(
            residual_s,
            method=robust_options.method,
            threshold=robust_options.threshold,
        )
        n_used_before = int(np.count_nonzero(current_used_mask))
        max_abs_centered = (
            float(np.max(np.abs(residual_s - center_s)))
            if residual_s.size
            else 0.0
        )
        if scale_s <= ROBUST_SCALE_FLOOR_S:
            stop_reason = 'zero_scale'
            iteration_summaries.append(
                ResidualStaticRobustIterationSummary(
                    iteration_index=iteration_index,
                    method=robust_options.method,
                    n_used_before=n_used_before,
                    n_rejected_this_iteration=0,
                    n_used_after=n_used_before,
                    residual_center_s=center_s,
                    residual_scale_s=scale_s,
                    residual_cutoff_s=cutoff_s,
                    max_abs_centered_residual_s=max_abs_centered,
                    converged=False,
                    stop_reason=stop_reason,
                )
            )
            break
        if not np.any(outlier_local):
            stop_reason = 'converged'
            iteration_summaries.append(
                ResidualStaticRobustIterationSummary(
                    iteration_index=iteration_index,
                    method=robust_options.method,
                    n_used_before=n_used_before,
                    n_rejected_this_iteration=0,
                    n_used_after=n_used_before,
                    residual_center_s=center_s,
                    residual_scale_s=scale_s,
                    residual_cutoff_s=cutoff_s,
                    max_abs_centered_residual_s=max_abs_centered,
                    converged=True,
                    stop_reason=stop_reason,
                )
            )
            break

        current_used_indices = np.flatnonzero(current_used_mask)
        rejected_indices = current_used_indices[outlier_local]
        proposed_used_mask = current_used_mask.copy()
        proposed_used_mask[rejected_indices] = False
        _validate_source_receiver_min_used_fraction(
            initial_used_mask,
            proposed_used_mask,
            min_used_fraction=robust_options.min_used_fraction,
        )
        current_used_mask = proposed_used_mask
        rejected_iteration[rejected_indices] = iteration_index
        n_rejected = int(rejected_indices.shape[0])
        summary_stop_reason: ResidualStaticRobustStopReason | None = None
        if iteration_index == robust_options.max_iterations - 1:
            summary_stop_reason = 'max_iterations'
        iteration_summaries.append(
            ResidualStaticRobustIterationSummary(
                iteration_index=iteration_index,
                method=robust_options.method,
                n_used_before=n_used_before,
                n_rejected_this_iteration=n_rejected,
                n_used_after=int(np.count_nonzero(current_used_mask)),
                residual_center_s=center_s,
                residual_scale_s=scale_s,
                residual_cutoff_s=cutoff_s,
                max_abs_centered_residual_s=max_abs_centered,
                converged=False,
                stop_reason=summary_stop_reason,
            )
        )
    else:
        stop_reason = 'max_iterations'

    if stop_reason == 'max_iterations' or not np.array_equal(
        final_solver_result.used_mask,
        current_used_mask,
    ):
        final_solver_result = _solve_source_receiver_once(
            lag=lag,
            source=source,
            receiver=receiver,
            base_valid_mask=base_valid_mask,
            used_mask=current_used_mask,
            weight=weight,
            stabilization_options=stabilization_options,
            lsmr_options=lsmr_options,
        )
    if stop_reason is None:
        raise RuntimeError('source/receiver robust solver did not set a stop reason')

    return _SourceReceiverRobustResult(
        initial_solver_result=initial_solver_result,
        final_solver_result=final_solver_result,
        stabilization_options=stabilization_options,
        robust_options=robust_options,
        initial_used_mask=initial_used_mask,
        rejected_iteration=rejected_iteration,
        iteration_summaries=tuple(iteration_summaries),
        stop_reason=stop_reason,
    )


def _solve_source_receiver_once(
    *,
    lag: np.ndarray,
    source: _EndpointIndex,
    receiver: _EndpointIndex,
    base_valid_mask: np.ndarray,
    used_mask: np.ndarray,
    weight: np.ndarray,
    stabilization_options: ResidualStaticStabilizationOptions,
    lsmr_options: ResidualStaticLsmrOptions | None,
) -> _SourceReceiverSolveResult:
    if not np.any(used_mask):
        raise ValueError('at least one used source/receiver observation is required')
    n_sources = int(source.unique_ids.shape[0])
    n_receivers = int(receiver.unique_ids.shape[0])
    n_model_parameters = n_sources + n_receivers
    graph = _build_source_receiver_graph_summary(
        source_index=source.index_sorted,
        receiver_index=receiver.index_sorted,
        n_sources=n_sources,
        n_receivers=n_receivers,
        used_mask=used_mask,
    )
    matrix, rhs_s, n_gauge_rows, n_damping_rows = _build_source_receiver_system(
        lag=lag,
        source_index=source.index_sorted,
        receiver_index=receiver.index_sorted,
        n_sources=n_sources,
        n_receivers=n_receivers,
        used_mask=used_mask,
        weight=weight,
        graph=graph,
        damping_lambda=stabilization_options.damping_lambda,
    )
    raw_result = run_sparse_lsmr(matrix, rhs_s, options=lsmr_options)
    parameter = np.ascontiguousarray(raw_result.parameter_vector, dtype=np.float64)
    source_delay_s = np.ascontiguousarray(parameter[:n_sources], dtype=np.float64)
    receiver_delay_s = np.ascontiguousarray(parameter[n_sources:], dtype=np.float64)
    trace_delay_s = np.ascontiguousarray(
        source_delay_s[source.index_sorted] + receiver_delay_s[receiver.index_sorted],
        dtype=np.float64,
    )
    residual_s = np.full(lag.shape, np.nan, dtype=np.float64)
    residual_s[base_valid_mask] = lag[base_valid_mask] - trace_delay_s[
        base_valid_mask
    ]
    max_abs_trace_delay_s = (
        float(np.max(np.abs(trace_delay_s[base_valid_mask])))
        if np.any(base_valid_mask)
        else 0.0
    )
    if max_abs_trace_delay_s > stabilization_options.max_abs_estimated_delay_ms / 1000.0:
        raise ValueError('trace_delay_s exceeds max_abs_estimated_delay_ms')
    minimum_data = _build_source_receiver_minimum_data_summary(
        source=source,
        receiver=receiver,
        base_valid_mask=base_valid_mask,
        used_mask=used_mask,
        weight=weight,
        graph=graph,
        stabilization_options=stabilization_options,
    )
    return _SourceReceiverSolveResult(
        source_delay_s=source_delay_s,
        receiver_delay_s=receiver_delay_s,
        trace_delay_s=trace_delay_s,
        residual_s=residual_s,
        used_mask=np.ascontiguousarray(used_mask, dtype=bool),
        diagnostics=raw_result.diagnostics,
        graph=graph,
        minimum_data=minimum_data,
        n_observations=int(np.count_nonzero(used_mask)),
        n_model_parameters=n_model_parameters,
        n_gauge_rows=n_gauge_rows,
        n_damping_rows=n_damping_rows,
        max_abs_trace_delay_s=max_abs_trace_delay_s,
    )


def _build_source_receiver_system(
    *,
    lag: np.ndarray,
    source_index: np.ndarray,
    receiver_index: np.ndarray,
    n_sources: int,
    n_receivers: int,
    used_mask: np.ndarray,
    weight: np.ndarray,
    graph: ResidualStaticObservationGraphSummary,
    damping_lambda: float,
):
    used_trace_indices = np.flatnonzero(used_mask)
    n_observation_rows = int(used_trace_indices.shape[0])
    n_cols = n_sources + n_receivers
    sqrt_weight = np.sqrt(weight[used_trace_indices])
    observation_rows = np.repeat(np.arange(n_observation_rows, dtype=np.int64), 2)
    observation_cols = np.empty(n_observation_rows * 2, dtype=np.int64)
    observation_cols[0::2] = source_index[used_trace_indices]
    observation_cols[1::2] = n_sources + receiver_index[used_trace_indices]
    observation_data = np.repeat(sqrt_weight, 2).astype(np.float64, copy=False)
    observation_matrix = sparse.coo_matrix(
        (observation_data, (observation_rows, observation_cols)),
        shape=(n_observation_rows, n_cols),
        dtype=np.float64,
    ).tocsr()
    observation_rhs = np.ascontiguousarray(
        lag[used_trace_indices] * sqrt_weight,
        dtype=np.float64,
    )

    gauge_matrix, gauge_rhs = _build_source_receiver_gauge_matrix(
        n_sources=n_sources,
        n_receivers=n_receivers,
        graph=graph,
    )
    damping_matrix, damping_rhs = _build_source_receiver_damping_matrix(
        n_sources=n_sources,
        n_receivers=n_receivers,
        damping_lambda=damping_lambda,
    )
    matrix = sparse.vstack(
        [observation_matrix, gauge_matrix, damping_matrix],
        format='csr',
        dtype=np.float64,
    )
    rhs_s = np.ascontiguousarray(
        np.concatenate([observation_rhs, gauge_rhs, damping_rhs]),
        dtype=np.float64,
    )
    return matrix, rhs_s, int(gauge_rhs.shape[0]), int(damping_rhs.shape[0])


def _build_source_receiver_gauge_matrix(
    *,
    n_sources: int,
    n_receivers: int,
    graph: ResidualStaticObservationGraphSummary,
):
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    for component in range(graph.n_components):
        source_nodes = np.flatnonzero(graph.source_component_index == component)
        receiver_nodes = np.flatnonzero(graph.receiver_component_index == component)
        row = component
        if source_nodes.size:
            scale = 1.0 / float(source_nodes.size)
            row_indices.extend([row] * int(source_nodes.size))
            col_indices.extend(source_nodes.astype(int).tolist())
            data.extend([scale] * int(source_nodes.size))
        elif receiver_nodes.size:
            scale = 1.0 / float(receiver_nodes.size)
            row_indices.extend([row] * int(receiver_nodes.size))
            col_indices.extend((n_sources + receiver_nodes).astype(int).tolist())
            data.extend([scale] * int(receiver_nodes.size))
    n_rows = graph.n_components
    matrix = sparse.coo_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(col_indices, dtype=np.int64),
            ),
        ),
        shape=(n_rows, n_sources + n_receivers),
        dtype=np.float64,
    ).tocsr()
    return matrix, np.zeros(n_rows, dtype=np.float64)


def _build_source_receiver_damping_matrix(
    *,
    n_sources: int,
    n_receivers: int,
    damping_lambda: float,
):
    damping = float(damping_lambda)
    if damping < 0.0 or not np.isfinite(damping):
        raise ValueError('damping_lambda must be non-negative')
    n_cols = n_sources + n_receivers
    if damping == 0.0:
        return sparse.csr_matrix((0, n_cols), dtype=np.float64), np.empty(
            0,
            dtype=np.float64,
        )
    row_indices = np.arange(n_cols, dtype=np.int64)
    col_indices = np.arange(n_cols, dtype=np.int64)
    data = np.full(n_cols, damping, dtype=np.float64)
    matrix = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_cols, n_cols),
        dtype=np.float64,
    ).tocsr()
    return matrix, np.zeros(n_cols, dtype=np.float64)


def _build_source_receiver_graph_summary(
    *,
    source_index: np.ndarray,
    receiver_index: np.ndarray,
    n_sources: int,
    n_receivers: int,
    used_mask: np.ndarray,
) -> ResidualStaticObservationGraphSummary:
    n_nodes = n_sources + n_receivers
    parent = np.arange(n_nodes, dtype=np.int64)
    rank = np.zeros(n_nodes, dtype=np.int8)

    def find(node: int) -> int:
        root = node
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[node]) != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    used_trace_indices = np.flatnonzero(used_mask)
    for trace_index in used_trace_indices:
        union(
            int(source_index[trace_index]),
            n_sources + int(receiver_index[trace_index]),
        )

    component_by_root: dict[int, int] = {}
    node_component = np.empty(n_nodes, dtype=np.int64)
    for node in range(n_nodes):
        root = find(node)
        component = component_by_root.setdefault(root, len(component_by_root))
        node_component[node] = component

    n_components = int(len(component_by_root))
    component_observation_counts = np.zeros(n_components, dtype=np.int64)
    for trace_index in used_trace_indices:
        component = node_component[int(source_index[trace_index])]
        component_observation_counts[component] += 1

    source_component_index = np.ascontiguousarray(
        node_component[:n_sources],
        dtype=np.int64,
    )
    receiver_component_index = np.ascontiguousarray(
        node_component[n_sources:],
        dtype=np.int64,
    )
    return ResidualStaticObservationGraphSummary(
        n_components=n_components,
        source_component_index=source_component_index,
        receiver_component_index=receiver_component_index,
        component_observation_counts=np.ascontiguousarray(
            component_observation_counts,
            dtype=np.int64,
        ),
        component_source_counts=np.bincount(
            source_component_index,
            minlength=n_components,
        ).astype(np.int64, copy=False),
        component_receiver_counts=np.bincount(
            receiver_component_index,
            minlength=n_components,
        ).astype(np.int64, copy=False),
    )


def _build_source_receiver_minimum_data_summary(
    *,
    source: _EndpointIndex,
    receiver: _EndpointIndex,
    base_valid_mask: np.ndarray,
    used_mask: np.ndarray,
    weight: np.ndarray,
    graph: ResidualStaticObservationGraphSummary,
    stabilization_options: ResidualStaticStabilizationOptions,
) -> SourceReceiverStaticsMinimumDataSummary:
    source_counts = np.bincount(
        source.index_sorted[used_mask],
        minlength=int(source.unique_ids.shape[0]),
    ).astype(np.int64, copy=False)
    receiver_counts = np.bincount(
        receiver.index_sorted[used_mask],
        minlength=int(receiver.unique_ids.shape[0]),
    ).astype(np.int64, copy=False)
    under_source = source_counts < stabilization_options.min_picks_per_source
    under_receiver = receiver_counts < stabilization_options.min_picks_per_receiver
    n_used = int(np.count_nonzero(used_mask))
    n_model_parameters = int(source.unique_ids.shape[0] + receiver.unique_ids.shape[0])
    n_effective_parameters = n_model_parameters - graph.n_components
    return SourceReceiverStaticsMinimumDataSummary(
        n_used_observations=n_used,
        n_sources=int(source.unique_ids.shape[0]),
        n_receivers=int(receiver.unique_ids.shape[0]),
        n_model_parameters=n_model_parameters,
        n_effective_parameters=n_effective_parameters,
        source_used_observation_counts=np.ascontiguousarray(
            source_counts,
            dtype=np.int64,
        ),
        receiver_used_observation_counts=np.ascontiguousarray(
            receiver_counts,
            dtype=np.int64,
        ),
        underconstrained_source_ids=np.ascontiguousarray(
            source.unique_ids[under_source],
            dtype=source.unique_ids.dtype,
        ),
        underconstrained_receiver_ids=np.ascontiguousarray(
            receiver.unique_ids[under_receiver],
            dtype=receiver.unique_ids.dtype,
        ),
        n_zero_weight_observations=int(
            np.count_nonzero(base_valid_mask & (weight == 0.0))
        ),
        rank_deficient_possible=bool(
            graph.n_components != 1
            or n_used < n_effective_parameters
            or n_used < stabilization_options.min_valid_picks
            or np.any(under_source)
            or np.any(under_receiver)
        ),
    )


def _validate_source_receiver_min_used_fraction(
    initial_used_mask: np.ndarray,
    proposed_used_mask: np.ndarray,
    *,
    min_used_fraction: float,
) -> None:
    n_initial = int(np.count_nonzero(initial_used_mask))
    n_proposed = int(np.count_nonzero(proposed_used_mask))
    if n_initial == 0:
        raise ValueError('initial_used_mask must contain at least one observation')
    if n_proposed / n_initial < min_used_fraction:
        raise ValueError('robust rejection would violate min_used_fraction')


def _source_receiver_public_result(
    robust_result: _SourceReceiverRobustResult,
    *,
    source: _EndpointIndex,
    receiver: _EndpointIndex,
    weight: np.ndarray,
) -> SourceReceiverStaticsResult:
    final = robust_result.final_solver_result
    return SourceReceiverStaticsResult(
        source_unique_ids=source.unique_ids,
        receiver_unique_ids=receiver.unique_ids,
        source_delay_s=final.source_delay_s,
        receiver_delay_s=final.receiver_delay_s,
        trace_delay_s=final.trace_delay_s,
        applied_shift_s=delay_to_applied_shift(final.trace_delay_s),
        residual_s=final.residual_s,
        initial_used_mask=robust_result.initial_used_mask,
        used_mask=robust_result.final_used_mask,
        rejected_mask=robust_result.rejected_mask,
        rejected_iteration=robust_result.rejected_iteration,
        weight=np.ascontiguousarray(weight, dtype=np.float64),
        diagnostics=final.diagnostics,
        minimum_data=final.minimum_data,
        graph=final.graph,
        stabilization_options=robust_result.stabilization_options,
        robust_options=robust_result.robust_options,
        robust_iteration_summaries=robust_result.iteration_summaries,
        robust_stop_reason=robust_result.stop_reason,
        n_initial_used_observations=robust_result.n_initial_used_observations,
        n_final_used_observations=robust_result.n_final_used_observations,
        n_rejected_total=robust_result.n_rejected_total,
        n_observations=final.n_observations,
        n_model_parameters=final.n_model_parameters,
        n_gauge_rows=final.n_gauge_rows,
        n_damping_rows=final.n_damping_rows,
        max_abs_trace_delay_s=final.max_abs_trace_delay_s,
    )


__all__ = [
    'delay_to_applied_shift',
    'solve_first_break_residual_statics',
    'solve_source_receiver_statics',
]
