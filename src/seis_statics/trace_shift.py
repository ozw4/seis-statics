"""Trace-shift validation and numerical application helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from seis_statics._validation import (
    coerce_1d_bool_array,
    coerce_1d_real_numeric_float64,
    coerce_1d_string_array,
    coerce_nonnegative_finite_float,
    coerce_positive_finite_float,
    coerce_positive_int,
)


@dataclass(frozen=True)
class StaticTraceShiftValidationResult:
    trace_shift_s_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    trace_static_status_counts: dict[str, int]
    max_abs_shift_ms: float
    max_abs_applied_shift_ms: float
    exceeds_max_abs_shift_count: int
    n_valid_trace_shifts: int
    n_invalid_trace_shifts: int
    n_zero_trace_shifts: int
    n_positive_trace_shifts: int
    n_negative_trace_shifts: int


def coerce_finite_float32_fill_value(fill_value: float) -> np.float32:
    """Validate and return ``fill_value`` as a finite float32 scalar."""
    fill = float(fill_value)
    if not np.isfinite(fill):
        raise ValueError('fill_value must be finite')
    if abs(fill) > float(np.finfo(np.float32).max):
        raise ValueError('fill_value must be representable as finite float32')
    return np.float32(fill)


def shift_traces_linear(
    traces: np.ndarray,
    shifts_s: np.ndarray,
    dt: float,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply per-trace event-time shifts with linear interpolation."""
    arr = np.asarray(traces)
    if arr.ndim != 2:
        raise ValueError('traces must be a 2D array')

    n_traces, n_samples = arr.shape
    if n_traces <= 0:
        raise ValueError('traces must contain at least one trace')
    if n_samples <= 0:
        raise ValueError('traces must contain at least one sample')

    shifts = np.asarray(shifts_s, dtype=np.float64)
    if shifts.shape != (n_traces,):
        raise ValueError('shifts_s shape must match trace count')

    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError('dt must be finite and greater than 0')
    if not np.all(np.isfinite(shifts)):
        raise ValueError('shifts_s must contain only finite values')

    fill = coerce_finite_float32_fill_value(fill_value)

    arr_f32 = np.asarray(arr, dtype=np.float32)
    out = np.full((n_traces, n_samples), fill, dtype=np.float32)
    sample_indices = np.arange(n_samples, dtype=np.float64)
    max_source = float(n_samples - 1)

    for trace_idx, shift_s in enumerate(shifts):
        source = sample_indices - (float(shift_s) / dt_value)
        valid = (source >= 0.0) & (source <= max_source)
        if not np.any(valid):
            continue

        source_valid = source[valid]
        lo = np.floor(source_valid).astype(np.int64)
        hi = np.minimum(lo + 1, n_samples - 1)
        frac = (source_valid - lo).astype(np.float32, copy=False)
        row = arr_f32[trace_idx]
        out[trace_idx, valid] = row[lo] * (1.0 - frac) + row[hi] * frac

    return np.ascontiguousarray(out, dtype=np.float32)


def apply_trace_shifts_to_array(
    *,
    traces: np.ndarray,
    sample_interval_s: float,
    trace_shift_s_sorted: np.ndarray,
    interpolation: Literal['linear'] = 'linear',
    fill_value: float = 0.0,
    output_dtype: np.dtype | str = np.float32,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    """Apply sorted per-trace shifts using ``corrected(t) = raw(t - shift_s)``."""
    if interpolation != 'linear':
        raise error_type('interpolation must be "linear"')
    dtype = np.dtype(output_dtype)
    if dtype != np.dtype('float32'):
        raise error_type('output_dtype must be "float32"')

    arr = np.asarray(traces)
    if arr.ndim != 2:
        raise error_type('traces must be a 2D array')
    dt = coerce_positive_finite_float(
        sample_interval_s,
        name='sample_interval_s',
        error_type=error_type,
    )
    try:
        shifted = shift_traces_linear(
            arr,
            np.asarray(trace_shift_s_sorted, dtype=np.float64),
            dt,
            fill_value=fill_value,
        )
    except ValueError as exc:
        raise error_type(str(exc)) from exc
    return np.ascontiguousarray(shifted, dtype=np.float32)


def validate_trace_shifts_for_application(
    *,
    trace_shift_s_sorted: np.ndarray,
    trace_static_valid_mask_sorted: np.ndarray,
    trace_static_status_sorted: np.ndarray,
    n_traces: int,
    max_abs_shift_ms: float,
    shift_field_name: str,
    valid_mask_name: str = 'trace_static_valid_mask_sorted',
    status_name: str = 'trace_static_status_sorted',
    require_all_traces_valid: bool = True,
    invalid_message_prefix: str = 'Static corrections contain invalid trace shifts',
    error_type: type[Exception] = ValueError,
) -> StaticTraceShiftValidationResult:
    """Validate one-dimensional static shift, mask, and status arrays."""
    expected_shape = (
        coerce_positive_int(n_traces, name='n_traces', error_type=error_type),
    )
    shifts = require_1d_float64(
        trace_shift_s_sorted,
        name=shift_field_name,
        expected_shape=expected_shape,
        allow_nonfinite=True,
        error_type=error_type,
    )
    valid_mask = require_1d_bool(
        trace_static_valid_mask_sorted,
        name=valid_mask_name,
        expected_shape=expected_shape,
        error_type=error_type,
    )
    statuses = require_1d_string(
        trace_static_status_sorted,
        name=status_name,
        expected_shape=expected_shape,
        error_type=error_type,
    )
    status_counts = status_counts_by_value(statuses)
    max_abs_ms = coerce_nonnegative_finite_float(
        max_abs_shift_ms,
        name='max_abs_shift_ms',
        error_type=error_type,
    )

    if require_all_traces_valid and not bool(np.all(valid_mask)):
        invalid_count = int(np.count_nonzero(~valid_mask))
        raise error_type(
            f'{invalid_message_prefix}; corrected TraceStore was not created. '
            f'invalid_trace_shift_count={invalid_count}; '
            f'trace_static_status_counts={status_counts}'
        )

    applied_mask = np.ones(expected_shape, dtype=bool) if require_all_traces_valid else valid_mask
    if not np.any(applied_mask):
        raise error_type('no trace shifts are selected for application')
    if not np.all(np.isfinite(shifts[applied_mask])):
        raise error_type(
            f'{shift_field_name} contains non-finite shifts for traces selected '
            'for application'
        )

    shift_ms = shifts * 1000.0
    exceeds_mask = np.abs(shift_ms) > max_abs_ms
    exceeds_count = int(np.count_nonzero(exceeds_mask & applied_mask))
    max_abs_applied_ms = float(np.max(np.abs(shift_ms[applied_mask])))
    if exceeds_count:
        raise error_type(
            f'{shift_field_name} exceeds max_abs_shift_ms: '
            f'{max_abs_applied_ms:.6g} > {max_abs_ms:.6g}; '
            f'exceeds_max_abs_shift_count={exceeds_count}; '
            f'trace_static_status_counts={status_counts}'
        )

    valid_count = int(np.count_nonzero(valid_mask))
    invalid_count = int(valid_mask.size - valid_count)
    return StaticTraceShiftValidationResult(
        trace_shift_s_sorted=np.ascontiguousarray(shifts, dtype=np.float64),
        trace_static_valid_mask_sorted=np.ascontiguousarray(valid_mask, dtype=bool),
        trace_static_status_sorted=np.ascontiguousarray(statuses),
        trace_static_status_counts=status_counts,
        max_abs_shift_ms=max_abs_ms,
        max_abs_applied_shift_ms=max_abs_applied_ms,
        exceeds_max_abs_shift_count=exceeds_count,
        n_valid_trace_shifts=valid_count,
        n_invalid_trace_shifts=invalid_count,
        n_zero_trace_shifts=int(np.count_nonzero(shift_ms == 0.0)),
        n_positive_trace_shifts=int(np.count_nonzero(shift_ms > 0.0)),
        n_negative_trace_shifts=int(np.count_nonzero(shift_ms < 0.0)),
    )


def require_1d_float64(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    allow_nonfinite: bool = False,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        raise error_type(f'{name} must not have object dtype')
    if not np.issubdtype(arr.dtype, np.floating):
        raise error_type(f'{name} must be a float array')
    return coerce_1d_real_numeric_float64(
        arr,
        name=name,
        expected_shape=expected_shape,
        allow_nonfinite=allow_nonfinite,
        error_type=error_type,
    )


def require_1d_bool(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    return coerce_1d_bool_array(
        value,
        name=name,
        expected_shape=expected_shape,
        error_type=error_type,
    )


def require_1d_string(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    return coerce_1d_string_array(
        value,
        name=name,
        expected_shape=expected_shape,
        reject_object_dtype=True,
        output_dtype=str,
        error_type=error_type,
    )


def status_counts_by_value(statuses: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(statuses, dtype=str), return_counts=True)
    return {str(status): int(count) for status, count in zip(unique, counts, strict=True)}


__all__ = [
    'StaticTraceShiftValidationResult',
    'apply_trace_shifts_to_array',
    'coerce_finite_float32_fill_value',
    'require_1d_bool',
    'require_1d_float64',
    'require_1d_string',
    'shift_traces_linear',
    'status_counts_by_value',
    'validate_trace_shifts_for_application',
]
