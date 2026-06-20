"""Compose time-term delay estimates into applied trace shifts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
    coerce_positive_int as _coerce_positive_int,
)


@dataclass(frozen=True)
class TimeTermAppliedShiftResult:
    trace_time_term_delay_s_sorted: np.ndarray
    applied_weathering_shift_s_sorted: np.ndarray
    final_applied_shift_s_sorted: np.ndarray
    valid_shift_mask_sorted: np.ndarray
    max_abs_final_applied_shift_s: float


def delay_to_applied_shift(delay_s):
    """Return the trace shift that applies the negative of a delay estimate."""
    delay = np.asarray(delay_s, dtype=np.float64)
    shift = -delay
    if shift.ndim == 0:
        return float(shift)
    return np.ascontiguousarray(shift, dtype=np.float64)


def compose_time_term_applied_shifts(
    *,
    trace_time_term_delay_s_sorted: np.ndarray,
    datum_applied_shift_s_sorted: np.ndarray,
    residual_applied_shift_s_sorted: np.ndarray,
    valid_trace_mask_sorted: np.ndarray | None = None,
    max_abs_final_applied_shift_ms: float | None = 1000.0,
) -> TimeTermAppliedShiftResult:
    """Compose datum, residual, and weathering applied trace shifts."""
    time_term_delay = _coerce_1d_real_numeric_float64(
        trace_time_term_delay_s_sorted,
        name='trace_time_term_delay_s_sorted',
    )
    n_traces = _coerce_positive_int(time_term_delay.shape[0], name='n_traces')
    expected_shape = (n_traces,)
    datum_shift = _coerce_1d_real_numeric_float64(
        datum_applied_shift_s_sorted,
        name='datum_applied_shift_s_sorted',
        expected_shape=expected_shape,
    )
    residual_shift = _coerce_1d_real_numeric_float64(
        residual_applied_shift_s_sorted,
        name='residual_applied_shift_s_sorted',
        expected_shape=expected_shape,
    )
    if valid_trace_mask_sorted is None:
        valid_mask = np.ones(expected_shape, dtype=bool)
    else:
        valid_mask = _coerce_1d_bool_array(
            valid_trace_mask_sorted,
            name='valid_trace_mask_sorted',
            expected_shape=expected_shape,
        )

    valid_shift_mask = np.ascontiguousarray(
        valid_mask
        & np.isfinite(time_term_delay)
        & np.isfinite(datum_shift)
        & np.isfinite(residual_shift),
        dtype=bool,
    )
    applied_weathering_shift = np.full(expected_shape, np.nan, dtype=np.float64)
    final_applied_shift = np.full(expected_shape, np.nan, dtype=np.float64)
    applied_weathering_shift[valid_shift_mask] = -time_term_delay[valid_shift_mask]
    final_applied_shift[valid_shift_mask] = (
        datum_shift[valid_shift_mask]
        + residual_shift[valid_shift_mask]
        + applied_weathering_shift[valid_shift_mask]
    )
    max_abs_shift = _validate_max_abs_final_shift(
        final_applied_shift,
        valid_shift_mask=valid_shift_mask,
        max_abs_final_applied_shift_ms=max_abs_final_applied_shift_ms,
    )

    return TimeTermAppliedShiftResult(
        trace_time_term_delay_s_sorted=np.ascontiguousarray(
            time_term_delay,
            dtype=np.float64,
        ),
        applied_weathering_shift_s_sorted=np.ascontiguousarray(
            applied_weathering_shift,
            dtype=np.float64,
        ),
        final_applied_shift_s_sorted=np.ascontiguousarray(
            final_applied_shift,
            dtype=np.float64,
        ),
        valid_shift_mask_sorted=valid_shift_mask,
        max_abs_final_applied_shift_s=max_abs_shift,
    )


def _validate_max_abs_final_shift(
    final_applied_shift: np.ndarray,
    *,
    valid_shift_mask: np.ndarray,
    max_abs_final_applied_shift_ms: float | None,
) -> float:
    if max_abs_final_applied_shift_ms is not None:
        max_abs_ms = _coerce_nonnegative_finite_float(
            max_abs_final_applied_shift_ms,
            name='max_abs_final_applied_shift_ms',
        )
    else:
        max_abs_ms = None
    if not np.any(valid_shift_mask):
        return 0.0
    finite_shift = final_applied_shift[valid_shift_mask]
    max_abs_s = float(np.max(np.abs(finite_shift)))
    if max_abs_ms is not None and max_abs_s > max_abs_ms / 1000.0:
        raise ValueError(
            'final_applied_shift_s_sorted exceeds max_abs_final_applied_shift_ms'
        )
    return max_abs_s


__all__ = [
    'TimeTermAppliedShiftResult',
    'compose_time_term_applied_shifts',
    'delay_to_applied_shift',
]
