from __future__ import annotations

import numpy as np
import pytest

from seis_statics.trace_shift import (
    apply_trace_shifts_to_array,
    shift_traces_linear,
    validate_trace_shifts_for_application,
)


def test_shift_traces_linear_integer_sample_positive_and_negative_spike_shifts() -> None:
    traces = np.zeros((2, 6), dtype=np.float32)
    traces[:, 2] = 1.0

    shifted = shift_traces_linear(
        traces,
        np.asarray([1.0, -1.0], dtype=np.float64),
        dt=1.0,
    )

    np.testing.assert_allclose(
        shifted,
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_shift_traces_linear_fractional_shift_interpolates_between_samples() -> None:
    traces = np.asarray([[0.0, 10.0, 20.0, 30.0]], dtype=np.float32)

    shifted = shift_traces_linear(
        traces,
        np.asarray([0.5], dtype=np.float64),
        dt=1.0,
        fill_value=-1.0,
    )

    np.testing.assert_allclose(
        shifted,
        np.asarray([[-1.0, 5.0, 15.0, 25.0]], dtype=np.float32),
    )


def test_shift_traces_linear_uses_fill_value_outside_trace_bounds() -> None:
    traces = np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32)

    shifted = shift_traces_linear(
        traces,
        np.asarray([2.0], dtype=np.float64),
        dt=1.0,
        fill_value=-7.0,
    )

    np.testing.assert_allclose(
        shifted,
        np.asarray([[-7.0, -7.0, 10.0]], dtype=np.float32),
    )


def test_validate_trace_shifts_for_application_rejects_non_finite_selected_shift() -> None:
    with pytest.raises(ValueError, match='contains non-finite shifts'):
        validate_trace_shifts_for_application(
            trace_shift_s_sorted=np.asarray([0.0, np.nan], dtype=np.float64),
            trace_static_valid_mask_sorted=np.asarray([True, True]),
            trace_static_status_sorted=np.asarray(['ok', 'ok']),
            n_traces=2,
            max_abs_shift_ms=100.0,
            shift_field_name='trace_shift_s_sorted',
        )


def test_apply_trace_shifts_to_array_uses_corrected_t_equals_raw_t_minus_shift() -> None:
    traces = np.zeros((1, 5), dtype=np.float32)
    traces[0, 1] = 1.0

    corrected = apply_trace_shifts_to_array(
        traces=traces,
        sample_interval_s=1.0,
        trace_shift_s_sorted=np.asarray([1.0], dtype=np.float64),
        fill_value=0.0,
    )

    np.testing.assert_allclose(
        corrected,
        np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )
