from __future__ import annotations

import numpy as np
import pytest

from seis_statics.time_term import (
    TimeTermAppliedShiftResult,
    compose_time_term_applied_shifts,
    delay_to_applied_shift,
)


def test_delay_to_applied_shift_negates_scalar_and_array() -> None:
    assert delay_to_applied_shift(0.012) == pytest.approx(-0.012)

    out = delay_to_applied_shift(np.asarray([0.010, -0.004, np.nan]))

    np.testing.assert_allclose(out, [-0.010, 0.004, np.nan], equal_nan=True)
    assert out.dtype == np.float64


def test_compose_time_term_applied_shifts_sums_datum_residual_and_weathering() -> None:
    result = compose_time_term_applied_shifts(
        trace_time_term_delay_s_sorted=np.asarray([0.010, -0.004, 0.0]),
        datum_applied_shift_s_sorted=np.asarray([0.001, 0.002, 0.003]),
        residual_applied_shift_s_sorted=np.asarray([-0.002, 0.004, 0.005]),
    )

    assert isinstance(result, TimeTermAppliedShiftResult)
    np.testing.assert_allclose(
        result.applied_weathering_shift_s_sorted,
        [-0.010, 0.004, -0.0],
    )
    np.testing.assert_allclose(
        result.final_applied_shift_s_sorted,
        [-0.011, 0.010, 0.008],
    )
    np.testing.assert_array_equal(result.valid_shift_mask_sorted, [True, True, True])
    assert result.max_abs_final_applied_shift_s == pytest.approx(0.011)


def test_invalid_trace_policy_returns_nan_and_no_op_mask() -> None:
    result = compose_time_term_applied_shifts(
        trace_time_term_delay_s_sorted=np.asarray([0.010, 0.020, np.nan, 0.030]),
        datum_applied_shift_s_sorted=np.asarray([0.001, 0.002, 0.003, np.nan]),
        residual_applied_shift_s_sorted=np.asarray([0.0, 0.0, 0.0, 0.0]),
        valid_trace_mask_sorted=np.asarray([True, False, True, True]),
    )

    np.testing.assert_allclose(
        result.applied_weathering_shift_s_sorted,
        [-0.010, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.final_applied_shift_s_sorted,
        [-0.009, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        result.valid_shift_mask_sorted,
        [True, False, False, False],
    )


def test_compose_time_term_applied_shifts_rejects_max_shift_violation() -> None:
    with pytest.raises(ValueError, match='max_abs_final_applied_shift_ms'):
        compose_time_term_applied_shifts(
            trace_time_term_delay_s_sorted=np.asarray([0.010]),
            datum_applied_shift_s_sorted=np.asarray([0.0]),
            residual_applied_shift_s_sorted=np.asarray([0.0]),
            max_abs_final_applied_shift_ms=5.0,
        )
