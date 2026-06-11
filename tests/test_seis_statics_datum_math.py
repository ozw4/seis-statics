from __future__ import annotations

import numpy as np
import pytest

from seis_statics.datum.math import (
    DatumStaticResult as AppDatumStaticResult,
    compute_datum_static_shifts as app_compute_datum_static_shifts,
)
from seis_statics.datum import DatumStaticResult, compute_datum_static_shifts
from seis_statics.datum.math import compute_datum_static_shifts as math_compute_datum_static_shifts


def test_datum_math_is_available_from_core_package() -> None:
    assert DatumStaticResult is AppDatumStaticResult
    assert compute_datum_static_shifts is math_compute_datum_static_shifts
    assert app_compute_datum_static_shifts is math_compute_datum_static_shifts


def test_positive_elevation_difference_produces_positive_shift() -> None:
    result = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=np.asarray([80.0], dtype=np.float64),
        receiver_elevation_m_sorted=np.asarray([60.0], dtype=np.float64),
        datum_elevation_m=100.0,
        replacement_velocity_m_s=2000.0,
    )

    np.testing.assert_allclose(result.source_shift_s_sorted, np.asarray([0.01]))
    np.testing.assert_allclose(result.receiver_shift_s_sorted, np.asarray([0.02]))
    np.testing.assert_allclose(result.trace_shift_s_sorted, np.asarray([0.03]))


def test_negative_elevation_difference_produces_negative_shift() -> None:
    result = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=np.asarray([120.0], dtype=np.float64),
        receiver_elevation_m_sorted=np.asarray([160.0], dtype=np.float64),
        datum_elevation_m=100.0,
        replacement_velocity_m_s=2000.0,
    )

    np.testing.assert_allclose(result.source_shift_s_sorted, np.asarray([-0.01]))
    np.testing.assert_allclose(result.receiver_shift_s_sorted, np.asarray([-0.03]))
    np.testing.assert_allclose(result.trace_shift_s_sorted, np.asarray([-0.04]))


def test_zero_elevation_difference_produces_zero_shift() -> None:
    result = compute_datum_static_shifts(
        source_surface_elevation_m_sorted=np.asarray([100.0], dtype=np.float64),
        receiver_elevation_m_sorted=np.asarray([100.0], dtype=np.float64),
        datum_elevation_m=100.0,
        replacement_velocity_m_s=2000.0,
    )

    np.testing.assert_allclose(result.source_shift_s_sorted, np.asarray([0.0]))
    np.testing.assert_allclose(result.receiver_shift_s_sorted, np.asarray([0.0]))
    np.testing.assert_allclose(result.trace_shift_s_sorted, np.asarray([0.0]))


@pytest.mark.parametrize('replacement_velocity_m_s', [0.0, -1.0, np.nan, np.inf])
def test_rejects_invalid_replacement_velocity(replacement_velocity_m_s: float) -> None:
    with pytest.raises(ValueError):
        compute_datum_static_shifts(
            source_surface_elevation_m_sorted=np.asarray([100.0], dtype=np.float64),
            receiver_elevation_m_sorted=np.asarray([100.0], dtype=np.float64),
            datum_elevation_m=100.0,
            replacement_velocity_m_s=replacement_velocity_m_s,
        )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('source_surface_elevation_m_sorted', np.nan),
        ('source_surface_elevation_m_sorted', np.inf),
        ('receiver_elevation_m_sorted', np.nan),
        ('receiver_elevation_m_sorted', np.inf),
        ('source_depth_m_sorted', np.nan),
        ('source_depth_m_sorted', np.inf),
        ('datum_elevation_m', np.nan),
        ('datum_elevation_m', np.inf),
    ],
)
def test_rejects_non_finite_inputs(field: str, value: float) -> None:
    kwargs = {
        'source_surface_elevation_m_sorted': np.asarray([100.0], dtype=np.float64),
        'receiver_elevation_m_sorted': np.asarray([100.0], dtype=np.float64),
        'source_depth_m_sorted': np.asarray([0.0], dtype=np.float64),
        'datum_elevation_m': 100.0,
        'replacement_velocity_m_s': 2000.0,
    }
    if field == 'datum_elevation_m':
        kwargs[field] = value
    else:
        kwargs[field] = np.asarray([value], dtype=np.float64)

    with pytest.raises(ValueError):
        compute_datum_static_shifts(**kwargs)
