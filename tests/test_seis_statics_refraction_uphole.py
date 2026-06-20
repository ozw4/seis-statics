from __future__ import annotations

import numpy as np
import pytest

from seis_statics.refraction.uphole import (
    compute_uphole_time_correction,
    compute_uphole_time_correction_from_result,
    resolve_refraction_uphole,
)


def _keys(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def _ids(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _times(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def test_uphole_header_time_loads_seconds() -> None:
    result = resolve_refraction_uphole(
        source_endpoint_key_sorted=_keys(['source:a', 'source:b']),
        source_endpoint_id_sorted=_ids([101, 102]),
        source_node_id_sorted=_ids([0, 1]),
        uphole_time_sorted=_times([0.010, 0.020]),
        mode='header_time',
        uphole_time_byte=95,
        uphole_time_unit='s',
    )

    np.testing.assert_allclose(result.uphole_time_s, [0.010, 0.020])
    np.testing.assert_array_equal(result.uphole_status, ['ok', 'ok'])
    assert result.qc['uphole_time_unit'] == 's'


def test_uphole_header_time_loads_milliseconds() -> None:
    result = resolve_refraction_uphole(
        source_endpoint_key_sorted=_keys(['source:a', 'source:b']),
        source_endpoint_id_sorted=_ids([101, 102]),
        source_node_id_sorted=_ids([0, 1]),
        uphole_time_sorted=_times([10.0, 25.0]),
        mode='header_time',
        uphole_time_byte=95,
        uphole_time_unit='ms',
    )

    np.testing.assert_allclose(result.uphole_time_s, [0.010, 0.025])
    assert result.qc['median_uphole_time_s'] == pytest.approx(0.0175)


def test_uphole_time_aggregates_per_source_endpoint() -> None:
    result = resolve_refraction_uphole(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a', 'source:b']),
        source_endpoint_id_sorted=_ids([101, 101, 102]),
        source_node_id_sorted=_ids([0, 0, 1]),
        uphole_time_sorted=_times([0.010, 0.014, 0.020]),
        mode='header_time',
        uphole_time_byte=95,
        inconsistency_tolerance_s=0.010,
    )

    np.testing.assert_array_equal(result.source_endpoint_key, ['source:a', 'source:b'])
    np.testing.assert_allclose(result.uphole_time_s, [0.012, 0.020])
    np.testing.assert_array_equal(result.uphole_pick_count, [2, 1])
    np.testing.assert_array_equal(result.uphole_trace_count, [2, 1])


def test_uphole_time_marks_inconsistent_values() -> None:
    result = resolve_refraction_uphole(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a']),
        source_endpoint_id_sorted=_ids([101, 101]),
        source_node_id_sorted=_ids([0, 0]),
        uphole_time_sorted=_times([0.010, 0.020]),
        mode='header_time',
        uphole_time_byte=95,
        inconsistency_tolerance_s=0.001,
    )

    assert result.uphole_status.tolist() == ['inconsistent_uphole_time']
    np.testing.assert_allclose(result.uphole_time_s, [0.015])


def test_uphole_time_status_codes_missing_invalid_exceeds_and_inactive() -> None:
    result = resolve_refraction_uphole(
        source_endpoint_key_sorted=_keys(
            [
                'source:ok',
                'source:missing',
                'source:invalid',
                'source:large',
                'source:inactive',
            ]
        ),
        source_endpoint_id_sorted=_ids([10, 11, 12, 13, 14]),
        source_node_id_sorted=_ids([0, 1, 2, 3, -1]),
        uphole_time_sorted=_times([0.010, np.nan, np.inf, 1.5, 0.020]),
        mode='header_time',
        uphole_time_byte=95,
        max_abs_uphole_time_s=1.0,
    )

    assert result.uphole_status.tolist() == [
        'ok',
        'missing_uphole_time',
        'invalid_uphole_time',
        'exceeds_max_abs_uphole_time',
        'inactive_source_endpoint',
    ]
    qc = result.qc
    assert qc['n_sources_with_uphole'] == 3
    assert qc['n_missing_uphole'] == 1
    assert qc['n_invalid_uphole'] == 1
    assert qc['n_exceeds_max_abs_uphole'] == 1
    assert qc['n_inactive_source_endpoints'] == 1


def test_uphole_time_shift_formula_and_sign() -> None:
    result = compute_uphole_time_correction(
        _times([0.000, 0.010, 0.025]),
        status=np.asarray(['ok', 'ok', 'ok'], dtype='<U32'),
        positive_time_means_delay=True,
    )

    shift = result.component_shift_s['uphole_shift_s']
    np.testing.assert_allclose(shift, [-0.000, -0.010, -0.025])
    np.testing.assert_array_equal(result.component_status['uphole_shift_s'], ['ok'] * 3)
    assert result.qc['uphole_shift_formula'] == 'uphole_shift_s = -uphole_time_s'
    assert result.qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'


def test_uphole_time_qc_contains_sign_convention() -> None:
    resolved = resolve_refraction_uphole(
        source_endpoint_key_sorted=_keys(['source:a']),
        source_endpoint_id_sorted=_ids([101]),
        source_node_id_sorted=_ids([0]),
        uphole_time_sorted=_times([0.010]),
        mode='header_time',
        uphole_time_byte=95,
        positive_time_means_delay=True,
    )
    result = compute_uphole_time_correction_from_result(
        resolved,
        positive_time_means_delay=True,
    )

    assert resolved.qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert result.qc['positive_time_means_delay'] is True
    assert result.qc['component_name'] == 'uphole_shift_s'


def test_uphole_time_positive_not_delay_uses_positive_shift() -> None:
    result = compute_uphole_time_correction(
        _times([0.010]),
        status=np.asarray(['ok'], dtype='<U32'),
        positive_time_means_delay=False,
    )

    np.testing.assert_allclose(result.component_shift_s['uphole_shift_s'], [0.010])
    assert result.qc['uphole_shift_formula'] == 'uphole_shift_s = +uphole_time_s'


def test_uphole_time_marks_exceeds_max_abs_shift() -> None:
    result = compute_uphole_time_correction(
        _times([1.5]),
        status=np.asarray(['ok'], dtype='<U32'),
        max_abs_uphole_time_s=1.0,
    )

    assert result.component_status['uphole_shift_s'].tolist() == [
        'exceeds_max_abs_uphole_time'
    ]
    assert np.isnan(result.component_shift_s['uphole_shift_s'][0])
    assert np.isnan(result.total_field_shift_s[0])
