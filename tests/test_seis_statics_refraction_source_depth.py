from __future__ import annotations

import numpy as np

from seis_statics.refraction.source_depth import (
    compute_source_depth_weathering_time_correction,
    compute_source_depth_weathering_time_correction_from_result,
    resolve_refraction_source_depth,
)


def _keys(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def _ids(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _depths(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def test_source_depth_resolves_median_per_source_endpoint() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a', 'source:b']),
        source_endpoint_id_sorted=_ids([101, 101, 102]),
        source_node_id_sorted=_ids([0, 0, 1]),
        source_depth_m_sorted=_depths([1.0, 3.0, 4.0]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
        inconsistency_tolerance_m=5.0,
    )

    np.testing.assert_array_equal(result.source_endpoint_key, ['source:a', 'source:b'])
    np.testing.assert_array_equal(result.source_endpoint_id, [101, 102])
    np.testing.assert_array_equal(result.source_node_id, [0, 1])
    np.testing.assert_allclose(result.source_depth_m, [2.0, 4.0])
    np.testing.assert_array_equal(result.source_depth_status, ['ok', 'ok'])
    np.testing.assert_array_equal(result.source_depth_pick_count, [2, 1])
    np.testing.assert_array_equal(result.source_depth_trace_count, [2, 1])


def test_source_depth_marks_missing_when_required() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a']),
        source_endpoint_id_sorted=_ids([101]),
        source_node_id_sorted=_ids([0]),
        source_depth_m_sorted=None,
        mode='weathering_velocity_time',
        source_depth_byte=115,
    )

    assert result.source_depth_status.tolist() == ['missing_source_depth']
    assert np.isnan(result.source_depth_m[0])
    assert result.source_depth_pick_count.tolist() == [0]


def test_source_depth_marks_missing_trace_with_resolved_endpoint_depth() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a']),
        source_endpoint_id_sorted=_ids([101, 101]),
        source_node_id_sorted=_ids([0, 0]),
        source_depth_m_sorted=_depths([np.nan, 4.0]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
    )

    assert result.source_depth_status.tolist() == ['missing_source_depth']
    np.testing.assert_allclose(result.source_depth_m, [4.0])
    assert result.source_depth_pick_count.tolist() == [1]
    assert result.source_depth_trace_count.tolist() == [2]


def test_source_depth_marks_inconsistent_values() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a']),
        source_endpoint_id_sorted=_ids([101, 101]),
        source_node_id_sorted=_ids([0, 0]),
        source_depth_m_sorted=_depths([2.0, 2.5]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
        inconsistency_tolerance_m=0.1,
    )

    assert result.source_depth_status.tolist() == ['inconsistent_source_depth']
    np.testing.assert_allclose(result.source_depth_m, [2.25])


def test_source_depth_marks_negative_depth_invalid_when_positive_down() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a', 'source:a']),
        source_endpoint_id_sorted=_ids([101, 101]),
        source_node_id_sorted=_ids([0, 0]),
        source_depth_m_sorted=_depths([-2.0, -4.0]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
        positive_down=True,
    )

    assert result.source_depth_status.tolist() == ['invalid_source_depth']
    assert np.isnan(result.source_depth_m[0])
    assert result.source_depth_pick_count.tolist() == [0]


def test_source_depth_qc_counts_statuses() -> None:
    result = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(
            [
                'source:ok',
                'source:ok',
                'source:missing',
                'source:invalid',
                'source:wide',
                'source:wide',
                'source:large',
                'source:inactive',
            ]
        ),
        source_endpoint_id_sorted=_ids([10, 10, 11, 12, 13, 13, 14, 15]),
        source_node_id_sorted=_ids([0, 0, 1, 2, 3, 3, 4, -1]),
        source_depth_m_sorted=_depths([2.0, 2.005, np.nan, -1.0, 4.0, 5.0, 101.0, 7.0]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
        max_abs_source_depth_m=100.0,
        inconsistency_tolerance_m=0.1,
    )

    assert result.source_depth_status.tolist() == [
        'ok',
        'missing_source_depth',
        'invalid_source_depth',
        'inconsistent_source_depth',
        'exceeds_max_abs_source_depth',
        'inactive_source_endpoint',
    ]
    qc = result.qc
    assert qc['source_depth_mode'] == 'weathering_velocity_time'
    assert qc['source_depth_byte'] == 115
    assert qc['n_source_endpoints'] == 6
    assert qc['n_sources_with_depth'] == 4
    assert qc['n_missing_source_depth'] == 1
    assert qc['n_invalid_source_depth'] == 1
    assert qc['n_inconsistent_source_depth'] == 1
    assert qc['n_exceeds_max_abs_source_depth'] == 1
    assert qc['n_inactive_source_endpoints'] == 1
    assert qc['status_counts']['ok'] == 1
    assert qc['min_source_depth_m'] == 2.0025
    assert qc['max_source_depth_m'] == 101.0


def test_source_depth_weathering_time_shift_formula_and_sign() -> None:
    result = compute_source_depth_weathering_time_correction(
        _depths([0.0, 10.0]),
        500.0,
        status=np.asarray(['ok', 'ok'], dtype='<U32'),
        max_abs_shift_s=None,
    )

    shift = result.component_shift_s['source_depth_shift_s']
    np.testing.assert_allclose(shift, [0.0, 0.020])
    np.testing.assert_array_equal(
        result.component_status['source_depth_shift_s'],
        ['ok', 'ok'],
    )
    assert (
        result.qc['source_depth_shift_formula']
        == 'source_depth_shift_s = +source_depth_m / V1_m_s'
    )
    assert result.qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'


def test_source_depth_weathering_time_uses_resolved_v1() -> None:
    resolved_depth = resolve_refraction_source_depth(
        source_endpoint_key_sorted=_keys(['source:a']),
        source_endpoint_id_sorted=_ids([101]),
        source_node_id_sorted=_ids([0]),
        source_depth_m_sorted=_depths([8.0]),
        mode='weathering_velocity_time',
        source_depth_byte=115,
    )

    result = compute_source_depth_weathering_time_correction_from_result(
        resolved_depth,
        800.0,
        max_abs_shift_s=None,
    )

    np.testing.assert_allclose(result.component_shift_s['source_depth_shift_s'], [0.010])
    assert result.qc['v1_m_s'] == 800.0


def test_source_depth_weathering_time_marks_missing_depth() -> None:
    result = compute_source_depth_weathering_time_correction(
        _depths([np.nan]),
        800.0,
        status=np.asarray(['missing_source_depth'], dtype='<U32'),
        max_abs_shift_s=None,
    )

    assert result.component_status['source_depth_shift_s'].tolist() == [
        'missing_source_depth'
    ]
    assert np.isnan(result.component_shift_s['source_depth_shift_s'][0])
    assert np.isnan(result.total_field_shift_s[0])


def test_source_depth_weathering_time_marks_exceeds_max_abs_shift() -> None:
    result = compute_source_depth_weathering_time_correction(
        _depths([10.0]),
        500.0,
        status=np.asarray(['ok'], dtype='<U32'),
        max_abs_shift_s=0.01,
    )

    assert result.component_status['source_depth_shift_s'].tolist() == [
        'exceeds_max_abs_source_depth_shift'
    ]
    assert np.isnan(result.component_shift_s['source_depth_shift_s'][0])
    assert np.isnan(result.total_field_shift_s[0])
