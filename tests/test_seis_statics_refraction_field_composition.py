from __future__ import annotations

import numpy as np
import pytest

from seis_statics.refraction.field_composition import (
    RefractionFieldCompositionError,
    compose_refraction_endpoint_field_corrections,
    compose_refraction_final_trace_shift,
    compose_refraction_trace_field_corrections,
)


def _keys(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def _ids(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _shifts(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _statuses(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype='<U64')


def _source_field() -> object:
    return compose_refraction_endpoint_field_corrections(
        endpoint_kind='source',
        endpoint_key=_keys(['s0', 's1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([10, 11]),
        source_depth_shift_s=_shifts([0.002, 0.003]),
        source_depth_status=_statuses(['ok', 'ok']),
        uphole_shift_s=_shifts([-0.001, -0.004]),
        uphole_status=_statuses(['ok', 'ok']),
        manual_static_shift_s=_shifts([0.0005, 0.0010]),
        manual_static_status=_statuses(['ok', 'ok']),
    )


def _receiver_field() -> object:
    return compose_refraction_endpoint_field_corrections(
        endpoint_kind='receiver',
        endpoint_key=_keys(['r0', 'r1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([20, 21]),
        manual_static_shift_s=_shifts([0.004, -0.002]),
        manual_static_status=_statuses(['ok', 'ok']),
    )


def _trace_field() -> object:
    return compose_refraction_trace_field_corrections(
        source_endpoint_field=_source_field(),
        receiver_endpoint_field=_receiver_field(),
        source_endpoint_key_sorted=_keys(['s0', 's1', 's0']),
        receiver_endpoint_key_sorted=_keys(['r0', 'r0', 'r1']),
    )


def test_field_composition_sums_source_and_receiver_components() -> None:
    source = _source_field()
    receiver = _receiver_field()

    np.testing.assert_allclose(source.total_field_shift_s, [0.0015, 0.0])
    np.testing.assert_array_equal(source.field_static_status, ['ok', 'ok'])
    np.testing.assert_allclose(receiver.total_field_shift_s, [0.004, -0.002])
    np.testing.assert_array_equal(receiver.field_static_status, ['ok', 'ok'])
    assert source.qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'


def test_field_composition_trace_shift_is_source_plus_receiver() -> None:
    trace = _trace_field()

    np.testing.assert_allclose(
        trace.trace_field_shift_s_sorted,
        trace.source_field_shift_s_sorted + trace.receiver_field_shift_s_sorted,
    )
    np.testing.assert_allclose(
        trace.trace_field_shift_s_sorted,
        [0.0055, 0.004, -0.0005],
    )
    np.testing.assert_array_equal(
        trace.trace_field_static_status_sorted,
        ['ok', 'ok', 'ok'],
    )


def test_field_composition_final_shift_adds_refraction_shift() -> None:
    trace = _trace_field()
    result = compose_refraction_final_trace_shift(
        refraction_trace_shift_s_sorted=_shifts([0.010, 0.020, -0.005]),
        trace_static_status_sorted=_statuses(['ok', 'ok', 'ok']),
        trace_static_valid_mask_sorted=np.asarray([True, True, True], dtype=bool),
        trace_field_correction=trace,
        apply_to_trace_shift=True,
        invalid_component_policy='fail',
    )

    np.testing.assert_allclose(
        result.final_trace_shift_s_sorted,
        [0.0155, 0.024, -0.0055],
    )
    np.testing.assert_allclose(
        result.base_refraction_trace_shift_s_sorted,
        [0.010, 0.020, -0.005],
    )
    assert result.qc['final_trace_shift_formula'] == (
        'final_trace_shift_s = refraction_trace_shift_s + trace_field_shift_s'
    )


def test_field_composition_apply_false_keeps_final_shift_at_base_refraction() -> None:
    trace = _trace_field()
    result = compose_refraction_final_trace_shift(
        refraction_trace_shift_s_sorted=_shifts([0.010, 0.020, -0.005]),
        trace_static_status_sorted=_statuses(['ok', 'muted_trace', 'ok']),
        trace_static_valid_mask_sorted=np.asarray([True, False, True], dtype=bool),
        trace_field_correction=trace,
        apply_to_trace_shift=False,
        invalid_component_policy='fail',
    )

    np.testing.assert_allclose(
        result.final_trace_shift_s_sorted,
        [0.010, 0.020, -0.005],
    )
    np.testing.assert_allclose(result.applied_field_shift_s_sorted, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        result.final_trace_static_status_sorted,
        ['ok', 'muted_trace', 'ok'],
    )
    np.testing.assert_array_equal(
        result.final_trace_static_valid_mask_sorted,
        [True, False, True],
    )
    assert result.qc['apply_to_trace_shift'] is False
    assert result.qc['final_trace_shift_formula'] == (
        'final_trace_shift_s = refraction_trace_shift_s'
    )


def test_field_composition_invalid_policy_fail() -> None:
    source = compose_refraction_endpoint_field_corrections(
        endpoint_kind='source',
        endpoint_key=_keys(['s0', 's1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([10, 11]),
        source_depth_shift_s=_shifts([0.002, np.nan]),
        source_depth_status=_statuses(['ok', 'missing_source_depth']),
    )
    trace = compose_refraction_trace_field_corrections(
        source_endpoint_field=source,
        receiver_endpoint_field=_receiver_field(),
        source_endpoint_key_sorted=_keys(['s0', 's1']),
        receiver_endpoint_key_sorted=_keys(['r0', 'r0']),
    )

    with pytest.raises(RefractionFieldCompositionError, match='invalid_trace_field'):
        compose_refraction_final_trace_shift(
            refraction_trace_shift_s_sorted=_shifts([0.010, 0.020]),
            trace_static_status_sorted=_statuses(['ok', 'ok']),
            trace_static_valid_mask_sorted=np.asarray([True, True], dtype=bool),
            trace_field_correction=trace,
            apply_to_trace_shift=True,
            invalid_component_policy='fail',
        )


def test_field_composition_source_only_manual_static_with_missing_allowed() -> None:
    source = compose_refraction_endpoint_field_corrections(
        endpoint_kind='source',
        endpoint_key=_keys(['s0', 's1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([10, 11]),
        manual_static_shift_s=_shifts([0.001, 0.0]),
        manual_static_status=_statuses(['ok', 'missing_manual_static']),
    )
    receiver = compose_refraction_endpoint_field_corrections(
        endpoint_kind='receiver',
        endpoint_key=_keys(['r0', 'r1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([20, 21]),
        manual_static_shift_s=_shifts([0.0, 0.0]),
        manual_static_status=_statuses(
            ['missing_manual_static', 'missing_manual_static'],
        ),
    )
    trace = compose_refraction_trace_field_corrections(
        source_endpoint_field=source,
        receiver_endpoint_field=receiver,
        source_endpoint_key_sorted=_keys(['s0', 's1']),
        receiver_endpoint_key_sorted=_keys(['r0', 'r1']),
    )
    result = compose_refraction_final_trace_shift(
        refraction_trace_shift_s_sorted=_shifts([0.010, 0.020]),
        trace_static_status_sorted=_statuses(['ok', 'ok']),
        trace_static_valid_mask_sorted=np.asarray([True, True], dtype=bool),
        trace_field_correction=trace,
        apply_to_trace_shift=True,
        invalid_component_policy='fail',
    )

    np.testing.assert_allclose(source.total_field_shift_s, [0.001, 0.0])
    np.testing.assert_array_equal(source.field_static_status, ['ok', 'ok'])
    np.testing.assert_allclose(receiver.total_field_shift_s, [0.0, 0.0])
    np.testing.assert_array_equal(receiver.field_static_status, ['ok', 'ok'])
    np.testing.assert_array_equal(trace.trace_field_static_status_sorted, ['ok', 'ok'])
    np.testing.assert_allclose(result.final_trace_shift_s_sorted, [0.011, 0.020])


def test_field_composition_receiver_only_manual_static_with_missing_allowed() -> None:
    source = compose_refraction_endpoint_field_corrections(
        endpoint_kind='source',
        endpoint_key=_keys(['s0', 's1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([10, 11]),
        manual_static_shift_s=_shifts([0.0, 0.0]),
        manual_static_status=_statuses(
            ['missing_manual_static', 'missing_manual_static'],
        ),
    )
    receiver = compose_refraction_endpoint_field_corrections(
        endpoint_kind='receiver',
        endpoint_key=_keys(['r0', 'r1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([20, 21]),
        manual_static_shift_s=_shifts([0.0, -0.002]),
        manual_static_status=_statuses(['missing_manual_static', 'ok']),
    )
    trace = compose_refraction_trace_field_corrections(
        source_endpoint_field=source,
        receiver_endpoint_field=receiver,
        source_endpoint_key_sorted=_keys(['s0', 's1']),
        receiver_endpoint_key_sorted=_keys(['r0', 'r1']),
    )
    result = compose_refraction_final_trace_shift(
        refraction_trace_shift_s_sorted=_shifts([0.010, 0.020]),
        trace_static_status_sorted=_statuses(['ok', 'ok']),
        trace_static_valid_mask_sorted=np.asarray([True, True], dtype=bool),
        trace_field_correction=trace,
        apply_to_trace_shift=True,
        invalid_component_policy='fail',
    )

    np.testing.assert_allclose(source.total_field_shift_s, [0.0, 0.0])
    np.testing.assert_array_equal(source.field_static_status, ['ok', 'ok'])
    np.testing.assert_allclose(receiver.total_field_shift_s, [0.0, -0.002])
    np.testing.assert_array_equal(receiver.field_static_status, ['ok', 'ok'])
    np.testing.assert_array_equal(trace.trace_field_static_status_sorted, ['ok', 'ok'])
    np.testing.assert_allclose(result.final_trace_shift_s_sorted, [0.010, 0.018])


def test_field_composition_invalid_policy_skip_invalid_traces() -> None:
    source = compose_refraction_endpoint_field_corrections(
        endpoint_kind='source',
        endpoint_key=_keys(['s0', 's1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([10, 11]),
        source_depth_shift_s=_shifts([0.002, np.nan]),
        source_depth_status=_statuses(['ok', 'missing_source_depth']),
    )
    trace = compose_refraction_trace_field_corrections(
        source_endpoint_field=source,
        receiver_endpoint_field=_receiver_field(),
        source_endpoint_key_sorted=_keys(['s0', 's1']),
        receiver_endpoint_key_sorted=_keys(['r0', 'r0']),
    )
    result = compose_refraction_final_trace_shift(
        refraction_trace_shift_s_sorted=_shifts([0.010, 0.020]),
        trace_static_status_sorted=_statuses(['ok', 'ok']),
        trace_static_valid_mask_sorted=np.asarray([True, True], dtype=bool),
        trace_field_correction=trace,
        apply_to_trace_shift=True,
        invalid_component_policy='skip_invalid_traces',
    )

    np.testing.assert_array_equal(
        trace.trace_field_static_status_sorted,
        ['ok', 'missing_source_depth'],
    )
    np.testing.assert_allclose(result.final_trace_shift_s_sorted, [0.016, 0.020])
    np.testing.assert_allclose(result.applied_field_shift_s_sorted, [0.006, 0.0])
    np.testing.assert_array_equal(
        result.final_trace_static_valid_mask_sorted,
        [True, True],
    )
