from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seis_statics.time_term import TimeTermInversionInputs
from seis_statics.time_term.moveout import (
    TimeTermMoveoutConfig,
    build_reciprocal_pair_index,
    compute_geometry_distance_m,
    compute_time_term_moveout,
    summarize_time_term_moveout,
)

N_TRACES = 5
N_SAMPLES = 64
DT = 0.004
GEOMETRY_DISTANCE = np.asarray([5.0, 5.0, 10.0, 10.0, 10.0], dtype=np.float64)
OFFSET = np.asarray([-5.0, 5.0, -10.0, 9.0, -12.0], dtype=np.float64)


def _inputs(**overrides: Any) -> TimeTermInversionInputs:
    pick_raw = np.asarray([0.020, 0.021, 0.022, 0.023, 0.024], dtype=np.float64)
    datum = np.asarray([0.001, 0.002, 0.003, 0.004, 0.005], dtype=np.float64)
    residual = np.asarray([-0.0005, 0.0, 0.0005, 0.001, -0.001], dtype=np.float64)
    payload: dict[str, Any] = {
        'n_traces': N_TRACES,
        'n_samples': N_SAMPLES,
        'dt': DT,
        'key1_byte': 189,
        'key2_byte': 193,
        'pick_time_raw_s_sorted': pick_raw,
        'valid_pick_mask_sorted': np.ones(N_TRACES, dtype=bool),
        'datum_trace_shift_s_sorted': datum,
        'residual_applied_shift_s_sorted': residual,
        'pick_time_after_static_s_sorted': pick_raw + datum + residual,
        'source_node_id_sorted': np.asarray([0, 1, 0, 2, 1], dtype=np.int64),
        'receiver_node_id_sorted': np.asarray([1, 0, 2, 0, 0], dtype=np.int64),
        'n_nodes': 3,
        'source_id_sorted': np.asarray([10, 20, 10, 30, 20], dtype=np.int64),
        'receiver_id_sorted': np.asarray([20, 10, 30, 10, 10], dtype=np.int64),
        'offset_sorted': OFFSET.copy(),
        'source_x_m_sorted': np.asarray([0.0, 3.0, 0.0, 6.0, 8.0]),
        'source_y_m_sorted': np.asarray([0.0, 4.0, 0.0, 8.0, 6.0]),
        'receiver_x_m_sorted': np.asarray([3.0, 0.0, 6.0, 0.0, 0.0]),
        'receiver_y_m_sorted': np.asarray([4.0, 0.0, 8.0, 0.0, 0.0]),
        'source_elevation_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'receiver_elevation_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'source_depth_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'input_file_id': 'file-id',
        'pick_source_description': 'test-picks',
        'datum_solution_path': Path('datum.npz'),
        'residual_solution_path': Path('residual.npz'),
        'linkage_artifact_path': Path('geometry_linkage.npz'),
    }
    payload.update(overrides)
    return TimeTermInversionInputs(**payload)


def _compute(
    inputs: TimeTermInversionInputs | None = None,
    **config_overrides: Any,
):
    config_payload: dict[str, Any] = {
        'model': 'head_wave_linear_offset',
        'refractor_velocity_m_s': 2500.0,
        'offset_byte': 37,
    }
    config_payload.update(config_overrides)
    return compute_time_term_moveout(
        _inputs() if inputs is None else inputs,
        TimeTermMoveoutConfig(**config_payload),
    )


def test_time_term_surface_imports() -> None:
    import seis_statics.time_term as time_term
    import seis_statics.time_term.moveout as moveout
    import seis_statics.time_term.types as types

    assert time_term.TimeTermInversionInputs is TimeTermInversionInputs
    assert moveout.TimeTermMoveoutConfig is TimeTermMoveoutConfig
    assert types.TimeTermInversionInputs is TimeTermInversionInputs
    assert 'compute_time_term_moveout' in time_term.__all__


def test_compute_geometry_distance_returns_horizontal_source_receiver_distance() -> None:
    distance = compute_geometry_distance_m(
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 2.0]),
        np.asarray([3.0, 1.0]),
        np.asarray([4.0, -2.0]),
    )

    np.testing.assert_allclose(distance, [5.0, 4.0])


def test_head_wave_moveout_uses_geometry_distance_over_refractor_velocity() -> None:
    result = _compute()

    np.testing.assert_allclose(result.distance_m_sorted, GEOMETRY_DISTANCE)
    np.testing.assert_allclose(result.moveout_time_s_sorted, GEOMETRY_DISTANCE / 2500.0)
    np.testing.assert_array_equal(result.valid_moveout_mask_sorted, np.ones(N_TRACES, dtype=bool))


def test_head_wave_moveout_uses_offset_header_distance_when_requested() -> None:
    result = _compute(distance_source='offset_header')

    np.testing.assert_allclose(result.distance_m_sorted, np.abs(OFFSET))
    np.testing.assert_allclose(result.moveout_time_s_sorted, np.abs(OFFSET) / 2500.0)


def test_offset_header_distance_does_not_validate_unused_bad_geometry() -> None:
    source_x = _inputs().source_x_m_sorted.copy()
    source_x[2] = np.nan

    result = _compute(
        _inputs(source_x_m_sorted=source_x),
        distance_source='offset_header',
    )

    np.testing.assert_allclose(result.distance_m_sorted, np.abs(OFFSET))
    np.testing.assert_allclose(result.moveout_time_s_sorted, np.abs(OFFSET) / 2500.0)
    assert result.geometry_offset_mismatch_m_sorted is None


def test_head_wave_moveout_auto_uses_geometry_when_available() -> None:
    result = _compute(distance_source='auto')

    np.testing.assert_allclose(result.distance_m_sorted, GEOMETRY_DISTANCE)
    assert result.distance_source == 'auto'


def test_head_wave_moveout_auto_uses_offset_when_geometry_unavailable() -> None:
    source_x = _inputs().source_x_m_sorted.copy()
    source_x[2] = np.nan

    result = _compute(_inputs(source_x_m_sorted=source_x), distance_source='auto')

    np.testing.assert_allclose(result.distance_m_sorted, np.abs(OFFSET))
    np.testing.assert_allclose(result.moveout_time_s_sorted, np.abs(OFFSET) / 2500.0)


def test_auto_distance_does_not_validate_unused_bad_offset_when_geometry_is_valid() -> None:
    offset = OFFSET.copy()
    offset[1] = np.inf

    result = _compute(_inputs(offset_sorted=offset), distance_source='auto')

    np.testing.assert_allclose(result.distance_m_sorted, GEOMETRY_DISTANCE)
    assert result.offset_abs_m_sorted is None


def test_head_wave_moveout_none_returns_zero_moveout() -> None:
    result = _compute(model='none', distance_source='offset_header')

    np.testing.assert_allclose(result.distance_m_sorted, np.zeros(N_TRACES))
    np.testing.assert_allclose(result.moveout_time_s_sorted, np.zeros(N_TRACES))
    np.testing.assert_array_equal(result.valid_moveout_mask_sorted, np.ones(N_TRACES, dtype=bool))


def test_none_moveout_does_not_require_distance_metadata() -> None:
    source_x = _inputs().source_x_m_sorted.copy()
    source_x[2] = np.nan

    result = _compute(
        _inputs(source_x_m_sorted=source_x, offset_sorted=None),
        model='none',
        distance_source='offset_header',
        offset_byte=None,
    )

    np.testing.assert_allclose(result.distance_m_sorted, np.zeros(N_TRACES))
    np.testing.assert_allclose(result.moveout_time_s_sorted, np.zeros(N_TRACES))
    np.testing.assert_array_equal(result.valid_moveout_mask_sorted, np.ones(N_TRACES, dtype=bool))


@pytest.mark.parametrize('velocity', [0.0, -1.0, np.nan, np.inf])
def test_head_wave_moveout_rejects_non_positive_refractor_velocity(
    velocity: float,
) -> None:
    with pytest.raises(ValueError, match='refractor_velocity_m_s'):
        _compute(refractor_velocity_m_s=velocity)


def test_head_wave_moveout_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match='unsupported moveout model'):
        _compute(model='quadratic')


def test_head_wave_moveout_rejects_unsupported_distance_source() -> None:
    with pytest.raises(ValueError, match='unsupported distance_source'):
        _compute(distance_source='shot_record')


def test_head_wave_moveout_rejects_non_finite_geometry() -> None:
    source_x = _inputs().source_x_m_sorted.copy()
    source_x[2] = np.nan

    with pytest.raises(ValueError, match='geometry_distance_m_sorted'):
        _compute(_inputs(source_x_m_sorted=source_x))


def test_head_wave_moveout_rejects_missing_offset_when_offset_requested() -> None:
    with pytest.raises(ValueError, match='offset_sorted is required'):
        _compute(_inputs(offset_sorted=None), distance_source='offset_header')


def test_head_wave_moveout_rejects_missing_offset_byte_when_offset_requested() -> None:
    with pytest.raises(ValueError, match='offset_byte is required'):
        _compute(distance_source='offset_header', offset_byte=None)


def test_head_wave_moveout_rejects_non_finite_offset() -> None:
    offset = OFFSET.copy()
    offset[1] = np.inf

    with pytest.raises(ValueError, match='offset_sorted'):
        _compute(_inputs(offset_sorted=offset), distance_source='offset_header')


def test_geometry_distance_does_not_validate_unused_bad_offset() -> None:
    offset = OFFSET.copy()
    offset[1] = np.inf

    result = _compute(_inputs(offset_sorted=offset), distance_source='geometry')

    np.testing.assert_allclose(result.distance_m_sorted, GEOMETRY_DISTANCE)
    assert result.offset_abs_m_sorted is None


def test_offset_header_allow_missing_offset_returns_invalid_moveout() -> None:
    result = _compute(
        _inputs(offset_sorted=None),
        distance_source='offset_header',
        offset_byte=None,
        allow_missing_offset=True,
    )

    assert np.all(np.isnan(result.distance_m_sorted))
    assert np.all(np.isnan(result.moveout_time_s_sorted))
    np.testing.assert_array_equal(
        result.valid_moveout_mask_sorted,
        np.zeros(N_TRACES, dtype=bool),
    )


def test_head_wave_moveout_returns_non_negative_distance_and_time() -> None:
    result = _compute(distance_source='offset_header')

    assert np.all(np.isfinite(result.distance_m_sorted))
    assert np.all(result.distance_m_sorted >= 0.0)
    assert np.all(np.isfinite(result.moveout_time_s_sorted))
    assert np.all(result.moveout_time_s_sorted >= 0.0)


def test_head_wave_moveout_preserves_sorted_trace_order() -> None:
    result = _compute(
        _inputs(
            source_x_m_sorted=np.asarray([0.0, 0.0, 0.0]),
            source_y_m_sorted=np.asarray([0.0, 0.0, 0.0]),
            receiver_x_m_sorted=np.asarray([1.0, 3.0, 2.0]),
            receiver_y_m_sorted=np.asarray([0.0, 0.0, 0.0]),
            source_node_id_sorted=np.asarray([0, 1, 2]),
            receiver_node_id_sorted=np.asarray([1, 2, 0]),
            source_id_sorted=np.asarray([10, 20, 30]),
            receiver_id_sorted=np.asarray([20, 30, 10]),
            offset_sorted=np.asarray([1.0, 3.0, 2.0]),
            n_traces=3,
            n_nodes=3,
            pick_time_raw_s_sorted=np.asarray([0.01, 0.02, 0.03]),
            valid_pick_mask_sorted=np.ones(3, dtype=bool),
            datum_trace_shift_s_sorted=np.zeros(3),
            residual_applied_shift_s_sorted=np.zeros(3),
            pick_time_after_static_s_sorted=np.asarray([0.01, 0.02, 0.03]),
            source_elevation_m_sorted=np.zeros(3),
            receiver_elevation_m_sorted=np.zeros(3),
            source_depth_m_sorted=np.zeros(3),
        )
    )

    np.testing.assert_allclose(result.distance_m_sorted, [1.0, 3.0, 2.0])


def test_head_wave_moveout_computes_geometry_offset_mismatch() -> None:
    result = _compute()

    np.testing.assert_allclose(result.offset_abs_m_sorted, np.abs(OFFSET))
    np.testing.assert_allclose(
        result.geometry_offset_mismatch_m_sorted,
        GEOMETRY_DISTANCE - np.abs(OFFSET),
    )


def test_head_wave_moveout_rejects_geometry_offset_mismatch_above_threshold() -> None:
    with pytest.raises(ValueError, match='geometry_offset_mismatch_m_sorted'):
        _compute(max_geometry_offset_mismatch_m=0.5)


def test_reciprocal_pair_mapping_finds_reverse_node_pair() -> None:
    result = _compute(model='reciprocal_head_wave')

    np.testing.assert_array_equal(result.reciprocal_pair_index_sorted[:4], [1, 0, 3, 2])
    np.testing.assert_array_equal(result.has_reciprocal_pair_mask_sorted, np.ones(N_TRACES, dtype=bool))


def test_reciprocal_pair_mapping_returns_minus_one_when_missing() -> None:
    pair_index = build_reciprocal_pair_index(
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(pair_index, [-1, -1])


def test_reciprocal_pair_mapping_ignores_self_pair() -> None:
    pair_index = build_reciprocal_pair_index(
        np.asarray([1], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )

    np.testing.assert_array_equal(pair_index, [-1])


def test_reciprocal_pair_mapping_uses_smallest_sorted_trace_index_for_duplicates() -> None:
    pair_index = build_reciprocal_pair_index(
        np.asarray([0, 1, 1], dtype=np.int64),
        np.asarray([1, 0, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(pair_index, [1, 0, 0])


def test_reciprocal_pair_mapping_validates_node_id_range() -> None:
    source_nodes = _inputs().source_node_id_sorted.copy()
    source_nodes[0] = 3

    with pytest.raises(ValueError, match='node ids must be less than n_nodes'):
        _compute(_inputs(source_node_id_sorted=source_nodes))


def test_reciprocal_head_wave_model_computes_moveout_and_pairs() -> None:
    result = _compute(model='reciprocal_head_wave')

    np.testing.assert_allclose(result.moveout_time_s_sorted, GEOMETRY_DISTANCE / 2500.0)
    np.testing.assert_array_equal(result.reciprocal_pair_index_sorted, [1, 0, 3, 2, 0])


def test_linear_offset_alias_matches_head_wave_formula() -> None:
    linear = _compute(model='linear_offset')
    head_wave = _compute(model='head_wave_linear_offset')

    np.testing.assert_allclose(
        linear.moveout_time_s_sorted,
        head_wave.moveout_time_s_sorted,
    )
    assert linear.model == 'linear_offset'


def test_summarize_time_term_moveout_is_json_safe() -> None:
    summary = summarize_time_term_moveout(_compute())

    json.dumps(summary, allow_nan=False)
    assert summary['model'] == 'head_wave_linear_offset'
    assert summary['n_traces'] == N_TRACES
    assert summary['n_valid_moveout'] == N_TRACES
    assert summary['valid_moveout_fraction'] == pytest.approx(1.0)
    assert summary['has_offset_header'] is True
    assert 'geometry_offset_mismatch_m' in summary


def test_moveout_time_is_not_applied_static_shift() -> None:
    base = _inputs()
    shifted = replace(
        base,
        datum_trace_shift_s_sorted=np.full(N_TRACES, 10.0),
        residual_applied_shift_s_sorted=np.full(N_TRACES, -5.0),
        pick_time_after_static_s_sorted=np.full(N_TRACES, 99.0),
    )

    np.testing.assert_allclose(
        _compute(shifted).moveout_time_s_sorted,
        _compute(base).moveout_time_s_sorted,
    )
