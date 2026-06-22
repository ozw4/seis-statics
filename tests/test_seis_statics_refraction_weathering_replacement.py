from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from seis_statics.refraction.weathering import (
    RefractionWeatheringEndpointComponents,
    RefractionWeatheringModel,
    compute_weathering_thickness_from_half_intercept_time_with_status,
)
from seis_statics.refraction.weathering_replacement import (
    RefractionWeatheringReplacementError,
    build_refraction_weathering_replacement_statics,
    compute_weathering_replacement_shift_s,
    compute_weathering_replacement_shift_scalar_s,
)


WEATHERING_VELOCITY_M_S = 800.0
BEDROCK_VELOCITY_M_S = 2500.0
BEDROCK_SLOWNESS_S_PER_M = 1.0 / BEDROCK_VELOCITY_M_S
SLOWNESS_DELTA_S_PER_M = 1.0 / BEDROCK_VELOCITY_M_S - 1.0 / WEATHERING_VELOCITY_M_S


def _endpoint(
    *,
    node_id: np.ndarray,
    thickness: np.ndarray,
    status: np.ndarray,
    v2_m_s: np.ndarray | None = None,
    v2_status: np.ndarray | None = None,
) -> RefractionWeatheringEndpointComponents:
    n_items = int(node_id.shape[0])
    x_m = node_id.astype(np.float64) * 100.0
    if v2_m_s is None:
        v2_m_s = np.full(n_items, BEDROCK_VELOCITY_M_S, dtype=np.float64)
    if v2_status is None:
        v2_status = np.full(n_items, 'ok', dtype='<U32')
    return RefractionWeatheringEndpointComponents(
        endpoint_key=np.asarray([f'endpoint:{int(node)}' for node in node_id]),
        endpoint_id=node_id.copy(),
        node_id=node_id.copy(),
        x_m=x_m,
        y_m=np.zeros(n_items, dtype=np.float64),
        surface_elevation_m=np.full(n_items, 100.0, dtype=np.float64),
        half_intercept_time_s=np.zeros(n_items, dtype=np.float64),
        v1_m_s=np.full(n_items, WEATHERING_VELOCITY_M_S, dtype=np.float64),
        v2_m_s=np.ascontiguousarray(v2_m_s, dtype=np.float64),
        weathering_thickness_m=np.ascontiguousarray(thickness, dtype=np.float64),
        refractor_elevation_m=100.0 - thickness,
        solution_status=status.copy(),
        local_v2_status=np.asarray(v2_status, dtype='<U32'),
        weathering_status=status.copy(),
        pick_count=np.asarray([4, 4, 3, 3, 2, 0], dtype=np.int64)[:n_items],
        used_observation_count=np.asarray([4, 3, 3, 2, 2, 0], dtype=np.int64)[
            :n_items
        ],
        rejected_observation_count=np.asarray([0, 1, 0, 1, 0, 0], dtype=np.int64)[
            :n_items
        ],
    )


def _weathering_model(
    *,
    node_thickness: np.ndarray | None = None,
    node_status: np.ndarray | None = None,
    node_v2_m_s: np.ndarray | None = None,
    node_v2_status: np.ndarray | None = None,
) -> RefractionWeatheringModel:
    node_id = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    thickness = (
        np.asarray([10.0, 12.0, 15.0, 18.0, 20.0, np.nan], dtype=np.float64)
        if node_thickness is None
        else np.asarray(node_thickness, dtype=np.float64)
    )
    status = (
        np.asarray(['ok', 'ok', 'ok', 'ok', 'ok', 'inactive'], dtype='<U32')
        if node_status is None
        else np.asarray(node_status, dtype='<U32')
    )
    if node_v2_m_s is None:
        node_v2_m_s = np.full(node_id.shape, BEDROCK_VELOCITY_M_S, dtype=np.float64)
    if node_v2_status is None:
        node_v2_status = np.full(node_id.shape, 'ok', dtype='<U32')
    source_sorted = np.asarray([0, 1, 2, 3, 4, 0, 5], dtype=np.int64)
    receiver_sorted = np.asarray([1, 2, 3, 4, 0, 2, 5], dtype=np.int64)
    sorted_trace_index = np.asarray([4, 2, 0, 1, 3, 5, 6], dtype=np.int64)
    node_pos = {int(node): index for index, node in enumerate(node_id.tolist())}

    def _map_float(values: np.ndarray, source: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, np.nan, dtype=np.float64)
        for index, raw_node in enumerate(values.tolist()):
            node_index = node_pos.get(int(raw_node))
            if node_index is not None:
                out[index] = source[node_index]
        return out

    def _map_status(values: np.ndarray, source: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, 'missing_node', dtype='<U32')
        for index, raw_node in enumerate(values.tolist()):
            node_index = node_pos.get(int(raw_node))
            if node_index is not None:
                out[index] = source[node_index]
        return out

    source_endpoint = _endpoint(
        node_id=node_id,
        thickness=thickness,
        status=status,
        v2_m_s=node_v2_m_s,
        v2_status=node_v2_status,
    )
    receiver_endpoint = _endpoint(
        node_id=node_id,
        thickness=thickness,
        status=status,
        v2_m_s=node_v2_m_s,
        v2_status=node_v2_status,
    )
    return RefractionWeatheringModel(
        file_id='weathering-replacement',
        n_traces=int(sorted_trace_index.shape[0]),
        bedrock_velocity_mode='solve_global',
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
        bedrock_slowness_s_per_m=BEDROCK_SLOWNESS_S_PER_M,
        bedrock_velocity_status='solved',
        v2_m_s=BEDROCK_VELOCITY_M_S,
        node_id=node_id,
        node_x_m=node_id.astype(np.float64) * 100.0,
        node_y_m=np.zeros(node_id.shape, dtype=np.float64),
        node_surface_elevation_m=np.full(node_id.shape, 100.0, dtype=np.float64),
        node_half_intercept_time_s=np.zeros(node_id.shape, dtype=np.float64),
        node_v1_m_s=np.full(node_id.shape, WEATHERING_VELOCITY_M_S, dtype=np.float64),
        node_v2_m_s=np.ascontiguousarray(node_v2_m_s, dtype=np.float64),
        node_weathering_thickness_m=thickness,
        node_refractor_elevation_m=100.0 - thickness,
        node_solution_status=status.copy(),
        node_local_v2_status=np.asarray(node_v2_status, dtype='<U32'),
        node_weathering_status=status.copy(),
        node_pick_count=np.asarray([4, 4, 3, 3, 2, 0], dtype=np.int64),
        node_used_observation_count=np.asarray([4, 3, 3, 2, 2, 0], dtype=np.int64),
        node_rejected_observation_count=np.asarray([0, 1, 0, 1, 0, 0], dtype=np.int64),
        source_endpoint=source_endpoint,
        receiver_endpoint=receiver_endpoint,
        trace_index_sorted=sorted_trace_index,
        source_endpoint_key_sorted=np.asarray(
            [f'endpoint:{int(node)}' for node in source_sorted],
            dtype=object,
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'endpoint:{int(node)}' for node in receiver_sorted],
            dtype=object,
        ),
        source_node_id_sorted=source_sorted,
        receiver_node_id_sorted=receiver_sorted,
        source_weathering_thickness_m_sorted=_map_float(source_sorted, thickness),
        receiver_weathering_thickness_m_sorted=_map_float(receiver_sorted, thickness),
        source_refractor_elevation_m_sorted=100.0 - _map_float(
            source_sorted,
            thickness,
        ),
        receiver_refractor_elevation_m_sorted=100.0 - _map_float(
            receiver_sorted,
            thickness,
        ),
        source_weathering_status_sorted=_map_status(source_sorted, status),
        receiver_weathering_status_sorted=_map_status(receiver_sorted, status),
        trace_weathering_thickness_m_sorted=(
            _map_float(source_sorted, thickness) + _map_float(receiver_sorted, thickness)
        ),
        trace_weathering_status_sorted=np.full(sorted_trace_index.shape, 'ok', dtype='<U32'),
        cell_id=np.asarray([], dtype=np.int64),
        cell_v2_m_s=np.asarray([], dtype=np.float64),
        cell_velocity_status=np.asarray([], dtype='<U32'),
        cell_observation_count=np.asarray([], dtype=np.int64),
        qc={'method': 'gli_variable_thickness'},
    )


def test_math_helper_computes_vector_scalar_and_preserves_nan() -> None:
    thickness = np.asarray([10.0, np.nan, 0.0], dtype=np.float64)

    shift = compute_weathering_replacement_shift_s(
        weathering_thickness_m=thickness,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
    )

    assert shift[0] == pytest.approx(10.0 * SLOWNESS_DELTA_S_PER_M)
    assert np.isnan(shift[1])
    assert shift[2] == pytest.approx(0.0)
    assert shift[0] < 0.0
    assert compute_weathering_replacement_shift_scalar_s(
        weathering_thickness_m=12.0,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=BEDROCK_VELOCITY_M_S,
    ) == pytest.approx(12.0 * SLOWNESS_DELTA_S_PER_M)


def test_math_helper_accepts_vector_bedrock_velocity() -> None:
    thickness = np.asarray([10.0, 12.0, 14.0], dtype=np.float64)
    bedrock_velocity = np.asarray([2200.0, 2500.0, 3000.0], dtype=np.float64)

    shift = compute_weathering_replacement_shift_s(
        weathering_thickness_m=thickness,
        weathering_velocity_m_s=WEATHERING_VELOCITY_M_S,
        bedrock_velocity_m_s=bedrock_velocity,
    )

    expected = thickness * (1.0 / bedrock_velocity - 1.0 / WEATHERING_VELOCITY_M_S)
    np.testing.assert_allclose(shift, expected, rtol=1.0e-12)
    assert np.all(shift < 0.0)


@pytest.mark.parametrize(
    ('weathering_velocity', 'bedrock_velocity', 'match'),
    [
        (np.nan, BEDROCK_VELOCITY_M_S, 'weathering_velocity_m_s'),
        (0.0, BEDROCK_VELOCITY_M_S, 'weathering_velocity_m_s'),
        (WEATHERING_VELOCITY_M_S, np.inf, 'bedrock_velocity_m_s'),
        (WEATHERING_VELOCITY_M_S, -1.0, 'bedrock_velocity_m_s'),
        (WEATHERING_VELOCITY_M_S, WEATHERING_VELOCITY_M_S, 'greater'),
    ],
)
def test_math_helper_rejects_invalid_velocity(
    weathering_velocity: float,
    bedrock_velocity: float,
    match: str,
) -> None:
    with pytest.raises(RefractionWeatheringReplacementError, match=match):
        compute_weathering_replacement_shift_s(
            weathering_thickness_m=np.asarray([10.0], dtype=np.float64),
            weathering_velocity_m_s=weathering_velocity,
            bedrock_velocity_m_s=bedrock_velocity,
        )


def test_build_computes_node_endpoint_and_trace_shifts_in_sorted_order() -> None:
    weathering = _weathering_model()

    result = build_refraction_weathering_replacement_statics(
        weathering_model=weathering,
        max_abs_shift_ms=250.0,
    )

    expected_node_shift = (
        weathering.node_weathering_thickness_m * SLOWNESS_DELTA_S_PER_M
    )
    np.testing.assert_allclose(
        result.node_weathering_replacement_shift_s[:5],
        expected_node_shift[:5],
        rtol=1.0e-12,
    )
    assert np.isnan(result.node_weathering_replacement_shift_s[5])
    expected_source = expected_node_shift[result.source_node_id_sorted[:6]]
    expected_receiver = expected_node_shift[result.receiver_node_id_sorted[:6]]
    np.testing.assert_allclose(
        result.source_weathering_replacement_shift_s_sorted[:6],
        expected_source,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.receiver_weathering_replacement_shift_s_sorted[:6],
        expected_receiver,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.weathering_replacement_trace_shift_s_sorted[:6],
        expected_source + expected_receiver,
        rtol=1.0e-12,
    )
    np.testing.assert_array_equal(
        result.trace_index_sorted,
        np.asarray([4, 2, 0, 1, 3, 5, 6], dtype=np.int64),
    )
    assert result.trace_static_valid_mask_sorted[:6].tolist() == [True] * 6
    assert result.trace_static_valid_mask_sorted[6] == np.False_


def test_statuses_preserve_inactive_invalid_and_inherited_conditions() -> None:
    thickness = np.asarray([10.0, np.nan, -1.0, 18.0, 20.0, np.nan])
    status = np.asarray(
        [
            'ok',
            'ok',
            'ok',
            'exceeds_max_thickness',
            'clipped_half_intercept_upper',
            'inactive',
        ],
        dtype='<U32',
    )

    result = build_refraction_weathering_replacement_statics(
        weathering_model=_weathering_model(
            node_thickness=thickness,
            node_status=status,
        ),
        max_abs_shift_ms=250.0,
    )

    assert result.node_static_status.tolist() == [
        'ok',
        'invalid_weathering_thickness',
        'negative_weathering_thickness',
        'exceeds_max_thickness',
        'clipped_half_intercept_upper',
        'inactive',
    ]
    assert np.isnan(result.node_weathering_replacement_shift_s[1])
    assert np.isnan(result.node_weathering_replacement_shift_s[2])
    assert np.isfinite(result.node_weathering_replacement_shift_s[3])
    assert result.trace_static_status_sorted[3] == 'exceeds_max_thickness'
    assert result.trace_static_valid_mask_sorted[3] == np.False_
    assert result.trace_static_status_sorted[4] == 'clipped_half_intercept_upper'
    assert result.trace_static_valid_mask_sorted[4] == np.False_
    assert result.qc['invalid_weathering_thickness_count'] == 1
    assert result.qc['exceeds_max_thickness_count'] == 1


def test_statuses_preserve_real_weathering_builder_nan_conditions() -> None:
    components = compute_weathering_thickness_from_half_intercept_time_with_status(
        half_intercept_time_s=np.full(6, 0.02, dtype=np.float64),
        surface_elevation_m=np.full(6, 100.0, dtype=np.float64),
        v1_m_s=WEATHERING_VELOCITY_M_S,
        v2_m_s=np.asarray(
            [
                BEDROCK_VELOCITY_M_S,
                BEDROCK_VELOCITY_M_S,
                BEDROCK_VELOCITY_M_S,
                BEDROCK_VELOCITY_M_S,
                WEATHERING_VELOCITY_M_S,
                BEDROCK_VELOCITY_M_S,
            ],
            dtype=np.float64,
        ),
        solution_status=np.asarray(
            [
                'low_fold',
                'clipped_half_intercept_lower',
                'clipped_half_intercept_upper',
                'ok',
                'ok',
                'inactive',
            ],
            dtype='<U32',
        ),
        max_weathering_thickness_m=1.0,
    )

    result = build_refraction_weathering_replacement_statics(
        weathering_model=_weathering_model(
            node_thickness=components.weathering_thickness_m,
            node_status=components.weathering_status,
        ),
        max_abs_shift_ms=250.0,
    )

    assert result.node_static_status.tolist() == [
        'low_fold',
        'clipped_half_intercept_lower',
        'clipped_half_intercept_upper',
        'exceeds_max_thickness',
        'invalid_velocity_order',
        'inactive',
    ]
    assert np.isnan(result.node_weathering_replacement_shift_s).all()
    assert 'invalid_shift' not in set(result.trace_static_status_sorted.tolist())
    assert result.qc['low_fold_node_count'] == 1
    assert result.qc['invalid_weathering_thickness_count'] == 0
    assert result.qc['exceeds_max_thickness_count'] == 1


def test_elevation_only_weathering_statuses_do_not_block_replacement_shift() -> None:
    status = np.asarray(
        [
            'invalid_surface_elevation',
            'invalid_refractor_elevation',
            'ok',
            'ok',
            'ok',
            'inactive',
        ],
        dtype='<U32',
    )

    result = build_refraction_weathering_replacement_statics(
        weathering_model=_weathering_model(node_status=status),
        max_abs_shift_ms=250.0,
    )

    expected_node_shift = np.asarray([10.0, 12.0], dtype=np.float64) * (
        SLOWNESS_DELTA_S_PER_M
    )
    np.testing.assert_allclose(
        result.node_weathering_replacement_shift_s[:2],
        expected_node_shift,
        rtol=1.0e-12,
    )
    assert result.node_static_status[:2].tolist() == ['ok', 'ok']
    assert result.trace_static_status_sorted[0] == 'ok'
    assert np.isfinite(result.weathering_replacement_trace_shift_s_sorted[0])


def test_max_abs_shift_marks_status_without_clipping() -> None:
    result = build_refraction_weathering_replacement_statics(
        weathering_model=_weathering_model(),
        max_abs_shift_ms=18.0,
    )
    expected_first = (10.0 + 12.0) * SLOWNESS_DELTA_S_PER_M

    assert result.weathering_replacement_trace_shift_s_sorted[0] == pytest.approx(
        expected_first
    )
    assert abs(expected_first) * 1000.0 > 18.0
    assert result.trace_static_status_sorted[0] == 'exceeds_max_abs_shift'
    assert result.trace_static_valid_mask_sorted[0] == np.False_
    assert result.qc['exceeds_max_abs_shift_count'] == 6
    assert result.qc['invalid_trace_shift_count'] == 1


def test_max_abs_shift_marks_node_and_endpoint_status_without_clipping() -> None:
    result = build_refraction_weathering_replacement_statics(
        weathering_model=_weathering_model(),
        max_abs_shift_ms=12.0,
    )
    expected_node_shift = np.asarray(
        [10.0, 12.0, 15.0, 18.0, 20.0],
        dtype=np.float64,
    ) * SLOWNESS_DELTA_S_PER_M

    np.testing.assert_allclose(
        result.node_weathering_replacement_shift_s[:5],
        expected_node_shift,
        rtol=1.0e-12,
    )
    assert result.node_static_status.tolist() == [
        'ok',
        'ok',
        'exceeds_max_abs_shift',
        'exceeds_max_abs_shift',
        'exceeds_max_abs_shift',
        'inactive',
    ]
    assert result.source_static_status[2] == 'exceeds_max_abs_shift'
    assert result.receiver_static_status[2] == 'exceeds_max_abs_shift'
    assert result.source_static_status_sorted[2] == 'exceeds_max_abs_shift'
    assert result.receiver_static_status_sorted[1] == 'exceeds_max_abs_shift'
    assert result.qc['node_static_status_counts']['exceeds_max_abs_shift'] == 3
    assert result.qc['source_static_status_counts']['exceeds_max_abs_shift'] == 3
    assert result.qc['receiver_static_status_counts']['exceeds_max_abs_shift'] == 3


def test_local_v2_values_and_status_overlay_are_preserved() -> None:
    local_v2 = np.asarray(
        [2200.0, 2500.0, 3000.0, np.nan, 2500.0, 2500.0],
        dtype=np.float64,
    )
    local_status = np.asarray(
        ['ok', 'ok', 'ok', 'low_fold_v2_cell', 'ok', 'inactive_v2_cell'],
        dtype='<U32',
    )

    result = build_refraction_weathering_replacement_statics(
        weathering_model=_weathering_model(
            node_v2_m_s=local_v2,
            node_v2_status=local_status,
        ),
        max_abs_shift_ms=250.0,
    )

    expected = 15.0 * (1.0 / 3000.0 - 1.0 / WEATHERING_VELOCITY_M_S)
    assert result.node_weathering_replacement_shift_s[2] == pytest.approx(expected)
    assert result.source_weathering_replacement_shift_s[2] == pytest.approx(expected)
    assert result.node_static_status[3] == 'low_fold_v2_cell'
    assert result.source_static_status[3] == 'low_fold_v2_cell'
    assert result.receiver_static_status_sorted[2] == 'low_fold_v2_cell'
    assert np.isnan(result.node_weathering_replacement_shift_s[3])
    assert result.trace_static_valid_mask_sorted[3] == np.False_


def test_solve_cell_replacement_uses_local_v2_without_scalar_global_v2() -> None:
    local_v2 = np.asarray(
        [2200.0, 2500.0, 3000.0, 2800.0, 2600.0, 2500.0],
        dtype=np.float64,
    )
    weathering = replace(
        _weathering_model(node_v2_m_s=local_v2),
        bedrock_velocity_mode='solve_cell',
        bedrock_velocity_m_s=float('nan'),
        bedrock_slowness_s_per_m=float('nan'),
        bedrock_velocity_status='cell',
        v2_m_s=float('nan'),
        cell_id=np.asarray([0, 1, 2], dtype=np.int64),
        cell_v2_m_s=np.asarray([2200.0, 2500.0, 3000.0], dtype=np.float64),
        cell_velocity_status=np.asarray(['solved', 'solved', 'solved'], dtype='<U32'),
        cell_observation_count=np.asarray([2, 2, 2], dtype=np.int64),
    )

    result = build_refraction_weathering_replacement_statics(
        weathering_model=weathering,
        max_abs_shift_ms=250.0,
    )

    expected = weathering.node_weathering_thickness_m[:5] * (
        1.0 / local_v2[:5] - 1.0 / WEATHERING_VELOCITY_M_S
    )
    np.testing.assert_allclose(
        result.node_weathering_replacement_shift_s[:5],
        expected,
        rtol=1.0e-12,
    )
    assert result.bedrock_velocity_mode == 'solve_cell'
    assert np.isnan(result.bedrock_velocity_m_s)
    assert result.qc['bedrock_velocity_m_s'] is None
    assert result.qc['bedrock_slowness_s_per_m'] is None
    assert result.qc['replacement_slowness_delta_s_per_m'] is None
    json.dumps(result.qc, allow_nan=False)


def test_qc_summary_is_json_safe_and_contains_sign_convention() -> None:
    result = build_refraction_weathering_replacement_statics(
        weathering_model=_weathering_model(),
        max_abs_shift_ms=250.0,
    )

    json.dumps(result.qc, allow_nan=False)
    assert result.qc['static_component'] == 'weathering_replacement'
    assert result.qc['bedrock_velocity_m_s'] == pytest.approx(BEDROCK_VELOCITY_M_S)
    assert result.qc['replacement_slowness_delta_s_per_m'] == pytest.approx(
        SLOWNESS_DELTA_S_PER_M
    )
    assert result.qc['trace_shift_p95_abs_ms'] is not None
    assert result.qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'
    assert result.qc['formula'] == 'shift = z * (1/vb - 1/vw)'


def test_velocity_context_validation_rejects_bad_slowness() -> None:
    weathering = replace(
        _weathering_model(),
        bedrock_slowness_s_per_m=BEDROCK_SLOWNESS_S_PER_M * 1.01,
    )

    with pytest.raises(RefractionWeatheringReplacementError, match='slowness'):
        build_refraction_weathering_replacement_statics(
            weathering_model=weathering,
            max_abs_shift_ms=250.0,
        )
