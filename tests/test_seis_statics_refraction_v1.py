from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import numpy as np
import pytest

from seis_statics.refraction import (
    RefractionEndpointTable,
    RefractionStaticFirstLayerOptions,
    RefractionStaticInputModel,
    RefractionV1EstimateResult,
    RefractionV1EstimationError,
    estimate_global_v1_from_direct_arrivals,
)


SYNTHETIC_V1_M_S = 800.0
SYNTHETIC_V1_TOLERANCE_M_S = 1.0e-6


def _first_layer(**overrides: Any) -> RefractionStaticFirstLayerOptions:
    payload = {
        'mode': 'estimate_direct_arrival',
        'min_weathering_velocity_m_s': 500.0,
        'max_weathering_velocity_m_s': 1200.0,
        'min_direct_offset_m': 20.0,
        'max_direct_offset_m': 140.0,
        'min_picks_per_fit': 5,
        'min_groups': 3,
        'robust_enabled': True,
        'robust_threshold': 3.5,
    }
    payload.update(overrides)
    return RefractionStaticFirstLayerOptions(**payload)


def _input_model(
    *,
    v1_m_s: float = SYNTHETIC_V1_M_S,
    intercept_by_source: tuple[float, ...] = (0.010, 0.012, 0.014),
    offsets_m: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0),
    pick_overrides: dict[tuple[int, int], float] | None = None,
) -> RefractionStaticInputModel:
    n_sources = len(intercept_by_source)
    n_offsets = len(offsets_m)
    n_traces = n_sources * n_offsets

    source_id = np.repeat(np.arange(100, 100 + n_sources), n_offsets)
    receiver_id = np.arange(1000, 1000 + n_traces)
    source_x = np.repeat(np.arange(n_sources, dtype=np.float64) * 1000.0, n_offsets)
    source_y = np.zeros(n_traces, dtype=np.float64)
    offsets = np.tile(np.asarray(offsets_m, dtype=np.float64), n_sources)
    receiver_x = source_x + offsets
    receiver_y = np.zeros(n_traces, dtype=np.float64)
    source_elevation = np.full(n_traces, 100.0, dtype=np.float64)
    receiver_elevation = np.full(n_traces, 95.0, dtype=np.float64)
    pick_time = np.empty(n_traces, dtype=np.float64)
    for source_index, intercept in enumerate(intercept_by_source):
        start = source_index * n_offsets
        stop = start + n_offsets
        pick_time[start:stop] = intercept + offsets[:n_offsets] / v1_m_s
    for (source_index, offset_index), value in (pick_overrides or {}).items():
        pick_time[source_index * n_offsets + offset_index] = float(value)

    source_endpoint_key = np.asarray([f's{source}' for source in source_id])
    receiver_endpoint_key = np.asarray([f'r{receiver}' for receiver in receiver_id])
    node_x = np.concatenate([np.unique(source_x), receiver_x])
    node_y = np.zeros(node_x.shape[0], dtype=np.float64)
    node_elevation = np.concatenate(
        [
            np.full(n_sources, 100.0, dtype=np.float64),
            receiver_elevation,
        ]
    )
    endpoint_table = RefractionEndpointTable(
        node_id=np.arange(n_sources + n_traces, dtype=np.int64),
        endpoint_id=np.arange(n_sources + n_traces, dtype=np.int64),
        x_m=node_x,
        y_m=node_y,
        elevation_m=node_elevation,
        kind=np.asarray(['source'] * n_sources + ['receiver'] * n_traces),
        pick_count=np.ones(n_sources + n_traces, dtype=np.int64),
    )

    return RefractionStaticInputModel(
        file_id='line-a',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=pick_time,
        valid_pick_mask_sorted=np.isfinite(pick_time),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=source_id,
        receiver_id_sorted=receiver_id,
        source_x_m_sorted=source_x,
        source_y_m_sorted=source_y,
        receiver_x_m_sorted=receiver_x,
        receiver_y_m_sorted=receiver_y,
        source_elevation_m_sorted=source_elevation,
        receiver_elevation_m_sorted=receiver_elevation,
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=offsets,
        offset_m_sorted=None,
        distance_m_sorted=offsets,
        source_endpoint_key_sorted=source_endpoint_key,
        receiver_endpoint_key_sorted=receiver_endpoint_key,
        source_node_id_sorted=np.repeat(np.arange(n_sources, dtype=np.int64), n_offsets),
        receiver_node_id_sorted=np.arange(n_sources, n_sources + n_traces, dtype=np.int64),
        node_x_m=node_x,
        node_y_m=node_y,
        node_elevation_m=node_elevation,
        node_kind=np.asarray(['source'] * n_sources + ['receiver'] * n_traces),
        rejection_reason_sorted=np.asarray([''] * n_traces),
        qc={},
        endpoint_table=endpoint_table,
        metadata={},
    )


def test_v1_estimate_global_from_direct_arrivals() -> None:
    model = _input_model(
        intercept_by_source=(0.010, 0.012, 0.014, 0.016, 0.018, 0.020),
        pick_overrides={(2, 3): 0.300},
    )

    result = estimate_global_v1_from_direct_arrivals(
        input_model=model,
        first_layer=_first_layer(robust_threshold=2.5),
    )

    assert isinstance(result, RefractionV1EstimateResult)
    assert result.resolved_weathering_velocity_m_s == pytest.approx(
        SYNTHETIC_V1_M_S,
        abs=SYNTHETIC_V1_TOLERANCE_M_S,
    )
    assert result.qc['v1_status'] == 'estimated'
    assert result.qc['n_used_groups'] == 6
    assert result.qc['group_status_counts'] == {'ok': 6}
    assert set(result.group_status.tolist()) == {'ok'}
    assert np.any(result.group_n_used < result.group_n_candidates)
    json.dumps(result.qc, allow_nan=False)


def test_v1_estimate_robust_to_outlier_picks() -> None:
    model = _input_model(pick_overrides={(1, 2): 0.300})

    result = estimate_global_v1_from_direct_arrivals(
        input_model=model,
        first_layer=_first_layer(robust_threshold=2.5),
    )

    assert result.resolved_weathering_velocity_m_s == pytest.approx(800.0)
    assert np.min(result.group_n_used) < np.max(result.group_n_candidates)


def test_v1_estimate_excludes_invalid_finite_picks() -> None:
    model = _input_model(
        pick_overrides={
            (source_index, 5): 0.300
            for source_index in range(3)
        }
    )
    valid_pick_mask = model.valid_pick_mask_sorted.copy()
    valid_pick_mask[np.arange(5, model.n_traces, 6)] = False
    model = replace(model, valid_pick_mask_sorted=valid_pick_mask)

    result = estimate_global_v1_from_direct_arrivals(
        input_model=model,
        first_layer=_first_layer(robust_enabled=False),
    )

    assert result.resolved_weathering_velocity_m_s == pytest.approx(
        SYNTHETIC_V1_M_S,
        abs=SYNTHETIC_V1_TOLERANCE_M_S,
    )
    assert result.qc['n_candidate_picks'] == 15
    assert result.group_n_candidates.tolist() == [5, 5, 5]


def test_v1_estimate_respects_direct_offset_gate() -> None:
    offsets = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 300.0, 400.0)
    overrides = {
        (source_index, offset_index): 0.020 + offsets[offset_index] / 2200.0
        for source_index in range(3)
        for offset_index in (6, 7)
    }
    model = _input_model(offsets_m=offsets, pick_overrides=overrides)

    result = estimate_global_v1_from_direct_arrivals(
        input_model=model,
        first_layer=_first_layer(max_direct_offset_m=120.0),
    )

    assert result.resolved_weathering_velocity_m_s == pytest.approx(800.0)
    assert result.group_n_candidates.tolist() == [6, 6, 6]
    assert result.group_offset_max_m.tolist() == [120.0, 120.0, 120.0]


def test_v1_estimate_fails_with_insufficient_picks() -> None:
    model = _input_model(offsets_m=(20.0, 40.0, 60.0, 80.0))

    with pytest.raises(RefractionV1EstimationError, match='Insufficient'):
        estimate_global_v1_from_direct_arrivals(
            input_model=model,
            first_layer=_first_layer(min_picks_per_fit=5),
        )


def test_v1_estimate_partial_failures_reports_status_counts() -> None:
    offsets = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0)
    overrides = {
        (2, offset_index): 0.016 + offset / 2200.0
        for offset_index, offset in enumerate(offsets)
    }
    for source_index in range(3, 7):
        for offset_index in (4, 5):
            overrides[(source_index, offset_index)] = np.nan
    model = _input_model(
        intercept_by_source=(0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022),
        offsets_m=offsets,
        pick_overrides=overrides,
    )

    with pytest.raises(RefractionV1EstimationError) as exc_info:
        estimate_global_v1_from_direct_arrivals(
            input_model=model,
            first_layer=_first_layer(
                min_groups=3,
                min_weathering_velocity_m_s=500.0,
                max_weathering_velocity_m_s=1200.0,
            ),
        )

    message = str(exc_info.value)
    assert message.startswith('Insufficient valid direct-arrival V1 groups')
    assert '2 valid groups, require at least 3' in message
    assert 'Status counts:' in message
    assert 'ok=2' in message
    assert 'velocity_out_of_bounds=1' in message
    assert 'insufficient_picks=4' in message
    assert 'No valid direct-arrival V1 groups remain within' not in message
    assert exc_info.value.n_valid_groups == 2
    assert exc_info.value.min_groups == 3
    assert exc_info.value.group_status_counts == {
        'insufficient_picks': 4,
        'ok': 2,
        'velocity_out_of_bounds': 1,
    }


def test_v1_estimate_all_out_of_bounds_error_mentions_velocity_bounds() -> None:
    model = _input_model(v1_m_s=2200.0)

    with pytest.raises(RefractionV1EstimationError) as exc_info:
        estimate_global_v1_from_direct_arrivals(
            input_model=model,
            first_layer=_first_layer(
                min_weathering_velocity_m_s=500.0,
                max_weathering_velocity_m_s=1200.0,
            ),
        )

    message = str(exc_info.value)
    assert message.startswith(
        'No valid direct-arrival V1 groups remain within '
        'model.first_layer velocity bounds'
    )
    assert '0 valid groups, require at least 3' in message
    assert 'Status counts: velocity_out_of_bounds=3.' in message
    assert exc_info.value.n_valid_groups == 0
    assert exc_info.value.min_groups == 3
    assert exc_info.value.group_status_counts == {'velocity_out_of_bounds': 3}


def test_v1_estimate_insufficient_groups_error_reports_valid_and_required_counts() -> None:
    model = _input_model(intercept_by_source=(0.010, 0.012))

    with pytest.raises(RefractionV1EstimationError) as exc_info:
        estimate_global_v1_from_direct_arrivals(
            input_model=model,
            first_layer=_first_layer(min_groups=3),
        )

    message = str(exc_info.value)
    assert message.startswith('Insufficient valid direct-arrival V1 groups')
    assert '2 valid groups, require at least 3' in message
    assert 'Status counts: ok=2.' in message
    assert 'velocity bounds' not in message
    assert exc_info.value.n_valid_groups == 2
    assert exc_info.value.min_groups == 3
    assert exc_info.value.group_status_counts == {'ok': 2}


def test_v1_estimate_rejects_non_estimate_first_layer_options() -> None:
    with pytest.raises(RefractionV1EstimationError, match='estimate_direct_arrival'):
        estimate_global_v1_from_direct_arrivals(
            input_model=_input_model(),
            first_layer=RefractionStaticFirstLayerOptions(
                mode='constant',
                weathering_velocity_m_s=800.0,
            ),
        )
