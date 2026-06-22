from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_statics.refraction import (
    RefractionEndpointTable,
    RefractionStaticInputModel,
    RefractionStaticModelOptions,
    RefractionStaticRefractorCellOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
    build_refraction_weathering_model_from_half_intercept_result,
    compute_weathering_thickness_from_half_intercept_time,
    compute_weathering_thickness_from_half_intercept_time_with_status,
    compute_weathering_thickness_scalar_from_half_intercept_time,
    estimate_refraction_half_intercept_from_input_model,
)


def _solver_options() -> RefractionStaticSolverOptions:
    return RefractionStaticSolverOptions(
        min_picks_per_node=1,
        half_intercept_damping_lambda=0.0,
        max_abs_half_intercept_time_ms=100.0,
        robust=RefractionStaticRobustOptions(enabled=False),
    )


def _global_model() -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode='solve_global',
        initial_bedrock_velocity_m_s=3000.0,
        min_bedrock_velocity_m_s=1200.0,
        max_bedrock_velocity_m_s=6000.0,
    )


def _cell_model(*, n_cells: int = 3) -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode='solve_cell',
        initial_bedrock_velocity_m_s=2600.0,
        min_bedrock_velocity_m_s=1200.0,
        max_bedrock_velocity_m_s=6000.0,
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=n_cells,
            size_of_cell_x_m=100.0,
            x_coordinate_origin_m=0.0,
            min_observations_per_cell=1,
        ),
    )


def _global_input_model() -> RefractionStaticInputModel:
    source_node_id = np.asarray([10, 10, 20, 20, 10, 20], dtype=np.int64)
    receiver_node_id = np.asarray([30, 40, 30, 40, 30, 40], dtype=np.int64)
    distance_m = np.asarray([500.0, 700.0, 600.0, 850.0, 900.0, 950.0])
    true_t1_by_node = {10: 0.03, 20: 0.05, 30: 0.035, 40: 0.045}
    pick_time = np.asarray(
        [
            true_t1_by_node[int(src)]
            + true_t1_by_node[int(rec)]
            + dist / 2500.0
            for src, rec, dist in zip(
                source_node_id,
                receiver_node_id,
                distance_m,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    valid_mask = np.asarray([True, True, True, True, True, False])
    endpoint_table = RefractionEndpointTable(
        node_id=np.asarray([10, 20, 30, 40], dtype=np.int64),
        endpoint_id=np.asarray([100, 200, 300, 400], dtype=np.int64),
        x_m=np.asarray([0.0, 100.0, 500.0, 700.0], dtype=np.float64),
        y_m=np.zeros(4, dtype=np.float64),
        elevation_m=np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
        kind=np.asarray(['source', 'source', 'receiver', 'receiver']),
        pick_count=np.asarray([3, 3, 3, 3], dtype=np.int64),
    )
    return RefractionStaticInputModel(
        file_id='weathering-global',
        n_traces=6,
        sorted_trace_index=np.arange(6, dtype=np.int64),
        pick_time_s_sorted=pick_time,
        valid_pick_mask_sorted=valid_mask,
        valid_observation_mask_sorted=valid_mask,
        source_id_sorted=source_node_id,
        receiver_id_sorted=receiver_node_id,
        source_x_m_sorted=np.asarray([0.0, 0.0, 100.0, 100.0, 0.0, 100.0]),
        source_y_m_sorted=np.zeros(6, dtype=np.float64),
        receiver_x_m_sorted=np.asarray([500.0, 700.0, 500.0, 700.0, 500.0, 700.0]),
        receiver_y_m_sorted=np.zeros(6, dtype=np.float64),
        source_elevation_m_sorted=np.full(6, 10.0, dtype=np.float64),
        receiver_elevation_m_sorted=np.full(6, 12.0, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=distance_m,
        offset_m_sorted=None,
        distance_m_sorted=distance_m,
        source_endpoint_key_sorted=np.asarray(
            [f'source:{value}' for value in source_node_id]
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{value}' for value in receiver_node_id]
        ),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        node_x_m=endpoint_table.x_m,
        node_y_m=endpoint_table.y_m,
        node_elevation_m=endpoint_table.elevation_m,
        node_kind=endpoint_table.kind,
        rejection_reason_sorted=np.where(valid_mask, 'ok', 'invalid_pick'),
        qc={},
        endpoint_table=endpoint_table,
        metadata={},
    )


def _cell_input_model() -> RefractionStaticInputModel:
    source_node_id = np.asarray([10, 30, 10, 10], dtype=np.int64)
    receiver_node_id = np.asarray([20, 20, 20, 20], dtype=np.int64)
    distance_m = np.asarray([500.0, 510.0, 700.0, 710.0], dtype=np.float64)
    midpoint_cell_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
    t1_by_node = {10: 0.03, 20: 0.04, 30: 0.05}
    slowness_by_cell = {0: 1.0 / 2400.0, 1: 1.0 / 3000.0}
    pick_time = np.asarray(
        [
            t1_by_node[int(src)]
            + t1_by_node[int(rec)]
            + dist * slowness_by_cell[int(cell)]
            for src, rec, dist, cell in zip(
                source_node_id,
                receiver_node_id,
                distance_m,
                midpoint_cell_id,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    endpoint_table = RefractionEndpointTable(
        node_id=np.asarray([10, 20, 30], dtype=np.int64),
        endpoint_id=np.asarray([100, 200, 300], dtype=np.int64),
        x_m=np.asarray([0.0, 100.0, 200.0], dtype=np.float64),
        y_m=np.zeros(3, dtype=np.float64),
        elevation_m=np.asarray([15.0, 16.0, 17.0], dtype=np.float64),
        kind=np.asarray(['source', 'receiver', 'source']),
        pick_count=np.asarray([3, 4, 1], dtype=np.int64),
    )
    return RefractionStaticInputModel(
        file_id='weathering-cell',
        n_traces=4,
        sorted_trace_index=np.arange(4, dtype=np.int64),
        pick_time_s_sorted=pick_time,
        valid_pick_mask_sorted=np.ones(4, dtype=bool),
        valid_observation_mask_sorted=np.ones(4, dtype=bool),
        source_id_sorted=np.asarray([1, 3, 1, 1], dtype=np.int64),
        receiver_id_sorted=np.asarray([2, 2, 2, 2], dtype=np.int64),
        source_x_m_sorted=np.asarray([0.0, 0.0, 100.0, 100.0], dtype=np.float64),
        source_y_m_sorted=np.zeros(4, dtype=np.float64),
        receiver_x_m_sorted=np.asarray([20.0, 20.0, 120.0, 120.0], dtype=np.float64),
        receiver_y_m_sorted=np.zeros(4, dtype=np.float64),
        source_elevation_m_sorted=np.zeros(4, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(4, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=distance_m,
        offset_m_sorted=None,
        distance_m_sorted=distance_m,
        source_endpoint_key_sorted=np.asarray(
            ['source:10', 'source:30', 'source:10', 'source:10']
        ),
        receiver_endpoint_key_sorted=np.asarray(
            ['receiver:20', 'receiver:20', 'receiver:20', 'receiver:20']
        ),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        node_x_m=endpoint_table.x_m,
        node_y_m=endpoint_table.y_m,
        node_elevation_m=endpoint_table.elevation_m,
        node_kind=endpoint_table.kind,
        rejection_reason_sorted=np.asarray(['ok', 'ok', 'ok', 'ok']),
        qc={},
        endpoint_table=endpoint_table,
        metadata={},
    )


def test_weathering_thickness_scalar_and_array_formula_match() -> None:
    scalar = compute_weathering_thickness_scalar_from_half_intercept_time(
        0.03,
        500.0,
        2500.0,
    )
    array = compute_weathering_thickness_from_half_intercept_time(
        np.asarray([0.03, 0.05]),
        500.0,
        2500.0,
    )

    expected = np.asarray([0.03, 0.05]) * 2500.0 * 500.0 / np.sqrt(
        2500.0**2 - 500.0**2
    )
    assert scalar == pytest.approx(expected[0])
    np.testing.assert_allclose(array, expected)


def test_weathering_with_status_marks_invalid_inputs_and_order() -> None:
    result = compute_weathering_thickness_from_half_intercept_time_with_status(
        half_intercept_time_s=np.asarray([np.nan, 0.01, -0.01]),
        surface_elevation_m=np.asarray([1.0, 1.0, 1.0]),
        v1_m_s=500.0,
        v2_m_s=np.asarray([2500.0, 400.0, 2500.0]),
    )

    np.testing.assert_array_equal(
        result.weathering_status,
        [
            'invalid_nonfinite_input',
            'invalid_velocity_order',
            'negative_weathering_thickness',
        ],
    )
    assert np.isnan(result.weathering_thickness_m).all()


def test_weathering_model_from_half_intercept_global_v2_maps_endpoints_and_traces() -> None:
    input_model = _global_input_model()
    half_intercept = estimate_refraction_half_intercept_from_input_model(
        input_model=input_model,
        model=_global_model(),
        solver_options=_solver_options(),
    )

    model = build_refraction_weathering_model_from_half_intercept_result(
        input_model=input_model,
        half_intercept_result=half_intercept,
        model=_global_model(),
    )

    np.testing.assert_array_equal(model.source_endpoint.endpoint_key, ['source:10', 'source:20'])
    np.testing.assert_array_equal(model.receiver_endpoint.endpoint_key, ['receiver:30', 'receiver:40'])
    expected_node_thickness = compute_weathering_thickness_from_half_intercept_time(
        np.asarray([0.03, 0.05, 0.035, 0.045]),
        500.0,
        2500.0,
    )
    np.testing.assert_allclose(
        model.node_weathering_thickness_m,
        expected_node_thickness,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        model.node_refractor_elevation_m,
        input_model.endpoint_table.elevation_m - expected_node_thickness,
        atol=1.0e-8,
    )
    np.testing.assert_array_equal(model.node_weathering_status, ['ok', 'ok', 'ok', 'ok'])
    np.testing.assert_allclose(
        model.trace_weathering_thickness_m_sorted[:2],
        [
            expected_node_thickness[0] + expected_node_thickness[2],
            expected_node_thickness[0] + expected_node_thickness[3],
        ],
        atol=1.0e-8,
    )
    assert model.qc['trace_weathering_status_counts'] == {'ok': 6}


def test_weathering_model_overlays_cell_local_v2_status() -> None:
    input_model = _cell_input_model()
    half_intercept = estimate_refraction_half_intercept_from_input_model(
        input_model=input_model,
        model=_cell_model(),
        solver_options=_solver_options(),
    )
    cell_status = half_intercept.cell_velocity_status.copy()
    cell_v2 = half_intercept.cell_v2_m_s.copy()
    cell_status[1] = 'low_fold'
    cell_v2[1] = np.nan
    cell_status[2] = 'inactive'
    half_intercept = replace(
        half_intercept,
        cell_velocity_status=cell_status,
        cell_v2_m_s=cell_v2,
    )

    model = build_refraction_weathering_model_from_half_intercept_result(
        input_model=input_model,
        half_intercept_result=half_intercept,
        model=_cell_model(),
    )

    np.testing.assert_allclose(model.source_endpoint.v2_m_s[0], 2400.0, atol=1.0e-3)
    assert model.source_endpoint.local_v2_status[0] == 'ok'
    assert model.source_endpoint.local_v2_status[1] == 'inactive_v2_cell'
    assert model.receiver_endpoint.local_v2_status[0] == 'low_fold_v2_cell'
    assert np.isnan(model.receiver_endpoint.weathering_thickness_m[0])
    assert model.receiver_endpoint.weathering_status[0] == 'low_fold_v2_cell'
    assert 'low_fold_v2_cell' in set(model.trace_weathering_status_sorted.tolist())
    assert model.source_endpoint.weathering_status[1] == 'inactive_v2_cell'
    assert 'mixed' in set(model.trace_weathering_status_sorted.tolist())
    design_qc = half_intercept.qc['design_matrix']
    for key in (
        'min_observations_per_cell',
        'n_low_fold_cells',
        'n_observations_rejected_by_low_fold_cell',
        'low_fold_cell_rejection_reason',
        'low_fold_cell_id',
        'cell_observation_count',
    ):
        assert model.qc[key] == design_qc[key]
