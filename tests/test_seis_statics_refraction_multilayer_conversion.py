from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import seis_statics.refraction.multilayer_conversion as multilayer_conversion_module
from seis_statics.refraction import (
    RefractionEndpointTable,
    RefractionMultilayerConversionError,
    RefractionStaticDatumOptions,
    RefractionStaticFirstLayerOptions,
    RefractionStaticInputModel,
    RefractionStaticLayerOptions,
    RefractionStaticModelOptions,
    RefractionStaticRefractorCellOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
    build_refraction_multilayer_conversion,
    compute_refraction_multilayer_datum_statics_from_input_model,
    solve_refraction_multilayer_time_terms,
)
from seis_statics.refraction.t1lsst import (
    compute_t1lsst_2layer_thicknesses_with_status,
    compute_t1lsst_3layer_thicknesses_with_status,
)


V1_M_S = 800.0
V2_M_S = 2400.0
V3_M_S = 3600.0
VSUB_M_S = 5000.0
T1_S = np.asarray([0.008, 0.010, 0.012, 0.014, 0.016], dtype=np.float64)
T2_S = np.asarray([0.020, 0.024, 0.022, 0.026, 0.028], dtype=np.float64)
T3_S = np.asarray([0.036, 0.040, 0.038, 0.043, 0.046], dtype=np.float64)
SOURCE_NODE = np.asarray([0, 0, 0, 1, 1, 2, 0, 1, 2, 3], dtype=np.int64)
RECEIVER_NODE = np.asarray([1, 2, 3, 2, 4, 4, 0, 1, 2, 3], dtype=np.int64)
V2_OFFSET_M = np.asarray(
    [320.0, 450.0, 600.0, 380.0, 700.0, 520.0, 260.0, 300.0, 340.0, 480.0],
    dtype=np.float64,
)
V3_OFFSET_M = np.asarray(
    [1050.0, 1240.0, 1390.0, 1160.0, 1550.0, 1320.0, 1100.0, 1450.0, 1700.0, 1280.0],
    dtype=np.float64,
)
VSUB_OFFSET_M = np.asarray(
    [2400.0, 2700.0, 3100.0, 2500.0, 3300.0, 2900.0, 2450.0, 2800.0, 3250.0, 3000.0],
    dtype=np.float64,
)


def test_multilayer_2layer_conversion_matches_t1lsst_endpoint_arrays() -> None:
    input_model = _input_model(layer_count=2)
    solve = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=_model(layer_count=2),
        solver_options=_solver_options(),
    )

    result = build_refraction_multilayer_conversion(
        input_model=input_model,
        model=_model(layer_count=2),
        solve_result=solve,
    )
    expected = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=T1_S,
        t2_s=T2_S,
        v1_m_s=V1_M_S,
        v2_m_s=V2_M_S,
        v3_m_s=V3_M_S,
    )

    assert result.layer_count == 2
    np.testing.assert_array_equal(result.source_endpoint.static_status, ['ok'] * 4)
    np.testing.assert_allclose(result.source_endpoint.sh1_m, expected.sh1_m[:4])
    np.testing.assert_allclose(result.source_endpoint.sh2_m, expected.sh2_m[:4])
    receiver_node_order = np.asarray([1, 2, 3, 4, 0], dtype=np.int64)
    np.testing.assert_allclose(
        result.receiver_endpoint.weathering_replacement_shift_s,
        expected.weathering_correction_s[receiver_node_order],
    )
    np.testing.assert_allclose(
        result.weathering_replacement_trace_shift_s_sorted,
        expected.weathering_correction_s[input_model.source_node_id_sorted]
        + expected.weathering_correction_s[input_model.receiver_node_id_sorted],
    )


def test_multilayer_3layer_datum_facade_composes_flat_final_shift() -> None:
    input_model = _input_model(layer_count=3, elevation_m=100.0)
    expected = compute_t1lsst_3layer_thicknesses_with_status(
        t1_s=T1_S,
        t2_s=T2_S,
        t3_s=T3_S,
        v1_m_s=V1_M_S,
        v2_m_s=V2_M_S,
        v3_m_s=V3_M_S,
        vsub_m_s=VSUB_M_S,
    )

    result = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=input_model,
        model=_model(layer_count=3),
        datum_options=RefractionStaticDatumOptions(
            mode='flat_only',
            flat_datum_elevation_m=80.0,
        ),
        solver_options=_solver_options(),
    )

    np.testing.assert_allclose(
        result.conversion.source_endpoint.sh3_m,
        expected.sh3_m[:4],
        atol=1.0e-9,
    )
    expected_endpoint_total = expected.weathering_correction_s - (20.0 / VSUB_M_S)
    np.testing.assert_allclose(
        result.datum.refraction_trace_shift_s_sorted,
        expected_endpoint_total[input_model.source_node_id_sorted]
        + expected_endpoint_total[input_model.receiver_node_id_sorted],
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.datum.final_trace_shift_s_sorted,
        result.datum.refraction_trace_shift_s_sorted,
    )
    np.testing.assert_array_equal(result.datum.trace_static_status_sorted, ['ok'] * 30)


def test_multilayer_datum_facade_uses_endpoint_replacement_velocities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_model = _input_model(layer_count=3, elevation_m=100.0)
    solve = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=_model(layer_count=3),
        solver_options=_solver_options(),
    )
    vsub_by_cell = np.asarray([4900.0, 5000.0, 5100.0, 5200.0, 5300.0], dtype=np.float64)
    vsub_layer = solve.layer_result_by_kind['vsub_t3']
    patched_vsub_solve = replace(
        vsub_layer.solve_result,
        bedrock_velocity_mode='solve_cell',
        cell_id=np.arange(vsub_by_cell.size, dtype=np.int64),
        cell_bedrock_velocity_m_s=vsub_by_cell,
        cell_velocity_status=np.full(vsub_by_cell.shape, 'solved', dtype='<U64'),
    )
    patched_vsub_layer = replace(vsub_layer, solve_result=patched_vsub_solve)
    patched_solve = replace(
        solve,
        layer_results=(
            solve.layer_result_by_kind['v2_t1'],
            solve.layer_result_by_kind['v3_t2'],
            patched_vsub_layer,
        ),
        layer_result_by_kind={
            'v2_t1': solve.layer_result_by_kind['v2_t1'],
            'v3_t2': solve.layer_result_by_kind['v3_t2'],
            'vsub_t3': patched_vsub_layer,
        },
    )
    monkeypatch.setattr(
        multilayer_conversion_module,
        'solve_refraction_multilayer_time_terms',
        lambda **_: patched_solve,
    )

    result = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=input_model,
        model=_model_with_cell_replacement_layer(),
        datum_options=RefractionStaticDatumOptions(
            mode='flat_only',
            flat_datum_elevation_m=80.0,
        ),
        solver_options=_solver_options(),
    )

    np.testing.assert_allclose(
        result.datum.source_endpoint_datum.replacement_velocity_m_s,
        vsub_by_cell[[0, 1, 2, 3]],
    )
    np.testing.assert_allclose(
        result.datum.receiver_endpoint_datum.replacement_velocity_m_s,
        vsub_by_cell[[1, 2, 3, 4, 0]],
    )
    np.testing.assert_allclose(
        result.datum.source_endpoint_datum.flat_datum_shift_s,
        -20.0 / vsub_by_cell[[0, 1, 2, 3]],
    )
    np.testing.assert_allclose(
        result.datum.receiver_endpoint_datum.flat_datum_shift_s,
        -20.0 / vsub_by_cell[[1, 2, 3, 4, 0]],
    )
    assert result.datum.replacement_velocity_m_s is None
    np.testing.assert_array_equal(result.datum.trace_static_status_sorted, ['ok'] * 30)


def test_multilayer_conversion_propagates_low_fold_endpoint_status() -> None:
    input_model = _input_model(layer_count=2)
    solve = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=_model(layer_count=2),
        solver_options=_solver_options(),
    )
    layer = solve.layer_result_by_kind['v2_t1']
    patched_layer_solve = replace(
        layer.solve_result,
        node_solution_status=np.asarray(
            ['solved', 'low_fold', 'solved', 'solved', 'solved'],
            dtype='<U64',
        ),
    )
    patched_solve = replace(
        solve,
        layer_results=(
            replace(layer, solve_result=patched_layer_solve),
            solve.layer_result_by_kind['v3_t2'],
        ),
        layer_result_by_kind={
            'v2_t1': replace(layer, solve_result=patched_layer_solve),
            'v3_t2': solve.layer_result_by_kind['v3_t2'],
        },
    )

    result = build_refraction_multilayer_conversion(
        input_model=input_model,
        model=_model(layer_count=2),
        solve_result=patched_solve,
    )

    assert result.source_endpoint.static_status.tolist()[1] == 'low_fold'
    assert np.isnan(result.source_endpoint.weathering_replacement_shift_s[1])
    assert result.trace_static_status_sorted[3] == 'low_fold'
    assert not result.trace_static_valid_mask_sorted[3]


def test_multilayer_conversion_propagates_solve_cell_v2_statuses() -> None:
    input_model = _input_model(layer_count=2)
    solve = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=_model(layer_count=2),
        solver_options=_solver_options(),
    )
    layer = solve.layer_result_by_kind['v2_t1']
    patched_layer_solve = replace(
        layer.solve_result,
        bedrock_velocity_mode='solve_cell',
        cell_id=np.arange(4, dtype=np.int64),
        cell_bedrock_velocity_m_s=np.asarray(
            [V1_M_S - 100.0, V2_M_S, V2_M_S, np.nan],
            dtype=np.float64,
        ),
        cell_velocity_status=np.asarray(
            ['solved', 'inactive', 'low_fold', 'solved'],
            dtype='<U64',
        ),
    )
    patched_v2_layer = replace(layer, solve_result=patched_layer_solve)
    patched_solve = replace(
        solve,
        layer_results=(patched_v2_layer, solve.layer_result_by_kind['v3_t2']),
        layer_result_by_kind={
            'v2_t1': patched_v2_layer,
            'v3_t2': solve.layer_result_by_kind['v3_t2'],
        },
    )

    result = build_refraction_multilayer_conversion(
        input_model=input_model,
        model=_model_with_cell_v2_layer(),
        solve_result=patched_solve,
    )

    np.testing.assert_array_equal(
        result.source_endpoint.static_status,
        [
            'v2_not_greater_than_v1',
            'inactive_v2_cell',
            'low_fold_v2_cell',
            'invalid_local_v2',
        ],
    )
    np.testing.assert_array_equal(
        result.receiver_endpoint.static_status,
        [
            'inactive_v2_cell',
            'low_fold_v2_cell',
            'invalid_local_v2',
            'outside_refractor_cell_grid',
            'v2_not_greater_than_v1',
        ],
    )
    assert result.trace_static_status_sorted[0] == 'v2_not_greater_than_v1'
    assert result.trace_static_status_sorted[4] == 'inactive_v2_cell'
    assert result.trace_static_status_sorted[9] == 'invalid_local_v2'
    assert not np.any(result.trace_static_valid_mask_sorted)
    assert np.all(np.isnan(result.source_endpoint.weathering_replacement_shift_s))


def test_multilayer_conversion_rejects_single_layer_results() -> None:
    input_model = _input_model(layer_count=2)
    solve = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=_model(layer_count=2),
        solver_options=_solver_options(),
    )

    with pytest.raises(RefractionMultilayerConversionError, match='layer_count'):
        build_refraction_multilayer_conversion(
            input_model=input_model,
            model=_model(layer_count=2),
            solve_result=solve,
            layer_count=1,
        )


def _solver_options() -> RefractionStaticSolverOptions:
    return RefractionStaticSolverOptions(
        half_intercept_damping_lambda=0.0,
        robust=RefractionStaticRobustOptions(enabled=False),
    )


def _model(*, layer_count: int) -> RefractionStaticModelOptions:
    layers = [
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=250.0,
            max_offset_m=800.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=V2_M_S,
        ),
        RefractionStaticLayerOptions(
            kind='v3_t2',
            min_offset_m=1000.0,
            max_offset_m=None if layer_count == 2 else 1900.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=V3_M_S,
        ),
    ]
    if layer_count == 3:
        layers.append(
            RefractionStaticLayerOptions(
                kind='vsub_t3',
                min_offset_m=2200.0,
                max_offset_m=None,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=VSUB_M_S,
            )
        )
    return RefractionStaticModelOptions(
        method='multilayer_time_term',
        first_layer=RefractionStaticFirstLayerOptions(
            mode='constant',
            weathering_velocity_m_s=V1_M_S,
        ),
        layers=tuple(layers),
    )


def _model_with_cell_replacement_layer() -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        method='multilayer_time_term',
        first_layer=RefractionStaticFirstLayerOptions(
            mode='constant',
            weathering_velocity_m_s=V1_M_S,
        ),
        layers=(
            RefractionStaticLayerOptions(
                kind='v2_t1',
                min_offset_m=250.0,
                max_offset_m=800.0,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=V2_M_S,
            ),
            RefractionStaticLayerOptions(
                kind='v3_t2',
                min_offset_m=1000.0,
                max_offset_m=1900.0,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=V3_M_S,
            ),
            RefractionStaticLayerOptions(
                kind='vsub_t3',
                min_offset_m=2200.0,
                max_offset_m=None,
                velocity_mode='solve_cell',
                initial_velocity_m_s=VSUB_M_S,
                min_velocity_m_s=4500.0,
                max_velocity_m_s=5600.0,
            ),
        ),
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=5,
            size_of_cell_x_m=1.0,
            x_coordinate_origin_m=-0.5,
            min_observations_per_cell=1,
        ),
    )


def _model_with_cell_v2_layer() -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        method='multilayer_time_term',
        first_layer=RefractionStaticFirstLayerOptions(
            mode='constant',
            weathering_velocity_m_s=V1_M_S,
        ),
        layers=(
            RefractionStaticLayerOptions(
                kind='v2_t1',
                min_offset_m=250.0,
                max_offset_m=800.0,
                velocity_mode='solve_cell',
                initial_velocity_m_s=V2_M_S,
                min_velocity_m_s=V1_M_S + 1.0,
                max_velocity_m_s=3000.0,
            ),
            RefractionStaticLayerOptions(
                kind='v3_t2',
                min_offset_m=1000.0,
                max_offset_m=None,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=V3_M_S,
            ),
        ),
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=4,
            size_of_cell_x_m=1.0,
            x_coordinate_origin_m=-0.5,
            min_observations_per_cell=1,
        ),
    )


def _input_model(*, layer_count: int, elevation_m: float = 0.0) -> RefractionStaticInputModel:
    pick_parts = [
        T1_S[SOURCE_NODE] + T1_S[RECEIVER_NODE] + V2_OFFSET_M / V2_M_S,
        T2_S[SOURCE_NODE] + T2_S[RECEIVER_NODE] + V3_OFFSET_M / V3_M_S,
    ]
    source_parts = [SOURCE_NODE, SOURCE_NODE]
    receiver_parts = [RECEIVER_NODE, RECEIVER_NODE]
    distance_parts = [V2_OFFSET_M, V3_OFFSET_M]
    if layer_count == 3:
        pick_parts.append(
            T3_S[SOURCE_NODE] + T3_S[RECEIVER_NODE] + VSUB_OFFSET_M / VSUB_M_S
        )
        source_parts.append(SOURCE_NODE)
        receiver_parts.append(RECEIVER_NODE)
        distance_parts.append(VSUB_OFFSET_M)
    source_node = np.concatenate(source_parts).astype(np.int64)
    receiver_node = np.concatenate(receiver_parts).astype(np.int64)
    distance = np.concatenate(distance_parts).astype(np.float64)
    pick = np.concatenate(pick_parts).astype(np.float64)
    n_traces = int(pick.shape[0])
    endpoint_table = _endpoint_table(source_node, receiver_node, elevation_m=elevation_m)
    return RefractionStaticInputModel(
        file_id='multilayer-conversion-unit',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=pick,
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=source_node,
        receiver_id_sorted=receiver_node,
        source_x_m_sorted=source_node.astype(np.float64),
        source_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_x_m_sorted=receiver_node.astype(np.float64),
        receiver_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_elevation_m_sorted=np.full(n_traces, elevation_m, dtype=np.float64),
        receiver_elevation_m_sorted=np.full(n_traces, elevation_m, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=distance,
        offset_m_sorted=distance,
        distance_m_sorted=distance,
        source_endpoint_key_sorted=np.asarray(
            [f'source:{value}' for value in source_node],
            dtype=object,
        ),
        receiver_endpoint_key_sorted=np.asarray(
            [f'receiver:{value}' for value in receiver_node],
            dtype=object,
        ),
        source_node_id_sorted=source_node,
        receiver_node_id_sorted=receiver_node,
        node_x_m=endpoint_table.x_m,
        node_y_m=endpoint_table.y_m,
        node_elevation_m=endpoint_table.elevation_m,
        node_kind=endpoint_table.kind,
        rejection_reason_sorted=np.full(n_traces, '', dtype='<U64'),
        qc={},
        endpoint_table=endpoint_table,
        metadata={},
    )


def _endpoint_table(
    source_node: np.ndarray,
    receiver_node: np.ndarray,
    *,
    elevation_m: float,
) -> RefractionEndpointTable:
    node_id = np.arange(T1_S.size, dtype=np.int64)
    pick_count = np.zeros(node_id.shape, dtype=np.int64)
    for node in np.concatenate((source_node, receiver_node)).tolist():
        pick_count[int(node)] += 1
    return RefractionEndpointTable(
        node_id=node_id,
        endpoint_id=node_id.copy(),
        x_m=node_id.astype(np.float64),
        y_m=np.zeros(node_id.shape, dtype=np.float64),
        elevation_m=np.full(node_id.shape, elevation_m, dtype=np.float64),
        kind=np.full(node_id.shape, 'linked', dtype='<U16'),
        pick_count=pick_count,
    )
