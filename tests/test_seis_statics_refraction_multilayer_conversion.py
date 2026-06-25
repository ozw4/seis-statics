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
NONIDENTITY_SORTED_TRACE_INDEX_5 = np.asarray([3, 0, 4, 1, 2], dtype=np.int64)


def test_multilayer_2layer_conversion_matches_t1lsst_endpoint_arrays() -> None:
    source_input_model = _input_model(layer_count=2)
    sorted_trace_index = _nonidentity_sorted_trace_index(source_input_model.n_traces)
    input_model = replace(source_input_model, sorted_trace_index=sorted_trace_index)
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

    np.testing.assert_array_equal(input_model.sorted_trace_index, sorted_trace_index)
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


@pytest.mark.parametrize(
    ('mode', 'flat_datum_elevation_m', 'datum_shift_s'),
    [
        ('floating_only', None, 0.0),
        ('floating_and_flat', 80.0, -20.0 / VSUB_M_S),
    ],
)
def test_multilayer_3layer_datum_facade_composes_floating_modes(
    mode: str,
    flat_datum_elevation_m: float | None,
    datum_shift_s: float,
) -> None:
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
            mode=mode,
            floating_datum_mode='surface',
            flat_datum_elevation_m=flat_datum_elevation_m,
        ),
        solver_options=_solver_options(),
    )

    np.testing.assert_allclose(
        result.datum.source_endpoint_datum.floating_datum_shift_s,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.datum.receiver_endpoint_datum.floating_datum_shift_s,
        0.0,
        atol=1.0e-12,
    )
    expected_endpoint_total = expected.weathering_correction_s + datum_shift_s
    np.testing.assert_allclose(
        result.datum.final_trace_shift_s_sorted,
        expected_endpoint_total[input_model.source_node_id_sorted]
        + expected_endpoint_total[input_model.receiver_node_id_sorted],
        atol=1.0e-9,
    )
    np.testing.assert_array_equal(result.datum.trace_static_status_sorted, ['ok'] * 30)


def test_multilayer_datum_facade_smooths_topography_in_node_order() -> None:
    node_elevation_m = np.asarray([100.0, 115.0, 90.0, 110.0, 95.0], dtype=np.float64)
    input_model = _with_node_elevation(
        _input_model(layer_count=3),
        node_elevation_m,
    )
    reordered = _reorder_input_model(
        input_model,
        np.arange(input_model.n_traces - 1, -1, -1, dtype=np.int64),
    )
    datum_options = RefractionStaticDatumOptions(
        mode='floating_only',
        floating_datum_mode='smoothed_topography',
        smoothing_window_nodes=3,
    )

    result = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=input_model,
        model=_model(layer_count=3),
        datum_options=datum_options,
        solver_options=_solver_options(),
    )
    reordered_result = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=reordered,
        model=_model(layer_count=3),
        datum_options=datum_options,
        solver_options=_solver_options(),
    )

    np.testing.assert_allclose(
        _values_by_node(
            result.conversion.source_endpoint.node_id,
            result.datum.source_endpoint_datum.floating_datum_elevation_m,
        ),
        _values_by_node(
            reordered_result.conversion.source_endpoint.node_id,
            reordered_result.datum.source_endpoint_datum.floating_datum_elevation_m,
        ),
    )
    np.testing.assert_allclose(
        _values_by_node(
            result.conversion.receiver_endpoint.node_id,
            result.datum.receiver_endpoint_datum.floating_datum_elevation_m,
        ),
        _values_by_node(
            reordered_result.conversion.receiver_endpoint.node_id,
            reordered_result.datum.receiver_endpoint_datum.floating_datum_elevation_m,
        ),
    )
    np.testing.assert_allclose(
        _trace_order_values(input_model, result.datum.final_trace_shift_s_sorted),
        _trace_order_values(
            reordered,
            reordered_result.datum.final_trace_shift_s_sorted,
        ),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        _trace_order_values(input_model, result.datum.trace_static_status_sorted),
        _trace_order_values(
            reordered,
            reordered_result.datum.trace_static_status_sorted,
        ),
    )


def test_multilayer_datum_facade_smooths_topography_by_radius() -> None:
    node_elevation_m = np.asarray([100.0, 130.0, 70.0, 110.0, 90.0], dtype=np.float64)
    input_model = _with_node_elevation(
        _input_model(layer_count=3),
        node_elevation_m,
    )
    datum_options = RefractionStaticDatumOptions(
        mode='floating_only',
        floating_datum_mode='smoothed_topography',
        smoothing_radius_m=1.0,
        smoothing_window_nodes=11,
    )

    result = compute_refraction_multilayer_datum_statics_from_input_model(
        input_model=input_model,
        model=_model(layer_count=3),
        datum_options=datum_options,
        solver_options=_solver_options(),
    )

    np.testing.assert_allclose(
        _values_by_node(
            result.conversion.source_endpoint.node_id,
            result.datum.source_endpoint_datum.floating_datum_elevation_m,
        ),
        np.asarray([115.0, 100.0, 310.0 / 3.0, 90.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        _values_by_node(
            result.conversion.receiver_endpoint.node_id,
            result.datum.receiver_endpoint_datum.floating_datum_elevation_m,
        ),
        np.asarray([115.0, 100.0, 310.0 / 3.0, 90.0, 100.0], dtype=np.float64),
    )


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


def _with_node_elevation(
    input_model: RefractionStaticInputModel,
    node_elevation_m: np.ndarray,
) -> RefractionStaticInputModel:
    elevation = np.ascontiguousarray(node_elevation_m, dtype=np.float64)
    return replace(
        input_model,
        source_elevation_m_sorted=elevation[input_model.source_node_id_sorted],
        receiver_elevation_m_sorted=elevation[input_model.receiver_node_id_sorted],
        node_elevation_m=elevation,
        endpoint_table=replace(input_model.endpoint_table, elevation_m=elevation),
    )


def _reorder_input_model(
    input_model: RefractionStaticInputModel,
    order: np.ndarray,
) -> RefractionStaticInputModel:
    def take(value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        return np.ascontiguousarray(value[order])

    return replace(
        input_model,
        sorted_trace_index=take(input_model.sorted_trace_index),
        pick_time_s_sorted=take(input_model.pick_time_s_sorted),
        valid_pick_mask_sorted=take(input_model.valid_pick_mask_sorted),
        valid_observation_mask_sorted=take(input_model.valid_observation_mask_sorted),
        source_id_sorted=take(input_model.source_id_sorted),
        receiver_id_sorted=take(input_model.receiver_id_sorted),
        source_x_m_sorted=take(input_model.source_x_m_sorted),
        source_y_m_sorted=take(input_model.source_y_m_sorted),
        receiver_x_m_sorted=take(input_model.receiver_x_m_sorted),
        receiver_y_m_sorted=take(input_model.receiver_y_m_sorted),
        source_elevation_m_sorted=take(input_model.source_elevation_m_sorted),
        receiver_elevation_m_sorted=take(input_model.receiver_elevation_m_sorted),
        source_depth_m_sorted=take(input_model.source_depth_m_sorted),
        geometry_distance_m_sorted=take(input_model.geometry_distance_m_sorted),
        offset_m_sorted=take(input_model.offset_m_sorted),
        distance_m_sorted=take(input_model.distance_m_sorted),
        source_endpoint_key_sorted=take(input_model.source_endpoint_key_sorted),
        receiver_endpoint_key_sorted=take(input_model.receiver_endpoint_key_sorted),
        source_node_id_sorted=take(input_model.source_node_id_sorted),
        receiver_node_id_sorted=take(input_model.receiver_node_id_sorted),
        rejection_reason_sorted=take(input_model.rejection_reason_sorted),
        source_endpoint_id_sorted=take(input_model.source_endpoint_id_sorted),
        receiver_endpoint_id_sorted=take(input_model.receiver_endpoint_id_sorted),
    )


def _nonidentity_sorted_trace_index(n_traces: int) -> np.ndarray:
    trace_index = np.arange(n_traces, dtype=np.int64)
    trace_index[: NONIDENTITY_SORTED_TRACE_INDEX_5.size] = (
        NONIDENTITY_SORTED_TRACE_INDEX_5
    )
    return trace_index


def _values_by_node(node_id: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(node_id, kind='stable')
    return np.ascontiguousarray(np.asarray(values)[order])


def _trace_order_values(
    input_model: RefractionStaticInputModel,
    values: np.ndarray,
) -> np.ndarray:
    value_array = np.asarray(values)
    out = np.empty(input_model.n_traces, dtype=value_array.dtype)
    out[np.asarray(input_model.sorted_trace_index, dtype=np.int64)] = value_array
    return out


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
