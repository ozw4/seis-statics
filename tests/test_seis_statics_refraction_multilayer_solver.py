from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_statics.refraction import (
    LOW_FOLD_CELL_REJECTION_REASON,
    LOW_FOLD_NODE_REJECTION_REASON,
    RefractionEndpointTable,
    RefractionStaticFirstLayerOptions,
    RefractionStaticInputModel,
    RefractionStaticLayerOptions,
    RefractionStaticModelOptions,
    RefractionStaticRefractorCellOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
    solve_refraction_multilayer_time_terms,
)


REFERENCE_V1_M_S = 800.0
REFERENCE_V2_M_S = 2400.0
REFERENCE_V3_M_S = 3600.0
REFERENCE_VSUB_M_S = 4800.0
REFERENCE_T1_S = np.asarray([0.008, 0.010, 0.012, 0.014, 0.016], dtype=np.float64)
REFERENCE_T2_S = np.asarray([0.020, 0.024, 0.022, 0.026, 0.028], dtype=np.float64)
REFERENCE_T3_S = np.asarray([0.036, 0.040, 0.038, 0.043, 0.046], dtype=np.float64)
REFERENCE_SOURCE_NODE = np.asarray(
    [0, 0, 0, 1, 1, 2, 0, 1, 2, 3],
    dtype=np.int64,
)
REFERENCE_RECEIVER_NODE = np.asarray(
    [1, 2, 3, 2, 4, 4, 0, 1, 2, 3],
    dtype=np.int64,
)
REFERENCE_V2_OFFSET_M = np.asarray(
    [320.0, 450.0, 600.0, 380.0, 700.0, 520.0, 260.0, 300.0, 340.0, 480.0],
    dtype=np.float64,
)
REFERENCE_V3_OFFSET_M = np.asarray(
    [
        1050.0,
        1240.0,
        1390.0,
        1160.0,
        1550.0,
        1320.0,
        1100.0,
        1450.0,
        1700.0,
        1280.0,
    ],
    dtype=np.float64,
)
REFERENCE_VSUB_OFFSET_M = np.asarray(
    [
        2400.0,
        2700.0,
        3100.0,
        2500.0,
        3300.0,
        2900.0,
        2450.0,
        2800.0,
        3250.0,
        3000.0,
    ],
    dtype=np.float64,
)
NONIDENTITY_SORTED_TRACE_INDEX_5 = np.asarray([3, 0, 4, 1, 2], dtype=np.int64)


def test_multilayer_solver_keeps_layer_results_in_sorted_position_order() -> None:
    input_model = _five_trace_input_model()
    original_sorted_trace_index = input_model.sorted_trace_index.copy()
    model = _three_layer_model(
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=0.0,
            max_offset_m=100.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=2500.0,
        ),
        RefractionStaticLayerOptions(
            kind='v3_t2',
            min_offset_m=100.0,
            max_offset_m=200.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=3500.0,
        ),
        RefractionStaticLayerOptions(
            kind='vsub_t3',
            min_offset_m=200.0,
            max_offset_m=None,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=5000.0,
        ),
    )

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=model,
        solver_options=_solver_options(),
    )

    np.testing.assert_array_equal(
        input_model.sorted_trace_index,
        original_sorted_trace_index,
    )
    assert result.qc['array_order'] == 'sorted_position'
    np.testing.assert_array_equal(result.used_observation_mask_sorted, [True] * 5)
    np.testing.assert_array_equal(result.rejected_observation_mask_sorted, [False] * 5)
    np.testing.assert_array_equal(
        result.layer_kind_sorted,
        ['v2_t1', 'v3_t2', 'vsub_t3', 'v2_t1', 'v3_t2'],
    )
    np.testing.assert_allclose(result.residual_s_sorted, 0.0, atol=1.0e-10)
    np.testing.assert_array_equal(
        result.layer_result_by_kind['v2_t1'].solve_result.used_observation_mask_sorted,
        [True, False, False, True, False],
    )
    np.testing.assert_array_equal(
        result.layer_result_by_kind['v2_t1'].solve_result.design.row_trace_index_sorted,
        [0, 3],
    )
    np.testing.assert_array_equal(
        result.layer_result_by_kind['v3_t2'].solve_result.used_observation_mask_sorted,
        [False, True, False, False, True],
    )
    np.testing.assert_array_equal(
        result.layer_result_by_kind['v3_t2'].solve_result.design.row_trace_index_sorted,
        [1, 4],
    )
    np.testing.assert_array_equal(
        result.layer_result_by_kind['vsub_t3'].solve_result.used_observation_mask_sorted,
        [False, False, True, False, False],
    )
    np.testing.assert_array_equal(
        result.layer_result_by_kind[
            'vsub_t3'
        ].solve_result.design.row_trace_index_sorted,
        [2],
    )


def test_multilayer_solver_velocity_order_rejection_uses_sorted_positions() -> None:
    input_model = _five_trace_input_model()
    model = _three_layer_model(
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=0.0,
            max_offset_m=100.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=2500.0,
        ),
        RefractionStaticLayerOptions(
            kind='v3_t2',
            min_offset_m=100.0,
            max_offset_m=None,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=2400.0,
        ),
    )

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=model,
        solver_options=_solver_options(),
    )

    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, False, False, True, False],
    )
    np.testing.assert_array_equal(
        result.rejected_observation_mask_sorted,
        [False, True, True, False, True],
    )
    np.testing.assert_array_equal(
        result.layer_kind_sorted,
        ['v2_t1', 'v3_t2', 'v3_t2', 'v2_t1', 'v3_t2'],
    )
    np.testing.assert_array_equal(
        result.rejection_reason_sorted,
        ['', 'invalid_velocity_order', 'invalid_velocity_order', '', 'invalid_velocity_order'],
    )


def test_multilayer_solver_combines_enabled_layer_trace_arrays() -> None:
    input_model = _three_layer_input_model()
    model = _three_layer_model(
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=0.0,
            max_offset_m=100.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=2500.0,
        ),
        RefractionStaticLayerOptions(
            kind='v3_t2',
            min_offset_m=100.0,
            max_offset_m=200.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=3500.0,
        ),
        RefractionStaticLayerOptions(
            kind='vsub_t3',
            min_offset_m=200.0,
            max_offset_m=None,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=5000.0,
        ),
    )

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=model,
        solver_options=_solver_options(),
    )

    assert [item.layer_kind for item in result.layer_results] == [
        'v2_t1',
        'v3_t2',
        'vsub_t3',
    ]
    assert result.qc['layer_observations']['layer_observation_count'] == {
        'v2_t1': 12,
        'v3_t2': 12,
        'vsub_t3': 12,
    }
    assert np.all(result.used_observation_mask_sorted)
    assert not np.any(result.rejected_observation_mask_sorted)
    assert result.layer_kind_sorted[:12].tolist() == ['v2_t1'] * 12
    assert result.layer_kind_sorted[12:24].tolist() == ['v3_t2'] * 12
    assert result.layer_kind_sorted[24:].tolist() == ['vsub_t3'] * 12

    for layer_result in result.layer_results:
        mask = layer_result.solve_result.used_observation_mask_sorted
        np.testing.assert_allclose(
            result.modeled_pick_time_s_sorted[mask],
            layer_result.solve_result.modeled_pick_time_s_sorted[mask],
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            result.residual_s_sorted[mask],
            layer_result.solve_result.residual_s_sorted[mask],
            atol=1.0e-10,
        )


def test_multilayer_solver_marks_invalid_deeper_velocity_order_rows() -> None:
    input_model = _three_layer_input_model(layer_count=2)
    model = _three_layer_model(
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=0.0,
            max_offset_m=100.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=2500.0,
        ),
        RefractionStaticLayerOptions(
            kind='v3_t2',
            min_offset_m=100.0,
            max_offset_m=None,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=2400.0,
        ),
    )

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=model,
        solver_options=_solver_options(),
    )

    invalid_order_rows = np.arange(12, 24)
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        np.asarray([True] * 12 + [False] * 12),
    )
    np.testing.assert_array_equal(
        result.rejected_observation_mask_sorted,
        np.asarray([False] * 12 + [True] * 12),
    )
    np.testing.assert_array_equal(
        result.rejection_reason_sorted[invalid_order_rows],
        np.asarray(['invalid_velocity_order'] * 12),
    )
    np.testing.assert_array_equal(
        result.layer_kind_sorted,
        np.asarray(['v2_t1'] * 12 + ['v3_t2'] * 12),
    )

    layer = result.layer_result_by_kind['v3_t2']
    assert not np.any(layer.solve_result.used_observation_mask_sorted[invalid_order_rows])
    assert np.all(layer.solve_result.rejected_observation_mask_sorted[invalid_order_rows])
    np.testing.assert_array_equal(
        layer.rejection_reason_sorted[invalid_order_rows],
        np.asarray(['invalid_velocity_order'] * 12),
    )
    assert result.qc['n_used_observations'] == 12
    assert result.qc['n_rejected_observations'] == 12


def test_multilayer_solver_preserves_layer_and_robust_rejection_reasons() -> None:
    input_model = _three_layer_input_model(layer_count=2)
    sorted_trace_index = _nonidentity_sorted_trace_index(input_model.n_traces)
    pick_time = input_model.pick_time_s_sorted.copy()
    valid_mask = input_model.valid_observation_mask_sorted.copy()
    rejection_reason = input_model.rejection_reason_sorted.copy()
    valid_mask[0] = False
    rejection_reason[0] = 'bad_pick'
    pick_time[-1] += 0.12
    input_model = replace(
        input_model,
        sorted_trace_index=sorted_trace_index,
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        rejection_reason_sorted=rejection_reason,
    )
    original_sorted_trace_index = input_model.sorted_trace_index.copy()
    model = _three_layer_model(
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=0.0,
            max_offset_m=100.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=2500.0,
        ),
        RefractionStaticLayerOptions(
            kind='v3_t2',
            min_offset_m=100.0,
            max_offset_m=None,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=3500.0,
        ),
    )

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=model,
        solver_options=_robust_solver_options(),
    )

    np.testing.assert_array_equal(
        input_model.sorted_trace_index,
        original_sorted_trace_index,
    )
    assert result.rejection_reason_sorted[0] == 'bad_pick'
    assert result.rejection_reason_sorted[-1] == 'robust_rejected'
    assert result.layer_kind_sorted[0] == 'v2_t1'
    assert result.layer_kind_sorted[-1] == 'v3_t2'
    assert result.rejected_observation_mask_sorted[0]
    assert result.rejected_observation_mask_sorted[-1]
    assert result.qc['n_rejected_observations'] == 2


def test_multilayer_solver_preserves_low_fold_design_rejection_reasons() -> None:
    good_pairs = [
        (10, 20),
        (10, 21),
        (10, 22),
        (11, 20),
        (11, 21),
        (11, 22),
        (12, 20),
        (12, 21),
        (12, 22),
    ]
    rows: list[tuple[int, int, float, float]] = []
    for index, (src, rec) in enumerate(good_pairs):
        distance = 50.0 + float(index)
        rows.append((src, rec, distance, 0.01 + 0.02 + distance / 2500.0))
    low_fold_index = len(rows)
    rows.append((12, 23, 75.0, 0.01 + 0.02 + 75.0 / 2500.0))
    for index, (src, rec) in enumerate(good_pairs):
        distance = 150.0 + float(index)
        rows.append((src, rec, distance, 0.01 + 0.02 + distance / 3500.0))
    input_model = _input_model_from_rows(rows)
    model = _three_layer_model(
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=0.0,
            max_offset_m=100.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=2500.0,
        ),
        RefractionStaticLayerOptions(
            kind='v3_t2',
            min_offset_m=100.0,
            max_offset_m=None,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=3500.0,
        ),
    )
    solver_options = replace(_solver_options(), min_picks_per_node=3)

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=model,
        solver_options=solver_options,
    )

    assert result.rejected_observation_mask_sorted[low_fold_index]
    assert (
        result.rejection_reason_sorted[low_fold_index]
        == LOW_FOLD_NODE_REJECTION_REASON
    )
    assert result.layer_kind_sorted[low_fold_index] == 'v2_t1'
    layer = result.layer_result_by_kind['v2_t1']
    assert (
        layer.rejection_reason_sorted[low_fold_index]
        == LOW_FOLD_NODE_REJECTION_REASON
    )


def test_multilayer_solver_preserves_low_fold_cell_design_rejection_reasons() -> None:
    source_node_id = np.asarray([10, 11], dtype=np.int64)
    receiver_node_id = np.asarray([20, 21], dtype=np.int64)
    source_t = {10: 0.010, 11: 0.012}
    receiver_t = {20: 0.020, 21: 0.022, 22: 0.024}
    rows: list[tuple[int, int, float, float, float, float]] = []
    for src_index, src in enumerate(source_node_id):
        for rec_index, rec in enumerate(receiver_node_id):
            for cell_index, (cell_velocity, src_x, rec_x) in enumerate(
                ((2200.0, 0.0, 20.0), (2800.0, 80.0, 100.0))
            ):
                distance = 50.0 + float(src_index * 4 + rec_index * 2 + cell_index)
                pick = source_t[int(src)] + receiver_t[int(rec)] + distance / cell_velocity
                rows.append((int(src), int(rec), distance, pick, src_x, rec_x))
    low_fold_index = len(rows)
    rows.append((10, 22, 75.0, source_t[10] + receiver_t[22] + 75.0 / 2500.0, 130.0, 150.0))
    for src_index, src in enumerate(source_node_id):
        for rec_index, rec in enumerate(receiver_node_id):
            for cell_index, (src_x, rec_x) in enumerate(((0.0, 20.0), (80.0, 100.0))):
                distance = 150.0 + float(src_index * 4 + rec_index * 2 + cell_index)
                pick = source_t[int(src)] + receiver_t[int(rec)] + distance / 3500.0
                rows.append((int(src), int(rec), distance, pick, src_x, rec_x))
    input_model = _input_model_from_rows_with_coordinates(rows)
    model = _three_layer_model(
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=0.0,
            max_offset_m=100.0,
            velocity_mode='solve_cell',
            initial_velocity_m_s=2500.0,
            min_velocity_m_s=1600.0,
            max_velocity_m_s=3200.0,
            min_observations_per_cell=3,
        ),
        RefractionStaticLayerOptions(
            kind='v3_t2',
            min_offset_m=100.0,
            max_offset_m=None,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=3500.0,
        ),
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=3,
            size_of_cell_x_m=50.0,
            x_coordinate_origin_m=0.0,
            min_observations_per_cell=3,
        ),
    )

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=model,
        solver_options=_solver_options(),
    )

    assert result.rejected_observation_mask_sorted[low_fold_index]
    assert (
        result.rejection_reason_sorted[low_fold_index]
        == LOW_FOLD_CELL_REJECTION_REASON
    )
    assert result.layer_kind_sorted[low_fold_index] == 'v2_t1'
    layer = result.layer_result_by_kind['v2_t1']
    assert (
        layer.rejection_reason_sorted[low_fold_index]
        == LOW_FOLD_CELL_REJECTION_REASON
    )


def test_multilayer_solver_projects_cell_velocity_to_deeper_layer_rows() -> None:
    source_input_model = _cell_projection_input_model()
    sorted_trace_index = _nonidentity_sorted_trace_index(source_input_model.n_traces)
    input_model = replace(source_input_model, sorted_trace_index=sorted_trace_index)
    model = _three_layer_model(
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=0.0,
            max_offset_m=100.0,
            velocity_mode='solve_cell',
            initial_velocity_m_s=2600.0,
            min_velocity_m_s=1800.0,
            max_velocity_m_s=3200.0,
            min_observations_per_cell=3,
        ),
        RefractionStaticLayerOptions(
            kind='v3_t2',
            min_offset_m=100.0,
            max_offset_m=None,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=3600.0,
        ),
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=2,
            size_of_cell_x_m=50.0,
            x_coordinate_origin_m=0.0,
            min_observations_per_cell=3,
        ),
    )

    result = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=model,
        solver_options=_solver_options(),
    )

    np.testing.assert_array_equal(input_model.sorted_trace_index, sorted_trace_index)
    v2 = result.layer_result_by_kind['v2_t1'].velocity_m_s_sorted
    assert np.all(np.isfinite(v2))
    assert np.all(v2[:8] < result.velocity_m_s_sorted[8:])
    assert result.qc['n_used_observations'] == 16
    np.testing.assert_array_equal(
        result.layer_kind_sorted,
        np.asarray(['v2_t1'] * 8 + ['v3_t2'] * 8),
    )


def test_multilayer_solver_v3_t2_solve_global_recovers_t2_terms() -> None:
    result = solve_refraction_multilayer_time_terms(
        input_model=_reference_input_model(layer_count=2),
        model=_reference_model(v3_velocity_mode='solve_global'),
        solver_options=_solver_options(),
    )

    layer = result.layer_result_by_kind['v3_t2']

    assert layer.layer_index == 2
    assert layer.solve_result.bedrock_velocity_mode == 'solve_global'
    assert layer.solve_result.bedrock_velocity_m_s == pytest.approx(
        REFERENCE_V3_M_S,
        rel=1.0e-9,
    )
    np.testing.assert_allclose(
        layer.solve_result.node_half_intercept_time_s,
        REFERENCE_T2_S,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        layer.solve_result.residual_s_sorted[
            layer.solve_result.used_observation_mask_sorted
        ],
        0.0,
        atol=1.0e-9,
    )


def test_multilayer_solver_vsub_t3_fixed_global_recovers_t3_terms() -> None:
    result = solve_refraction_multilayer_time_terms(
        input_model=_reference_input_model(layer_count=3),
        model=_reference_model(
            v3_velocity_mode='fixed_global',
            vsub_velocity_mode='fixed_global',
        ),
        solver_options=_solver_options(),
    )

    layer = result.layer_result_by_kind['vsub_t3']

    assert layer.layer_index == 3
    assert layer.solve_result.bedrock_velocity_mode == 'fixed_global'
    assert layer.solve_result.bedrock_velocity_m_s == pytest.approx(
        REFERENCE_VSUB_M_S
    )
    np.testing.assert_allclose(
        layer.solve_result.node_half_intercept_time_s,
        REFERENCE_T3_S,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        layer.solve_result.residual_s_sorted[
            layer.solve_result.used_observation_mask_sorted
        ],
        0.0,
        atol=1.0e-9,
    )


def _solver_options() -> RefractionStaticSolverOptions:
    return RefractionStaticSolverOptions(
        half_intercept_damping_lambda=0.0,
        robust=RefractionStaticRobustOptions(enabled=False),
    )


def _robust_solver_options() -> RefractionStaticSolverOptions:
    return RefractionStaticSolverOptions(
        half_intercept_damping_lambda=0.0,
        robust=RefractionStaticRobustOptions(
            enabled=True,
            method='mad',
            threshold=2.0,
            scale_floor_ms=0.0,
            max_iterations=5,
            min_used_fraction=0.5,
            min_used_observations=1,
        ),
    )


def _three_layer_model(
    *layers: RefractionStaticLayerOptions,
    refractor_cell: RefractionStaticRefractorCellOptions | None = None,
) -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        method='multilayer_time_term',
        first_layer=RefractionStaticFirstLayerOptions(
            mode='constant',
            weathering_velocity_m_s=800.0,
        ),
        layers=layers,
        refractor_cell=refractor_cell,
    )


def _reference_model(
    *,
    v3_velocity_mode: str,
    vsub_velocity_mode: str | None = None,
) -> RefractionStaticModelOptions:
    v3_layer = RefractionStaticLayerOptions(
        kind='v3_t2',
        min_offset_m=1000.0,
        max_offset_m=None if vsub_velocity_mode is None else 1900.0,
        velocity_mode=v3_velocity_mode,
        fixed_velocity_m_s=(
            REFERENCE_V3_M_S if v3_velocity_mode == 'fixed_global' else None
        ),
        initial_velocity_m_s=(
            REFERENCE_V3_M_S if v3_velocity_mode != 'fixed_global' else None
        ),
        min_velocity_m_s=2600.0,
        max_velocity_m_s=4800.0,
    )
    layers = [
        RefractionStaticLayerOptions(
            kind='v2_t1',
            min_offset_m=250.0,
            max_offset_m=800.0,
            velocity_mode='fixed_global',
            fixed_velocity_m_s=REFERENCE_V2_M_S,
            min_velocity_m_s=1600.0,
            max_velocity_m_s=3200.0,
        ),
        v3_layer,
    ]
    if vsub_velocity_mode is not None:
        layers.append(
            RefractionStaticLayerOptions(
                kind='vsub_t3',
                min_offset_m=2200.0,
                max_offset_m=None,
                velocity_mode=vsub_velocity_mode,
                fixed_velocity_m_s=(
                    REFERENCE_VSUB_M_S
                    if vsub_velocity_mode == 'fixed_global'
                    else None
                ),
                initial_velocity_m_s=(
                    REFERENCE_VSUB_M_S
                    if vsub_velocity_mode != 'fixed_global'
                    else None
                ),
                min_velocity_m_s=3000.0,
                max_velocity_m_s=6200.0,
            )
        )
    return RefractionStaticModelOptions(
        method='multilayer_time_term',
        first_layer=RefractionStaticFirstLayerOptions(
            mode='constant',
            weathering_velocity_m_s=REFERENCE_V1_M_S,
        ),
        layers=tuple(layers),
    )


def _three_layer_input_model(layer_count: int = 3) -> RefractionStaticInputModel:
    source_node_id = np.asarray([10, 11, 12], dtype=np.int64)
    receiver_node_id = np.asarray([20, 21, 22, 23], dtype=np.int64)
    source_t = {10: 0.010, 11: 0.012, 12: 0.014}
    receiver_t = {20: 0.020, 21: 0.022, 22: 0.024, 23: 0.026}
    layer_specs = [
        ('v2_t1', 2500.0, 50.0),
        ('v3_t2', 3500.0, 150.0),
        ('vsub_t3', 5000.0, 250.0),
    ][:layer_count]
    rows: list[tuple[int, int, float, float]] = []
    for _, velocity, offset_base in layer_specs:
        for src in source_node_id:
            for rec_index, rec in enumerate(receiver_node_id):
                distance = offset_base + float(rec_index)
                pick = source_t[int(src)] + receiver_t[int(rec)] + distance / velocity
                rows.append((int(src), int(rec), distance, pick))
    return _input_model_from_rows(rows)


def _five_trace_input_model() -> RefractionStaticInputModel:
    source_t = {10: 0.010, 11: 0.012, 12: 0.014}
    receiver_t = {20: 0.020, 21: 0.022, 22: 0.024}
    rows = [
        _synthetic_row(source_t, receiver_t, 10, 20, 50.0, 2500.0),
        _synthetic_row(source_t, receiver_t, 10, 21, 150.0, 3500.0),
        _synthetic_row(source_t, receiver_t, 11, 21, 250.0, 5000.0),
        _synthetic_row(source_t, receiver_t, 11, 22, 60.0, 2500.0),
        _synthetic_row(source_t, receiver_t, 12, 22, 160.0, 3500.0),
    ]
    return replace(
        _input_model_from_rows(rows),
        sorted_trace_index=NONIDENTITY_SORTED_TRACE_INDEX_5.copy(),
    )


def _synthetic_row(
    source_t: dict[int, float],
    receiver_t: dict[int, float],
    source_node: int,
    receiver_node: int,
    distance_m: float,
    velocity_m_s: float,
) -> tuple[int, int, float, float]:
    return (
        source_node,
        receiver_node,
        distance_m,
        source_t[source_node] + receiver_t[receiver_node] + distance_m / velocity_m_s,
    )


def _nonidentity_sorted_trace_index(n_traces: int) -> np.ndarray:
    trace_index = np.arange(n_traces, dtype=np.int64)
    trace_index[: NONIDENTITY_SORTED_TRACE_INDEX_5.size] = (
        NONIDENTITY_SORTED_TRACE_INDEX_5
    )
    return trace_index


def _cell_projection_input_model() -> RefractionStaticInputModel:
    source_node_id = np.asarray([10, 11], dtype=np.int64)
    receiver_node_id = np.asarray([20, 21], dtype=np.int64)
    source_t = {10: 0.010, 11: 0.012, 12: 0.014}
    receiver_t = {20: 0.020, 21: 0.022}
    rows: list[tuple[int, int, float, float, float, float]] = []
    for offset_base, cell_velocity in (
        (50.0, (2200.0, 2800.0)),
        (150.0, (3600.0, 3600.0)),
    ):
        for src_index, src in enumerate(source_node_id):
            for rec_index, rec in enumerate(receiver_node_id):
                for cell_index, (src_x, rec_x) in enumerate(((0.0, 20.0), (80.0, 100.0))):
                    distance = offset_base + float(src_index * 4 + rec_index * 2 + cell_index)
                    velocity = cell_velocity[cell_index]
                    pick = source_t[int(src)] + receiver_t[int(rec)] + distance / velocity
                    rows.append(
                        (
                            int(src),
                            int(rec),
                            distance,
                            pick,
                            src_x,
                            rec_x,
                        )
                    )
    return _input_model_from_rows_with_coordinates(rows)


def _reference_input_model(layer_count: int) -> RefractionStaticInputModel:
    pick_parts = [
        (
            REFERENCE_T1_S[REFERENCE_SOURCE_NODE]
            + REFERENCE_T1_S[REFERENCE_RECEIVER_NODE]
            + REFERENCE_V2_OFFSET_M / REFERENCE_V2_M_S
        ),
        (
            REFERENCE_T2_S[REFERENCE_SOURCE_NODE]
            + REFERENCE_T2_S[REFERENCE_RECEIVER_NODE]
            + REFERENCE_V3_OFFSET_M / REFERENCE_V3_M_S
        ),
    ]
    source_parts = [REFERENCE_SOURCE_NODE, REFERENCE_SOURCE_NODE]
    receiver_parts = [REFERENCE_RECEIVER_NODE, REFERENCE_RECEIVER_NODE]
    distance_parts = [REFERENCE_V2_OFFSET_M, REFERENCE_V3_OFFSET_M]
    if layer_count == 3:
        pick_parts.append(
            REFERENCE_T3_S[REFERENCE_SOURCE_NODE]
            + REFERENCE_T3_S[REFERENCE_RECEIVER_NODE]
            + REFERENCE_VSUB_OFFSET_M / REFERENCE_VSUB_M_S
        )
        source_parts.append(REFERENCE_SOURCE_NODE)
        receiver_parts.append(REFERENCE_RECEIVER_NODE)
        distance_parts.append(REFERENCE_VSUB_OFFSET_M)
    source_node = np.concatenate(source_parts).astype(np.int64)
    receiver_node = np.concatenate(receiver_parts).astype(np.int64)
    distance = np.concatenate(distance_parts).astype(np.float64)
    pick = np.concatenate(pick_parts).astype(np.float64)
    rows = [
        (int(src), int(rec), float(dist), float(time), 0.0, float(dist))
        for src, rec, dist, time in zip(
            source_node.tolist(),
            receiver_node.tolist(),
            distance.tolist(),
            pick.tolist(),
            strict=True,
        )
    ]
    return _input_model_from_reference_rows(rows)


def _input_model_from_reference_rows(
    rows: list[tuple[int, int, float, float, float, float]],
) -> RefractionStaticInputModel:
    n_traces = len(rows)
    source_node = np.asarray([row[0] for row in rows], dtype=np.int64)
    receiver_node = np.asarray([row[1] for row in rows], dtype=np.int64)
    distance = np.asarray([row[2] for row in rows], dtype=np.float64)
    pick = np.asarray([row[3] for row in rows], dtype=np.float64)
    source_x = np.asarray([row[4] for row in rows], dtype=np.float64)
    receiver_x = np.asarray([row[5] for row in rows], dtype=np.float64)
    endpoint_table = _reference_endpoint_table(source_node, receiver_node)
    return RefractionStaticInputModel(
        file_id='multilayer-source-parity-unit',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=pick,
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=source_node,
        receiver_id_sorted=receiver_node,
        source_x_m_sorted=source_x,
        source_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_x_m_sorted=receiver_x,
        receiver_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
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


def _reference_endpoint_table(
    source_node: np.ndarray,
    receiver_node: np.ndarray,
) -> RefractionEndpointTable:
    node_id = np.arange(REFERENCE_T1_S.size, dtype=np.int64)
    pick_count = np.zeros(node_id.shape, dtype=np.int64)
    for node in np.concatenate((source_node, receiver_node)).tolist():
        pick_count[int(node)] += 1
    return RefractionEndpointTable(
        node_id=node_id,
        endpoint_id=node_id.copy(),
        x_m=node_id.astype(np.float64),
        y_m=np.zeros(node_id.shape, dtype=np.float64),
        elevation_m=np.zeros(node_id.shape, dtype=np.float64),
        kind=np.full(node_id.shape, 'linked', dtype='<U16'),
        pick_count=pick_count,
    )


def _input_model_from_rows(
    rows: list[tuple[int, int, float, float]],
) -> RefractionStaticInputModel:
    rows_with_coordinates = [
        (src, rec, distance, pick, 0.0, float((rec - 20) * 20 + 20))
        for src, rec, distance, pick in rows
    ]
    return _input_model_from_rows_with_coordinates(rows_with_coordinates)


def _input_model_from_rows_with_coordinates(
    rows: list[tuple[int, int, float, float, float, float]],
) -> RefractionStaticInputModel:
    n_traces = len(rows)
    source_node = np.asarray([row[0] for row in rows], dtype=np.int64)
    receiver_node = np.asarray([row[1] for row in rows], dtype=np.int64)
    distance = np.asarray([row[2] for row in rows], dtype=np.float64)
    pick = np.asarray([row[3] for row in rows], dtype=np.float64)
    source_x = np.asarray([row[4] for row in rows], dtype=np.float64)
    receiver_x = np.asarray([row[5] for row in rows], dtype=np.float64)
    node_id = np.asarray([10, 11, 12, 20, 21, 22, 23], dtype=np.int64)
    endpoint_table = RefractionEndpointTable(
        node_id=node_id,
        endpoint_id=node_id + 100,
        x_m=np.asarray([0.0, 0.0, 0.0, 20.0, 40.0, 120.0, 140.0]),
        y_m=np.zeros(node_id.size, dtype=np.float64),
        elevation_m=np.zeros(node_id.size, dtype=np.float64),
        kind=np.asarray(['source', 'source', 'source', 'receiver', 'receiver', 'receiver', 'receiver']),
        pick_count=np.full(node_id.size, 12, dtype=np.int64),
    )
    return RefractionStaticInputModel(
        file_id='multilayer-unit',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=pick,
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=source_node,
        receiver_id_sorted=receiver_node,
        source_x_m_sorted=source_x,
        source_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_x_m_sorted=receiver_x,
        receiver_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_elevation_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=distance,
        offset_m_sorted=distance,
        distance_m_sorted=distance,
        source_endpoint_key_sorted=np.asarray([f'source:{value}' for value in source_node]),
        receiver_endpoint_key_sorted=np.asarray([f'receiver:{value}' for value in receiver_node]),
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
