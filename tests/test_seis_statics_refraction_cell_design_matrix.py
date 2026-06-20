from __future__ import annotations

import numpy as np
from scipy import sparse

from seis_statics.refraction import (
    LOW_FOLD_CELL_REJECTION_REASON,
    LOW_FOLD_NODE_REJECTION_REASON,
    OUTSIDE_REFRACTOR_CELL_GRID_REASON,
    RefractionEndpointTable,
    RefractionStaticInputModel,
    RefractionStaticModelOptions,
    RefractionStaticRefractorCellOptions,
    build_refraction_static_design_matrix,
    build_refraction_static_design_matrix_from_arrays,
)


def test_refraction_cell_design_matrix_filters_low_fold_and_outside_cells() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20, 0.21, 0.30, 0.40, 0.50]),
        valid_observation_mask_sorted=np.asarray([True, True, True, True, True]),
        source_node_id_sorted=np.asarray([10, 20, 10, 30, 10]),
        receiver_node_id_sorted=np.asarray([20, 30, 30, 20, 30]),
        distance_m_sorted=np.asarray([500.0, 510.0, 700.0, 800.0, 900.0]),
        node_id=np.asarray([10, 20, 30]),
        sorted_trace_index=np.asarray([42, 43, 44, 45, 46]),
        bedrock_velocity_mode='solve_cell',
        midpoint_cell_id_sorted=np.asarray([0, 0, 1, 2, -1]),
        n_total_cells=3,
        number_of_cell_x=3,
        number_of_cell_y=1,
        cell_assignment_mode='midpoint',
        min_observations_per_cell=2,
        rejection_reason_sorted=np.asarray(['ok', 'ok', 'ok', 'ok', 'ok']),
    )

    assert sparse.isspmatrix_csr(design.matrix)
    assert design.matrix.dtype == np.float64
    np.testing.assert_array_equal(design.row_trace_index_sorted, [42, 43])
    np.testing.assert_array_equal(design.active_cell_id, [0])
    np.testing.assert_array_equal(design.inactive_cell_id, [1, 2])
    assert design.cell_id_to_col == {0: 3}
    np.testing.assert_array_equal(design.row_midpoint_cell_id, [0, 0])
    np.testing.assert_array_equal(design.row_midpoint_cell_col, [3, 3])
    np.testing.assert_allclose(
        design.matrix.toarray(),
        np.asarray(
            [
                [1.0, 1.0, 0.0, 500.0],
                [0.0, 1.0, 1.0, 510.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(design.rhs_s, [0.20, 0.21])
    np.testing.assert_array_equal(
        design.rejection_reason_sorted,
        [
            'ok',
            'ok',
            LOW_FOLD_CELL_REJECTION_REASON,
            LOW_FOLD_CELL_REJECTION_REASON,
            OUTSIDE_REFRACTOR_CELL_GRID_REASON,
        ],
    )
    assert design.qc['cell_observation_count'] == [2, 1, 1]
    assert design.qc['low_fold_cell_id'] == [1, 2]
    assert design.qc['n_observations_outside_grid'] == 1
    assert design.qc['n_observations_rejected_by_low_fold_cell'] == 2
    assert design.qc['n_observations_used'] == 2


def test_refraction_cell_design_matrix_rechecks_cell_fold_after_node_pruning() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20, 0.21, 0.30, 0.31]),
        valid_observation_mask_sorted=np.asarray([True, True, True, True]),
        source_node_id_sorted=np.asarray([10, 30, 10, 10]),
        receiver_node_id_sorted=np.asarray([20, 20, 20, 20]),
        distance_m_sorted=np.asarray([500.0, 510.0, 700.0, 710.0]),
        node_id=np.asarray([10, 20, 30]),
        sorted_trace_index=np.asarray([100, 101, 102, 103]),
        bedrock_velocity_mode='solve_cell',
        midpoint_cell_id_sorted=np.asarray([0, 0, 1, 1]),
        n_total_cells=2,
        number_of_cell_x=2,
        number_of_cell_y=1,
        cell_assignment_mode='midpoint',
        min_observations_per_cell=2,
        min_observations_per_node=2,
        rejection_reason_sorted=np.asarray(['ok', 'ok', 'ok', 'ok']),
    )

    np.testing.assert_array_equal(design.row_trace_index_sorted, [102, 103])
    np.testing.assert_array_equal(design.active_cell_id, [1])
    np.testing.assert_array_equal(design.inactive_cell_id, [0])
    assert design.cell_id_to_col == {1: 2}
    np.testing.assert_array_equal(design.row_midpoint_cell_id, [1, 1])
    np.testing.assert_array_equal(
        design.rejection_reason_sorted,
        [
            LOW_FOLD_CELL_REJECTION_REASON,
            LOW_FOLD_NODE_REJECTION_REASON,
            'ok',
            'ok',
        ],
    )
    np.testing.assert_array_equal(design.low_fold_node_id, [30])
    assert design.qc['cell_observation_count'] == [1, 2]
    assert design.qc['low_fold_cell_id'] == [0]
    assert design.qc['n_observations_rejected_by_low_fold_cell'] == 1
    assert design.qc['n_observations_rejected_by_low_fold_node'] == 1
    assert design.qc['min_observations_per_active_cell'] == 2


def test_refraction_cell_design_matrix_from_model_uses_midpoint_assignment() -> None:
    model = RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode='solve_cell',
        initial_bedrock_velocity_m_s=2000.0,
        min_bedrock_velocity_m_s=1200.0,
        max_bedrock_velocity_m_s=4000.0,
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=3,
            size_of_cell_x_m=100.0,
            x_coordinate_origin_m=0.0,
            min_observations_per_cell=2,
        ),
    )
    input_model = _input_model_for_cell_assignment()

    design = build_refraction_static_design_matrix(
        input_model=input_model,
        model=model,
        include_diagnostics=True,
    )

    np.testing.assert_array_equal(design.row_trace_index_sorted, [0, 1])
    np.testing.assert_array_equal(design.row_midpoint_cell_id, [0, 0])
    assert design.qc['coordinate_mode'] == 'grid_3d'
    assert design.qc['n_low_fold_cells'] == 2
    diagnostics = {item.node_id: item for item in design.node_diagnostics}
    assert diagnostics[10].status == 'ok'
    assert diagnostics[20].status == 'ok'
    assert diagnostics[30].status == 'ok'


def _input_model_for_cell_assignment() -> RefractionStaticInputModel:
    n_traces = 5
    endpoint_table = RefractionEndpointTable(
        node_id=np.asarray([10, 20, 30], dtype=np.int64),
        endpoint_id=np.asarray([100, 200, 300], dtype=np.int64),
        x_m=np.asarray([0.0, 100.0, 200.0], dtype=np.float64),
        y_m=np.zeros(3, dtype=np.float64),
        elevation_m=np.asarray([10.0, 11.0, 12.0], dtype=np.float64),
        kind=np.asarray(['source', 'receiver', 'linked']),
        pick_count=np.asarray([3, 4, 3], dtype=np.int64),
    )
    return RefractionStaticInputModel(
        file_id='unit',
        n_traces=n_traces,
        sorted_trace_index=np.arange(n_traces, dtype=np.int64),
        pick_time_s_sorted=np.asarray([0.20, 0.21, 0.30, 0.40, 0.50]),
        valid_pick_mask_sorted=np.ones(n_traces, dtype=bool),
        valid_observation_mask_sorted=np.ones(n_traces, dtype=bool),
        source_id_sorted=np.asarray([1, 2, 1, 3, 1], dtype=np.int64),
        receiver_id_sorted=np.asarray([2, 3, 3, 2, 3], dtype=np.int64),
        source_x_m_sorted=np.asarray([0.0, 0.0, 100.0, 200.0, 300.0]),
        source_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        receiver_x_m_sorted=np.asarray([20.0, 40.0, 140.0, 240.0, 400.0]),
        receiver_y_m_sorted=np.zeros(n_traces, dtype=np.float64),
        source_elevation_m_sorted=np.full(n_traces, 10.0, dtype=np.float64),
        receiver_elevation_m_sorted=np.full(n_traces, 12.0, dtype=np.float64),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.asarray([500.0, 510.0, 700.0, 800.0, 900.0]),
        offset_m_sorted=None,
        distance_m_sorted=np.asarray([500.0, 510.0, 700.0, 800.0, 900.0]),
        source_endpoint_key_sorted=np.asarray(
            ['source:10', 'source:20', 'source:10', 'source:30', 'source:10']
        ),
        receiver_endpoint_key_sorted=np.asarray(
            ['receiver:20', 'receiver:30', 'receiver:30', 'receiver:20', 'receiver:30']
        ),
        source_node_id_sorted=np.asarray([10, 20, 10, 30, 10], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20, 30, 30, 20, 30], dtype=np.int64),
        node_x_m=endpoint_table.x_m,
        node_y_m=endpoint_table.y_m,
        node_elevation_m=endpoint_table.elevation_m,
        node_kind=endpoint_table.kind,
        rejection_reason_sorted=np.asarray(['ok', 'ok', 'ok', 'ok', 'ok']),
        qc={},
        endpoint_table=endpoint_table,
        metadata={},
    )
