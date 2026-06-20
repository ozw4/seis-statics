from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from seis_statics.refraction import (
    LOW_FOLD_NODE_REJECTION_REASON,
    LOW_FOLD_NODE_STATUS,
    RefractionStaticDesignMatrixError,
    build_refraction_design_matrix_node_diagnostics,
    build_refraction_static_design_matrix_from_arrays,
    summarize_refraction_static_design_matrix,
)


def test_refraction_design_matrix_solve_global_layout_matches_gli_equation() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20, 0.25, 0.30]),
        valid_observation_mask_sorted=np.asarray([True, True, False]),
        source_node_id_sorted=np.asarray([10, 10, 30]),
        receiver_node_id_sorted=np.asarray([20, 20, 20]),
        distance_m_sorted=np.asarray([500.0, 600.0, 700.0]),
        node_id=np.asarray([10, 20, 30]),
        bedrock_velocity_mode='solve_global',
    )

    assert sparse.isspmatrix_csr(design.matrix)
    assert design.matrix.dtype == np.float64
    np.testing.assert_allclose(
        design.matrix.toarray(),
        np.asarray(
            [
                [1.0, 1.0, 500.0],
                [1.0, 1.0, 600.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(design.rhs_s, [0.20, 0.25])
    assert design.node_id_to_col == {10: 0, 20: 1}
    np.testing.assert_array_equal(design.active_node_id, [10, 20])
    np.testing.assert_array_equal(design.inactive_node_id, [30])
    assert design.bedrock_slowness_col == 2
    assert design.qc['slowness_column_present'] is True


def test_refraction_design_matrix_row_map_uses_sorted_trace_index_values() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20, 0.25, 0.30, 0.35]),
        valid_observation_mask_sorted=np.asarray([True, False, True, False]),
        source_node_id_sorted=np.asarray([10, 10, 30, 10]),
        receiver_node_id_sorted=np.asarray([20, 20, 20, 30]),
        distance_m_sorted=np.asarray([500.0, 600.0, 700.0, 800.0]),
        node_id=np.asarray([10, 20, 30]),
        sorted_trace_index=np.asarray([41, 17, 99, 23]),
        bedrock_velocity_mode='solve_global',
    )

    np.testing.assert_array_equal(design.row_trace_index_sorted, [41, 99])
    np.testing.assert_allclose(design.observed_pick_time_s, [0.20, 0.30])
    np.testing.assert_allclose(design.row_distance_m, [500.0, 700.0])


def test_refraction_design_matrix_fixed_global_moves_distance_term_to_rhs() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20, 0.25]),
        valid_observation_mask_sorted=np.asarray([True, True]),
        source_node_id_sorted=np.asarray([10, 10]),
        receiver_node_id_sorted=np.asarray([20, 20]),
        distance_m_sorted=np.asarray([500.0, 600.0]),
        node_id=np.asarray([10, 20]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
    )

    np.testing.assert_allclose(
        design.matrix.toarray(),
        np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(design.rhs_s, [0.0, 0.01])
    assert design.bedrock_slowness_col is None
    assert design.fixed_bedrock_slowness_s_per_m == pytest.approx(1.0 / 2500.0)
    assert design.qc['fixed_bedrock_velocity_m_s'] == 2500.0


def test_refraction_design_matrix_same_source_receiver_node_sums_coefficients() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20]),
        valid_observation_mask_sorted=np.asarray([True]),
        source_node_id_sorted=np.asarray([10]),
        receiver_node_id_sorted=np.asarray([10]),
        distance_m_sorted=np.asarray([500.0]),
        node_id=np.asarray([10]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
    )

    np.testing.assert_allclose(design.matrix.toarray(), [[2.0]])
    assert design.matrix.nnz == 1


def test_refraction_design_matrix_node_diagnostics_count_filtered_rows() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20, 0.25, 0.30]),
        valid_observation_mask_sorted=np.asarray([True, True, False]),
        source_node_id_sorted=np.asarray([10, 10, 30]),
        receiver_node_id_sorted=np.asarray([20, 20, 20]),
        source_endpoint_key_sorted=np.asarray(
            ['source:1001', 'source:1001', 'source:3001']
        ),
        receiver_endpoint_key_sorted=np.asarray(
            ['receiver:2001', 'receiver:2001', 'receiver:2001']
        ),
        distance_m_sorted=np.asarray([500.0, 600.0, 700.0]),
        node_id=np.asarray([10, 20, 30]),
        node_kind=np.asarray(['source', 'receiver', 'source']),
        sorted_trace_index=np.asarray([101, 102, 103]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        rejection_reason_sorted=np.asarray(['ok', 'ok', 'offset_gate']),
        include_diagnostics=True,
    )

    diagnostics = {item.node_id: item for item in design.node_diagnostics}
    assert diagnostics[10].status == 'ok'
    assert diagnostics[10].first_trace_indices_pre_filter == (101, 102)
    assert diagnostics[30].active is False
    assert diagnostics[30].reason == 'all_observations_filtered_by_offset_gate'
    assert design.design_matrix_qc['node_status_counts'] == {'ok': 2, 'inactive': 1}
    assert summarize_refraction_static_design_matrix(design)['n_active_nodes'] == 2


def test_refraction_design_matrix_filters_low_fold_nodes() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20, 0.25, 0.35]),
        valid_observation_mask_sorted=np.asarray([True, True, True]),
        source_node_id_sorted=np.asarray([10, 10, 30]),
        receiver_node_id_sorted=np.asarray([20, 20, 20]),
        distance_m_sorted=np.asarray([500.0, 600.0, 700.0]),
        node_id=np.asarray([10, 20, 30]),
        sorted_trace_index=np.asarray([101, 102, 103]),
        bedrock_velocity_mode='solve_global',
        min_observations_per_node=2,
        rejection_reason_sorted=np.asarray(['ok', 'ok', 'ok']),
        include_diagnostics=True,
    )

    np.testing.assert_array_equal(design.row_trace_index_sorted, [101, 102])
    np.testing.assert_array_equal(design.active_node_id, [10, 20])
    np.testing.assert_array_equal(design.inactive_node_id, [30])
    np.testing.assert_array_equal(design.low_fold_node_id, [30])
    np.testing.assert_array_equal(design.node_observation_count, [2, 2, 0])
    np.testing.assert_array_equal(
        design.rejection_reason_sorted,
        ['ok', 'ok', LOW_FOLD_NODE_REJECTION_REASON],
    )
    assert design.n_observations_rejected_by_low_fold_node == 1
    assert design.qc['min_observations_per_node'] == 2
    assert design.qc['n_low_fold_nodes'] == 1
    assert design.qc['low_fold_node_id'] == [30]
    assert design.qc['n_observations_rejected_by_low_fold_node'] == 1
    diagnostics = {item.node_id: item for item in design.node_diagnostics}
    assert diagnostics[30].status == LOW_FOLD_NODE_STATUS
    assert diagnostics[30].reason == LOW_FOLD_NODE_REJECTION_REASON
    assert design.design_matrix_qc['node_status_counts'] == {
        'ok': 2,
        LOW_FOLD_NODE_STATUS: 1,
    }


def test_refraction_design_matrix_diagnostics_are_lazy_by_default() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.20]),
        valid_observation_mask_sorted=np.asarray([True]),
        source_node_id_sorted=np.asarray([17]),
        receiver_node_id_sorted=np.asarray([21]),
        distance_m_sorted=np.asarray([500.0]),
        node_id=np.asarray([17, 21]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
    )

    assert design.node_diagnostics == ()
    assert design.design_matrix_qc['node_status_counts'] == {}
    assert [item.node_id for item in build_refraction_design_matrix_node_diagnostics(design)] == [
        17,
        21,
    ]


def test_refraction_design_matrix_rejects_invalid_selected_values() -> None:
    with pytest.raises(RefractionStaticDesignMatrixError, match='unknown node ID 99'):
        build_refraction_static_design_matrix_from_arrays(
            pick_time_s_sorted=np.asarray([0.20]),
            valid_observation_mask_sorted=np.asarray([True]),
            source_node_id_sorted=np.asarray([99]),
            receiver_node_id_sorted=np.asarray([21]),
            distance_m_sorted=np.asarray([500.0]),
            node_id=np.asarray([17, 21]),
            bedrock_velocity_mode='fixed_global',
            fixed_bedrock_velocity_m_s=2500.0,
        )


def test_refraction_design_matrix_cell_inputs_are_rejected_outside_cell_mode() -> None:
    with pytest.raises(RefractionStaticDesignMatrixError, match='only allowed'):
        build_refraction_static_design_matrix_from_arrays(
            pick_time_s_sorted=np.asarray([0.20]),
            valid_observation_mask_sorted=np.asarray([True]),
            source_node_id_sorted=np.asarray([17]),
            receiver_node_id_sorted=np.asarray([21]),
            distance_m_sorted=np.asarray([500.0]),
            node_id=np.asarray([17, 21]),
            bedrock_velocity_mode='fixed_global',
            fixed_bedrock_velocity_m_s=2500.0,
            midpoint_cell_id_sorted=np.asarray([0]),
        )
