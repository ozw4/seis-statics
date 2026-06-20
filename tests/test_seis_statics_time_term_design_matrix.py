from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from seis_statics.time_term import (
    TimeTermDesignMatrixOptions,
    TimeTermInversionInputs,
    TimeTermMoveoutResult,
    build_time_term_design_matrix,
    summarize_time_term_design_matrix,
)

N_TRACES = 4
N_SAMPLES = 64
DT = 0.004
N_NODES = 3

SOURCE_NODE_ID = np.asarray([0, 0, 1, 2], dtype=np.int64)
RECEIVER_NODE_ID = np.asarray([1, 2, 2, 2], dtype=np.int64)
PICK_TIME_AFTER_STATIC = np.asarray([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
MOVEOUT_TIME = np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
VALID_PICK_MASK = np.asarray([True, True, False, True])
VALID_MOVEOUT_MASK = np.asarray([True, True, True, True])


def _inputs(**overrides: Any) -> TimeTermInversionInputs:
    payload: dict[str, Any] = {
        'n_traces': N_TRACES,
        'n_samples': N_SAMPLES,
        'dt': DT,
        'key1_byte': 189,
        'key2_byte': 193,
        'pick_time_raw_s_sorted': PICK_TIME_AFTER_STATIC.copy(),
        'valid_pick_mask_sorted': VALID_PICK_MASK.copy(),
        'datum_trace_shift_s_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'residual_applied_shift_s_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'pick_time_after_static_s_sorted': PICK_TIME_AFTER_STATIC.copy(),
        'source_node_id_sorted': SOURCE_NODE_ID.copy(),
        'receiver_node_id_sorted': RECEIVER_NODE_ID.copy(),
        'n_nodes': N_NODES,
        'source_id_sorted': np.asarray([10, 10, 20, 30], dtype=np.int64),
        'receiver_id_sorted': np.asarray([20, 30, 30, 30], dtype=np.int64),
        'offset_sorted': np.asarray([100.0, 200.0, 300.0, 0.0]),
        'source_x_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'source_y_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'receiver_x_m_sorted': np.ones(N_TRACES, dtype=np.float64),
        'receiver_y_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
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


def _moveout(**overrides: Any) -> TimeTermMoveoutResult:
    payload: dict[str, Any] = {
        'model': 'head_wave_linear_offset',
        'refractor_velocity_m_s': 2500.0,
        'distance_source': 'geometry',
        'distance_m_sorted': MOVEOUT_TIME * 2500.0,
        'moveout_time_s_sorted': MOVEOUT_TIME.copy(),
        'valid_moveout_mask_sorted': VALID_MOVEOUT_MASK.copy(),
        'reciprocal_pair_index_sorted': np.full(N_TRACES, -1, dtype=np.int64),
        'has_reciprocal_pair_mask_sorted': np.zeros(N_TRACES, dtype=bool),
        'geometry_distance_m_sorted': MOVEOUT_TIME * 2500.0,
        'offset_abs_m_sorted': None,
        'geometry_offset_mismatch_m_sorted': None,
    }
    payload.update(overrides)
    return TimeTermMoveoutResult(**payload)


def _build(
    inputs: TimeTermInversionInputs | None = None,
    moveout: TimeTermMoveoutResult | None = None,
    *,
    options: TimeTermDesignMatrixOptions | None = None,
):
    return build_time_term_design_matrix(
        _inputs() if inputs is None else inputs,
        _moveout() if moveout is None else moveout,
        options=options,
    )


def test_time_term_design_matrix_builds_csr_matrix_for_simple_nodes() -> None:
    design = _build()

    assert sparse.isspmatrix_csr(design.matrix)
    assert design.matrix.shape == (3, 3)
    np.testing.assert_allclose(
        design.matrix.toarray(),
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
    )


def test_time_term_design_matrix_data_vector_subtracts_moveout() -> None:
    design = _build()

    np.testing.assert_allclose(design.data_s, [0.09, 0.18, 0.36])
    np.testing.assert_allclose(design.row_data_s, [0.09, 0.18, 0.36])
    np.testing.assert_allclose(
        design.row_pick_time_after_static_s - design.row_moveout_time_s,
        design.data_s,
    )


def test_time_term_design_matrix_uses_only_valid_pick_and_moveout_rows() -> None:
    design = _build(
        inputs=_inputs(valid_pick_mask_sorted=np.asarray([True, False, True, True])),
        moveout=_moveout(
            valid_moveout_mask_sorted=np.asarray([True, True, False, True])
        ),
    )

    np.testing.assert_array_equal(design.row_trace_index_sorted, [0, 3])
    np.testing.assert_array_equal(
        design.used_trace_mask_sorted,
        [True, False, False, True],
    )


def test_time_term_design_matrix_preserves_sorted_trace_row_order() -> None:
    design = _build(
        inputs=_inputs(valid_pick_mask_sorted=np.asarray([False, True, True, False])),
        moveout=_moveout(valid_moveout_mask_sorted=np.ones(N_TRACES, dtype=bool)),
    )

    np.testing.assert_array_equal(design.row_trace_index_sorted, [1, 2])


def test_time_term_design_matrix_builds_trace_to_row_index() -> None:
    design = _build()

    np.testing.assert_array_equal(design.trace_to_row_index_sorted, [0, 1, -1, 2])


def test_time_term_design_matrix_row_has_source_and_receiver_coefficients() -> None:
    design = _build()

    np.testing.assert_allclose(design.matrix.toarray()[0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(design.matrix.toarray()[1], [1.0, 0.0, 1.0])


def test_time_term_design_matrix_same_source_receiver_node_has_coefficient_two() -> None:
    design = _build()

    assert design.matrix[2, 2] == pytest.approx(2.0)
    assert design.matrix.nnz == 5


def test_time_term_design_matrix_counts_source_receiver_observations_by_node() -> None:
    design = _build()

    np.testing.assert_array_equal(
        design.source_observation_count_by_node,
        [2, 0, 1],
    )
    np.testing.assert_array_equal(
        design.receiver_observation_count_by_node,
        [0, 1, 2],
    )
    np.testing.assert_array_equal(
        design.total_observation_count_by_node,
        [2, 1, 3],
    )


def test_time_term_design_matrix_counts_same_node_trace_as_two_total_contributions() -> None:
    design = _build()

    assert design.total_observation_count_by_node[2] == 3


def test_time_term_design_matrix_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match='moveout_time_s_sorted shape mismatch'):
        _build(
            moveout=_moveout(
                moveout_time_s_sorted=np.asarray([0.01, 0.02], dtype=np.float64)
            )
        )


def test_time_term_design_matrix_rejects_non_bool_valid_masks() -> None:
    with pytest.raises(ValueError, match='valid_pick_mask_sorted'):
        _build(inputs=_inputs(valid_pick_mask_sorted=np.asarray([1, 1, 0, 1])))

    with pytest.raises(ValueError, match='valid_moveout_mask_sorted'):
        _build(moveout=_moveout(valid_moveout_mask_sorted=np.asarray([1, 1, 1, 1])))


def test_time_term_design_matrix_rejects_node_id_out_of_range() -> None:
    with pytest.raises(ValueError, match='node ids must be less than n_nodes'):
        _build(inputs=_inputs(source_node_id_sorted=np.asarray([0, 0, 3, 2])))


def test_time_term_design_matrix_rejects_negative_moveout_on_used_trace() -> None:
    moveout_time = MOVEOUT_TIME.copy()
    moveout_time[0] = -0.01

    with pytest.raises(ValueError, match='moveout_time_s_sorted'):
        _build(moveout=_moveout(moveout_time_s_sorted=moveout_time))


def test_time_term_design_matrix_rejects_non_finite_moveout_on_used_trace() -> None:
    moveout_time = MOVEOUT_TIME.copy()
    moveout_time[0] = np.inf

    with pytest.raises(ValueError, match='moveout_time_s_sorted'):
        _build(moveout=_moveout(moveout_time_s_sorted=moveout_time))


def test_time_term_design_matrix_rejects_non_finite_pick_on_used_trace() -> None:
    pick_time = PICK_TIME_AFTER_STATIC.copy()
    pick_time[0] = np.nan

    with pytest.raises(ValueError, match='pick_time_after_static_s_sorted'):
        _build(inputs=_inputs(pick_time_after_static_s_sorted=pick_time))


def test_time_term_design_matrix_allows_nan_pick_on_unused_trace() -> None:
    pick_time = PICK_TIME_AFTER_STATIC.copy()
    pick_time[2] = np.nan

    design = _build(inputs=_inputs(pick_time_after_static_s_sorted=pick_time))

    np.testing.assert_array_equal(design.row_trace_index_sorted, [0, 1, 3])


def test_time_term_design_matrix_rejects_no_usable_observations() -> None:
    with pytest.raises(ValueError, match='at least one usable'):
        _build(inputs=_inputs(valid_pick_mask_sorted=np.zeros(N_TRACES, dtype=bool)))


def test_time_term_design_matrix_respects_min_observations() -> None:
    with pytest.raises(ValueError, match='not enough usable'):
        _build(options=TimeTermDesignMatrixOptions(min_observations=4))


def test_time_term_design_matrix_can_include_invalid_pick_rows_when_requested() -> None:
    design = _build(
        options=TimeTermDesignMatrixOptions(include_only_valid_picks=False)
    )

    np.testing.assert_array_equal(design.row_trace_index_sorted, [0, 1, 2, 3])
    assert design.n_observations == 4


def test_time_term_design_matrix_does_not_add_gauge_or_damping_rows() -> None:
    design = _build()

    assert design.matrix.shape[0] == design.n_observations
    assert design.matrix.shape[0] == design.row_trace_index_sorted.shape[0]


def test_summarize_time_term_design_matrix_is_json_safe() -> None:
    summary = summarize_time_term_design_matrix(_build())

    json.dumps(summary, allow_nan=False)
    assert summary['n_traces'] == N_TRACES
    assert summary['n_observations'] == 3
    assert summary['n_nodes'] == N_NODES
    assert summary['observation_fraction'] == pytest.approx(0.75)
    assert summary['matrix_shape'] == [3, 3]
    assert summary['matrix_nnz'] == 5
    assert summary['n_nodes_with_any_observations'] == 3
    assert summary['n_nodes_without_observations'] == 0
    assert summary['data_ms']['count'] == 3
