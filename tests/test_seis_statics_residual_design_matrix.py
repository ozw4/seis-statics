from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import seis_statics.residual as residual
from seis_statics.residual import ResidualStaticSolverInputs

N_TRACES = 4
PICK_TIME_AFTER_DATUM = np.asarray([0.100, 0.110, 0.120, 0.130])
VALID_PICK_MASK = np.asarray([True, True, True, True])
SOURCE_INDEX = np.asarray([0, 0, 1, 1], dtype=np.int64)
RECEIVER_INDEX = np.asarray([0, 1, 0, 1], dtype=np.int64)
ABS_OFFSET = np.asarray([10.0, 20.0, 30.0, 40.0])


def _inputs(**overrides: Any) -> ResidualStaticSolverInputs:
    payload: dict[str, Any] = {
        'picks_time_s_sorted': PICK_TIME_AFTER_DATUM.copy(),
        'valid_pick_mask_sorted': VALID_PICK_MASK.copy(),
        'pick_time_after_datum_s_sorted': PICK_TIME_AFTER_DATUM.copy(),
        'datum_trace_shift_s_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'source_id_sorted': np.asarray([10, 10, 20, 20], dtype=np.int64),
        'receiver_id_sorted': np.asarray([1, 2, 1, 2], dtype=np.int64),
        'source_unique_ids': np.asarray([10, 20], dtype=np.int64),
        'receiver_unique_ids': np.asarray([1, 2], dtype=np.int64),
        'source_index_sorted': SOURCE_INDEX.copy(),
        'receiver_index_sorted': RECEIVER_INDEX.copy(),
        'source_valid_pick_counts': np.asarray([2, 2], dtype=np.int64),
        'receiver_valid_pick_counts': np.asarray([2, 2], dtype=np.int64),
        'offset_sorted': np.asarray([-10.0, 20.0, -30.0, 40.0]),
        'abs_offset_sorted': ABS_OFFSET.copy(),
        'key1_sorted': np.asarray([100, 100, 200, 200], dtype=np.int64),
        'key2_sorted': np.asarray([1, 2, 1, 2], dtype=np.int64),
        'source_elevation_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'receiver_elevation_m_sorted': np.zeros(N_TRACES, dtype=np.float64),
        'dt': 0.004,
        'n_traces': N_TRACES,
        'n_samples': 64,
        'key1_byte': 189,
        'key2_byte': 193,
        'source_id_byte': 17,
        'receiver_id_byte': 13,
        'offset_byte': 37,
        'moveout_model': 'linear_abs_offset',
        'input_file_id': 'input-file',
        'datum_source_file_id': 'datum-source-file',
        'datum_job_id': 'datum-job',
        'pick_source_kind': 'batch_npz',
        'metadata': {},
    }
    payload.update(overrides)
    return ResidualStaticSolverInputs(**payload)


def _moveout_none_inputs(**overrides: Any) -> ResidualStaticSolverInputs:
    payload: dict[str, Any] = {
        'moveout_model': 'none',
        'offset_byte': None,
        'offset_sorted': None,
        'abs_offset_sorted': None,
    }
    payload.update(overrides)
    return _inputs(**payload)


def test_moveout_none_column_count_and_row_triplets() -> None:
    triplets = residual.build_residual_static_observation_matrix_triplets(
        _moveout_none_inputs()
    )

    assert triplets.n_cols == 5
    assert triplets.layout.slowness_col is None
    np.testing.assert_array_equal(
        triplets.row_indices,
        np.repeat(np.arange(N_TRACES, dtype=np.int64), 3),
    )
    np.testing.assert_array_equal(
        triplets.col_indices,
        [0, 1, 3, 0, 1, 4, 0, 2, 3, 0, 2, 4],
    )
    np.testing.assert_allclose(triplets.data, np.ones(12, dtype=np.float64))
    np.testing.assert_allclose(triplets.rhs_s, PICK_TIME_AFTER_DATUM)


def test_linear_abs_offset_column_count_and_slowness_column() -> None:
    triplets = residual.build_residual_static_observation_matrix_triplets(_inputs())

    assert triplets.n_cols == 6
    assert triplets.layout.slowness_col == 1
    np.testing.assert_array_equal(
        triplets.col_indices,
        [0, 1, 2, 4, 0, 1, 2, 5, 0, 1, 3, 4, 0, 1, 3, 5],
    )
    np.testing.assert_allclose(triplets.data[1::4], ABS_OFFSET)
    np.testing.assert_allclose(triplets.data[0::4], np.ones(N_TRACES))
    np.testing.assert_allclose(triplets.data[2::4], np.ones(N_TRACES))
    np.testing.assert_allclose(triplets.data[3::4], np.ones(N_TRACES))


def test_invalid_source_and_receiver_indices_raise_errors() -> None:
    with pytest.raises(ValueError, match='source_index_sorted'):
        residual.build_residual_static_column_layout(
            _inputs(source_index_sorted=np.asarray([0, 0, 1, 2], dtype=np.int64))
        )
    with pytest.raises(ValueError, match='receiver_index_sorted'):
        residual.build_residual_static_column_layout(
            _inputs(receiver_index_sorted=np.asarray([0, 1, 0, 2], dtype=np.int64))
        )


def test_used_mask_preserves_row_to_trace_mapping() -> None:
    used_mask = np.asarray([False, True, False, True])

    triplets = residual.build_residual_static_observation_matrix_triplets(
        _inputs(),
        used_mask_sorted=used_mask,
    )

    np.testing.assert_array_equal(triplets.row_to_sorted_trace_index, [1, 3])
    np.testing.assert_array_equal(triplets.used_mask_sorted, used_mask)
    np.testing.assert_allclose(triplets.rhs_s, [0.110, 0.130])
    np.testing.assert_array_equal(triplets.row_indices, [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(triplets.col_indices, [0, 1, 2, 5, 0, 1, 3, 5])
    np.testing.assert_allclose(triplets.data, [1.0, 20.0, 1.0, 1.0, 1.0, 40.0, 1.0, 1.0])
