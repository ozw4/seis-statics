"""Design matrix builder for source/receiver node time-term inversion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy import sparse

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_nonnegative_int as _coerce_nonnegative_int,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics.time_term.moveout import TimeTermMoveoutResult
from seis_statics.time_term.types import TimeTermInversionInputs

_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    nonfinite_message='must contain only finite values',
)


@dataclass(frozen=True)
class TimeTermDesignMatrixOptions:
    min_observations: int = 1
    include_only_valid_picks: bool = True
    include_only_valid_moveout: bool = True


@dataclass(frozen=True)
class TimeTermDesignMatrix:
    matrix: sparse.csr_matrix
    data_s: np.ndarray

    n_traces: int
    n_observations: int
    n_nodes: int

    used_trace_mask_sorted: np.ndarray
    row_trace_index_sorted: np.ndarray
    trace_to_row_index_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray

    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    row_pick_time_after_static_s: np.ndarray
    row_moveout_time_s: np.ndarray
    row_data_s: np.ndarray

    source_observation_count_by_node: np.ndarray
    receiver_observation_count_by_node: np.ndarray
    total_observation_count_by_node: np.ndarray


def build_time_term_design_matrix(
    inputs: TimeTermInversionInputs,
    moveout: TimeTermMoveoutResult,
    *,
    options: TimeTermDesignMatrixOptions | None = None,
) -> TimeTermDesignMatrix:
    """Build observation rows for node time-term unknowns in sorted trace order."""
    opts = _validate_options(options)
    n_traces = _coerce_positive_int(inputs.n_traces, name='n_traces')
    n_nodes = _coerce_positive_int(inputs.n_nodes, name='n_nodes')
    _coerce_positive_finite_float(inputs.dt, name='dt')
    expected_shape = (n_traces,)

    valid_pick_mask = _coerce_1d_bool_array(
        inputs.valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=expected_shape,
    )
    valid_moveout_mask = _coerce_1d_bool_array(
        moveout.valid_moveout_mask_sorted,
        name='valid_moveout_mask_sorted',
        expected_shape=expected_shape,
    )
    pick_time_after_static = _coerce_1d_real_numeric_float64(
        inputs.pick_time_after_static_s_sorted,
        name='pick_time_after_static_s_sorted',
        expected_shape=expected_shape,
    )
    moveout_time = _coerce_1d_real_numeric_float64(
        moveout.moveout_time_s_sorted,
        name='moveout_time_s_sorted',
        expected_shape=expected_shape,
    )
    source_node_id = _coerce_1d_integer_int64(
        inputs.source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=expected_shape,
    )
    receiver_node_id = _coerce_1d_integer_int64(
        inputs.receiver_node_id_sorted,
        name='receiver_node_id_sorted',
        expected_shape=expected_shape,
    )
    _validate_node_range(source_node_id, receiver_node_id, n_nodes=n_nodes)

    candidate_mask = _candidate_trace_mask(
        valid_pick_mask=valid_pick_mask,
        valid_moveout_mask=valid_moveout_mask,
        options=opts,
    )
    _validate_finite_at_mask(
        pick_time_after_static,
        mask=candidate_mask,
        name='pick_time_after_static_s_sorted',
        mask_name='candidate traces',
    )
    _validate_finite_nonnegative_at_mask(
        moveout_time,
        mask=candidate_mask,
        name='moveout_time_s_sorted',
        mask_name='candidate traces',
    )
    used_trace_mask = np.ascontiguousarray(
        candidate_mask
        & np.isfinite(pick_time_after_static)
        & np.isfinite(moveout_time)
        & (moveout_time >= 0.0),
        dtype=bool,
    )
    row_trace_index = np.ascontiguousarray(
        np.flatnonzero(used_trace_mask),
        dtype=np.int64,
    )
    n_observations = int(row_trace_index.shape[0])
    if n_observations <= 0:
        raise ValueError('at least one usable time-term observation is required')
    if n_observations < opts.min_observations:
        raise ValueError(
            'not enough usable time-term observations: '
            f'{n_observations} < {opts.min_observations}'
        )

    trace_to_row_index = np.full(n_traces, -1, dtype=np.int64)
    trace_to_row_index[row_trace_index] = np.arange(n_observations, dtype=np.int64)

    row_source_node_id = np.ascontiguousarray(
        source_node_id[row_trace_index],
        dtype=np.int64,
    )
    row_receiver_node_id = np.ascontiguousarray(
        receiver_node_id[row_trace_index],
        dtype=np.int64,
    )
    row_pick_time_after_static = np.ascontiguousarray(
        pick_time_after_static[row_trace_index],
        dtype=np.float64,
    )
    row_moveout_time = np.ascontiguousarray(
        moveout_time[row_trace_index],
        dtype=np.float64,
    )
    row_data_s = np.ascontiguousarray(
        row_pick_time_after_static - row_moveout_time,
        dtype=np.float64,
    )

    matrix = _build_sparse_matrix(
        row_source_node_id,
        row_receiver_node_id,
        n_observations=n_observations,
        n_nodes=n_nodes,
    )
    source_count = np.bincount(row_source_node_id, minlength=n_nodes).astype(
        np.int64,
        copy=False,
    )
    receiver_count = np.bincount(row_receiver_node_id, minlength=n_nodes).astype(
        np.int64,
        copy=False,
    )
    total_count = np.ascontiguousarray(source_count + receiver_count, dtype=np.int64)

    return TimeTermDesignMatrix(
        matrix=matrix,
        data_s=row_data_s,
        n_traces=n_traces,
        n_observations=n_observations,
        n_nodes=n_nodes,
        used_trace_mask_sorted=used_trace_mask,
        row_trace_index_sorted=row_trace_index,
        trace_to_row_index_sorted=trace_to_row_index,
        source_node_id_sorted=np.ascontiguousarray(source_node_id, dtype=np.int64),
        receiver_node_id_sorted=np.ascontiguousarray(receiver_node_id, dtype=np.int64),
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        row_pick_time_after_static_s=row_pick_time_after_static,
        row_moveout_time_s=row_moveout_time,
        row_data_s=row_data_s,
        source_observation_count_by_node=np.ascontiguousarray(
            source_count,
            dtype=np.int64,
        ),
        receiver_observation_count_by_node=np.ascontiguousarray(
            receiver_count,
            dtype=np.int64,
        ),
        total_observation_count_by_node=total_count,
    )


def summarize_time_term_design_matrix(
    design: TimeTermDesignMatrix,
) -> dict[str, object]:
    """Return a JSON-safe summary for future artifacts and job logs."""
    n_traces = _coerce_positive_int(design.n_traces, name='n_traces')
    n_nodes = _coerce_positive_int(design.n_nodes, name='n_nodes')
    n_observations = _coerce_positive_int(
        design.n_observations,
        name='n_observations',
    )
    if design.matrix.shape != (n_observations, n_nodes):
        raise ValueError('matrix shape does not match design dimensions')
    data_s = _coerce_1d_real_numeric_float64(
        design.data_s,
        name='data_s',
        expected_shape=(n_observations,),
    )
    source_count = _coerce_1d_integer_int64(
        design.source_observation_count_by_node,
        name='source_observation_count_by_node',
        expected_shape=(n_nodes,),
    )
    receiver_count = _coerce_1d_integer_int64(
        design.receiver_observation_count_by_node,
        name='receiver_observation_count_by_node',
        expected_shape=(n_nodes,),
    )
    total_count = _coerce_1d_integer_int64(
        design.total_observation_count_by_node,
        name='total_observation_count_by_node',
        expected_shape=(n_nodes,),
    )
    n_nodes_with_source = int(np.count_nonzero(source_count > 0))
    n_nodes_with_receiver = int(np.count_nonzero(receiver_count > 0))
    n_nodes_with_any = int(np.count_nonzero(total_count > 0))

    return {
        'n_traces': n_traces,
        'n_observations': n_observations,
        'n_nodes': n_nodes,
        'observation_fraction': _fraction(n_observations, n_traces),
        'matrix_shape': [int(design.matrix.shape[0]), int(design.matrix.shape[1])],
        'matrix_nnz': int(design.matrix.nnz),
        'source_observation_count_by_node': _stats_payload(source_count),
        'receiver_observation_count_by_node': _stats_payload(receiver_count),
        'total_observation_count_by_node': _stats_payload(total_count),
        'data_s': _stats_payload(data_s),
        'data_ms': _stats_payload(data_s * 1000.0),
        'n_nodes_with_source_observations': n_nodes_with_source,
        'n_nodes_with_receiver_observations': n_nodes_with_receiver,
        'n_nodes_with_any_observations': n_nodes_with_any,
        'n_nodes_without_observations': int(n_nodes - n_nodes_with_any),
    }


def _validate_options(
    options: TimeTermDesignMatrixOptions | None,
) -> TimeTermDesignMatrixOptions:
    opts = TimeTermDesignMatrixOptions() if options is None else options
    min_observations = _coerce_nonnegative_int(
        opts.min_observations,
        name='min_observations',
    )
    include_only_valid_picks = _coerce_bool(
        opts.include_only_valid_picks,
        name='include_only_valid_picks',
    )
    include_only_valid_moveout = _coerce_bool(
        opts.include_only_valid_moveout,
        name='include_only_valid_moveout',
    )
    return TimeTermDesignMatrixOptions(
        min_observations=min_observations,
        include_only_valid_picks=include_only_valid_picks,
        include_only_valid_moveout=include_only_valid_moveout,
    )


def _candidate_trace_mask(
    *,
    valid_pick_mask: np.ndarray,
    valid_moveout_mask: np.ndarray,
    options: TimeTermDesignMatrixOptions,
) -> np.ndarray:
    candidate_mask = np.ones(valid_pick_mask.shape, dtype=bool)
    if options.include_only_valid_picks:
        candidate_mask &= valid_pick_mask
    if options.include_only_valid_moveout:
        candidate_mask &= valid_moveout_mask
    return np.ascontiguousarray(candidate_mask, dtype=bool)


def _build_sparse_matrix(
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
    *,
    n_observations: int,
    n_nodes: int,
) -> sparse.csr_matrix:
    row_indices = np.repeat(np.arange(n_observations, dtype=np.int64), 2)
    col_indices = np.empty(n_observations * 2, dtype=np.int64)
    col_indices[0::2] = row_source_node_id
    col_indices[1::2] = row_receiver_node_id
    data = np.ones(n_observations * 2, dtype=np.float64)
    matrix = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_observations, n_nodes),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()

    expected_nnz = int(
        n_observations + np.count_nonzero(row_source_node_id != row_receiver_node_id)
    )
    if matrix.shape != (n_observations, n_nodes):
        raise ValueError('time-term design matrix shape mismatch')
    if matrix.nnz != expected_nnz:
        raise ValueError('time-term design matrix nnz mismatch')
    return matrix


def _validate_node_range(
    source: np.ndarray,
    receiver: np.ndarray,
    *,
    n_nodes: int,
) -> None:
    if np.any(source < 0) or np.any(receiver < 0):
        raise ValueError('node ids must be non-negative')
    if np.any(source >= n_nodes) or np.any(receiver >= n_nodes):
        raise ValueError('node ids must be less than n_nodes')


def _validate_finite_at_mask(
    values: np.ndarray,
    *,
    mask: np.ndarray,
    name: str,
    mask_name: str,
) -> None:
    if np.any(~np.isfinite(values[mask])):
        raise ValueError(f'{name} must be finite for {mask_name}')


def _validate_finite_nonnegative_at_mask(
    values: np.ndarray,
    *,
    mask: np.ndarray,
    name: str,
    mask_name: str,
) -> None:
    _validate_finite_at_mask(values, mask=mask, name=name, mask_name=mask_name)
    if np.any(values[mask] < 0.0):
        raise ValueError(f'{name} must be non-negative for {mask_name}')


def _coerce_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be bool')
    return bool(value)


def _stats_payload(values: np.ndarray) -> dict[str, float | int | None]:
    arr = _coerce_1d_real_numeric_float64(values, name='summary values')
    finite = np.ascontiguousarray(arr[np.isfinite(arr)], dtype=np.float64)
    count = int(finite.shape[0])
    if count == 0:
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'max_abs': None,
        }
    return {
        'count': count,
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
        'median': float(np.median(finite)),
        'std': float(np.std(finite, ddof=0)),
        'max_abs': float(np.max(np.abs(finite))),
    }


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


__all__ = [
    'TimeTermDesignMatrix',
    'TimeTermDesignMatrixOptions',
    'build_time_term_design_matrix',
    'summarize_time_term_design_matrix',
]
