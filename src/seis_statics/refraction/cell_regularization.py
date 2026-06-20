"""Neighbor-cell regularization helpers for refraction-static V2 solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from seis_statics.validation import (
    coerce_nonnegative_finite_float,
    coerce_nonnegative_int,
    coerce_positive_finite_float,
    coerce_positive_int,
)


@dataclass(frozen=True)
class CellSlownessSmoothingRows:
    """Sparse smoothing rows for active refractor-cell slowness parameters."""

    matrix: sparse.csr_matrix
    rhs_s: np.ndarray
    edge_cell_id: np.ndarray
    active_cell_neighbor_count: np.ndarray
    reference_distance_m: float | None
    row_scale: float
    qc: dict[str, Any]

    @property
    def n_edges(self) -> int:
        return int(self.edge_cell_id.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.matrix.shape[0])


def build_cell_slowness_smoothing_rows(
    *,
    active_cell_id: np.ndarray,
    velocity_smoothing_weight: float,
    smoothing_reference_distance_m: float | None = None,
    row_distance_m: np.ndarray | None = None,
    n_total_cells: int | None = None,
    number_of_cell_x: int | None = None,
    number_of_cell_y: int | None = None,
    cell_id_to_col: dict[int, int] | None = None,
    bedrock_slowness_cell_col_start: int | None = None,
    n_parameters: int | None = None,
) -> CellSlownessSmoothingRows:
    """Build one slowness-difference row for each active 4-neighbor edge."""
    active_cells = _coerce_unique_active_cell_id(active_cell_id)
    weight = coerce_nonnegative_finite_float(
        velocity_smoothing_weight,
        name='velocity_smoothing_weight',
    )
    total_cells, n_x, n_y = _validate_grid_shape(
        n_total_cells=n_total_cells,
        number_of_cell_x=number_of_cell_x,
        number_of_cell_y=number_of_cell_y,
        active_cell_id=active_cells,
    )
    _validate_active_cell_bounds(active_cells, n_total_cells=total_cells)
    col_by_cell = _build_cell_column_mapping(
        active_cells,
        cell_id_to_col=cell_id_to_col,
        bedrock_slowness_cell_col_start=bedrock_slowness_cell_col_start,
    )
    n_cols = _resolve_n_parameters(n_parameters, col_by_cell=col_by_cell)

    if weight == 0.0:
        return _empty_smoothing_rows(
            n_parameters=n_cols,
            velocity_smoothing_weight=0.0,
            smoothing_reference_distance_m=smoothing_reference_distance_m,
            row_scale=0.0,
        )

    reference_distance = _resolve_reference_distance(
        smoothing_reference_distance_m=smoothing_reference_distance_m,
        row_distance_m=row_distance_m,
    )
    row_scale = float(weight * reference_distance)
    if row_scale == 0.0:
        return _empty_smoothing_rows(
            n_parameters=n_cols,
            velocity_smoothing_weight=weight,
            smoothing_reference_distance_m=reference_distance,
            row_scale=0.0,
        )

    edges = _build_active_neighbor_edges(
        active_cells,
        number_of_cell_x=n_x,
        number_of_cell_y=n_y,
    )
    neighbor_count = _neighbor_count_by_active_cell(active_cells, edges)
    if edges.shape[0] == 0:
        return _empty_smoothing_rows(
            n_parameters=n_cols,
            velocity_smoothing_weight=weight,
            smoothing_reference_distance_m=reference_distance,
            row_scale=row_scale,
            active_cell_neighbor_count=neighbor_count,
        )

    row_index = np.repeat(np.arange(edges.shape[0], dtype=np.int64), 2)
    col_index = np.empty(edges.shape[0] * 2, dtype=np.int64)
    data = np.empty(edges.shape[0] * 2, dtype=np.float64)
    col_index[0::2] = [col_by_cell[int(cell_id)] for cell_id in edges[:, 0]]
    col_index[1::2] = [col_by_cell[int(cell_id)] for cell_id in edges[:, 1]]
    data[0::2] = row_scale
    data[1::2] = -row_scale
    matrix = sparse.coo_matrix(
        (data, (row_index, col_index)),
        shape=(int(edges.shape[0]), n_cols),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    rhs_s = np.zeros(edges.shape[0], dtype=np.float64)
    qc = _build_smoothing_qc(
        velocity_smoothing_weight=weight,
        smoothing_reference_distance_m=reference_distance,
        n_edges=int(edges.shape[0]),
        row_scale=row_scale,
        active_cell_neighbor_count=neighbor_count,
    )
    return CellSlownessSmoothingRows(
        matrix=matrix,
        rhs_s=np.ascontiguousarray(rhs_s, dtype=np.float64),
        edge_cell_id=np.ascontiguousarray(edges, dtype=np.int64),
        active_cell_neighbor_count=np.ascontiguousarray(neighbor_count, dtype=np.int64),
        reference_distance_m=reference_distance,
        row_scale=row_scale,
        qc=qc,
    )


def augment_design_matrix_with_cell_smoothing(
    matrix: sparse.csr_matrix,
    rhs_s: np.ndarray,
    smoothing_rows: CellSlownessSmoothingRows,
) -> tuple[sparse.csr_matrix, np.ndarray, int]:
    """Append cell smoothing rows to an existing sparse least-squares system."""
    if smoothing_rows.n_rows == 0:
        return matrix, rhs_s, 0
    if not sparse.isspmatrix_csr(matrix):
        raise ValueError('matrix must be CSR')
    rhs = np.asarray(rhs_s)
    if rhs.shape != (matrix.shape[0],):
        raise ValueError('rhs_s shape mismatch')
    if smoothing_rows.matrix.shape[1] != matrix.shape[1]:
        raise ValueError('smoothing row column count must match matrix')
    matrix_aug = sparse.vstack((matrix, smoothing_rows.matrix), format='csr')
    rhs_aug = np.concatenate((rhs, smoothing_rows.rhs_s))
    return (
        matrix_aug,
        np.ascontiguousarray(rhs_aug, dtype=np.float64),
        smoothing_rows.n_rows,
    )


def _build_active_neighbor_edges(
    active_cell_id: np.ndarray,
    *,
    number_of_cell_x: int,
    number_of_cell_y: int,
) -> np.ndarray:
    active = {int(cell_id) for cell_id in active_cell_id.tolist()}
    edges: list[tuple[int, int]] = []
    for cell_id in sorted(active):
        ix = cell_id % number_of_cell_x
        iy = cell_id // number_of_cell_x
        if ix + 1 < number_of_cell_x:
            right = cell_id + 1
            if right in active:
                edges.append((cell_id, right))
        if iy + 1 < number_of_cell_y:
            down = cell_id + number_of_cell_x
            if down in active:
                edges.append((cell_id, down))
    return np.asarray(edges, dtype=np.int64).reshape((-1, 2))


def _neighbor_count_by_active_cell(
    active_cell_id: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    counts_by_cell = {int(cell_id): 0 for cell_id in active_cell_id.tolist()}
    for left, right in edges.tolist():
        counts_by_cell[int(left)] += 1
        counts_by_cell[int(right)] += 1
    return np.asarray(
        [counts_by_cell[int(cell_id)] for cell_id in active_cell_id.tolist()],
        dtype=np.int64,
    )


def _empty_smoothing_rows(
    *,
    n_parameters: int,
    velocity_smoothing_weight: float,
    smoothing_reference_distance_m: float | None,
    row_scale: float,
    active_cell_neighbor_count: np.ndarray | None = None,
) -> CellSlownessSmoothingRows:
    if active_cell_neighbor_count is None:
        active_cell_neighbor_count = np.empty(0, dtype=np.int64)
    matrix = sparse.csr_matrix((0, n_parameters), dtype=np.float64)
    qc = _build_smoothing_qc(
        velocity_smoothing_weight=velocity_smoothing_weight,
        smoothing_reference_distance_m=smoothing_reference_distance_m,
        n_edges=0,
        row_scale=row_scale,
        active_cell_neighbor_count=active_cell_neighbor_count,
    )
    return CellSlownessSmoothingRows(
        matrix=matrix,
        rhs_s=np.empty(0, dtype=np.float64),
        edge_cell_id=np.empty((0, 2), dtype=np.int64),
        active_cell_neighbor_count=np.ascontiguousarray(
            active_cell_neighbor_count,
            dtype=np.int64,
        ),
        reference_distance_m=smoothing_reference_distance_m,
        row_scale=float(row_scale),
        qc=qc,
    )


def _build_smoothing_qc(
    *,
    velocity_smoothing_weight: float,
    smoothing_reference_distance_m: float | None,
    n_edges: int,
    row_scale: float,
    active_cell_neighbor_count: np.ndarray,
) -> dict[str, Any]:
    if active_cell_neighbor_count.size:
        neighbor_min: int | None = int(np.min(active_cell_neighbor_count))
        neighbor_median: float | None = float(np.median(active_cell_neighbor_count))
        neighbor_max: int | None = int(np.max(active_cell_neighbor_count))
    else:
        neighbor_min = None
        neighbor_median = None
        neighbor_max = None
    return {
        'velocity_smoothing_weight': float(velocity_smoothing_weight),
        'smoothing_reference_distance_m': _json_optional_float(
            smoothing_reference_distance_m
        ),
        'n_cell_smoothing_edges': int(n_edges),
        'n_cell_smoothing_rows': int(n_edges),
        'smoothing_row_scale': float(row_scale),
        'active_cell_neighbor_count_min': neighbor_min,
        'active_cell_neighbor_count_median': neighbor_median,
        'active_cell_neighbor_count_max': neighbor_max,
    }


def _resolve_reference_distance(
    *,
    smoothing_reference_distance_m: float | None,
    row_distance_m: np.ndarray | None,
) -> float:
    if smoothing_reference_distance_m is not None:
        return coerce_positive_finite_float(
            smoothing_reference_distance_m,
            name='smoothing_reference_distance_m',
        )
    if row_distance_m is None:
        raise ValueError(
            'row_distance_m is required when smoothing_reference_distance_m is omitted'
        )
    distance = _coerce_1d_float(row_distance_m, name='row_distance_m')
    if distance.size == 0:
        raise ValueError('row_distance_m must contain at least one value')
    if np.any(distance <= 0.0):
        raise ValueError('row_distance_m values must be positive')
    return float(np.median(distance))


def _build_cell_column_mapping(
    active_cell_id: np.ndarray,
    *,
    cell_id_to_col: dict[int, int] | None,
    bedrock_slowness_cell_col_start: int | None,
) -> dict[int, int]:
    if cell_id_to_col is not None and bedrock_slowness_cell_col_start is not None:
        raise ValueError(
            'cell_id_to_col and bedrock_slowness_cell_col_start are mutually exclusive'
        )
    if cell_id_to_col is not None:
        out = {int(cell_id): int(col) for cell_id, col in cell_id_to_col.items()}
        missing = [int(cell_id) for cell_id in active_cell_id if int(cell_id) not in out]
        if missing:
            raise ValueError('cell_id_to_col is missing an active cell ID')
        return out
    if bedrock_slowness_cell_col_start is None:
        start = 0
    else:
        start = coerce_nonnegative_int(
            bedrock_slowness_cell_col_start,
            name='bedrock_slowness_cell_col_start',
        )
    return {
        int(cell_id): start + index
        for index, cell_id in enumerate(active_cell_id.tolist())
    }


def _resolve_n_parameters(
    value: int | None,
    *,
    col_by_cell: dict[int, int],
) -> int:
    min_value = max(col_by_cell.values(), default=-1) + 1
    if value is None:
        return min_value
    out = coerce_positive_int(value, name='n_parameters')
    if out < min_value:
        raise ValueError('n_parameters is too small for cell columns')
    return out


def _validate_grid_shape(
    *,
    n_total_cells: int | None,
    number_of_cell_x: int | None,
    number_of_cell_y: int | None,
    active_cell_id: np.ndarray,
) -> tuple[int, int, int]:
    if number_of_cell_y is None:
        n_y = 1
    else:
        n_y = coerce_positive_int(number_of_cell_y, name='number_of_cell_y')
    if number_of_cell_x is None:
        if n_total_cells is not None:
            total_hint = coerce_positive_int(n_total_cells, name='n_total_cells')
            if total_hint % n_y != 0:
                raise ValueError('n_total_cells must be divisible by number_of_cell_y')
            n_x = int(total_hint // n_y)
        elif active_cell_id.size:
            total_hint = int(np.max(active_cell_id)) + 1
            if total_hint % n_y != 0:
                raise ValueError(
                    'inferred total cell count must be divisible by number_of_cell_y'
                )
            n_x = int(total_hint // n_y)
        else:
            n_x = 1
    else:
        n_x = coerce_positive_int(number_of_cell_x, name='number_of_cell_x')
    total = n_x * n_y
    if n_total_cells is not None:
        expected = coerce_positive_int(n_total_cells, name='n_total_cells')
        if expected != total:
            raise ValueError('n_total_cells must equal number_of_cell_x * number_of_cell_y')
        total = expected
    return total, n_x, n_y


def _validate_active_cell_bounds(
    active_cell_id: np.ndarray,
    *,
    n_total_cells: int,
) -> None:
    if np.any(active_cell_id < 0) or np.any(active_cell_id >= n_total_cells):
        raise ValueError('active_cell_id contains a cell outside the grid')


def _coerce_unique_active_cell_id(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError('active_cell_id must be a 1D array')
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError('active_cell_id must contain integer values')
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError('active_cell_id must contain integer values')
    out = np.ascontiguousarray(arr, dtype=np.int64)
    if np.unique(out).shape[0] != out.shape[0]:
        raise ValueError('active_cell_id values must be unique')
    return out


def _coerce_1d_float(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must contain numeric values')
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if np.any(~np.isfinite(out)):
        raise ValueError(f'{name} values must be finite')
    return out


def _json_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    'CellSlownessSmoothingRows',
    'augment_design_matrix_with_cell_smoothing',
    'build_cell_slowness_smoothing_rows',
]
