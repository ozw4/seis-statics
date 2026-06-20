"""Refractor cell grid utilities for refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seis_statics.refraction.options import RefractionStaticRefractorCellOptions


_FINAL_BOUNDARY_ATOL = 1.0e-9


@dataclass(frozen=True)
class RefractionCellGrid:
    cell_id: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    x_min_m: np.ndarray
    x_max_m: np.ndarray
    y_min_m: np.ndarray
    y_max_m: np.ndarray
    x_center_m: np.ndarray
    y_center_m: np.ndarray
    number_of_cell_x: int
    number_of_cell_y: int
    size_of_cell_x_m: float
    size_of_cell_y_m: float
    x_coordinate_origin_m: float
    y_coordinate_origin_m: float


@dataclass(frozen=True)
class RefractionCellAssignment:
    cell_id: np.ndarray
    inside_grid_mask: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    qc: dict[str, Any]


def build_refraction_cell_grid(
    config: RefractionStaticRefractorCellOptions,
) -> RefractionCellGrid:
    """Build a deterministic row-major refractor cell grid."""
    number_of_cell_x = _positive_int(
        getattr(config, 'number_of_cell_x', None),
        name='model.refractor_cell.number_of_cell_x',
    )
    number_of_cell_y = _positive_int(
        getattr(config, 'number_of_cell_y', None),
        name='model.refractor_cell.number_of_cell_y',
    )
    size_of_cell_x_m = _positive_finite_float(
        getattr(config, 'size_of_cell_x_m', None),
        name='model.refractor_cell.size_of_cell_x_m',
    )
    x_coordinate_origin_m = _finite_float(
        getattr(config, 'x_coordinate_origin_m', None),
        name='model.refractor_cell.x_coordinate_origin_m',
    )
    y_coordinate_origin_m = _finite_float(
        getattr(config, 'y_coordinate_origin_m', 0.0),
        name='model.refractor_cell.y_coordinate_origin_m',
    )

    raw_size_y = getattr(config, 'size_of_cell_y_m', None)
    if raw_size_y is None:
        if number_of_cell_y > 1:
            raise ValueError(
                'model.refractor_cell.size_of_cell_y_m is required when '
                'model.refractor_cell.number_of_cell_y > 1'
            )
        size_of_cell_y_m = float('inf')
        y_min_by_index = np.asarray([-np.inf], dtype=np.float64)
        y_max_by_index = np.asarray([np.inf], dtype=np.float64)
        y_center_by_index = np.asarray([y_coordinate_origin_m], dtype=np.float64)
    else:
        size_of_cell_y_m = _positive_finite_float(
            raw_size_y,
            name='model.refractor_cell.size_of_cell_y_m',
        )
        y_index = np.arange(number_of_cell_y, dtype=np.float64)
        y_min_by_index = y_coordinate_origin_m + y_index * size_of_cell_y_m
        y_max_by_index = y_min_by_index + size_of_cell_y_m
        y_center_by_index = y_min_by_index + 0.5 * size_of_cell_y_m

    ix_grid, iy_grid = np.meshgrid(
        np.arange(number_of_cell_x, dtype=np.int64),
        np.arange(number_of_cell_y, dtype=np.int64),
        indexing='xy',
    )
    ix = ix_grid.ravel()
    iy = iy_grid.ravel()
    cell_id = iy * number_of_cell_x + ix

    x_min_m = x_coordinate_origin_m + ix.astype(np.float64) * size_of_cell_x_m
    x_max_m = x_min_m + size_of_cell_x_m
    x_center_m = x_min_m + 0.5 * size_of_cell_x_m
    y_min_m = y_min_by_index[iy]
    y_max_m = y_max_by_index[iy]
    y_center_m = y_center_by_index[iy]

    return RefractionCellGrid(
        cell_id=cell_id.astype(np.int64, copy=False),
        ix=ix.astype(np.int64, copy=False),
        iy=iy.astype(np.int64, copy=False),
        x_min_m=x_min_m.astype(np.float64, copy=False),
        x_max_m=x_max_m.astype(np.float64, copy=False),
        y_min_m=y_min_m.astype(np.float64, copy=False),
        y_max_m=y_max_m.astype(np.float64, copy=False),
        x_center_m=x_center_m.astype(np.float64, copy=False),
        y_center_m=y_center_m.astype(np.float64, copy=False),
        number_of_cell_x=number_of_cell_x,
        number_of_cell_y=number_of_cell_y,
        size_of_cell_x_m=size_of_cell_x_m,
        size_of_cell_y_m=size_of_cell_y_m,
        x_coordinate_origin_m=x_coordinate_origin_m,
        y_coordinate_origin_m=y_coordinate_origin_m,
    )


def assign_points_to_refraction_cells(
    grid: RefractionCellGrid,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> RefractionCellAssignment:
    """Assign endpoint coordinates to cells without clipping outside points."""
    x = _as_1d_float_array(x_m, name='x_m')
    y = _as_1d_float_array(y_m, name='y_m')
    if x.shape != y.shape:
        raise ValueError('x_m and y_m must have the same shape')

    ix, inside_x = _axis_cell_indices(
        x,
        origin_m=grid.x_coordinate_origin_m,
        size_m=grid.size_of_cell_x_m,
        count=grid.number_of_cell_x,
    )
    iy, inside_y = _axis_cell_indices(
        y,
        origin_m=grid.y_coordinate_origin_m,
        size_m=grid.size_of_cell_y_m,
        count=grid.number_of_cell_y,
    )

    inside_grid_mask = inside_x & inside_y
    cell_id = np.full(x.shape, -1, dtype=np.int64)
    cell_id[inside_grid_mask] = (
        iy[inside_grid_mask] * grid.number_of_cell_x + ix[inside_grid_mask]
    )

    qc = _assignment_qc(
        grid=grid,
        cell_id=cell_id,
        inside_grid_mask=inside_grid_mask,
        x_m=x,
        y_m=y,
    )
    return RefractionCellAssignment(
        cell_id=cell_id,
        inside_grid_mask=inside_grid_mask,
        ix=ix,
        iy=iy,
        x_m=x,
        y_m=y,
        qc=qc,
    )


def compute_source_receiver_midpoints(
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-receiver midpoint coordinates."""
    source_x = _as_1d_float_array(source_x_m, name='source_x_m')
    source_y = _as_1d_float_array(source_y_m, name='source_y_m')
    receiver_x = _as_1d_float_array(receiver_x_m, name='receiver_x_m')
    receiver_y = _as_1d_float_array(receiver_y_m, name='receiver_y_m')
    expected_shape = source_x.shape
    for name, values in (
        ('source_y_m', source_y),
        ('receiver_x_m', receiver_x),
        ('receiver_y_m', receiver_y),
    ):
        if values.shape != expected_shape:
            raise ValueError(f'{name} must have the same shape as source_x_m')

    return (
        0.5 * (source_x + receiver_x),
        0.5 * (source_y + receiver_y),
    )


def assign_observation_midpoint_cells(
    grid: RefractionCellGrid,
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
) -> RefractionCellAssignment:
    """Assign first-break observations by source-receiver midpoint."""
    midpoint_x_m, midpoint_y_m = compute_source_receiver_midpoints(
        source_x_m=source_x_m,
        source_y_m=source_y_m,
        receiver_x_m=receiver_x_m,
        receiver_y_m=receiver_y_m,
    )
    return assign_points_to_refraction_cells(
        grid,
        x_m=midpoint_x_m,
        y_m=midpoint_y_m,
    )


def _assignment_qc(
    *,
    grid: RefractionCellGrid,
    cell_id: np.ndarray,
    inside_grid_mask: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> dict[str, Any]:
    n_points = int(cell_id.size)
    n_inside_grid = int(np.count_nonzero(inside_grid_mask))
    n_outside_grid = n_points - n_inside_grid
    inside_cell_id = cell_id[inside_grid_mask]
    total_cell_count = grid.number_of_cell_x * grid.number_of_cell_y
    if inside_cell_id.size:
        point_count_by_cell = np.bincount(
            inside_cell_id,
            minlength=total_cell_count,
        )
        active_counts = point_count_by_cell[point_count_by_cell > 0]
        assigned_x = x_m[inside_grid_mask]
        assigned_y = y_m[inside_grid_mask]
        x_min_m = float(np.min(assigned_x))
        x_max_m = float(np.max(assigned_x))
        y_min_m = float(np.min(assigned_y))
        y_max_m = float(np.max(assigned_y))
        min_points_per_active_cell: int | None = int(np.min(active_counts))
        median_points_per_active_cell: float | None = float(np.median(active_counts))
        max_points_per_active_cell: int | None = int(np.max(active_counts))
    else:
        active_counts = np.asarray([], dtype=np.int64)
        x_min_m = None
        x_max_m = None
        y_min_m = None
        y_max_m = None
        min_points_per_active_cell = None
        median_points_per_active_cell = None
        max_points_per_active_cell = None

    active_cell_count = int(active_counts.size)
    return {
        'n_points': n_points,
        'n_inside_grid': n_inside_grid,
        'n_outside_grid': n_outside_grid,
        'inside_grid_fraction': float(n_inside_grid / n_points) if n_points else None,
        'x_min_m': x_min_m,
        'x_max_m': x_max_m,
        'y_min_m': y_min_m,
        'y_max_m': y_max_m,
        'active_cell_count': active_cell_count,
        'inactive_cell_count': total_cell_count - active_cell_count,
        'min_points_per_active_cell': min_points_per_active_cell,
        'median_points_per_active_cell': median_points_per_active_cell,
        'max_points_per_active_cell': max_points_per_active_cell,
    }


def _axis_cell_indices(
    values: np.ndarray,
    *,
    origin_m: float,
    size_m: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.full(values.shape, -1, dtype=np.int64)
    finite_mask = np.isfinite(values)
    if np.isinf(size_m):
        indices[finite_mask] = 0
        return indices, finite_mask

    end_m = origin_m + count * size_m
    tolerance_m = _final_boundary_tolerance(
        origin_m=origin_m,
        end_m=end_m,
        size_m=size_m,
    )
    on_final_boundary = np.isclose(
        values,
        end_m,
        rtol=0.0,
        atol=tolerance_m,
    )
    inside_mask = finite_mask & (values >= origin_m) & (
        (values < end_m) | on_final_boundary
    )
    if np.any(inside_mask):
        relative = (values[inside_mask] - origin_m) / size_m
        inside_indices = np.floor(relative).astype(np.int64)
        indices[inside_mask] = np.minimum(inside_indices, count - 1)
    return indices, inside_mask


def _final_boundary_tolerance(
    *,
    origin_m: float,
    end_m: float,
    size_m: float,
) -> float:
    scale = max(1.0, abs(origin_m), abs(end_m), abs(size_m))
    return _FINAL_BOUNDARY_ATOL * scale


def _as_1d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric array') from exc
    if array.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    return array


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be a positive integer')
    number = int(value)
    if number <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return number


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not np.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


def _positive_finite_float(value: Any, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return number


__all__ = [
    'RefractionCellAssignment',
    'RefractionCellGrid',
    'assign_observation_midpoint_cells',
    'assign_points_to_refraction_cells',
    'build_refraction_cell_grid',
    'compute_source_receiver_midpoints',
]
