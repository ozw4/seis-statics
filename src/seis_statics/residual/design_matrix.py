"""Design-matrix helpers for residual static estimation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_finite_float as _coerce_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics.residual.types import MoveoutModel, ResidualStaticSolverInputs

_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    allow_integer_like_float=False,
)


@dataclass(frozen=True)
class ResidualStaticColumnLayout:
    moveout_model: MoveoutModel
    n_model_parameters: int

    intercept_col: int
    slowness_col: int | None

    source_delay_start_col: int
    receiver_delay_start_col: int

    n_sources: int
    n_receivers: int
    source_delay_cols: np.ndarray
    receiver_delay_cols: np.ndarray


@dataclass(frozen=True)
class ResidualStaticObservationMatrixTriplets:
    row_indices: np.ndarray
    col_indices: np.ndarray
    data: np.ndarray
    rhs_s: np.ndarray

    row_to_sorted_trace_index: np.ndarray
    used_mask_sorted: np.ndarray

    n_rows: int
    n_cols: int
    layout: ResidualStaticColumnLayout


@dataclass(frozen=True)
class ResidualStaticParameterParts:
    intercept_s: float
    slowness_s_per_offset_unit: float | None
    source_delay_s: np.ndarray
    receiver_delay_s: np.ndarray


@dataclass(frozen=True)
class ResidualStaticModelEvaluation:
    moveout_model_time_s_sorted: np.ndarray
    estimated_trace_delay_s_sorted: np.ndarray
    modeled_pick_time_s_sorted: np.ndarray
    residual_s_sorted: np.ndarray
    residual_valid_mask_sorted: np.ndarray


def build_residual_static_column_layout(
    inputs: ResidualStaticSolverInputs,
) -> ResidualStaticColumnLayout:
    """Build the deterministic parameter-column layout for residual statics."""
    moveout_model = _validate_moveout_model(inputs.moveout_model)
    n_traces = _input_n_traces(inputs)
    source_unique_ids = _coerce_1d_integer_int64(
        inputs.source_unique_ids,
        name='source_unique_ids',
    )
    receiver_unique_ids = _coerce_1d_integer_int64(
        inputs.receiver_unique_ids,
        name='receiver_unique_ids',
    )
    n_sources = _non_empty_array_size(source_unique_ids, name='source_unique_ids')
    n_receivers = _non_empty_array_size(
        receiver_unique_ids,
        name='receiver_unique_ids',
    )

    source_index = _coerce_1d_integer_int64(
        inputs.source_index_sorted,
        name='source_index_sorted',
        expected_shape=(n_traces,),
    )
    receiver_index = _coerce_1d_integer_int64(
        inputs.receiver_index_sorted,
        name='receiver_index_sorted',
        expected_shape=(n_traces,),
    )
    _validate_index_range(source_index, n_unique=n_sources, name='source_index_sorted')
    _validate_index_range(
        receiver_index,
        n_unique=n_receivers,
        name='receiver_index_sorted',
    )

    intercept_col = 0
    if moveout_model == 'linear_abs_offset':
        slowness_col: int | None = 1
        source_delay_start_col = 2
    else:
        slowness_col = None
        source_delay_start_col = 1
    receiver_delay_start_col = source_delay_start_col + n_sources
    n_model_parameters = receiver_delay_start_col + n_receivers

    source_delay_cols = np.arange(
        source_delay_start_col,
        source_delay_start_col + n_sources,
        dtype=np.int64,
    )
    receiver_delay_cols = np.arange(
        receiver_delay_start_col,
        receiver_delay_start_col + n_receivers,
        dtype=np.int64,
    )
    return ResidualStaticColumnLayout(
        moveout_model=moveout_model,
        n_model_parameters=n_model_parameters,
        intercept_col=intercept_col,
        slowness_col=slowness_col,
        source_delay_start_col=source_delay_start_col,
        receiver_delay_start_col=receiver_delay_start_col,
        n_sources=n_sources,
        n_receivers=n_receivers,
        source_delay_cols=source_delay_cols,
        receiver_delay_cols=receiver_delay_cols,
    )


def build_residual_static_observation_matrix_triplets(
    inputs: ResidualStaticSolverInputs,
    *,
    used_mask_sorted: np.ndarray | None = None,
) -> ResidualStaticObservationMatrixTriplets:
    """Build deterministic COO triplets for the residual static observations."""
    layout = build_residual_static_column_layout(inputs)
    n_traces = _input_n_traces(inputs)
    source_index = _coerce_1d_integer_int64(
        inputs.source_index_sorted,
        name='source_index_sorted',
        expected_shape=(n_traces,),
    )
    receiver_index = _coerce_1d_integer_int64(
        inputs.receiver_index_sorted,
        name='receiver_index_sorted',
        expected_shape=(n_traces,),
    )
    valid_mask = _coerce_1d_bool_array(
        inputs.valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=(n_traces,),
    )
    used_mask = _used_mask_or_default(
        valid_mask,
        used_mask_sorted=used_mask_sorted,
        n_traces=n_traces,
    )
    _validate_mask_subset(
        used_mask,
        valid_mask,
        mask_name='used_mask_sorted',
        base_name='valid_pick_mask_sorted',
    )
    row_to_sorted_trace_index = np.ascontiguousarray(
        np.flatnonzero(used_mask),
        dtype=np.int64,
    )
    n_rows = int(row_to_sorted_trace_index.shape[0])
    if n_rows == 0:
        raise ValueError('at least one used residual static pick is required')

    pick_time_after_datum = _coerce_1d_real_numeric_float64(
        inputs.pick_time_after_datum_s_sorted,
        name='pick_time_after_datum_s_sorted',
        expected_shape=(n_traces,),
    )
    _validate_no_inf(
        pick_time_after_datum,
        name='pick_time_after_datum_s_sorted',
    )
    _validate_finite_at_mask(
        pick_time_after_datum,
        mask=used_mask,
        name='pick_time_after_datum_s_sorted',
        mask_name='used picks',
    )

    if layout.moveout_model == 'linear_abs_offset':
        abs_offset = _required_abs_offset(inputs, expected_shape=(n_traces,))
        _validate_finite_nonnegative_at_mask(
            abs_offset,
            mask=used_mask,
            name='abs_offset_sorted',
            mask_name='used traces',
        )
        entries_per_row = 4
    else:
        abs_offset = None
        entries_per_row = 3

    row_indices = np.repeat(
        np.arange(n_rows, dtype=np.int64),
        entries_per_row,
    )
    col_indices = np.empty(n_rows * entries_per_row, dtype=np.int64)
    data = np.empty(n_rows * entries_per_row, dtype=np.float64)

    if layout.moveout_model == 'linear_abs_offset':
        if layout.slowness_col is None:
            raise ValueError('linear_abs_offset layout requires a slowness column')
        col_indices[0::4] = layout.intercept_col
        col_indices[1::4] = layout.slowness_col
        col_indices[2::4] = layout.source_delay_cols[
            source_index[row_to_sorted_trace_index]
        ]
        col_indices[3::4] = layout.receiver_delay_cols[
            receiver_index[row_to_sorted_trace_index]
        ]
        data[0::4] = 1.0
        data[1::4] = abs_offset[row_to_sorted_trace_index]
        data[2::4] = 1.0
        data[3::4] = 1.0
    else:
        col_indices[0::3] = layout.intercept_col
        col_indices[1::3] = layout.source_delay_cols[
            source_index[row_to_sorted_trace_index]
        ]
        col_indices[2::3] = layout.receiver_delay_cols[
            receiver_index[row_to_sorted_trace_index]
        ]
        data[0::3] = 1.0
        data[1::3] = 1.0
        data[2::3] = 1.0

    rhs_s = np.ascontiguousarray(
        pick_time_after_datum[row_to_sorted_trace_index],
        dtype=np.float64,
    )
    return ResidualStaticObservationMatrixTriplets(
        row_indices=row_indices,
        col_indices=col_indices,
        data=data,
        rhs_s=rhs_s,
        row_to_sorted_trace_index=row_to_sorted_trace_index,
        used_mask_sorted=np.ascontiguousarray(used_mask, dtype=bool),
        n_rows=n_rows,
        n_cols=layout.n_model_parameters,
        layout=layout,
    )


def compute_linear_abs_offset_moveout_s(
    abs_offset_sorted: np.ndarray,
    *,
    intercept_s: float,
    slowness_s_per_offset_unit: float,
) -> np.ndarray:
    """Evaluate ``intercept + slowness * abs(offset)`` in seconds."""
    abs_offset = _coerce_1d_real_numeric_float64(
        abs_offset_sorted,
        name='abs_offset_sorted',
    )
    _validate_finite_nonnegative(abs_offset, name='abs_offset_sorted')
    intercept = _coerce_finite_float(intercept_s, name='intercept_s')
    slowness = _coerce_finite_float(
        slowness_s_per_offset_unit,
        name='slowness_s_per_offset_unit',
    )
    with np.errstate(over='ignore', invalid='ignore'):
        moveout_s = intercept + slowness * abs_offset
    moveout_s = np.ascontiguousarray(moveout_s, dtype=np.float64)
    _validate_all_finite(moveout_s, name='moveout_s')
    return moveout_s


def pack_residual_static_parameters(
    *,
    layout: ResidualStaticColumnLayout,
    intercept_s: float,
    slowness_s_per_offset_unit: float | None,
    source_delay_s: np.ndarray,
    receiver_delay_s: np.ndarray,
) -> np.ndarray:
    """Pack parameter parts according to a residual static column layout."""
    moveout_model = _validate_moveout_model(layout.moveout_model)
    _validate_layout_sizes(layout)
    intercept = _coerce_finite_float(intercept_s, name='intercept_s')
    source_delay = _coerce_1d_real_numeric_float64(
        source_delay_s,
        name='source_delay_s',
        expected_shape=(layout.n_sources,),
    )
    receiver_delay = _coerce_1d_real_numeric_float64(
        receiver_delay_s,
        name='receiver_delay_s',
        expected_shape=(layout.n_receivers,),
    )
    _validate_all_finite(source_delay, name='source_delay_s')
    _validate_all_finite(receiver_delay, name='receiver_delay_s')

    parameter_vector = np.zeros(layout.n_model_parameters, dtype=np.float64)
    parameter_vector[layout.intercept_col] = intercept
    if moveout_model == 'linear_abs_offset':
        if layout.slowness_col is None:
            raise ValueError('linear_abs_offset layout requires a slowness column')
        if slowness_s_per_offset_unit is None:
            raise ValueError(
                'slowness_s_per_offset_unit is required for linear_abs_offset'
            )
        parameter_vector[layout.slowness_col] = _coerce_finite_float(
            slowness_s_per_offset_unit,
            name='slowness_s_per_offset_unit',
        )
    elif slowness_s_per_offset_unit is not None:
        raise ValueError('slowness_s_per_offset_unit must be None for none moveout')

    parameter_vector[layout.source_delay_cols] = source_delay
    parameter_vector[layout.receiver_delay_cols] = receiver_delay
    return np.ascontiguousarray(parameter_vector, dtype=np.float64)


def unpack_residual_static_parameters(
    layout: ResidualStaticColumnLayout,
    parameter_vector: np.ndarray,
) -> ResidualStaticParameterParts:
    """Unpack a residual static parameter vector according to its layout."""
    moveout_model = _validate_moveout_model(layout.moveout_model)
    _validate_layout_sizes(layout)
    vector = _coerce_1d_real_numeric_float64(
        parameter_vector,
        name='parameter_vector',
        expected_shape=(layout.n_model_parameters,),
    )
    _validate_all_finite(vector, name='parameter_vector')
    intercept_s = float(vector[layout.intercept_col])
    if moveout_model == 'linear_abs_offset':
        if layout.slowness_col is None:
            raise ValueError('linear_abs_offset layout requires a slowness column')
        slowness_s_per_offset_unit = float(vector[layout.slowness_col])
    else:
        slowness_s_per_offset_unit = None

    return ResidualStaticParameterParts(
        intercept_s=intercept_s,
        slowness_s_per_offset_unit=slowness_s_per_offset_unit,
        source_delay_s=np.ascontiguousarray(
            vector[layout.source_delay_cols],
            dtype=np.float64,
        ),
        receiver_delay_s=np.ascontiguousarray(
            vector[layout.receiver_delay_cols],
            dtype=np.float64,
        ),
    )


def evaluate_residual_static_model(
    inputs: ResidualStaticSolverInputs,
    layout: ResidualStaticColumnLayout,
    parameter_vector: np.ndarray,
    *,
    residual_mask_sorted: np.ndarray | None = None,
) -> ResidualStaticModelEvaluation:
    """Evaluate moveout, estimated delays, modeled picks, and residuals."""
    input_moveout_model = _validate_moveout_model(inputs.moveout_model)
    layout_moveout_model = _validate_moveout_model(layout.moveout_model)
    if layout_moveout_model != input_moveout_model:
        raise ValueError('layout moveout_model does not match inputs.moveout_model')
    parts = unpack_residual_static_parameters(layout, parameter_vector)

    n_traces = _input_n_traces(inputs)
    source_index = _coerce_1d_integer_int64(
        inputs.source_index_sorted,
        name='source_index_sorted',
        expected_shape=(n_traces,),
    )
    receiver_index = _coerce_1d_integer_int64(
        inputs.receiver_index_sorted,
        name='receiver_index_sorted',
        expected_shape=(n_traces,),
    )
    _validate_index_range(
        source_index,
        n_unique=layout.n_sources,
        name='source_index_sorted',
    )
    _validate_index_range(
        receiver_index,
        n_unique=layout.n_receivers,
        name='receiver_index_sorted',
    )

    valid_mask = _coerce_1d_bool_array(
        inputs.valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=(n_traces,),
    )
    residual_mask = _residual_mask_or_default(
        valid_mask,
        residual_mask_sorted=residual_mask_sorted,
        n_traces=n_traces,
    )
    _validate_mask_subset(
        residual_mask,
        valid_mask,
        mask_name='residual_mask_sorted',
        base_name='valid_pick_mask_sorted',
    )
    pick_time_after_datum = _coerce_1d_real_numeric_float64(
        inputs.pick_time_after_datum_s_sorted,
        name='pick_time_after_datum_s_sorted',
        expected_shape=(n_traces,),
    )
    _validate_no_inf(
        pick_time_after_datum,
        name='pick_time_after_datum_s_sorted',
    )
    _validate_finite_at_mask(
        pick_time_after_datum,
        mask=residual_mask,
        name='pick_time_after_datum_s_sorted',
        mask_name='residual-valid traces',
    )

    if layout_moveout_model == 'linear_abs_offset':
        if parts.slowness_s_per_offset_unit is None:
            raise ValueError('linear_abs_offset parameters require slowness')
        abs_offset = _required_abs_offset(inputs, expected_shape=(n_traces,))
        _validate_finite_nonnegative(abs_offset, name='abs_offset_sorted')
        moveout_model_time_s = compute_linear_abs_offset_moveout_s(
            abs_offset,
            intercept_s=parts.intercept_s,
            slowness_s_per_offset_unit=parts.slowness_s_per_offset_unit,
        )
    else:
        moveout_model_time_s = np.full(
            n_traces,
            parts.intercept_s,
            dtype=np.float64,
        )

    estimated_trace_delay_s = np.ascontiguousarray(
        parts.source_delay_s[source_index] + parts.receiver_delay_s[receiver_index],
        dtype=np.float64,
    )
    modeled_pick_time_s = np.ascontiguousarray(
        moveout_model_time_s + estimated_trace_delay_s,
        dtype=np.float64,
    )
    residual_s = np.ascontiguousarray(
        pick_time_after_datum - modeled_pick_time_s,
        dtype=np.float64,
    )
    residual_s[~residual_mask] = np.nan
    _validate_finite_at_mask(
        residual_s,
        mask=residual_mask,
        name='residual_s_sorted',
        mask_name='residual-valid traces',
    )

    return ResidualStaticModelEvaluation(
        moveout_model_time_s_sorted=np.ascontiguousarray(
            moveout_model_time_s,
            dtype=np.float64,
        ),
        estimated_trace_delay_s_sorted=estimated_trace_delay_s,
        modeled_pick_time_s_sorted=modeled_pick_time_s,
        residual_s_sorted=residual_s,
        residual_valid_mask_sorted=np.ascontiguousarray(residual_mask, dtype=bool),
    )


def _validate_moveout_model(value: object) -> MoveoutModel:
    if value == 'linear_abs_offset':
        return 'linear_abs_offset'
    if value == 'none':
        return 'none'
    raise ValueError(f'unsupported moveout_model: {value!r}')


def _input_n_traces(inputs: ResidualStaticSolverInputs) -> int:
    return _coerce_positive_int(inputs.n_traces, name='n_traces')


def _validate_layout_sizes(layout: ResidualStaticColumnLayout) -> None:
    _coerce_positive_int(layout.n_model_parameters, name='layout.n_model_parameters')
    _coerce_positive_int(layout.n_sources, name='layout.n_sources')
    _coerce_positive_int(layout.n_receivers, name='layout.n_receivers')
    _coerce_positive_int(layout.intercept_col + 1, name='layout.intercept_col')
    _coerce_positive_int(
        layout.source_delay_start_col + 1,
        name='layout.source_delay_start_col',
    )
    _coerce_positive_int(
        layout.receiver_delay_start_col + 1,
        name='layout.receiver_delay_start_col',
    )
    source_delay_cols = _coerce_1d_integer_int64(
        layout.source_delay_cols,
        name='layout.source_delay_cols',
        expected_shape=(layout.n_sources,),
    )
    receiver_delay_cols = _coerce_1d_integer_int64(
        layout.receiver_delay_cols,
        name='layout.receiver_delay_cols',
        expected_shape=(layout.n_receivers,),
    )
    _validate_index_range(
        source_delay_cols,
        n_unique=layout.n_model_parameters,
        name='layout.source_delay_cols',
    )
    _validate_index_range(
        receiver_delay_cols,
        n_unique=layout.n_model_parameters,
        name='layout.receiver_delay_cols',
    )


def _used_mask_or_default(
    valid_mask: np.ndarray,
    *,
    used_mask_sorted: np.ndarray | None,
    n_traces: int,
) -> np.ndarray:
    if used_mask_sorted is None:
        return np.ascontiguousarray(valid_mask, dtype=bool)
    return _coerce_1d_bool_array(
        used_mask_sorted,
        name='used_mask_sorted',
        expected_shape=(n_traces,),
    )


def _residual_mask_or_default(
    valid_mask: np.ndarray,
    *,
    residual_mask_sorted: np.ndarray | None,
    n_traces: int,
) -> np.ndarray:
    if residual_mask_sorted is None:
        return np.ascontiguousarray(valid_mask, dtype=bool)
    return _coerce_1d_bool_array(
        residual_mask_sorted,
        name='residual_mask_sorted',
        expected_shape=(n_traces,),
    )


def _validate_mask_subset(
    mask: np.ndarray,
    base_mask: np.ndarray,
    *,
    mask_name: str,
    base_name: str,
) -> None:
    if np.any(mask & ~base_mask):
        raise ValueError(f'{mask_name} must be a subset of {base_name}')


def _required_abs_offset(
    inputs: ResidualStaticSolverInputs,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if inputs.abs_offset_sorted is None:
        raise ValueError('abs_offset_sorted is required for linear_abs_offset moveout')
    return _coerce_1d_real_numeric_float64(
        inputs.abs_offset_sorted,
        name='abs_offset_sorted',
        expected_shape=expected_shape,
    )


def _non_empty_array_size(values: np.ndarray, *, name: str) -> int:
    size = int(values.shape[0])
    if size <= 0:
        raise ValueError(f'{name} must be non-empty')
    return size


def _validate_index_range(
    indices: np.ndarray,
    *,
    n_unique: int,
    name: str,
) -> None:
    if np.any(indices < 0):
        raise ValueError(f'{name} must be greater than or equal to 0')
    if np.any(indices >= n_unique):
        raise ValueError(f'{name} contains values outside 0..{n_unique - 1}')


def _validate_no_inf(values: np.ndarray, *, name: str) -> None:
    if np.any(np.isinf(values)):
        raise ValueError(f'{name} contains inf')


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


def _validate_finite_nonnegative(values: np.ndarray, *, name: str) -> None:
    _validate_all_finite(values, name=name)
    if np.any(values < 0.0):
        raise ValueError(f'{name} must be non-negative')


def _validate_all_finite(values: np.ndarray, *, name: str) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f'{name} must contain only finite values')


__all__ = [
    'MoveoutModel',
    'ResidualStaticColumnLayout',
    'ResidualStaticModelEvaluation',
    'ResidualStaticObservationMatrixTriplets',
    'ResidualStaticParameterParts',
    'build_residual_static_column_layout',
    'build_residual_static_observation_matrix_triplets',
    'compute_linear_abs_offset_moveout_s',
    'evaluate_residual_static_model',
    'pack_residual_static_parameters',
    'unpack_residual_static_parameters',
]
