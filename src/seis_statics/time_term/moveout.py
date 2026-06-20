"""Moveout terms for future time-term static inversion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics.time_term.types import TimeTermInversionInputs

_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    nonfinite_message='must contain only finite values',
)

TimeTermMoveoutModel = Literal[
    'head_wave_linear_offset',
    'reciprocal_head_wave',
    'linear_offset',
    'none',
]
MoveoutDistanceSource = Literal['geometry', 'offset_header', 'auto']

_MOVEOUT_MODELS: set[str] = {
    'head_wave_linear_offset',
    'reciprocal_head_wave',
    'linear_offset',
    'none',
}
_DISTANCE_SOURCES: set[str] = {'geometry', 'offset_header', 'auto'}


@dataclass(frozen=True)
class TimeTermMoveoutConfig:
    model: TimeTermMoveoutModel
    refractor_velocity_m_s: float
    distance_source: MoveoutDistanceSource = 'geometry'
    offset_byte: int | None = None
    allow_missing_offset: bool = False
    max_geometry_offset_mismatch_m: float | None = None


@dataclass(frozen=True)
class TimeTermMoveoutResult:
    model: TimeTermMoveoutModel
    refractor_velocity_m_s: float
    distance_source: MoveoutDistanceSource

    distance_m_sorted: np.ndarray
    moveout_time_s_sorted: np.ndarray
    valid_moveout_mask_sorted: np.ndarray

    reciprocal_pair_index_sorted: np.ndarray
    has_reciprocal_pair_mask_sorted: np.ndarray

    geometry_distance_m_sorted: np.ndarray
    offset_abs_m_sorted: np.ndarray | None
    geometry_offset_mismatch_m_sorted: np.ndarray | None


def compute_time_term_moveout(
    inputs: TimeTermInversionInputs,
    config: TimeTermMoveoutConfig,
) -> TimeTermMoveoutResult:
    """Compute positive propagation moveout time in TraceStore sorted order."""
    n_traces = _coerce_positive_int(inputs.n_traces, name='n_traces')
    model = _validate_model(config.model)
    distance_source = _validate_distance_source(config.distance_source)
    offset_byte = _validate_optional_trace_header_byte(
        config.offset_byte,
        name='offset_byte',
    )
    allow_missing_offset = _validate_bool(
        config.allow_missing_offset,
        name='allow_missing_offset',
    )
    if (
        model != 'none'
        and distance_source == 'offset_header'
        and offset_byte is None
        and not allow_missing_offset
    ):
        raise ValueError('offset_byte is required for offset_header distance')
    refractor_velocity = _coerce_positive_finite_float(
        config.refractor_velocity_m_s,
        name='refractor_velocity_m_s',
    )
    max_mismatch = _validate_optional_nonnegative_finite_float(
        config.max_geometry_offset_mismatch_m,
        name='max_geometry_offset_mismatch_m',
    )
    expected_shape = (n_traces,)

    source_x = _coerce_1d_real_numeric_float64(
        inputs.source_x_m_sorted,
        name='source_x_m_sorted',
        expected_shape=expected_shape,
    )
    source_y = _coerce_1d_real_numeric_float64(
        inputs.source_y_m_sorted,
        name='source_y_m_sorted',
        expected_shape=expected_shape,
    )
    receiver_x = _coerce_1d_real_numeric_float64(
        inputs.receiver_x_m_sorted,
        name='receiver_x_m_sorted',
        expected_shape=expected_shape,
    )
    receiver_y = _coerce_1d_real_numeric_float64(
        inputs.receiver_y_m_sorted,
        name='receiver_y_m_sorted',
        expected_shape=expected_shape,
    )
    geometry_distance = compute_geometry_distance_m(
        source_x,
        source_y,
        receiver_x,
        receiver_y,
    )
    geometry_is_valid = _is_finite_nonnegative(geometry_distance)
    geometry_required = model != 'none' and distance_source == 'geometry'
    if geometry_required and not geometry_is_valid:
        _validate_finite_nonnegative(
            geometry_distance,
            name='geometry_distance_m_sorted',
        )

    offset_required = (
        model != 'none'
        and distance_source == 'offset_header'
        and not allow_missing_offset
    )
    offset_abs = _optional_offset_abs(
        inputs.offset_sorted,
        expected_shape=expected_shape,
        required=offset_required,
        allow_invalid=not _must_validate_optional_offset(
            model=model,
            distance_source=distance_source,
            geometry_is_valid=geometry_is_valid,
            max_geometry_offset_mismatch_m=max_mismatch,
        ),
    )
    mismatch = None
    if model != 'none' and offset_abs is not None and geometry_is_valid:
        mismatch = np.ascontiguousarray(geometry_distance - offset_abs, dtype=np.float64)
        if max_mismatch is not None and np.any(np.abs(mismatch) > max_mismatch):
            raise ValueError(
                'geometry_offset_mismatch_m_sorted exceeds '
                'max_geometry_offset_mismatch_m'
            )
    elif max_mismatch is not None and model != 'none':
        if not geometry_is_valid:
            _validate_finite_nonnegative(
                geometry_distance,
                name='geometry_distance_m_sorted',
            )
        raise ValueError(
            'offset_sorted is required for max_geometry_offset_mismatch_m'
        )

    source_nodes = _coerce_1d_integer_int64(
        inputs.source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=expected_shape,
    )
    receiver_nodes = _coerce_1d_integer_int64(
        inputs.receiver_node_id_sorted,
        name='receiver_node_id_sorted',
        expected_shape=expected_shape,
    )
    _validate_node_range(source_nodes, receiver_nodes, n_nodes=inputs.n_nodes)
    reciprocal_pair_index = build_reciprocal_pair_index(source_nodes, receiver_nodes)
    _validate_reciprocal_pair_index(reciprocal_pair_index, n_traces=n_traces)

    if model == 'none':
        distance = np.zeros(n_traces, dtype=np.float64)
        moveout_time = np.zeros(n_traces, dtype=np.float64)
    else:
        distance = _select_distance(
            distance_source=distance_source,
            geometry_distance=geometry_distance,
            geometry_is_valid=geometry_is_valid,
            offset_abs=offset_abs,
            allow_missing_offset=allow_missing_offset,
            n_traces=n_traces,
        )
        if not allow_missing_offset or not np.any(~np.isfinite(distance)):
            _validate_finite_nonnegative(distance, name='distance_m_sorted')
        # This is a positive propagation-time term, not an applied static shift.
        moveout_time = np.ascontiguousarray(distance / refractor_velocity, dtype=np.float64)
        if not allow_missing_offset or not np.any(~np.isfinite(moveout_time)):
            _validate_finite_nonnegative(
                moveout_time,
                name='moveout_time_s_sorted',
            )

    valid_moveout_mask = np.ascontiguousarray(
        np.isfinite(distance)
        & (distance >= 0.0)
        & np.isfinite(moveout_time)
        & (moveout_time >= 0.0),
        dtype=bool,
    )
    if valid_moveout_mask.shape != expected_shape:
        raise ValueError('valid_moveout_mask_sorted shape mismatch')

    return TimeTermMoveoutResult(
        model=model,
        refractor_velocity_m_s=refractor_velocity,
        distance_source=distance_source,
        distance_m_sorted=np.ascontiguousarray(distance, dtype=np.float64),
        moveout_time_s_sorted=np.ascontiguousarray(moveout_time, dtype=np.float64),
        valid_moveout_mask_sorted=valid_moveout_mask,
        reciprocal_pair_index_sorted=reciprocal_pair_index,
        has_reciprocal_pair_mask_sorted=np.ascontiguousarray(
            reciprocal_pair_index >= 0,
            dtype=bool,
        ),
        geometry_distance_m_sorted=geometry_distance,
        offset_abs_m_sorted=offset_abs,
        geometry_offset_mismatch_m_sorted=mismatch,
    )


def compute_geometry_distance_m(
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
) -> np.ndarray:
    """Return horizontal source-receiver distance in meters."""
    source_x = np.asarray(source_x_m, dtype=np.float64)
    source_y = np.asarray(source_y_m, dtype=np.float64)
    receiver_x = np.asarray(receiver_x_m, dtype=np.float64)
    receiver_y = np.asarray(receiver_y_m, dtype=np.float64)
    dx = receiver_x - source_x
    dy = receiver_y - source_y
    return np.ascontiguousarray(np.hypot(dx, dy), dtype=np.float64)


def build_reciprocal_pair_index(
    source_node_id_sorted: np.ndarray,
    receiver_node_id_sorted: np.ndarray,
) -> np.ndarray:
    """Return the smallest reverse-key trace index for each sorted trace."""
    source = _coerce_1d_integer_int64(
        source_node_id_sorted,
        name='source_node_id_sorted',
    )
    receiver = _coerce_1d_integer_int64(
        receiver_node_id_sorted,
        name='receiver_node_id_sorted',
        expected_shape=source.shape,
    )
    pair_to_indices: dict[tuple[int, int], list[int]] = {}
    for trace_index, pair in enumerate(zip(source, receiver, strict=True)):
        pair_to_indices.setdefault((int(pair[0]), int(pair[1])), []).append(trace_index)

    pair_index = np.full(source.shape[0], -1, dtype=np.int64)
    for trace_index, pair in enumerate(zip(source, receiver, strict=True)):
        reverse_key = (int(pair[1]), int(pair[0]))
        for candidate_index in pair_to_indices.get(reverse_key, ()):
            if candidate_index != trace_index:
                pair_index[trace_index] = int(candidate_index)
                break
    return np.ascontiguousarray(pair_index, dtype=np.int64)


def summarize_time_term_moveout(result: TimeTermMoveoutResult) -> dict[str, object]:
    """Return a JSON-safe summary for QC artifacts and job logs."""
    distance = _coerce_1d_real_numeric_float64(
        result.distance_m_sorted,
        name='distance_m_sorted',
    )
    moveout = _coerce_1d_real_numeric_float64(
        result.moveout_time_s_sorted,
        name='moveout_time_s_sorted',
        expected_shape=distance.shape,
    )
    valid = _coerce_1d_bool_array(
        result.valid_moveout_mask_sorted,
        name='valid_moveout_mask_sorted',
        expected_shape=distance.shape,
    )
    pairs = _coerce_1d_integer_int64(
        result.reciprocal_pair_index_sorted,
        name='reciprocal_pair_index_sorted',
        expected_shape=distance.shape,
    )
    has_pairs = _coerce_1d_bool_array(
        result.has_reciprocal_pair_mask_sorted,
        name='has_reciprocal_pair_mask_sorted',
        expected_shape=distance.shape,
    )
    n_traces = int(distance.shape[0])
    n_valid = int(np.count_nonzero(valid))
    n_pairs = int(np.count_nonzero(has_pairs))

    payload: dict[str, object] = {
        'model': str(result.model),
        'refractor_velocity_m_s': float(result.refractor_velocity_m_s),
        'distance_source': str(result.distance_source),
        'n_traces': n_traces,
        'n_valid_moveout': n_valid,
        'valid_moveout_fraction': _fraction(n_valid, n_traces),
        'distance_m': _stats_payload(distance),
        'moveout_time_s': _stats_payload(moveout),
        'moveout_time_ms': _stats_payload(moveout * 1000.0),
        'n_reciprocal_pairs': n_pairs,
        'reciprocal_pair_fraction': _fraction(n_pairs, n_traces),
        'has_offset_header': result.offset_abs_m_sorted is not None,
    }
    _validate_reciprocal_pair_index(pairs, n_traces=n_traces)

    if result.offset_abs_m_sorted is not None:
        offset = _coerce_1d_real_numeric_float64(
            result.offset_abs_m_sorted,
            name='offset_abs_m_sorted',
            expected_shape=distance.shape,
        )
        payload['offset_abs_m'] = _stats_payload(offset)
    if result.geometry_offset_mismatch_m_sorted is not None:
        mismatch = _coerce_1d_real_numeric_float64(
            result.geometry_offset_mismatch_m_sorted,
            name='geometry_offset_mismatch_m_sorted',
            expected_shape=distance.shape,
        )
        payload['geometry_offset_mismatch_m'] = _stats_payload(mismatch)
    return payload


def _select_distance(
    *,
    distance_source: MoveoutDistanceSource,
    geometry_distance: np.ndarray,
    geometry_is_valid: bool,
    offset_abs: np.ndarray | None,
    allow_missing_offset: bool,
    n_traces: int,
) -> np.ndarray:
    if distance_source == 'geometry':
        return np.ascontiguousarray(geometry_distance, dtype=np.float64)
    if distance_source == 'auto':
        if geometry_is_valid:
            return np.ascontiguousarray(geometry_distance, dtype=np.float64)
        if offset_abs is not None:
            return np.ascontiguousarray(offset_abs, dtype=np.float64)
        if allow_missing_offset:
            return np.full(int(n_traces), np.nan, dtype=np.float64)
        raise ValueError('auto distance requires finite geometry or offset_sorted')
    if distance_source == 'offset_header':
        if offset_abs is None:
            if allow_missing_offset:
                return np.full(int(n_traces), np.nan, dtype=np.float64)
            raise ValueError('offset_sorted is required for offset_header distance')
        return np.ascontiguousarray(offset_abs, dtype=np.float64)
    raise ValueError(f'unsupported distance_source: {distance_source!r}')


def _optional_offset_abs(
    values: np.ndarray | None,
    *,
    expected_shape: tuple[int, ...],
    required: bool,
    allow_invalid: bool,
) -> np.ndarray | None:
    if values is None:
        if required:
            raise ValueError('offset_sorted is required for offset_header distance')
        return None
    try:
        offset = _coerce_1d_real_numeric_float64(
            values,
            name='offset_sorted',
            expected_shape=expected_shape,
        )
        if not np.all(np.isfinite(offset)):
            raise ValueError('offset_sorted must contain only finite values')
        offset_abs = np.ascontiguousarray(np.abs(offset), dtype=np.float64)
        _validate_finite_nonnegative(offset_abs, name='offset_abs_m_sorted')
    except ValueError:
        if allow_invalid and not required:
            return None
        raise
    return offset_abs


def _must_validate_optional_offset(
    *,
    model: TimeTermMoveoutModel,
    distance_source: MoveoutDistanceSource,
    geometry_is_valid: bool,
    max_geometry_offset_mismatch_m: float | None,
) -> bool:
    if model == 'none':
        return False
    if max_geometry_offset_mismatch_m is not None:
        return True
    if distance_source == 'offset_header':
        return True
    return distance_source == 'auto' and not geometry_is_valid


def _validate_model(value: object) -> TimeTermMoveoutModel:
    if value in _MOVEOUT_MODELS:
        return value  # type: ignore[return-value]
    raise ValueError(f'unsupported moveout model: {value!r}')


def _validate_distance_source(value: object) -> MoveoutDistanceSource:
    if value in _DISTANCE_SOURCES:
        return value  # type: ignore[return-value]
    raise ValueError(f'unsupported distance_source: {value!r}')


def _validate_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be a bool')
    return value


def _validate_optional_trace_header_byte(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer SEG-Y trace header byte')
    byte = int(value)
    if byte < 1 or byte > 240:
        raise ValueError(f'{name} must be in the range 1..240')
    return byte


def _validate_node_range(
    source: np.ndarray,
    receiver: np.ndarray,
    *,
    n_nodes: int,
) -> None:
    node_count = _coerce_positive_int(n_nodes, name='n_nodes')
    if np.any(source < 0) or np.any(receiver < 0):
        raise ValueError('node ids must be non-negative')
    if np.any(source >= node_count) or np.any(receiver >= node_count):
        raise ValueError('node ids must be less than n_nodes')


def _validate_reciprocal_pair_index(values: np.ndarray, *, n_traces: int) -> None:
    arr = _coerce_1d_integer_int64(
        values,
        name='reciprocal_pair_index_sorted',
        expected_shape=(n_traces,),
    )
    if np.any((arr < -1) | (arr >= n_traces)):
        raise ValueError('reciprocal_pair_index_sorted contains out-of-range indices')


def _validate_finite_nonnegative(values: np.ndarray, *, name: str) -> None:
    if not _is_finite_nonnegative(values):
        if not np.all(np.isfinite(values)):
            raise ValueError(f'{name} must contain only finite values')
        raise ValueError(f'{name} must be non-negative')


def _is_finite_nonnegative(values: np.ndarray) -> bool:
    if not np.all(np.isfinite(values)):
        return False
    if np.any(values < 0.0):
        return False
    return True


def _validate_optional_nonnegative_finite_float(
    value: object,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    return _coerce_nonnegative_finite_float(value, name=name)


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
    'MoveoutDistanceSource',
    'TimeTermMoveoutConfig',
    'TimeTermMoveoutModel',
    'TimeTermMoveoutResult',
    'build_reciprocal_pair_index',
    'compute_geometry_distance_m',
    'compute_time_term_moveout',
    'summarize_time_term_moveout',
]
