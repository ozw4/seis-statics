"""Source-depth resolution for refraction field corrections."""

from __future__ import annotations

from typing import Any

import numpy as np

from seis_statics._validation import (
    coerce_1d_integer_int64 as _coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_1d_string_array as _coerce_1d_string_array,
    coerce_finite_float as _coerce_finite_float,
)
from seis_statics.refraction.types import (
    REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES,
    RefractionEndpointFieldCorrectionResult,
    RefractionSourceDepthResult,
    RefractionStaticInputModel,
)


_STATUS_DTYPE = '<U32'
_FIELD_STATUS_DTYPE = '<U48'
_ENDPOINT_KEY_DTYPE = object
_DEFAULT_INCONSISTENCY_TOLERANCE_M = 0.01
_SOURCE_DEPTH_STATUSES = (
    'ok',
    'missing_source_depth',
    'invalid_source_depth',
    'inconsistent_source_depth',
    'exceeds_max_abs_source_depth',
    'inactive_source_endpoint',
)
_SOURCE_DEPTH_COMPONENT = 'source_depth_shift_s'
_NOT_APPLICABLE_STATUS = 'not_applicable'
_SOURCE_DEPTH_SHIFT_FORMULA = 'source_depth_shift_s = +source_depth_m / V1_m_s'
_SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'


def resolve_refraction_source_depth_for_input_model(
    *,
    input_model: RefractionStaticInputModel,
    mode: str,
    source_depth_byte: int | None,
    positive_down: bool = True,
    max_abs_source_depth_m: float = 100.0,
    inconsistency_tolerance_m: float = _DEFAULT_INCONSISTENCY_TOLERANCE_M,
) -> RefractionSourceDepthResult:
    """Resolve trace-order source depths to source endpoint rows."""
    return resolve_refraction_source_depth(
        source_endpoint_key_sorted=input_model.source_endpoint_key_sorted,
        source_endpoint_id_sorted=input_model.source_endpoint_id_sorted,
        source_node_id_sorted=input_model.source_node_id_sorted,
        source_depth_m_sorted=input_model.source_depth_m_sorted,
        mode=mode,
        source_depth_byte=source_depth_byte,
        positive_down=positive_down,
        max_abs_source_depth_m=max_abs_source_depth_m,
        inconsistency_tolerance_m=inconsistency_tolerance_m,
    )


def resolve_refraction_source_depth(
    *,
    source_endpoint_key_sorted: np.ndarray,
    source_node_id_sorted: np.ndarray,
    mode: str,
    source_depth_byte: int | None,
    source_depth_m_sorted: np.ndarray | None = None,
    source_endpoint_id_sorted: np.ndarray | None = None,
    positive_down: bool = True,
    max_abs_source_depth_m: float = 100.0,
    inconsistency_tolerance_m: float = _DEFAULT_INCONSISTENCY_TOLERANCE_M,
) -> RefractionSourceDepthResult:
    """Aggregate positive-down source depth samples by source endpoint."""
    mode_text = _coerce_mode(mode)
    keys = _coerce_1d_string(source_endpoint_key_sorted, name='source_endpoint_key_sorted')
    n_traces = int(keys.shape[0])
    nodes = _coerce_1d_integer(
        source_node_id_sorted,
        name='source_node_id_sorted',
        expected_shape=(n_traces,),
    )
    endpoint_ids = _coerce_optional_1d_integer(
        source_endpoint_id_sorted,
        name='source_endpoint_id_sorted',
        expected_shape=(n_traces,),
    )
    if source_depth_m_sorted is None:
        raw_depth = np.full(n_traces, np.nan, dtype=np.float64)
        depth_array_present = False
    else:
        raw_depth = _coerce_1d_float(
            source_depth_m_sorted,
            name='source_depth_m_sorted',
            expected_shape=(n_traces,),
        )
        depth_array_present = True
    depth = raw_depth if positive_down else -raw_depth
    max_abs = _positive_finite_float(
        max_abs_source_depth_m,
        name='max_abs_source_depth_m',
    )
    tolerance = _nonnegative_finite_float(
        inconsistency_tolerance_m,
        name='inconsistency_tolerance_m',
    )

    groups, first_positions = _source_endpoint_groups(keys)
    n_endpoints = int(first_positions.shape[0])
    out_key = keys[first_positions].astype(_ENDPOINT_KEY_DTYPE, copy=False)
    out_id = np.empty(n_endpoints, dtype=np.int64)
    out_node = np.empty(n_endpoints, dtype=np.int64)
    out_depth = np.full(n_endpoints, np.nan, dtype=np.float64)
    out_status = np.full(n_endpoints, 'ok', dtype=_STATUS_DTYPE)
    out_pick_count = np.zeros(n_endpoints, dtype=np.int64)
    out_trace_count = np.zeros(n_endpoints, dtype=np.int64)
    depth_required = mode_text != 'none'

    for endpoint_index, endpoint_key in enumerate(out_key.tolist()):
        trace_indices = np.asarray(groups[str(endpoint_key)], dtype=np.int64)
        out_trace_count[endpoint_index] = int(trace_indices.shape[0])
        out_id[endpoint_index] = _endpoint_id_for_group(
            endpoint_ids=endpoint_ids,
            trace_indices=trace_indices,
            fallback=endpoint_index,
        )
        node_values = nodes[trace_indices]
        out_node[endpoint_index] = _node_id_for_group(node_values)

        values = depth[trace_indices]
        finite = np.isfinite(values)
        nonnegative = finite & (values >= 0.0)
        valid_values = values[nonnegative]
        out_pick_count[endpoint_index] = int(valid_values.shape[0])
        if valid_values.size:
            out_depth[endpoint_index] = float(np.median(valid_values))

        out_status[endpoint_index] = _source_depth_status(
            node_id=int(out_node[endpoint_index]),
            values=values,
            valid_values=valid_values,
            depth_array_present=depth_array_present,
            depth_required=depth_required,
            max_abs_source_depth_m=max_abs,
            inconsistency_tolerance_m=tolerance,
        )

    qc = _source_depth_qc(
        mode=mode_text,
        source_depth_byte=source_depth_byte,
        positive_down=positive_down,
        max_abs_source_depth_m=max_abs,
        inconsistency_tolerance_m=tolerance,
        source_depth_m=out_depth,
        status=out_status,
        n_source_endpoints=n_endpoints,
    )
    return RefractionSourceDepthResult(
        source_endpoint_key=np.ascontiguousarray(out_key, dtype=_ENDPOINT_KEY_DTYPE),
        source_endpoint_id=np.ascontiguousarray(out_id, dtype=np.int64),
        source_node_id=np.ascontiguousarray(out_node, dtype=np.int64),
        source_depth_m=np.ascontiguousarray(out_depth, dtype=np.float64),
        source_depth_status=np.ascontiguousarray(out_status, dtype=_STATUS_DTYPE),
        source_depth_pick_count=np.ascontiguousarray(out_pick_count, dtype=np.int64),
        source_depth_trace_count=np.ascontiguousarray(out_trace_count, dtype=np.int64),
        qc=qc,
    )


def compute_source_depth_weathering_time_correction(
    source_depth_m: np.ndarray,
    v1_m_s: float,
    *,
    status: np.ndarray,
    max_abs_shift_s: float | None,
) -> RefractionEndpointFieldCorrectionResult:
    """Compute source-depth weathering-time shifts for source endpoints."""
    depth = _coerce_1d_float(
        source_depth_m,
        name='source_depth_m',
        expected_shape=np.asarray(source_depth_m).shape,
    )
    if depth.ndim != 1:
        raise ValueError('source_depth_m must be one-dimensional')
    endpoint_count = int(depth.shape[0])
    status_array = _coerce_1d_string_status(
        status,
        name='status',
        expected_shape=(endpoint_count,),
    )
    endpoint_id = np.arange(endpoint_count, dtype=np.int64)
    endpoint_key = endpoint_id.astype('<U32')
    return _compute_source_depth_weathering_time_correction_for_endpoints(
        endpoint_key=endpoint_key,
        endpoint_id=endpoint_id,
        node_id=endpoint_id,
        source_depth_m=depth,
        source_depth_status=status_array,
        v1_m_s=v1_m_s,
        max_abs_shift_s=max_abs_shift_s,
    )


def compute_source_depth_weathering_time_correction_from_result(
    result: RefractionSourceDepthResult,
    v1_m_s: float,
    *,
    max_abs_shift_s: float | None,
) -> RefractionEndpointFieldCorrectionResult:
    """Compute source-depth field shifts from resolved endpoint depth rows."""
    return _compute_source_depth_weathering_time_correction_for_endpoints(
        endpoint_key=result.source_endpoint_key,
        endpoint_id=result.source_endpoint_id,
        node_id=result.source_node_id,
        source_depth_m=result.source_depth_m,
        source_depth_status=result.source_depth_status,
        v1_m_s=v1_m_s,
        max_abs_shift_s=max_abs_shift_s,
    )


def _compute_source_depth_weathering_time_correction_for_endpoints(
    *,
    endpoint_key: np.ndarray,
    endpoint_id: np.ndarray,
    node_id: np.ndarray,
    source_depth_m: np.ndarray,
    source_depth_status: np.ndarray,
    v1_m_s: float,
    max_abs_shift_s: float | None,
) -> RefractionEndpointFieldCorrectionResult:
    keys = _coerce_1d_string(endpoint_key, name='endpoint_key')
    endpoint_count = int(keys.shape[0])
    ids = _coerce_1d_integer(endpoint_id, name='endpoint_id', expected_shape=(endpoint_count,))
    nodes = _coerce_1d_integer(node_id, name='node_id', expected_shape=(endpoint_count,))
    depth = _coerce_1d_float(
        source_depth_m,
        name='source_depth_m',
        expected_shape=(endpoint_count,),
    )
    depth_status = _coerce_1d_string_status(
        source_depth_status,
        name='source_depth_status',
        expected_shape=(endpoint_count,),
    )
    v1 = _positive_finite_float(v1_m_s, name='v1_m_s')
    max_shift = (
        None
        if max_abs_shift_s is None
        else _nonnegative_finite_float(max_abs_shift_s, name='max_abs_shift_s')
    )

    component_status = depth_status.astype(_FIELD_STATUS_DTYPE, copy=True)
    component_status[component_status == 'exceeds_max_abs_source_depth'] = (
        'exceeds_max_abs_source_depth_shift'
    )
    shift = np.full(endpoint_count, np.nan, dtype=np.float64)

    ok = component_status == 'ok'
    missing = ok & np.isnan(depth)
    component_status[missing] = 'missing_source_depth'
    invalid = ok & (~np.isnan(depth)) & (~np.isfinite(depth) | (depth < 0.0))
    component_status[invalid] = 'invalid_source_depth'
    ok = component_status == 'ok'
    shift[ok] = depth[ok] / v1
    if max_shift is not None:
        exceeds = ok & (np.abs(shift) > max_shift)
        component_status[exceeds] = 'exceeds_max_abs_source_depth_shift'
        shift[exceeds] = np.nan

    total = np.where(component_status == 'ok', shift, np.nan).astype(np.float64)
    zeros = np.zeros(endpoint_count, dtype=np.float64)
    inactive_status = np.full(
        endpoint_count,
        _NOT_APPLICABLE_STATUS,
        dtype=_FIELD_STATUS_DTYPE,
    )
    component_shift_s = {
        'source_depth_shift_s': np.ascontiguousarray(shift, dtype=np.float64),
        'uphole_shift_s': zeros.copy(),
        'manual_static_shift_s': zeros.copy(),
    }
    component_status_by_name = {
        'source_depth_shift_s': np.ascontiguousarray(
            component_status,
            dtype=_FIELD_STATUS_DTYPE,
        ),
        'uphole_shift_s': inactive_status.copy(),
        'manual_static_shift_s': inactive_status.copy(),
    }
    qc = _source_depth_shift_qc(
        source_depth_m=depth,
        shift_s=shift,
        status=component_status,
        v1_m_s=v1,
        max_abs_shift_s=max_shift,
        n_source_endpoints=endpoint_count,
    )
    return RefractionEndpointFieldCorrectionResult(
        endpoint_kind=np.full(endpoint_count, 'source', dtype='<U16'),
        endpoint_key=np.ascontiguousarray(keys, dtype=object),
        endpoint_id=np.ascontiguousarray(ids, dtype=np.int64),
        node_id=np.ascontiguousarray(nodes, dtype=np.int64),
        component_shift_s=component_shift_s,
        component_status=component_status_by_name,
        total_field_shift_s=np.ascontiguousarray(total, dtype=np.float64),
        field_static_status=np.ascontiguousarray(component_status, dtype=_FIELD_STATUS_DTYPE),
        qc=qc,
    )


def _source_depth_shift_qc(
    *,
    source_depth_m: np.ndarray,
    shift_s: np.ndarray,
    status: np.ndarray,
    v1_m_s: float,
    max_abs_shift_s: float | None,
    n_source_endpoints: int,
) -> dict[str, Any]:
    finite_shift = shift_s[np.isfinite(shift_s)]
    finite_depth = source_depth_m[np.isfinite(source_depth_m)]
    return {
        'source_depth_mode': 'weathering_velocity_time',
        'component_name': _SOURCE_DEPTH_COMPONENT,
        'component_names': list(REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES),
        'source_depth_shift_formula': _SOURCE_DEPTH_SHIFT_FORMULA,
        'sign_convention': _SIGN_CONVENTION,
        'positive_shift': 'event appears later in corrected data',
        'negative_shift': 'event appears earlier in corrected data',
        'v1_m_s': float(v1_m_s),
        'max_abs_shift_s': None if max_abs_shift_s is None else float(max_abs_shift_s),
        'n_source_endpoints': int(n_source_endpoints),
        'n_ok_source_depth_shifts': int(np.count_nonzero(status == 'ok')),
        'n_invalid_source_depth_shifts': int(np.count_nonzero(status != 'ok')),
        'status_counts': _status_counts(status),
        'min_source_depth_m': _finite_stat(finite_depth, 'min'),
        'median_source_depth_m': _finite_stat(finite_depth, 'median'),
        'max_source_depth_m': _finite_stat(finite_depth, 'max'),
        'min_source_depth_shift_s': _finite_stat(finite_shift, 'min'),
        'median_source_depth_shift_s': _finite_stat(finite_shift, 'median'),
        'max_source_depth_shift_s': _finite_stat(finite_shift, 'max'),
    }


def _source_endpoint_groups(keys: np.ndarray) -> tuple[dict[str, list[int]], np.ndarray]:
    groups: dict[str, list[int]] = {}
    first_positions: list[int] = []
    for index, raw_key in enumerate(keys.tolist()):
        key = str(raw_key)
        if not key:
            continue
        if key not in groups:
            groups[key] = []
            first_positions.append(index)
        groups[key].append(index)
    return groups, np.asarray(first_positions, dtype=np.int64)


def _endpoint_id_for_group(
    *,
    endpoint_ids: np.ndarray | None,
    trace_indices: np.ndarray,
    fallback: int,
) -> int:
    if endpoint_ids is None:
        return int(fallback)
    return int(endpoint_ids[int(trace_indices[0])])


def _node_id_for_group(node_values: np.ndarray) -> int:
    nonnegative = node_values[node_values >= 0]
    if nonnegative.size:
        return int(nonnegative[0])
    if node_values.size:
        return int(node_values[0])
    return -1


def _source_depth_status(
    *,
    node_id: int,
    values: np.ndarray,
    valid_values: np.ndarray,
    depth_array_present: bool,
    depth_required: bool,
    max_abs_source_depth_m: float,
    inconsistency_tolerance_m: float,
) -> str:
    if not depth_required:
        return 'ok'
    if node_id < 0:
        return 'inactive_source_endpoint'
    if not depth_array_present:
        return 'missing_source_depth'

    finite = np.isfinite(values)
    nonfinite_invalid = np.isinf(values)
    negative = finite & (values < 0.0)
    if bool(np.any(nonfinite_invalid | negative)):
        return 'invalid_source_depth'
    if bool(np.any(np.isnan(values))):
        return 'missing_source_depth'
    if valid_values.size == 0:
        return 'missing_source_depth'
    if bool(np.any(np.abs(valid_values) > max_abs_source_depth_m)):
        return 'exceeds_max_abs_source_depth'
    if (
        valid_values.size > 1
        and float(np.max(valid_values) - np.min(valid_values))
        > inconsistency_tolerance_m
    ):
        return 'inconsistent_source_depth'
    return 'ok'


def _source_depth_qc(
    *,
    mode: str,
    source_depth_byte: int | None,
    positive_down: bool,
    max_abs_source_depth_m: float,
    inconsistency_tolerance_m: float,
    source_depth_m: np.ndarray,
    status: np.ndarray,
    n_source_endpoints: int,
) -> dict[str, Any]:
    finite_depth = source_depth_m[np.isfinite(source_depth_m)]
    status_counts = {
        item: int(np.count_nonzero(status == item)) for item in _SOURCE_DEPTH_STATUSES
    }
    return {
        'source_depth_mode': mode,
        'source_depth_byte': None if source_depth_byte is None else int(source_depth_byte),
        'source_depth_positive_down': bool(positive_down),
        'sign_convention': _SIGN_CONVENTION,
        'max_abs_source_depth_m': float(max_abs_source_depth_m),
        'source_depth_inconsistency_tolerance_m': float(inconsistency_tolerance_m),
        'n_source_endpoints': int(n_source_endpoints),
        'n_sources_with_depth': int(finite_depth.shape[0]),
        'n_missing_source_depth': status_counts['missing_source_depth'],
        'n_invalid_source_depth': status_counts['invalid_source_depth'],
        'n_inconsistent_source_depth': status_counts['inconsistent_source_depth'],
        'n_exceeds_max_abs_source_depth': status_counts['exceeds_max_abs_source_depth'],
        'n_inactive_source_endpoints': status_counts['inactive_source_endpoint'],
        'min_source_depth_m': _finite_stat(finite_depth, 'min'),
        'median_source_depth_m': _finite_stat(finite_depth, 'median'),
        'max_source_depth_m': _finite_stat(finite_depth, 'max'),
        'status_counts': status_counts,
    }


def _coerce_mode(value: object) -> str:
    if value == 'none':
        return 'none'
    if value == 'weathering_velocity_time':
        return 'weathering_velocity_time'
    raise ValueError(f'unsupported source_depth mode: {value!r}')


def _coerce_1d_string(values: object, *, name: str) -> np.ndarray:
    return _coerce_1d_string_array(
        values,
        name=name,
        allow_non_string_dtype=True,
        output_dtype=object,
    )


def _coerce_1d_string_status(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    return _coerce_1d_string_array(
        values,
        name=name,
        expected_shape=expected_shape,
        allow_non_string_dtype=True,
        output_dtype=_FIELD_STATUS_DTYPE,
    )


def _coerce_1d_integer(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    return _coerce_1d_integer_int64(
        values,
        name=name,
        expected_shape=expected_shape,
        allow_integer_like_float=False,
    )


def _coerce_optional_1d_integer(
    values: object | None,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    if values is None:
        return None
    return _coerce_1d_integer(values, name=name, expected_shape=expected_shape)


def _coerce_1d_float(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    return _coerce_1d_real_numeric_float64(
        values,
        name=name,
        expected_shape=expected_shape,
    )


def _positive_finite_float(value: object, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out <= 0.0:
        raise ValueError(f'{name} must be positive')
    return out


def _nonnegative_finite_float(value: object, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out < 0.0:
        raise ValueError(f'{name} must be nonnegative')
    return out


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be a finite number')
    return _coerce_finite_float(value, name=name)


def _finite_stat(values: np.ndarray, stat: str) -> float | None:
    if values.size == 0:
        return None
    if stat == 'min':
        return float(np.min(values))
    if stat == 'median':
        return float(np.median(values))
    if stat == 'max':
        return float(np.max(values))
    raise ValueError(f'unsupported finite stat: {stat}')


def _status_counts(status: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw_status in status.tolist():
        item = str(raw_status)
        out[item] = int(out.get(item, 0) + 1)
    return out


__all__ = [
    'compute_source_depth_weathering_time_correction',
    'compute_source_depth_weathering_time_correction_from_result',
    'resolve_refraction_source_depth',
    'resolve_refraction_source_depth_for_input_model',
]
