"""Compose refraction field-correction components into trace shifts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_1d_string_array as _coerce_1d_string_array,
    coerce_finite_float as _coerce_finite_float,
)
from seis_statics.refraction.types import (
    REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES,
    RefractionEndpointFieldCorrectionResult,
    RefractionFieldCorrectionComponentName,
    RefractionTraceFieldCorrectionResult,
)


RefractionFieldInvalidComponentPolicy = Literal['fail', 'skip_invalid_traces']

SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'

_STATUS_DTYPE = '<U64'
_ENDPOINT_KEY_DTYPE = object
_OK_STATUS = 'ok'
_NOT_APPLICABLE_STATUS = 'not_applicable'
_NOOP_STATUSES = {_OK_STATUS, _NOT_APPLICABLE_STATUS, 'not_enabled', 'none', ''}
_STATUS_PRIORITY = (
    'missing_manual_static',
    'missing_uphole_time',
    'missing_source_depth',
    'invalid_manual_static',
    'invalid_uphole_time',
    'invalid_source_depth',
    'inconsistent_uphole_time',
    'inconsistent_source_depth',
    'exceeds_max_abs_field_shift',
    'exceeds_max_abs_uphole_time',
    'exceeds_max_abs_source_depth',
    'inactive_source_endpoint',
    'missing_source_endpoint',
    'missing_receiver_endpoint',
    'invalid_field_shift',
)
_STATUS_NORMALIZATION = {
    'invalid_manual_static_value': 'invalid_manual_static',
    'exceeds_max_abs_source_depth_shift': 'exceeds_max_abs_source_depth',
}


class RefractionFieldCompositionError(ValueError):
    """Raised when field-correction composition cannot produce final shifts."""


@dataclass(frozen=True)
class RefractionFieldComposedTraceShiftResult:
    """Final trace-shift arrays after optional field-correction composition."""

    base_refraction_trace_shift_s_sorted: np.ndarray
    final_trace_shift_s_sorted: np.ndarray
    final_trace_static_status_sorted: np.ndarray
    final_trace_static_valid_mask_sorted: np.ndarray
    applied_field_shift_s_sorted: np.ndarray
    qc: dict[str, Any]


def compose_refraction_endpoint_field_corrections(
    *,
    endpoint_kind: Literal['source', 'receiver'],
    endpoint_key: np.ndarray,
    endpoint_id: np.ndarray | None,
    node_id: np.ndarray,
    source_depth_shift_s: np.ndarray | None = None,
    source_depth_status: np.ndarray | None = None,
    uphole_shift_s: np.ndarray | None = None,
    uphole_status: np.ndarray | None = None,
    manual_static_shift_s: np.ndarray | None = None,
    manual_static_status: np.ndarray | None = None,
    max_abs_field_shift_s: float | None = None,
) -> RefractionEndpointFieldCorrectionResult:
    """Sum endpoint field components into one endpoint-level field shift."""
    kind = _coerce_endpoint_kind(endpoint_kind)
    keys = _coerce_1d_string(endpoint_key, name=f'{kind}_endpoint_key')
    endpoint_count = int(keys.shape[0])
    ids = _endpoint_ids(endpoint_id, endpoint_count=endpoint_count)
    nodes = _coerce_1d_integer(
        node_id,
        name=f'{kind}_node_id',
        expected_shape=(endpoint_count,),
    )
    component_shift_s: dict[RefractionFieldCorrectionComponentName, np.ndarray] = {}
    component_status: dict[RefractionFieldCorrectionComponentName, np.ndarray] = {}
    for name, shift_values, status_values in (
        ('source_depth_shift_s', source_depth_shift_s, source_depth_status),
        ('uphole_shift_s', uphole_shift_s, uphole_status),
        ('manual_static_shift_s', manual_static_shift_s, manual_static_status),
    ):
        shift, status = _component_arrays(
            name=name,
            shift_s=shift_values,
            status=status_values,
            expected_shape=(endpoint_count,),
        )
        component_shift_s[name] = shift
        component_status[name] = status

    max_abs = _optional_nonnegative_finite_float(
        max_abs_field_shift_s,
        name='max_abs_field_shift_s',
    )
    total, field_status = _sum_endpoint_components(
        component_shift_s=component_shift_s,
        component_status=component_status,
        max_abs_field_shift_s=max_abs,
    )
    qc = _endpoint_qc(
        endpoint_kind=kind,
        total_field_shift_s=total,
        field_static_status=field_status,
        component_shift_s=component_shift_s,
        component_status=component_status,
        max_abs_field_shift_s=max_abs,
    )
    return RefractionEndpointFieldCorrectionResult(
        endpoint_kind=np.full(endpoint_count, kind, dtype='<U16'),
        endpoint_key=np.ascontiguousarray(keys, dtype=_ENDPOINT_KEY_DTYPE),
        endpoint_id=ids,
        node_id=nodes,
        component_shift_s=component_shift_s,
        component_status=component_status,
        total_field_shift_s=total,
        field_static_status=field_status,
        qc=qc,
    )


def compose_refraction_trace_field_corrections(
    *,
    source_endpoint_field: RefractionEndpointFieldCorrectionResult,
    receiver_endpoint_field: RefractionEndpointFieldCorrectionResult,
    source_endpoint_key_sorted: np.ndarray,
    receiver_endpoint_key_sorted: np.ndarray,
) -> RefractionTraceFieldCorrectionResult:
    """Map endpoint field shifts into sorted trace order."""
    source_keys = _coerce_1d_string(
        source_endpoint_key_sorted,
        name='source_endpoint_key_sorted',
    )
    receiver_keys = _coerce_1d_string(
        receiver_endpoint_key_sorted,
        name='receiver_endpoint_key_sorted',
        expected_shape=source_keys.shape,
    )
    source_shift, source_status = _map_endpoint_field_to_trace_order(
        endpoint_field=source_endpoint_field,
        endpoint_key_sorted=source_keys,
        missing_status='missing_source_endpoint',
    )
    receiver_shift, receiver_status = _map_endpoint_field_to_trace_order(
        endpoint_field=receiver_endpoint_field,
        endpoint_key_sorted=receiver_keys,
        missing_status='missing_receiver_endpoint',
    )
    trace_shift = np.full(source_keys.shape, np.nan, dtype=np.float64)
    trace_status = np.full(source_keys.shape, _OK_STATUS, dtype=_STATUS_DTYPE)
    for index in range(int(source_keys.shape[0])):
        status = _prioritized_status((source_status[index], receiver_status[index]))
        if status != _OK_STATUS:
            trace_status[index] = status
            continue
        source_value = source_shift[index]
        receiver_value = receiver_shift[index]
        if not np.isfinite(source_value) or not np.isfinite(receiver_value):
            trace_status[index] = 'invalid_field_shift'
            continue
        trace_shift[index] = float(source_value + receiver_value)
    qc = _trace_field_qc(
        source_field_shift_s_sorted=source_shift,
        receiver_field_shift_s_sorted=receiver_shift,
        trace_field_shift_s_sorted=trace_shift,
        trace_field_static_status_sorted=trace_status,
    )
    return RefractionTraceFieldCorrectionResult(
        source_endpoint_key_sorted=np.ascontiguousarray(
            source_keys,
            dtype=_ENDPOINT_KEY_DTYPE,
        ),
        receiver_endpoint_key_sorted=np.ascontiguousarray(
            receiver_keys,
            dtype=_ENDPOINT_KEY_DTYPE,
        ),
        source_field_shift_s_sorted=source_shift,
        receiver_field_shift_s_sorted=receiver_shift,
        trace_field_shift_s_sorted=np.ascontiguousarray(trace_shift, dtype=np.float64),
        trace_field_static_status_sorted=np.ascontiguousarray(
            trace_status,
            dtype=_STATUS_DTYPE,
        ),
        qc=qc,
    )


def compose_refraction_final_trace_shift(
    *,
    refraction_trace_shift_s_sorted: np.ndarray,
    trace_static_status_sorted: np.ndarray,
    trace_static_valid_mask_sorted: np.ndarray,
    trace_field_correction: RefractionTraceFieldCorrectionResult,
    apply_to_trace_shift: bool,
    invalid_component_policy: RefractionFieldInvalidComponentPolicy,
) -> RefractionFieldComposedTraceShiftResult:
    """Optionally add trace field shifts to the existing refraction shifts."""
    base = _coerce_1d_float(
        refraction_trace_shift_s_sorted,
        name='refraction_trace_shift_s_sorted',
    )
    n_traces = int(base.shape[0])
    base_status = _coerce_1d_string(
        trace_static_status_sorted,
        name='trace_static_status_sorted',
        expected_shape=(n_traces,),
    )
    base_valid = _coerce_1d_bool(
        trace_static_valid_mask_sorted,
        name='trace_static_valid_mask_sorted',
        expected_shape=(n_traces,),
    )
    field_shift = _coerce_1d_float(
        trace_field_correction.trace_field_shift_s_sorted,
        name='trace_field_shift_s_sorted',
        expected_shape=(n_traces,),
    )
    field_status = _coerce_1d_string(
        trace_field_correction.trace_field_static_status_sorted,
        name='trace_field_static_status_sorted',
        expected_shape=(n_traces,),
    )
    policy = _coerce_invalid_component_policy(invalid_component_policy)
    apply_field = bool(apply_to_trace_shift)
    field_valid = (field_status == _OK_STATUS) & np.isfinite(field_shift)
    invalid_field_count = int(np.count_nonzero(~field_valid))
    if apply_field and policy == 'fail' and invalid_field_count:
        raise RefractionFieldCompositionError(
            'invalid field-correction components prevent trace-shift '
            'composition; invalid_trace_field_shift_count='
            f'{invalid_field_count}; trace_field_static_status_counts='
            f'{_status_counts(field_status)}'
        )

    final = np.ascontiguousarray(base.copy(), dtype=np.float64)
    final_status = np.ascontiguousarray(base_status.copy(), dtype=_STATUS_DTYPE)
    final_valid = np.ascontiguousarray(base_valid.copy(), dtype=bool)
    applied_field = np.zeros(n_traces, dtype=np.float64)
    if apply_field:
        add_mask = base_valid & np.isfinite(base) & field_valid
        final[add_mask] = base[add_mask] + field_shift[add_mask]
        applied_field[add_mask] = field_shift[add_mask]
        invalid_final = add_mask & ~np.isfinite(final)
        if bool(np.any(invalid_final)):
            final_status[invalid_final] = 'invalid_field_shift'
            final_valid[invalid_final] = False

    qc = _final_trace_qc(
        base_refraction_trace_shift_s_sorted=base,
        field_shift_s_sorted=field_shift,
        applied_field_shift_s_sorted=applied_field,
        final_trace_shift_s_sorted=final,
        trace_field_static_status_sorted=field_status,
        apply_to_trace_shift=apply_field,
        invalid_component_policy=policy,
        invalid_trace_field_shift_count=invalid_field_count,
    )
    return RefractionFieldComposedTraceShiftResult(
        base_refraction_trace_shift_s_sorted=np.ascontiguousarray(
            base,
            dtype=np.float64,
        ),
        final_trace_shift_s_sorted=final,
        final_trace_static_status_sorted=final_status,
        final_trace_static_valid_mask_sorted=final_valid,
        applied_field_shift_s_sorted=np.ascontiguousarray(
            applied_field,
            dtype=np.float64,
        ),
        qc=qc,
    )


def _sum_endpoint_components(
    *,
    component_shift_s: dict[RefractionFieldCorrectionComponentName, np.ndarray],
    component_status: dict[RefractionFieldCorrectionComponentName, np.ndarray],
    max_abs_field_shift_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    endpoint_count = int(next(iter(component_shift_s.values())).shape[0])
    total = np.zeros(endpoint_count, dtype=np.float64)
    status = np.full(endpoint_count, _OK_STATUS, dtype=_STATUS_DTYPE)
    for index in range(endpoint_count):
        raw_statuses = [
            _component_value_status(
                component_name=component,
                shift_s=float(component_shift_s[component][index]),
                status=str(component_status[component][index]),
            )
            for component in REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES
        ]
        field_status = _prioritized_status(raw_statuses)
        if field_status != _OK_STATUS:
            total[index] = np.nan
            status[index] = field_status
            continue
        total[index] = float(
            sum(
                component_shift_s[component][index]
                for component in REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES
                if str(component_status[component][index]) == _OK_STATUS
            )
        )
        if (
            max_abs_field_shift_s is not None
            and abs(total[index]) > max_abs_field_shift_s
        ):
            total[index] = np.nan
            status[index] = 'exceeds_max_abs_field_shift'
    return (
        np.ascontiguousarray(total, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )


def _component_value_status(
    *,
    component_name: str,
    shift_s: float,
    status: str,
) -> str:
    normalized = _normalize_status(status)
    if (
        component_name == 'manual_static_shift_s'
        and normalized == 'missing_manual_static'
        and np.isfinite(shift_s)
        and shift_s == 0.0
    ):
        return _OK_STATUS
    if normalized not in _NOOP_STATUSES:
        return normalized
    if normalized == _OK_STATUS and not np.isfinite(shift_s):
        if component_name == 'manual_static_shift_s':
            return 'invalid_manual_static'
        if component_name == 'uphole_shift_s':
            return 'invalid_uphole_time'
        if component_name == 'source_depth_shift_s':
            return 'invalid_source_depth'
        return 'invalid_field_shift'
    return _OK_STATUS


def _map_endpoint_field_to_trace_order(
    *,
    endpoint_field: RefractionEndpointFieldCorrectionResult,
    endpoint_key_sorted: np.ndarray,
    missing_status: str,
) -> tuple[np.ndarray, np.ndarray]:
    endpoint_keys = _coerce_1d_string(
        endpoint_field.endpoint_key,
        name='endpoint_field.endpoint_key',
    )
    endpoint_shift = _coerce_1d_float(
        endpoint_field.total_field_shift_s,
        name='endpoint_field.total_field_shift_s',
        expected_shape=endpoint_keys.shape,
    )
    endpoint_status = _coerce_1d_string(
        endpoint_field.field_static_status,
        name='endpoint_field.field_static_status',
        expected_shape=endpoint_keys.shape,
    )
    shift_by_key = {
        str(key): float(endpoint_shift[index])
        for index, key in enumerate(endpoint_keys.tolist())
    }
    status_by_key = {
        str(key): str(endpoint_status[index])
        for index, key in enumerate(endpoint_keys.tolist())
    }
    trace_count = int(endpoint_key_sorted.shape[0])
    shift = np.full(trace_count, np.nan, dtype=np.float64)
    status = np.full(trace_count, missing_status, dtype=_STATUS_DTYPE)
    for index, raw_key in enumerate(endpoint_key_sorted.tolist()):
        key = str(raw_key)
        if key not in shift_by_key:
            continue
        shift[index] = shift_by_key[key]
        status[index] = _normalize_status(status_by_key[key])
    return (
        np.ascontiguousarray(shift, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )


def _component_arrays(
    *,
    name: str,
    shift_s: np.ndarray | None,
    status: np.ndarray | None,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if shift_s is None and status is None:
        return (
            np.zeros(expected_shape, dtype=np.float64),
            np.full(expected_shape, _NOT_APPLICABLE_STATUS, dtype=_STATUS_DTYPE),
        )
    if shift_s is None or status is None:
        raise ValueError(f'{name} shift and status arrays must be supplied together')
    return (
        _coerce_1d_float(shift_s, name=name, expected_shape=expected_shape),
        _coerce_1d_status(
            status,
            name=f'{name}_status',
            expected_shape=expected_shape,
        ),
    )


def _endpoint_qc(
    *,
    endpoint_kind: str,
    total_field_shift_s: np.ndarray,
    field_static_status: np.ndarray,
    component_shift_s: dict[RefractionFieldCorrectionComponentName, np.ndarray],
    component_status: dict[RefractionFieldCorrectionComponentName, np.ndarray],
    max_abs_field_shift_s: float | None,
) -> dict[str, Any]:
    finite_total = total_field_shift_s[np.isfinite(total_field_shift_s)]
    return {
        'endpoint_kind': endpoint_kind,
        'component_names': list(REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES),
        'sign_convention': SIGN_CONVENTION,
        'field_shift_formula': (
            'total_field_shift_s = source_depth_shift_s + uphole_shift_s + '
            'manual_static_shift_s'
        ),
        'max_abs_field_shift_s': (
            None if max_abs_field_shift_s is None else float(max_abs_field_shift_s)
        ),
        'n_endpoints': int(total_field_shift_s.shape[0]),
        'n_ok_field_shifts': int(np.count_nonzero(field_static_status == _OK_STATUS)),
        'n_invalid_field_shifts': int(
            np.count_nonzero(field_static_status != _OK_STATUS)
        ),
        'field_static_status_counts': _status_counts(field_static_status),
        'component_status_counts': {
            component: _status_counts(component_status[component])
            for component in REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES
        },
        'component_shift_summary_s': {
            component: _shift_summary(component_shift_s[component])
            for component in REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES
        },
        'total_field_shift_summary_s': _shift_summary(finite_total),
    }


def _trace_field_qc(
    *,
    source_field_shift_s_sorted: np.ndarray,
    receiver_field_shift_s_sorted: np.ndarray,
    trace_field_shift_s_sorted: np.ndarray,
    trace_field_static_status_sorted: np.ndarray,
) -> dict[str, Any]:
    valid = trace_field_static_status_sorted == _OK_STATUS
    return {
        'sign_convention': SIGN_CONVENTION,
        'trace_field_shift_formula': (
            'trace_field_shift_s = source_field_shift_s + '
            'receiver_field_shift_s'
        ),
        'n_traces': int(trace_field_shift_s_sorted.shape[0]),
        'n_ok_trace_field_shifts': int(np.count_nonzero(valid)),
        'n_invalid_trace_field_shifts': int(np.count_nonzero(~valid)),
        'trace_field_static_status_counts': _status_counts(
            trace_field_static_status_sorted
        ),
        'source_field_shift_summary_s': _shift_summary(source_field_shift_s_sorted),
        'receiver_field_shift_summary_s': _shift_summary(
            receiver_field_shift_s_sorted
        ),
        'trace_field_shift_summary_s': _shift_summary(trace_field_shift_s_sorted),
    }


def _final_trace_qc(
    *,
    base_refraction_trace_shift_s_sorted: np.ndarray,
    field_shift_s_sorted: np.ndarray,
    applied_field_shift_s_sorted: np.ndarray,
    final_trace_shift_s_sorted: np.ndarray,
    trace_field_static_status_sorted: np.ndarray,
    apply_to_trace_shift: bool,
    invalid_component_policy: str,
    invalid_trace_field_shift_count: int,
) -> dict[str, Any]:
    return {
        'composition_enabled': True,
        'apply_to_trace_shift': bool(apply_to_trace_shift),
        'invalid_component_policy': invalid_component_policy,
        'sign_convention': SIGN_CONVENTION,
        'final_trace_shift_formula': _final_trace_shift_formula(
            apply_to_trace_shift=apply_to_trace_shift
        ),
        'n_traces': int(final_trace_shift_s_sorted.shape[0]),
        'invalid_trace_field_shift_count': int(invalid_trace_field_shift_count),
        'trace_field_static_status_counts': _status_counts(
            trace_field_static_status_sorted
        ),
        'base_refraction_trace_shift_summary_s': _shift_summary(
            base_refraction_trace_shift_s_sorted
        ),
        'trace_field_shift_summary_s': _shift_summary(field_shift_s_sorted),
        'applied_field_shift_summary_s': _shift_summary(applied_field_shift_s_sorted),
        'final_trace_shift_summary_s': _shift_summary(final_trace_shift_s_sorted),
    }


def _final_trace_shift_formula(*, apply_to_trace_shift: bool) -> str:
    if apply_to_trace_shift:
        return (
            'final_trace_shift_s = refraction_trace_shift_s + '
            'trace_field_shift_s'
        )
    return 'final_trace_shift_s = refraction_trace_shift_s'


def _prioritized_status(statuses: list[str] | tuple[str, ...]) -> str:
    normalized = [
        _normalize_status(status)
        for status in statuses
        if _normalize_status(status) not in _NOOP_STATUSES
    ]
    if not normalized:
        return _OK_STATUS
    for candidate in _STATUS_PRIORITY:
        if candidate in normalized:
            return candidate
    return normalized[0]


def _normalize_status(status: object) -> str:
    text = str(status)
    return _STATUS_NORMALIZATION.get(text, text)


def _shift_summary(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {'count': 0, 'min': None, 'median': None, 'max': None}
    return {
        'count': int(finite.shape[0]),
        'min': float(np.min(finite)),
        'median': float(np.median(finite)),
        'max': float(np.max(finite)),
    }


def _status_counts(values: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw in np.asarray(values).tolist():
        status = str(raw)
        counts[status] = int(counts.get(status, 0) + 1)
    return counts


def _coerce_endpoint_kind(value: object) -> Literal['source', 'receiver']:
    if value == 'source':
        return 'source'
    if value == 'receiver':
        return 'receiver'
    raise ValueError(f'endpoint_kind must be source or receiver, got {value!r}')


def _coerce_invalid_component_policy(
    value: object,
) -> RefractionFieldInvalidComponentPolicy:
    if value in {'fail', 'skip_invalid_traces'}:
        return value  # type: ignore[return-value]
    raise ValueError(f'unsupported invalid_component_policy: {value!r}')


def _endpoint_ids(
    values: np.ndarray | None,
    *,
    endpoint_count: int,
) -> np.ndarray:
    if values is None:
        return np.arange(endpoint_count, dtype=np.int64)
    return _coerce_1d_integer(
        values,
        name='endpoint_id',
        expected_shape=(endpoint_count,),
    )


def _coerce_1d_string(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    return _coerce_1d_string_array(
        values,
        name=name,
        expected_shape=expected_shape,
        allow_non_string_dtype=True,
        output_dtype=_ENDPOINT_KEY_DTYPE,
    )


def _coerce_1d_status(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    return _coerce_1d_string_array(
        values,
        name=name,
        expected_shape=expected_shape,
        allow_non_string_dtype=True,
        output_dtype=_STATUS_DTYPE,
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


def _coerce_1d_float(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    return _coerce_1d_real_numeric_float64(
        values,
        name=name,
        expected_shape=expected_shape,
    )


def _coerce_1d_bool(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    return _coerce_1d_bool_array(
        values,
        name=name,
        expected_shape=expected_shape,
    )


def _optional_nonnegative_finite_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be a finite number')
    out = _coerce_finite_float(value, name=name)
    if out < 0.0:
        raise ValueError(f'{name} must be nonnegative')
    return out


__all__ = [
    'RefractionFieldComposedTraceShiftResult',
    'RefractionFieldCompositionError',
    'RefractionFieldInvalidComponentPolicy',
    'SIGN_CONVENTION',
    'compose_refraction_endpoint_field_corrections',
    'compose_refraction_final_trace_shift',
    'compose_refraction_trace_field_corrections',
]
