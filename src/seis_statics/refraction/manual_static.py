"""Manual source/receiver static row normalization and endpoint matching."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, is_dataclass
from typing import Any, Literal, cast

import numpy as np

from seis_statics._validation import (
    coerce_1d_integer_int64 as _coerce_1d_integer_int64,
    coerce_1d_string_array as _coerce_1d_string_array,
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
)
from seis_statics.refraction.types import (
    REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES,
    RefractionManualStaticResult,
)


ManualStaticSignConvention = Literal['applied_shift_s', 'delay_positive_ms']

_STATUS_DTYPE = '<U48'
_ENDPOINT_KEY_DTYPE = object
_SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'
_MANUAL_STATIC_COMPONENT = 'manual_static_shift_s'
_VALID_ENDPOINT_KINDS = {'source', 'receiver'}
_VALID_SIGN_CONVENTIONS = {'applied_shift_s', 'delay_positive_ms'}


@dataclass(frozen=True)
class RefractionManualStaticTableRow:
    """One parsed manual-static row before endpoint matching."""

    endpoint_kind: str
    endpoint_key: str | None
    endpoint_id: int | None
    station_id: int | None
    node_id: int | None
    x_m: float | None
    y_m: float | None
    manual_static_input_s: float
    status: str
    comment: str | None
    source_name: str
    row_number: int


def normalize_refraction_manual_static_rows(
    rows: Iterable[object] | None,
    *,
    default_endpoint_kind: str | None = None,
    source_name: str = 'mapping_table',
) -> tuple[RefractionManualStaticTableRow, ...]:
    """Normalize parsed mapping/dataclass rows without filesystem access."""
    default_kind = _coerce_optional_endpoint_kind(default_endpoint_kind)
    normalized: list[RefractionManualStaticTableRow] = []
    for index, raw in enumerate(rows or (), start=1):
        normalized.append(
            _normalize_manual_static_row(
                raw,
                default_endpoint_kind=default_kind,
                default_source_name=source_name,
                default_row_number=index,
            )
        )
    return tuple(normalized)


def manual_static_inline_rows(
    *,
    endpoint_kind: Literal['source', 'receiver'],
    entries: Iterable[object] | None,
    sign_convention: ManualStaticSignConvention | str = 'applied_shift_s',
) -> tuple[RefractionManualStaticTableRow, ...]:
    """Convert inline endpoint/value entries to manual-static rows."""
    kind = _coerce_endpoint_kind(endpoint_kind)
    sign = _coerce_sign_convention(sign_convention)
    rows: list[RefractionManualStaticTableRow] = []
    for index, entry in enumerate(entries or (), start=1):
        endpoint_id = int(_field(entry, 'endpoint_id'))
        raw_value = float(_field(entry, 'value'))
        value_s = raw_value / 1000.0 if sign == 'delay_positive_ms' else raw_value
        status = 'ok' if np.isfinite(value_s) else 'invalid_manual_static_value'
        rows.append(
            RefractionManualStaticTableRow(
                endpoint_kind=kind,
                endpoint_key=None,
                endpoint_id=endpoint_id,
                station_id=None,
                node_id=None,
                x_m=None,
                y_m=None,
                manual_static_input_s=float(value_s),
                status=status,
                comment=None,
                source_name='inline_table',
                row_number=index,
            )
        )
    return tuple(rows)


def resolve_refraction_manual_static(
    *,
    source_endpoint_key: np.ndarray,
    source_endpoint_id: np.ndarray | None,
    source_node_id: np.ndarray,
    receiver_endpoint_key: np.ndarray,
    receiver_endpoint_id: np.ndarray | None,
    receiver_node_id: np.ndarray,
    rows: Iterable[object],
    mode: Literal['artifact_table', 'inline_table'],
    sign_convention: ManualStaticSignConvention | str | None,
    allow_missing_endpoints: bool = True,
    max_abs_manual_static_shift_s: float | None = None,
) -> RefractionManualStaticResult:
    """Match manual static rows to endpoint rows and convert to applied shifts."""
    sign = _coerce_sign_convention(sign_convention)
    mode_text = _coerce_mode(mode)
    max_abs_shift = (
        None
        if max_abs_manual_static_shift_s is None
        else _coerce_nonnegative_finite_float(
            max_abs_manual_static_shift_s,
            name='max_abs_manual_static_shift_s',
        )
    )
    manual_rows = normalize_refraction_manual_static_rows(rows)
    source_keys = _coerce_1d_string(source_endpoint_key, name='source_endpoint_key')
    receiver_keys = _coerce_1d_string(
        receiver_endpoint_key,
        name='receiver_endpoint_key',
    )
    source_count = int(source_keys.shape[0])
    receiver_count = int(receiver_keys.shape[0])
    source_ids = _coerce_optional_endpoint_ids(
        source_endpoint_id,
        endpoint_count=source_count,
        name='source_endpoint_id',
    )
    receiver_ids = _coerce_optional_endpoint_ids(
        receiver_endpoint_id,
        endpoint_count=receiver_count,
        name='receiver_endpoint_id',
    )
    source_nodes = _coerce_1d_integer(
        source_node_id,
        name='source_node_id',
        expected_shape=(source_count,),
    )
    receiver_nodes = _coerce_1d_integer(
        receiver_node_id,
        name='receiver_node_id',
        expected_shape=(receiver_count,),
    )

    source_shift = np.full(source_count, np.nan, dtype=np.float64)
    receiver_shift = np.full(receiver_count, np.nan, dtype=np.float64)
    source_status = np.full(
        source_count,
        'missing_manual_static',
        dtype=_STATUS_DTYPE,
    )
    receiver_status = np.full(
        receiver_count,
        'missing_manual_static',
        dtype=_STATUS_DTYPE,
    )
    endpoint_maps = _endpoint_maps(
        source_endpoint_key=source_keys,
        source_endpoint_id=source_ids,
        receiver_endpoint_key=receiver_keys,
        receiver_endpoint_id=receiver_ids,
    )

    matched_endpoint: set[tuple[str, int]] = set()
    duplicate_count = 0
    unmatched_count = 0
    invalid_value_count = 0
    exceeds_max_abs_count = 0
    matched_source_count = 0
    matched_receiver_count = 0
    row_status_counts: dict[str, int] = {}

    for row in manual_rows:
        match = _match_manual_static_row(row, endpoint_maps)
        row_status = row.status
        if match is None:
            row_status = 'unmatched_manual_static_row'
            unmatched_count += 1
            _increment(row_status_counts, row_status)
            continue

        endpoint_key = (match.endpoint_kind, match.endpoint_index)
        if endpoint_key in matched_endpoint:
            duplicate_count += 1
            row_status = 'duplicate_manual_static_row'
            _increment(row_status_counts, row_status)
            continue
        matched_endpoint.add(endpoint_key)

        if row.status != 'ok':
            shift_value = np.nan
            row_status = 'invalid_manual_static_value'
            invalid_value_count += 1
        else:
            shift_value = _manual_static_shift_s(
                row.manual_static_input_s,
                sign_convention=sign,
            )
            if not np.isfinite(shift_value):
                row_status = 'invalid_manual_static_value'
                invalid_value_count += 1
                shift_value = np.nan
            elif max_abs_shift is not None and abs(shift_value) > max_abs_shift:
                row_status = 'exceeds_max_abs_manual_static_shift'
                exceeds_max_abs_count += 1
                shift_value = np.nan

        if match.endpoint_kind == 'source':
            source_shift[match.endpoint_index] = shift_value
            source_status[match.endpoint_index] = row_status
            matched_source_count += 1
        else:
            receiver_shift[match.endpoint_index] = shift_value
            receiver_status[match.endpoint_index] = row_status
            matched_receiver_count += 1
        _increment(row_status_counts, row_status)

    if duplicate_count:
        raise ValueError(
            'duplicate_manual_static_row: manual static table contains duplicate '
            'rows for one or more endpoints'
        )

    source_missing_mask = source_status == 'missing_manual_static'
    receiver_missing_mask = receiver_status == 'missing_manual_static'
    n_missing_source = int(np.count_nonzero(source_missing_mask))
    n_missing_receiver = int(np.count_nonzero(receiver_missing_mask))
    if not bool(allow_missing_endpoints) and (n_missing_source or n_missing_receiver):
        raise ValueError(
            'missing_manual_static: manual static values are required for every '
            'source and receiver endpoint'
        )
    if bool(allow_missing_endpoints):
        source_shift[source_missing_mask] = 0.0
        receiver_shift[receiver_missing_mask] = 0.0

    qc = _manual_static_qc(
        mode=mode_text,
        sign_convention=sign,
        rows=manual_rows,
        matched_source_count=matched_source_count,
        matched_receiver_count=matched_receiver_count,
        unmatched_count=unmatched_count,
        duplicate_count=duplicate_count,
        invalid_value_count=invalid_value_count,
        exceeds_max_abs_count=exceeds_max_abs_count,
        row_status_counts=row_status_counts,
        source_status=source_status,
        receiver_status=receiver_status,
        source_shift=source_shift,
        receiver_shift=receiver_shift,
        allow_missing_endpoints=bool(allow_missing_endpoints),
        max_abs_manual_static_shift_s=max_abs_shift,
    )
    return RefractionManualStaticResult(
        source_endpoint_key=np.ascontiguousarray(source_keys, dtype=_ENDPOINT_KEY_DTYPE),
        source_endpoint_id=(
            None
            if source_ids is None
            else np.ascontiguousarray(source_ids, dtype=np.int64)
        ),
        source_node_id=np.ascontiguousarray(source_nodes, dtype=np.int64),
        source_manual_static_shift_s=np.ascontiguousarray(
            source_shift,
            dtype=np.float64,
        ),
        source_manual_static_status=np.ascontiguousarray(
            source_status,
            dtype=_STATUS_DTYPE,
        ),
        receiver_endpoint_key=np.ascontiguousarray(
            receiver_keys,
            dtype=_ENDPOINT_KEY_DTYPE,
        ),
        receiver_endpoint_id=(
            None
            if receiver_ids is None
            else np.ascontiguousarray(receiver_ids, dtype=np.int64)
        ),
        receiver_node_id=np.ascontiguousarray(receiver_nodes, dtype=np.int64),
        receiver_manual_static_shift_s=np.ascontiguousarray(
            receiver_shift,
            dtype=np.float64,
        ),
        receiver_manual_static_status=np.ascontiguousarray(
            receiver_status,
            dtype=_STATUS_DTYPE,
        ),
        qc=qc,
    )


def _normalize_manual_static_row(
    raw: object,
    *,
    default_endpoint_kind: str | None,
    default_source_name: str,
    default_row_number: int,
) -> RefractionManualStaticTableRow:
    if isinstance(raw, RefractionManualStaticTableRow):
        return raw
    endpoint_kind = _coerce_endpoint_kind(
        _first_nonblank(_maybe_field(raw, 'endpoint_kind'), default_endpoint_kind)
    )
    value_s, value_status = _manual_static_input_seconds(raw)
    status = _optional_str(_maybe_field(raw, 'status')) or value_status
    if not np.isfinite(value_s):
        status = 'invalid_manual_static_value'
    return RefractionManualStaticTableRow(
        endpoint_kind=endpoint_kind,
        endpoint_key=_optional_str(_maybe_field(raw, 'endpoint_key')),
        endpoint_id=_optional_int(_maybe_field(raw, 'endpoint_id'), name='endpoint_id'),
        station_id=_optional_int(_maybe_field(raw, 'station_id'), name='station_id'),
        node_id=_optional_int(_maybe_field(raw, 'node_id'), name='node_id'),
        x_m=_optional_float(_maybe_field(raw, 'x_m'), name='x_m'),
        y_m=_optional_float(_maybe_field(raw, 'y_m'), name='y_m'),
        manual_static_input_s=value_s,
        status=status,
        comment=_optional_str(_maybe_field(raw, 'comment')),
        source_name=(
            _optional_str(_maybe_field(raw, 'source_name')) or str(default_source_name)
        ),
        row_number=(
            _optional_int(_maybe_field(raw, 'row_number'), name='row_number')
            or int(default_row_number)
        ),
    )


def _manual_static_input_seconds(raw: object) -> tuple[float, str]:
    direct_value = _maybe_field(raw, 'manual_static_input_s')
    if direct_value is not None:
        try:
            value = float(direct_value)
        except (TypeError, ValueError):
            return np.nan, 'invalid_manual_static_value'
        return (
            (float(value), 'ok')
            if np.isfinite(value)
            else (np.nan, 'invalid_manual_static_value')
        )

    seconds_text = _optional_str(_maybe_field(raw, 'manual_static_s'))
    millis_text = _optional_str(_maybe_field(raw, 'manual_static_ms'))
    try:
        if seconds_text is not None:
            value = float(seconds_text)
        elif millis_text is not None:
            value = float(millis_text) / 1000.0
        else:
            return np.nan, 'invalid_manual_static_value'
    except ValueError:
        return np.nan, 'invalid_manual_static_value'
    if not np.isfinite(value):
        return np.nan, 'invalid_manual_static_value'
    return float(value), 'ok'


@dataclass(frozen=True)
class _EndpointMatch:
    endpoint_kind: Literal['source', 'receiver']
    endpoint_index: int


def _endpoint_maps(
    *,
    source_endpoint_key: np.ndarray,
    source_endpoint_id: np.ndarray | None,
    receiver_endpoint_key: np.ndarray,
    receiver_endpoint_id: np.ndarray | None,
) -> dict[str, dict[object, _EndpointMatch]]:
    maps: dict[str, dict[object, _EndpointMatch]] = {
        'source_key': {},
        'source_id': {},
        'receiver_key': {},
        'receiver_id': {},
    }
    for index, key in enumerate(source_endpoint_key.tolist()):
        maps['source_key'][str(key)] = _EndpointMatch('source', int(index))
    if source_endpoint_id is not None:
        for index, endpoint_id in enumerate(source_endpoint_id.tolist()):
            maps['source_id'][int(endpoint_id)] = _EndpointMatch('source', int(index))
    for index, key in enumerate(receiver_endpoint_key.tolist()):
        maps['receiver_key'][str(key)] = _EndpointMatch('receiver', int(index))
    if receiver_endpoint_id is not None:
        for index, endpoint_id in enumerate(receiver_endpoint_id.tolist()):
            maps['receiver_id'][int(endpoint_id)] = _EndpointMatch(
                'receiver',
                int(index),
            )
    return maps


def _match_manual_static_row(
    row: RefractionManualStaticTableRow,
    endpoint_maps: dict[str, dict[object, _EndpointMatch]],
) -> _EndpointMatch | None:
    if row.endpoint_key is not None:
        match = endpoint_maps[f'{row.endpoint_kind}_key'].get(row.endpoint_key)
        if match is not None:
            return match
    if row.endpoint_id is not None:
        return endpoint_maps[f'{row.endpoint_kind}_id'].get(row.endpoint_id)
    return None


def _manual_static_shift_s(
    value_s: float,
    *,
    sign_convention: ManualStaticSignConvention,
) -> float:
    if sign_convention == 'applied_shift_s':
        return float(value_s)
    if sign_convention == 'delay_positive_ms':
        return -float(value_s)
    raise ValueError(
        f'invalid_manual_static_sign_convention: {sign_convention!r}'
    )


def _manual_static_qc(
    *,
    mode: str,
    sign_convention: str,
    rows: tuple[RefractionManualStaticTableRow, ...],
    matched_source_count: int,
    matched_receiver_count: int,
    unmatched_count: int,
    duplicate_count: int,
    invalid_value_count: int,
    exceeds_max_abs_count: int,
    row_status_counts: dict[str, int],
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    source_shift: np.ndarray,
    receiver_shift: np.ndarray,
    allow_missing_endpoints: bool,
    max_abs_manual_static_shift_s: float | None,
) -> dict[str, Any]:
    source_rows = [row for row in rows if row.endpoint_kind == 'source']
    receiver_rows = [row for row in rows if row.endpoint_kind == 'receiver']
    finite_shift = np.concatenate(
        (
            source_shift[np.isfinite(source_shift)],
            receiver_shift[np.isfinite(receiver_shift)],
        )
    )
    source_status_counts = _status_counts(source_status)
    receiver_status_counts = _status_counts(receiver_status)
    return {
        'manual_static_mode': mode,
        'manual_static_sign_convention': sign_convention,
        'component_name': _MANUAL_STATIC_COMPONENT,
        'component_names': list(REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES),
        'manual_static_shift_formula': _manual_static_shift_formula(sign_convention),
        'sign_convention': _SIGN_CONVENTION,
        'positive_shift': 'event appears later in corrected data',
        'negative_shift': 'event appears earlier in corrected data',
        'allow_missing_endpoints': bool(allow_missing_endpoints),
        'max_abs_manual_static_shift_s': (
            None
            if max_abs_manual_static_shift_s is None
            else float(max_abs_manual_static_shift_s)
        ),
        'node_id_matching_enabled': False,
        'n_manual_source_rows': int(len(source_rows)),
        'n_manual_receiver_rows': int(len(receiver_rows)),
        'n_matched_source_rows': int(matched_source_count),
        'n_matched_receiver_rows': int(matched_receiver_count),
        'n_unmatched_rows': int(unmatched_count),
        'n_duplicate_rows': int(duplicate_count),
        'n_invalid_manual_static_values': int(invalid_value_count),
        'n_exceeds_max_abs_manual_static_shifts': int(exceeds_max_abs_count),
        'n_missing_source_endpoints': int(
            source_status_counts.get('missing_manual_static', 0)
        ),
        'n_missing_receiver_endpoints': int(
            receiver_status_counts.get('missing_manual_static', 0)
        ),
        'row_status_counts': dict(sorted(row_status_counts.items())),
        'source_status_counts': source_status_counts,
        'receiver_status_counts': receiver_status_counts,
        'min_manual_static_shift_s': _finite_stat(finite_shift, 'min'),
        'median_manual_static_shift_s': _finite_stat(finite_shift, 'median'),
        'max_manual_static_shift_s': _finite_stat(finite_shift, 'max'),
    }


def _manual_static_shift_formula(sign_convention: str) -> str:
    if sign_convention == 'applied_shift_s':
        return 'manual_static_shift_s = manual_static_s'
    if sign_convention == 'delay_positive_ms':
        return 'manual_static_shift_s = -manual_static_delay_s'
    raise ValueError(
        f'invalid_manual_static_sign_convention: {sign_convention!r}'
    )


def _coerce_mode(value: object) -> str:
    if value in {'artifact_table', 'inline_table'}:
        return str(value)
    raise ValueError(f'unsupported manual_static mode: {value!r}')


def _coerce_sign_convention(value: object) -> ManualStaticSignConvention:
    if value in _VALID_SIGN_CONVENTIONS:
        return cast(ManualStaticSignConvention, value)
    raise ValueError(
        'invalid_manual_static_sign_convention: manual static input must '
        'declare sign_convention applied_shift_s or delay_positive_ms'
    )


def _coerce_endpoint_kind(value: object) -> Literal['source', 'receiver']:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VALID_ENDPOINT_KINDS:
            return cast(Literal['source', 'receiver'], normalized)
    raise ValueError('endpoint_kind must be source or receiver')


def _coerce_optional_endpoint_kind(value: object) -> str | None:
    if value is None:
        return None
    return _coerce_endpoint_kind(value)


def _coerce_optional_endpoint_ids(
    values: object | None,
    *,
    endpoint_count: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    return _coerce_1d_integer(
        values,
        name=name,
        expected_shape=(endpoint_count,),
    )


def _coerce_1d_string(values: object, *, name: str) -> np.ndarray:
    return _coerce_1d_string_array(
        values,
        name=name,
        allow_non_string_dtype=True,
        output_dtype=object,
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


def _field(row: object, name: str) -> object:
    value = _maybe_field(row, name)
    if value is None:
        raise ValueError(f'{name} is required')
    return value


def _maybe_field(row: object, name: str) -> object | None:
    if isinstance(row, Mapping):
        return row.get(name)
    if is_dataclass(row) or hasattr(row, name):
        return getattr(row, name, None)
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _first_nonblank(*values: object) -> object | None:
    for value in values:
        if _optional_str(value) is not None:
            return value
    return None


def _optional_int(value: object, *, name: str) -> int | None:
    text = _optional_str(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer') from exc


def _optional_float(value: object, *, name: str) -> float | None:
    text = _optional_str(value)
    if text is None:
        return None
    try:
        out = float(text)
    except ValueError as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not np.isfinite(out):
        raise ValueError(f'{name} must be finite')
    return out


def _increment(counts: dict[str, int], status: str) -> None:
    counts[status] = int(counts.get(status, 0) + 1)


def _status_counts(status: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_status in status.tolist():
        item = str(raw_status)
        counts[item] = int(counts.get(item, 0) + 1)
    return dict(sorted(counts.items()))


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


__all__ = [
    'ManualStaticSignConvention',
    'RefractionManualStaticTableRow',
    'manual_static_inline_rows',
    'normalize_refraction_manual_static_rows',
    'resolve_refraction_manual_static',
]
