"""Pure refraction datum static composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from seis_statics._validation import (
    coerce_1d_integer_int64 as _coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_1d_string_array as _coerce_1d_string_array,
    coerce_finite_float as _coerce_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics.refraction.field_composition import (
    RefractionFieldInvalidComponentPolicy,
    compose_refraction_final_trace_shift,
)
from seis_statics.refraction.options import RefractionStaticDatumMode
from seis_statics.refraction.types import RefractionTraceFieldCorrectionResult


_STATUS_DTYPE = '<U64'
_ENDPOINT_KEY_DTYPE = object
_SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'
_ELEVATION_FORMULA = 'shift_s = (datum_elevation_m - elevation_m) / replacement_velocity_m_s'
_OK_STATUS = 'ok'
_NOOP_STATUSES = {_OK_STATUS, 'solved', 'zero_thickness'}
_STATUS_PRIORITY = {
    'ok': 0,
    'solved': 0,
    'zero_thickness': 0,
    'not_observed': 1,
    'clipped_half_intercept_lower': 2,
    'clipped_half_intercept_upper': 3,
    'low_fold': 4,
    'exceeds_max_thickness': 5,
    'exceeds_max_abs_shift': 6,
    'invalid_shift': 7,
    'invalid_datum_shift': 8,
    'invalid_floating_datum_elevation': 9,
    'invalid_flat_datum_elevation': 10,
    'floating_datum_below_refractor': 11,
    'flat_datum_below_refractor': 12,
    'negative_weathering_thickness': 13,
    'invalid_weathering_thickness': 14,
    'inactive': 15,
    'invalid_velocity': 16,
    'invalid_nonfinite_input': 17,
    'invalid_velocity_order': 18,
    'invalid_surface_elevation': 19,
    'invalid_refractor_elevation': 20,
    'outside_refractor_cell_grid': 21,
    'inactive_v2_cell': 22,
    'low_fold_v2_cell': 23,
    'invalid_local_v2': 24,
    'v2_not_greater_than_v1': 25,
    'missing_endpoint': 26,
    'missing_node': 27,
}


class RefractionDatumError(ValueError):
    """Raised when refraction datum statics cannot be composed."""


@dataclass(frozen=True)
class ResolvedFloatingDatum:
    """Resolved floating datum elevations supplied by a caller."""

    source_elevation_m: np.ndarray | float
    receiver_elevation_m: np.ndarray | float


@dataclass(frozen=True)
class RefractionDatumEndpointResult:
    """Endpoint-level datum and replacement static composition."""

    endpoint_kind: np.ndarray
    endpoint_key: np.ndarray
    endpoint_id: np.ndarray | None
    node_id: np.ndarray
    surface_elevation_m: np.ndarray
    refractor_elevation_m: np.ndarray
    floating_datum_elevation_m: np.ndarray
    flat_datum_elevation_m: np.ndarray
    weathering_replacement_shift_s: np.ndarray
    floating_datum_shift_s: np.ndarray
    flat_datum_shift_s: np.ndarray
    total_refraction_shift_s: np.ndarray
    datum_static_status: np.ndarray
    datum_static_valid_mask: np.ndarray
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionDatumStaticsResult:
    """Trace-order refraction datum static composition result."""

    mode: RefractionStaticDatumMode
    replacement_velocity_m_s: float
    source_endpoint_datum: RefractionDatumEndpointResult
    receiver_endpoint_datum: RefractionDatumEndpointResult
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_refraction_shift_s_sorted: np.ndarray
    receiver_refraction_shift_s_sorted: np.ndarray
    source_floating_datum_shift_s_sorted: np.ndarray
    receiver_floating_datum_shift_s_sorted: np.ndarray
    source_flat_datum_shift_s_sorted: np.ndarray
    receiver_flat_datum_shift_s_sorted: np.ndarray
    refraction_trace_shift_s_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    applied_field_shift_s_sorted: np.ndarray
    final_trace_shift_s_sorted: np.ndarray
    final_trace_static_status_sorted: np.ndarray
    final_trace_static_valid_mask_sorted: np.ndarray
    qc: dict[str, Any]


def compute_refraction_datum_elevation_shift_scalar_s(
    *,
    elevation_m: float,
    datum_elevation_m: float,
    replacement_velocity_m_s: float,
) -> float:
    """Compute one datum elevation shift using the package sign convention."""
    elevation = _coerce_finite_float(elevation_m, name='elevation_m')
    datum = _coerce_finite_float(datum_elevation_m, name='datum_elevation_m')
    velocity = _coerce_positive_finite_float(
        replacement_velocity_m_s,
        name='replacement_velocity_m_s',
    )
    return float((datum - elevation) / velocity)


def compute_refraction_datum_elevation_shift_s(
    *,
    elevation_m: np.ndarray,
    datum_elevation_m: np.ndarray | float,
    replacement_velocity_m_s: float,
) -> np.ndarray:
    """Compute datum elevation shifts for a 1-D endpoint array."""
    elevation = _coerce_1d_float(elevation_m, name='elevation_m')
    datum = _coerce_datum_elevation(
        datum_elevation_m,
        name='datum_elevation_m',
        expected_shape=elevation.shape,
    )
    velocity = _coerce_positive_finite_float(
        replacement_velocity_m_s,
        name='replacement_velocity_m_s',
    )
    out = (datum - elevation) / velocity
    return np.ascontiguousarray(out, dtype=np.float64)


def smooth_refraction_floating_datum_elevation(
    elevation_m: np.ndarray,
    *,
    window_nodes: int,
    method: Literal['moving_average', 'median'] = 'moving_average',
) -> np.ndarray:
    """Smooth a 1-D resolved floating datum elevation array."""
    values = _coerce_1d_float(elevation_m, name='elevation_m')
    window = _coerce_positive_int(window_nodes, name='window_nodes')
    if window % 2 == 0:
        raise RefractionDatumError('window_nodes must be odd')
    if method not in {'moving_average', 'median'}:
        raise RefractionDatumError(
            "method must be 'moving_average' or 'median'"
        )
    if window == 1 or values.size == 0:
        return np.ascontiguousarray(values.copy(), dtype=np.float64)

    radius = window // 2
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(int(values.shape[0])):
        start = max(0, index - radius)
        stop = min(int(values.shape[0]), index + radius + 1)
        sample = values[start:stop]
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            continue
        if method == 'moving_average':
            out[index] = float(np.mean(sample))
        else:
            out[index] = float(np.median(sample))
    return np.ascontiguousarray(out, dtype=np.float64)


def build_refraction_endpoint_datum_statics(
    *,
    endpoint_kind: Literal['source', 'receiver'],
    endpoint_key: np.ndarray,
    endpoint_id: np.ndarray | None,
    node_id: np.ndarray,
    surface_elevation_m: np.ndarray,
    refractor_elevation_m: np.ndarray,
    weathering_replacement_shift_s: np.ndarray,
    weathering_replacement_status: np.ndarray,
    mode: RefractionStaticDatumMode,
    replacement_velocity_m_s: float,
    floating_datum_elevation_m: np.ndarray | float | None = None,
    flat_datum_elevation_m: np.ndarray | float | None = None,
    allow_flat_datum_above_topography: bool = True,
    allow_flat_datum_below_refractor: bool = False,
    max_abs_datum_shift_ms: float | None = None,
) -> RefractionDatumEndpointResult:
    """Compose replacement, floating datum, and flat datum shifts per endpoint."""
    kind = _coerce_endpoint_kind(endpoint_kind)
    datum_mode = _coerce_mode(mode)
    velocity = _coerce_positive_finite_float(
        replacement_velocity_m_s,
        name='replacement_velocity_m_s',
    )
    max_abs_shift_s = _optional_max_abs_shift_s(max_abs_datum_shift_ms)
    keys = _coerce_1d_string(endpoint_key, name=f'{kind}_endpoint_key')
    endpoint_count = int(keys.shape[0])
    ids = _coerce_optional_endpoint_ids(
        endpoint_id,
        endpoint_count=endpoint_count,
        name=f'{kind}_endpoint_id',
    )
    nodes = _coerce_1d_integer(
        node_id,
        name=f'{kind}_node_id',
        expected_shape=(endpoint_count,),
    )
    surface = _coerce_1d_float(
        surface_elevation_m,
        name=f'{kind}_surface_elevation_m',
        expected_shape=(endpoint_count,),
    )
    refractor = _coerce_1d_float(
        refractor_elevation_m,
        name=f'{kind}_refractor_elevation_m',
        expected_shape=(endpoint_count,),
    )
    replacement = _coerce_1d_float(
        weathering_replacement_shift_s,
        name=f'{kind}_weathering_replacement_shift_s',
        expected_shape=(endpoint_count,),
    )
    replacement_status = _coerce_1d_status(
        weathering_replacement_status,
        name=f'{kind}_weathering_replacement_status',
        expected_shape=(endpoint_count,),
    )
    floating = _resolved_datum(
        floating_datum_elevation_m,
        name=f'{kind}_floating_datum_elevation_m',
        expected_shape=(endpoint_count,),
        required=datum_mode in {'floating_only', 'floating_and_flat'},
    )
    flat = _resolved_datum(
        flat_datum_elevation_m,
        name=f'{kind}_flat_datum_elevation_m',
        expected_shape=(endpoint_count,),
        required=datum_mode in {'flat_only', 'floating_and_flat'},
    )

    floating_shift = np.zeros(endpoint_count, dtype=np.float64)
    flat_shift = np.zeros(endpoint_count, dtype=np.float64)
    total = np.full(endpoint_count, np.nan, dtype=np.float64)
    status = np.full(endpoint_count, _OK_STATUS, dtype=_STATUS_DTYPE)

    for index in range(endpoint_count):
        status[index] = _endpoint_status(
            replacement_status=str(replacement_status[index]),
            surface_elevation_m=float(surface[index]),
            refractor_elevation_m=float(refractor[index]),
            replacement_shift_s=float(replacement[index]),
            mode=datum_mode,
            floating_datum_elevation_m=float(floating[index]),
            flat_datum_elevation_m=float(flat[index]),
            allow_flat_datum_above_topography=bool(allow_flat_datum_above_topography),
            allow_flat_datum_below_refractor=bool(allow_flat_datum_below_refractor),
        )
        if status[index] != _OK_STATUS:
            continue

        if datum_mode in {'floating_only', 'floating_and_flat'}:
            floating_shift[index] = compute_refraction_datum_elevation_shift_scalar_s(
                elevation_m=float(surface[index]),
                datum_elevation_m=float(floating[index]),
                replacement_velocity_m_s=velocity,
            )
        if datum_mode == 'flat_only':
            flat_elevation_reference = float(surface[index])
        else:
            flat_elevation_reference = float(floating[index])
        if datum_mode in {'flat_only', 'floating_and_flat'}:
            flat_shift[index] = compute_refraction_datum_elevation_shift_scalar_s(
                elevation_m=flat_elevation_reference,
                datum_elevation_m=float(flat[index]),
                replacement_velocity_m_s=velocity,
            )

        endpoint_total = (
            float(replacement[index])
            + float(floating_shift[index])
            + float(flat_shift[index])
        )
        if not np.isfinite(endpoint_total):
            status[index] = 'invalid_datum_shift'
            continue
        if max_abs_shift_s is not None and abs(endpoint_total) > max_abs_shift_s:
            status[index] = 'invalid_datum_shift'
            continue
        total[index] = endpoint_total

    valid = (status == _OK_STATUS) & np.isfinite(total)
    qc = _endpoint_qc(
        endpoint_kind=kind,
        mode=datum_mode,
        replacement_velocity_m_s=velocity,
        surface_elevation_m=surface,
        refractor_elevation_m=refractor,
        replacement_shift_s=replacement,
        floating_shift_s=floating_shift,
        flat_shift_s=flat_shift,
        total_shift_s=total,
        status=status,
        max_abs_datum_shift_ms=max_abs_datum_shift_ms,
    )
    return RefractionDatumEndpointResult(
        endpoint_kind=np.full(endpoint_count, kind, dtype='<U16'),
        endpoint_key=np.ascontiguousarray(keys, dtype=_ENDPOINT_KEY_DTYPE),
        endpoint_id=ids,
        node_id=nodes,
        surface_elevation_m=np.ascontiguousarray(surface, dtype=np.float64),
        refractor_elevation_m=np.ascontiguousarray(refractor, dtype=np.float64),
        floating_datum_elevation_m=np.ascontiguousarray(floating, dtype=np.float64),
        flat_datum_elevation_m=np.ascontiguousarray(flat, dtype=np.float64),
        weathering_replacement_shift_s=np.ascontiguousarray(
            replacement,
            dtype=np.float64,
        ),
        floating_datum_shift_s=np.ascontiguousarray(
            floating_shift,
            dtype=np.float64,
        ),
        flat_datum_shift_s=np.ascontiguousarray(flat_shift, dtype=np.float64),
        total_refraction_shift_s=np.ascontiguousarray(total, dtype=np.float64),
        datum_static_status=np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
        datum_static_valid_mask=np.ascontiguousarray(valid, dtype=bool),
        qc=qc,
    )


def build_refraction_datum_statics(
    *,
    source_endpoint_key: np.ndarray,
    source_endpoint_id: np.ndarray | None,
    source_node_id: np.ndarray,
    source_surface_elevation_m: np.ndarray,
    source_refractor_elevation_m: np.ndarray,
    source_weathering_replacement_shift_s: np.ndarray,
    source_weathering_replacement_status: np.ndarray,
    receiver_endpoint_key: np.ndarray,
    receiver_endpoint_id: np.ndarray | None,
    receiver_node_id: np.ndarray,
    receiver_surface_elevation_m: np.ndarray,
    receiver_refractor_elevation_m: np.ndarray,
    receiver_weathering_replacement_shift_s: np.ndarray,
    receiver_weathering_replacement_status: np.ndarray,
    source_endpoint_key_sorted: np.ndarray,
    receiver_endpoint_key_sorted: np.ndarray,
    mode: RefractionStaticDatumMode,
    replacement_velocity_m_s: float,
    floating_datum: ResolvedFloatingDatum | None = None,
    source_floating_datum_elevation_m: np.ndarray | float | None = None,
    receiver_floating_datum_elevation_m: np.ndarray | float | None = None,
    flat_datum_elevation_m: np.ndarray | float | None = None,
    source_flat_datum_elevation_m: np.ndarray | float | None = None,
    receiver_flat_datum_elevation_m: np.ndarray | float | None = None,
    allow_flat_datum_above_topography: bool = True,
    allow_flat_datum_below_refractor: bool = False,
    max_abs_datum_shift_ms: float | None = None,
    trace_field_correction: RefractionTraceFieldCorrectionResult | None = None,
    apply_field_correction_to_trace_shift: bool = False,
    invalid_field_component_policy: RefractionFieldInvalidComponentPolicy = 'fail',
) -> RefractionDatumStaticsResult:
    """Build sorted endpoint/trace final shifts from resolved datum arrays."""
    datum_mode = _coerce_mode(mode)
    source_floating, receiver_floating = _floating_inputs(
        floating_datum=floating_datum,
        source_value=source_floating_datum_elevation_m,
        receiver_value=receiver_floating_datum_elevation_m,
    )
    source_flat = (
        flat_datum_elevation_m
        if source_flat_datum_elevation_m is None
        else source_flat_datum_elevation_m
    )
    receiver_flat = (
        flat_datum_elevation_m
        if receiver_flat_datum_elevation_m is None
        else receiver_flat_datum_elevation_m
    )
    velocity = _coerce_positive_finite_float(
        replacement_velocity_m_s,
        name='replacement_velocity_m_s',
    )

    source = build_refraction_endpoint_datum_statics(
        endpoint_kind='source',
        endpoint_key=source_endpoint_key,
        endpoint_id=source_endpoint_id,
        node_id=source_node_id,
        surface_elevation_m=source_surface_elevation_m,
        refractor_elevation_m=source_refractor_elevation_m,
        weathering_replacement_shift_s=source_weathering_replacement_shift_s,
        weathering_replacement_status=source_weathering_replacement_status,
        mode=datum_mode,
        replacement_velocity_m_s=velocity,
        floating_datum_elevation_m=source_floating,
        flat_datum_elevation_m=source_flat,
        allow_flat_datum_above_topography=allow_flat_datum_above_topography,
        allow_flat_datum_below_refractor=allow_flat_datum_below_refractor,
        max_abs_datum_shift_ms=max_abs_datum_shift_ms,
    )
    receiver = build_refraction_endpoint_datum_statics(
        endpoint_kind='receiver',
        endpoint_key=receiver_endpoint_key,
        endpoint_id=receiver_endpoint_id,
        node_id=receiver_node_id,
        surface_elevation_m=receiver_surface_elevation_m,
        refractor_elevation_m=receiver_refractor_elevation_m,
        weathering_replacement_shift_s=receiver_weathering_replacement_shift_s,
        weathering_replacement_status=receiver_weathering_replacement_status,
        mode=datum_mode,
        replacement_velocity_m_s=velocity,
        floating_datum_elevation_m=receiver_floating,
        flat_datum_elevation_m=receiver_flat,
        allow_flat_datum_above_topography=allow_flat_datum_above_topography,
        allow_flat_datum_below_refractor=allow_flat_datum_below_refractor,
        max_abs_datum_shift_ms=max_abs_datum_shift_ms,
    )

    source_keys_sorted = _coerce_1d_string(
        source_endpoint_key_sorted,
        name='source_endpoint_key_sorted',
    )
    receiver_keys_sorted = _coerce_1d_string(
        receiver_endpoint_key_sorted,
        name='receiver_endpoint_key_sorted',
        expected_shape=source_keys_sorted.shape,
    )
    source_shift_sorted = _map_endpoint_float_to_trace(
        source_keys_sorted,
        source.endpoint_key,
        source.total_refraction_shift_s,
    )
    receiver_shift_sorted = _map_endpoint_float_to_trace(
        receiver_keys_sorted,
        receiver.endpoint_key,
        receiver.total_refraction_shift_s,
    )
    source_status_sorted = _map_endpoint_status_to_trace(
        source_keys_sorted,
        source.endpoint_key,
        source.datum_static_status,
    )
    receiver_status_sorted = _map_endpoint_status_to_trace(
        receiver_keys_sorted,
        receiver.endpoint_key,
        receiver.datum_static_status,
    )
    source_floating_sorted = _map_endpoint_float_to_trace(
        source_keys_sorted,
        source.endpoint_key,
        source.floating_datum_shift_s,
    )
    receiver_floating_sorted = _map_endpoint_float_to_trace(
        receiver_keys_sorted,
        receiver.endpoint_key,
        receiver.floating_datum_shift_s,
    )
    source_flat_sorted = _map_endpoint_float_to_trace(
        source_keys_sorted,
        source.endpoint_key,
        source.flat_datum_shift_s,
    )
    receiver_flat_sorted = _map_endpoint_float_to_trace(
        receiver_keys_sorted,
        receiver.endpoint_key,
        receiver.flat_datum_shift_s,
    )
    trace_shift = _combine_trace_shifts(source_shift_sorted, receiver_shift_sorted)
    trace_status = _classify_trace_status(
        source_status=source_status_sorted,
        receiver_status=receiver_status_sorted,
        trace_shift_s=trace_shift,
        max_abs_datum_shift_ms=max_abs_datum_shift_ms,
    )
    trace_valid = (trace_status == _OK_STATUS) & np.isfinite(trace_shift)
    final_shift = trace_shift.copy()
    final_status = trace_status.copy()
    final_valid = trace_valid.copy()
    applied_field = np.zeros(trace_shift.shape, dtype=np.float64)
    if trace_field_correction is not None:
        field_result = compose_refraction_final_trace_shift(
            refraction_trace_shift_s_sorted=trace_shift,
            trace_static_status_sorted=trace_status,
            trace_static_valid_mask_sorted=trace_valid,
            trace_field_correction=trace_field_correction,
            apply_to_trace_shift=bool(apply_field_correction_to_trace_shift),
            invalid_component_policy=invalid_field_component_policy,
        )
        final_shift = field_result.final_trace_shift_s_sorted
        final_status = field_result.final_trace_static_status_sorted
        final_valid = field_result.final_trace_static_valid_mask_sorted
        applied_field = field_result.applied_field_shift_s_sorted

    qc = _trace_qc(
        mode=datum_mode,
        replacement_velocity_m_s=velocity,
        source=source,
        receiver=receiver,
        trace_shift_s=trace_shift,
        trace_status=trace_status,
        trace_valid=trace_valid,
        final_trace_shift_s=final_shift,
        final_trace_status=final_status,
        final_trace_valid=final_valid,
        applied_field_shift_s=applied_field,
        max_abs_datum_shift_ms=max_abs_datum_shift_ms,
        field_composition_enabled=trace_field_correction is not None,
        apply_field_correction_to_trace_shift=apply_field_correction_to_trace_shift,
    )
    return RefractionDatumStaticsResult(
        mode=datum_mode,
        replacement_velocity_m_s=float(velocity),
        source_endpoint_datum=source,
        receiver_endpoint_datum=receiver,
        source_endpoint_key_sorted=np.ascontiguousarray(
            source_keys_sorted,
            dtype=_ENDPOINT_KEY_DTYPE,
        ),
        receiver_endpoint_key_sorted=np.ascontiguousarray(
            receiver_keys_sorted,
            dtype=_ENDPOINT_KEY_DTYPE,
        ),
        source_refraction_shift_s_sorted=source_shift_sorted,
        receiver_refraction_shift_s_sorted=receiver_shift_sorted,
        source_floating_datum_shift_s_sorted=source_floating_sorted,
        receiver_floating_datum_shift_s_sorted=receiver_floating_sorted,
        source_flat_datum_shift_s_sorted=source_flat_sorted,
        receiver_flat_datum_shift_s_sorted=receiver_flat_sorted,
        refraction_trace_shift_s_sorted=np.ascontiguousarray(
            trace_shift,
            dtype=np.float64,
        ),
        trace_static_status_sorted=np.ascontiguousarray(
            trace_status,
            dtype=_STATUS_DTYPE,
        ),
        trace_static_valid_mask_sorted=np.ascontiguousarray(trace_valid, dtype=bool),
        applied_field_shift_s_sorted=np.ascontiguousarray(
            applied_field,
            dtype=np.float64,
        ),
        final_trace_shift_s_sorted=np.ascontiguousarray(
            final_shift,
            dtype=np.float64,
        ),
        final_trace_static_status_sorted=np.ascontiguousarray(
            final_status,
            dtype=_STATUS_DTYPE,
        ),
        final_trace_static_valid_mask_sorted=np.ascontiguousarray(
            final_valid,
            dtype=bool,
        ),
        qc=qc,
    )


def _endpoint_status(
    *,
    replacement_status: str,
    surface_elevation_m: float,
    refractor_elevation_m: float,
    replacement_shift_s: float,
    mode: RefractionStaticDatumMode,
    floating_datum_elevation_m: float,
    flat_datum_elevation_m: float,
    allow_flat_datum_above_topography: bool,
    allow_flat_datum_below_refractor: bool,
) -> str:
    inherited = _normalize_status(replacement_status)
    if inherited not in _NOOP_STATUSES:
        return inherited
    if not np.isfinite(surface_elevation_m):
        return 'invalid_surface_elevation'
    if not np.isfinite(refractor_elevation_m):
        return 'invalid_refractor_elevation'
    if not np.isfinite(replacement_shift_s):
        return 'invalid_datum_shift'
    if mode in {'floating_only', 'floating_and_flat'}:
        if not np.isfinite(floating_datum_elevation_m):
            return 'invalid_floating_datum_elevation'
        if floating_datum_elevation_m < refractor_elevation_m:
            return 'floating_datum_below_refractor'
        if floating_datum_elevation_m > surface_elevation_m:
            return 'invalid_floating_datum_elevation'
    if mode in {'flat_only', 'floating_and_flat'}:
        if not np.isfinite(flat_datum_elevation_m):
            return 'invalid_flat_datum_elevation'
        if (
            not allow_flat_datum_below_refractor
            and flat_datum_elevation_m < refractor_elevation_m
        ):
            return 'flat_datum_below_refractor'
        if (
            not allow_flat_datum_above_topography
            and flat_datum_elevation_m > surface_elevation_m
        ):
            return 'invalid_flat_datum_elevation'
    return _OK_STATUS


def _classify_trace_status(
    *,
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    trace_shift_s: np.ndarray,
    max_abs_datum_shift_ms: float | None,
) -> np.ndarray:
    n_traces = int(trace_shift_s.shape[0])
    status = np.full(n_traces, _OK_STATUS, dtype=_STATUS_DTYPE)
    source = np.asarray(source_status).astype(str, copy=False)
    receiver = np.asarray(receiver_status).astype(str, copy=False)
    for index in range(n_traces):
        status[index] = _highest_priority_status(source[index], receiver[index])
    otherwise_ok = (status == _OK_STATUS)
    status[(~np.isfinite(trace_shift_s)) & otherwise_ok] = 'invalid_datum_shift'
    max_abs_shift_s = _optional_max_abs_shift_s(max_abs_datum_shift_ms)
    if max_abs_shift_s is not None:
        exceeds = np.isfinite(trace_shift_s) & (np.abs(trace_shift_s) > max_abs_shift_s)
        status[exceeds] = 'invalid_datum_shift'
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _floating_inputs(
    *,
    floating_datum: ResolvedFloatingDatum | None,
    source_value: np.ndarray | float | None,
    receiver_value: np.ndarray | float | None,
) -> tuple[np.ndarray | float | None, np.ndarray | float | None]:
    if floating_datum is None:
        return source_value, receiver_value
    if source_value is not None or receiver_value is not None:
        raise RefractionDatumError(
            'provide either floating_datum or source/receiver floating datum arrays'
        )
    return floating_datum.source_elevation_m, floating_datum.receiver_elevation_m


def _resolved_datum(
    value: np.ndarray | float | None,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    required: bool,
) -> np.ndarray:
    if value is None:
        if required:
            raise RefractionDatumError(f'{name} is required for datum mode')
        return np.full(expected_shape, np.nan, dtype=np.float64)
    return _coerce_datum_elevation(value, name=name, expected_shape=expected_shape)


def _coerce_datum_elevation(
    value: np.ndarray | float,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = float(arr)
        return np.full(expected_shape, scalar, dtype=np.float64)
    return _coerce_1d_float(value, name=name, expected_shape=expected_shape)


def _map_endpoint_float_to_trace(
    trace_key: np.ndarray,
    endpoint_key: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    lookup = {str(key): float(value) for key, value in zip(endpoint_key, values, strict=True)}
    return np.ascontiguousarray(
        np.asarray([lookup.get(str(key), np.nan) for key in trace_key], dtype=np.float64)
    )


def _map_endpoint_status_to_trace(
    trace_key: np.ndarray,
    endpoint_key: np.ndarray,
    status: np.ndarray,
) -> np.ndarray:
    lookup = {str(key): str(value) for key, value in zip(endpoint_key, status, strict=True)}
    return np.ascontiguousarray(
        np.asarray(
            [lookup.get(str(key), 'missing_endpoint') for key in trace_key],
            dtype=_STATUS_DTYPE,
        )
    )


def _combine_trace_shifts(
    source_shift_s: np.ndarray,
    receiver_shift_s: np.ndarray,
) -> np.ndarray:
    out = np.full(source_shift_s.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(source_shift_s) & np.isfinite(receiver_shift_s)
    out[finite] = source_shift_s[finite] + receiver_shift_s[finite]
    return np.ascontiguousarray(out, dtype=np.float64)


def _endpoint_qc(
    *,
    endpoint_kind: str,
    mode: RefractionStaticDatumMode,
    replacement_velocity_m_s: float,
    surface_elevation_m: np.ndarray,
    refractor_elevation_m: np.ndarray,
    replacement_shift_s: np.ndarray,
    floating_shift_s: np.ndarray,
    flat_shift_s: np.ndarray,
    total_shift_s: np.ndarray,
    status: np.ndarray,
    max_abs_datum_shift_ms: float | None,
) -> dict[str, Any]:
    return {
        'endpoint_kind': endpoint_kind,
        'mode': mode,
        'sign_convention': _SIGN_CONVENTION,
        'elevation_shift_formula': _ELEVATION_FORMULA,
        'endpoint_shift_formula': (
            'total_refraction_shift_s = weathering_replacement_shift_s + '
            'floating_datum_shift_s + flat_datum_shift_s'
        ),
        'replacement_velocity_m_s': float(replacement_velocity_m_s),
        'max_abs_datum_shift_ms': (
            None if max_abs_datum_shift_ms is None else float(max_abs_datum_shift_ms)
        ),
        'n_endpoints': int(total_shift_s.shape[0]),
        'n_valid_endpoints': int(np.count_nonzero(status == _OK_STATUS)),
        'surface_elevation_summary_m': _shift_summary(surface_elevation_m),
        'refractor_elevation_summary_m': _shift_summary(refractor_elevation_m),
        'weathering_replacement_shift_summary_s': _shift_summary(replacement_shift_s),
        'floating_datum_shift_summary_s': _shift_summary(floating_shift_s),
        'flat_datum_shift_summary_s': _shift_summary(flat_shift_s),
        'total_refraction_shift_summary_s': _shift_summary(total_shift_s),
        'datum_static_status_counts': _status_counts(status),
    }


def _trace_qc(
    *,
    mode: RefractionStaticDatumMode,
    replacement_velocity_m_s: float,
    source: RefractionDatumEndpointResult,
    receiver: RefractionDatumEndpointResult,
    trace_shift_s: np.ndarray,
    trace_status: np.ndarray,
    trace_valid: np.ndarray,
    final_trace_shift_s: np.ndarray,
    final_trace_status: np.ndarray,
    final_trace_valid: np.ndarray,
    applied_field_shift_s: np.ndarray,
    max_abs_datum_shift_ms: float | None,
    field_composition_enabled: bool,
    apply_field_correction_to_trace_shift: bool,
) -> dict[str, Any]:
    return {
        'mode': mode,
        'sign_convention': _SIGN_CONVENTION,
        'elevation_shift_formula': _ELEVATION_FORMULA,
        'trace_shift_formula': (
            'refraction_trace_shift_s = source_refraction_shift_s + '
            'receiver_refraction_shift_s'
        ),
        'final_trace_shift_formula': _final_trace_formula(
            apply_field=field_composition_enabled and apply_field_correction_to_trace_shift
        ),
        'replacement_velocity_m_s': float(replacement_velocity_m_s),
        'max_abs_datum_shift_ms': (
            None if max_abs_datum_shift_ms is None else float(max_abs_datum_shift_ms)
        ),
        'field_composition_enabled': bool(field_composition_enabled),
        'apply_field_correction_to_trace_shift': bool(apply_field_correction_to_trace_shift),
        'n_source_endpoints': int(source.endpoint_key.shape[0]),
        'n_receiver_endpoints': int(receiver.endpoint_key.shape[0]),
        'n_traces': int(trace_shift_s.shape[0]),
        'n_valid_trace_shifts': int(np.count_nonzero(trace_valid)),
        'n_valid_final_trace_shifts': int(np.count_nonzero(final_trace_valid)),
        'source_datum_static_status_counts': _status_counts(source.datum_static_status),
        'receiver_datum_static_status_counts': _status_counts(receiver.datum_static_status),
        'trace_static_status_counts': _status_counts(trace_status),
        'final_trace_static_status_counts': _status_counts(final_trace_status),
        'refraction_trace_shift_summary_s': _shift_summary(trace_shift_s),
        'applied_field_shift_summary_s': _shift_summary(applied_field_shift_s),
        'final_trace_shift_summary_s': _shift_summary(final_trace_shift_s),
    }


def _final_trace_formula(*, apply_field: bool) -> str:
    if apply_field:
        return 'final_trace_shift_s = refraction_trace_shift_s + trace_field_shift_s'
    return 'final_trace_shift_s = refraction_trace_shift_s'


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
    out: dict[str, int] = {}
    for raw in np.asarray(values).tolist():
        key = str(raw)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _highest_priority_status(left: str, right: str) -> str:
    left_status = _normalize_status(left)
    right_status = _normalize_status(right)
    if _STATUS_PRIORITY.get(left_status, -1) >= _STATUS_PRIORITY.get(right_status, -1):
        return left_status
    return right_status


def _normalize_status(status: object) -> str:
    text = str(status)
    if text == 'not_applied':
        return 'invalid_datum_shift'
    return text


def _coerce_endpoint_kind(value: object) -> Literal['source', 'receiver']:
    if value == 'source':
        return 'source'
    if value == 'receiver':
        return 'receiver'
    raise RefractionDatumError(f'endpoint_kind must be source or receiver, got {value!r}')


def _coerce_mode(value: object) -> RefractionStaticDatumMode:
    if value in {'floating_and_flat', 'floating_only', 'flat_only', 'none'}:
        return value  # type: ignore[return-value]
    raise RefractionDatumError(f'unsupported datum mode: {value!r}')


def _optional_max_abs_shift_s(value: float | None) -> float | None:
    if value is None:
        return None
    out = _coerce_finite_float(value, name='max_abs_datum_shift_ms')
    if out < 0.0:
        raise RefractionDatumError('max_abs_datum_shift_ms must be non-negative')
    return float(out / 1000.0)


def _coerce_optional_endpoint_ids(
    values: np.ndarray | None,
    *,
    endpoint_count: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    return _coerce_1d_integer(values, name=name, expected_shape=(endpoint_count,))


def _coerce_1d_integer(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    return _coerce_1d_integer_int64(
        values,
        name=name,
        expected_shape=expected_shape,
        error_type=RefractionDatumError,
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
        error_type=RefractionDatumError,
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
        error_type=RefractionDatumError,
    )


def _coerce_1d_status(
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
        output_dtype=_STATUS_DTYPE,
        error_type=RefractionDatumError,
    )


__all__ = [
    'RefractionDatumEndpointResult',
    'RefractionDatumError',
    'RefractionDatumStaticsResult',
    'ResolvedFloatingDatum',
    'build_refraction_datum_statics',
    'build_refraction_endpoint_datum_statics',
    'compute_refraction_datum_elevation_shift_s',
    'compute_refraction_datum_elevation_shift_scalar_s',
    'smooth_refraction_floating_datum_elevation',
]
