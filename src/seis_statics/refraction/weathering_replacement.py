"""Weathering-replacement static shifts for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from seis_statics._validation import coerce_positive_finite_float
from seis_statics.refraction.status import LOCAL_V2_STATUS_VALUES
from seis_statics.refraction.t1lsst import (
    RefractionT1LSSTError,
    compute_t1lsst_1layer_weathering_correction,
)
from seis_statics.refraction.weathering import (
    RefractionWeatheringEndpointComponents,
    RefractionWeatheringModel,
)


_STATUS_DTYPE = '<U32'
_ZERO_SHIFT_ATOL_S = 1.0e-12
_SLOWNESS_RTOL = 1.0e-8
_FORMULA_TEXT = 'shift = z * (1/vb - 1/vw)'
_SIGN_CONVENTION_TEXT = 'corrected(t) = raw(t - shift_s)'

_STATUS_PRIORITY = {
    'ok': 0,
    'not_observed': 1,
    'zero_thickness': 1,
    'clipped_half_intercept_lower': 2,
    'clipped_half_intercept_upper': 3,
    'low_fold': 4,
    'exceeds_max_thickness': 5,
    'exceeds_max_abs_shift': 6,
    'invalid_shift': 7,
    'negative_weathering_thickness': 8,
    'invalid_weathering_thickness': 9,
    'inactive': 10,
    'invalid_velocity': 11,
    'invalid_nonfinite_input': 12,
    'invalid_velocity_order': 13,
    'invalid_surface_elevation': 14,
    'invalid_refractor_elevation': 15,
    'outside_refractor_cell_grid': 16,
    'inactive_v2_cell': 17,
    'low_fold_v2_cell': 18,
    'invalid_local_v2': 19,
    'v2_not_greater_than_v1': 20,
    'missing_endpoint': 21,
    'missing_node': 22,
}


class RefractionWeatheringReplacementError(ValueError):
    """Raised when weathering-replacement static outputs cannot be built."""


@dataclass(frozen=True)
class RefractionWeatheringReplacementResult:
    """Weathering-replacement static component from a weathering model."""

    file_id: str
    n_traces: int
    bedrock_velocity_mode: Literal['solve_global', 'fixed_global', 'solve_cell']
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float
    replacement_slowness_delta_s_per_m: float

    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_surface_elevation_m: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_weathering_thickness_m: np.ndarray
    node_refractor_elevation_m: np.ndarray
    node_solution_status: np.ndarray
    node_weathering_status: np.ndarray
    node_v2_m_s: np.ndarray
    node_v2_status: np.ndarray
    node_weathering_replacement_shift_s: np.ndarray
    node_weathering_replacement_shift_ms: np.ndarray
    node_static_status: np.ndarray
    node_pick_count: np.ndarray
    node_used_observation_count: np.ndarray
    node_rejected_observation_count: np.ndarray

    source_endpoint_key: np.ndarray
    source_endpoint_id: np.ndarray | None
    source_node_id: np.ndarray
    source_weathering_thickness_m: np.ndarray
    source_refractor_elevation_m: np.ndarray
    source_v2_m_s: np.ndarray
    source_v2_status: np.ndarray
    source_weathering_replacement_shift_s: np.ndarray
    source_static_status: np.ndarray

    receiver_endpoint_key: np.ndarray
    receiver_endpoint_id: np.ndarray | None
    receiver_node_id: np.ndarray
    receiver_weathering_thickness_m: np.ndarray
    receiver_refractor_elevation_m: np.ndarray
    receiver_v2_m_s: np.ndarray
    receiver_v2_status: np.ndarray
    receiver_weathering_replacement_shift_s: np.ndarray
    receiver_static_status: np.ndarray

    trace_index_sorted: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_weathering_thickness_m_sorted: np.ndarray
    receiver_weathering_thickness_m_sorted: np.ndarray
    source_weathering_replacement_shift_s_sorted: np.ndarray
    receiver_weathering_replacement_shift_s_sorted: np.ndarray
    weathering_replacement_trace_shift_s_sorted: np.ndarray
    source_static_status_sorted: np.ndarray
    receiver_static_status_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray

    qc: dict[str, Any]


def compute_weathering_replacement_shift_s(
    *,
    weathering_thickness_m: np.ndarray,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: float | np.ndarray,
) -> np.ndarray:
    """Compute ``shift = z * (1/vb - 1/vw)`` in seconds."""
    try:
        return compute_t1lsst_1layer_weathering_correction(
            sh1_m=weathering_thickness_m,
            v1_m_s=weathering_velocity_m_s,
            v2_m_s=bedrock_velocity_m_s,
        )
    except RefractionT1LSSTError as exc:
        if str(exc) == 'v2_m_s must be greater than v1_m_s':
            raise RefractionWeatheringReplacementError(
                'bedrock_velocity_m_s must be greater than weathering_velocity_m_s'
            ) from exc
        raise RefractionWeatheringReplacementError(
            str(exc)
            .replace('sh1_m', 'weathering_thickness_m')
            .replace('v1_m_s', 'weathering_velocity_m_s')
            .replace('v2_m_s', 'bedrock_velocity_m_s')
        ) from exc


def compute_weathering_replacement_shift_scalar_s(
    *,
    weathering_thickness_m: float,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: float,
) -> float:
    """Scalar wrapper around ``compute_weathering_replacement_shift_s``."""
    value = compute_weathering_replacement_shift_s(
        weathering_thickness_m=np.asarray([weathering_thickness_m], dtype=np.float64),
        weathering_velocity_m_s=weathering_velocity_m_s,
        bedrock_velocity_m_s=bedrock_velocity_m_s,
    )
    return float(value[0])


def build_refraction_weathering_replacement_statics(
    *,
    weathering_model: RefractionWeatheringModel,
    max_abs_shift_ms: float | None = None,
) -> RefractionWeatheringReplacementResult:
    """Compute weathering-replacement statics from a weathering model."""
    if not isinstance(weathering_model, RefractionWeatheringModel):
        raise RefractionWeatheringReplacementError(
            'weathering_model must be a RefractionWeatheringModel instance'
        )
    max_abs = _optional_positive_finite_float(
        max_abs_shift_ms,
        name='max_abs_shift_ms',
    )
    velocity = _velocity_context(weathering_model)

    node_shift, node_status = _component_shift_and_status(
        weathering_thickness_m=weathering_model.node_weathering_thickness_m,
        weathering_status=weathering_model.node_weathering_status,
        local_v2_m_s=weathering_model.node_v2_m_s,
        local_v2_status=weathering_model.node_local_v2_status,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        max_abs_shift_ms=max_abs,
    )
    source_shift, source_status = _endpoint_shift_and_status(
        endpoint=weathering_model.source_endpoint,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        max_abs_shift_ms=max_abs,
    )
    receiver_shift, receiver_status = _endpoint_shift_and_status(
        endpoint=weathering_model.receiver_endpoint,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        max_abs_shift_ms=max_abs,
    )

    source_shift_sorted = _map_endpoint_float_to_trace(
        weathering_model.source_endpoint_key_sorted,
        weathering_model.source_endpoint.endpoint_key,
        source_shift,
    )
    receiver_shift_sorted = _map_endpoint_float_to_trace(
        weathering_model.receiver_endpoint_key_sorted,
        weathering_model.receiver_endpoint.endpoint_key,
        receiver_shift,
    )
    source_status_sorted = _map_endpoint_status_to_trace(
        weathering_model.source_endpoint_key_sorted,
        weathering_model.source_endpoint.endpoint_key,
        source_status,
    )
    receiver_status_sorted = _map_endpoint_status_to_trace(
        weathering_model.receiver_endpoint_key_sorted,
        weathering_model.receiver_endpoint.endpoint_key,
        receiver_status,
    )
    trace_shift = _combine_trace_shifts(source_shift_sorted, receiver_shift_sorted)
    trace_status = _classify_trace_status(
        source_status=source_status_sorted,
        receiver_status=receiver_status_sorted,
        trace_shift_s=trace_shift,
        max_abs_shift_ms=max_abs,
    )
    trace_valid = _trace_valid_mask(
        trace_shift_s=trace_shift,
        trace_static_status=trace_status,
        max_abs_shift_ms=max_abs,
    )
    qc = _build_qc(
        weathering_model=weathering_model,
        velocity=velocity,
        node_shift_s=node_shift,
        node_static_status=node_status,
        source_shift_s=source_shift,
        source_static_status=source_status,
        receiver_shift_s=receiver_shift,
        receiver_static_status=receiver_status,
        trace_shift_s=trace_shift,
        trace_static_status=trace_status,
        trace_static_valid_mask=trace_valid,
        max_abs_shift_ms=max_abs,
    )

    return RefractionWeatheringReplacementResult(
        file_id=weathering_model.file_id,
        n_traces=weathering_model.n_traces,
        bedrock_velocity_mode=velocity.mode,
        bedrock_slowness_s_per_m=velocity.bedrock_slowness_s_per_m,
        bedrock_velocity_m_s=velocity.bedrock_velocity_m_s,
        weathering_velocity_m_s=velocity.weathering_velocity_m_s,
        replacement_slowness_delta_s_per_m=(
            velocity.replacement_slowness_delta_s_per_m
        ),
        node_id=np.ascontiguousarray(weathering_model.node_id, dtype=np.int64),
        node_x_m=np.ascontiguousarray(weathering_model.node_x_m, dtype=np.float64),
        node_y_m=np.ascontiguousarray(weathering_model.node_y_m, dtype=np.float64),
        node_surface_elevation_m=np.ascontiguousarray(
            weathering_model.node_surface_elevation_m,
            dtype=np.float64,
        ),
        node_half_intercept_time_s=np.ascontiguousarray(
            weathering_model.node_half_intercept_time_s,
            dtype=np.float64,
        ),
        node_weathering_thickness_m=np.ascontiguousarray(
            weathering_model.node_weathering_thickness_m,
            dtype=np.float64,
        ),
        node_refractor_elevation_m=np.ascontiguousarray(
            weathering_model.node_refractor_elevation_m,
            dtype=np.float64,
        ),
        node_solution_status=np.ascontiguousarray(
            weathering_model.node_solution_status,
            dtype=_STATUS_DTYPE,
        ),
        node_weathering_status=np.ascontiguousarray(
            weathering_model.node_weathering_status,
            dtype=_STATUS_DTYPE,
        ),
        node_v2_m_s=np.ascontiguousarray(weathering_model.node_v2_m_s, dtype=np.float64),
        node_v2_status=np.ascontiguousarray(
            weathering_model.node_local_v2_status,
            dtype=_STATUS_DTYPE,
        ),
        node_weathering_replacement_shift_s=node_shift,
        node_weathering_replacement_shift_ms=np.ascontiguousarray(
            node_shift * 1000.0,
            dtype=np.float64,
        ),
        node_static_status=node_status,
        node_pick_count=np.ascontiguousarray(
            weathering_model.node_pick_count,
            dtype=np.int64,
        ),
        node_used_observation_count=np.ascontiguousarray(
            weathering_model.node_used_observation_count,
            dtype=np.int64,
        ),
        node_rejected_observation_count=np.ascontiguousarray(
            weathering_model.node_rejected_observation_count,
            dtype=np.int64,
        ),
        source_endpoint_key=np.ascontiguousarray(
            weathering_model.source_endpoint.endpoint_key
        ),
        source_endpoint_id=_optional_int_array(
            weathering_model.source_endpoint.endpoint_id
        ),
        source_node_id=np.ascontiguousarray(
            weathering_model.source_endpoint.node_id,
            dtype=np.int64,
        ),
        source_weathering_thickness_m=np.ascontiguousarray(
            weathering_model.source_endpoint.weathering_thickness_m,
            dtype=np.float64,
        ),
        source_refractor_elevation_m=np.ascontiguousarray(
            weathering_model.source_endpoint.refractor_elevation_m,
            dtype=np.float64,
        ),
        source_v2_m_s=np.ascontiguousarray(
            weathering_model.source_endpoint.v2_m_s,
            dtype=np.float64,
        ),
        source_v2_status=np.ascontiguousarray(
            weathering_model.source_endpoint.local_v2_status,
            dtype=_STATUS_DTYPE,
        ),
        source_weathering_replacement_shift_s=source_shift,
        source_static_status=source_status,
        receiver_endpoint_key=np.ascontiguousarray(
            weathering_model.receiver_endpoint.endpoint_key
        ),
        receiver_endpoint_id=_optional_int_array(
            weathering_model.receiver_endpoint.endpoint_id
        ),
        receiver_node_id=np.ascontiguousarray(
            weathering_model.receiver_endpoint.node_id,
            dtype=np.int64,
        ),
        receiver_weathering_thickness_m=np.ascontiguousarray(
            weathering_model.receiver_endpoint.weathering_thickness_m,
            dtype=np.float64,
        ),
        receiver_refractor_elevation_m=np.ascontiguousarray(
            weathering_model.receiver_endpoint.refractor_elevation_m,
            dtype=np.float64,
        ),
        receiver_v2_m_s=np.ascontiguousarray(
            weathering_model.receiver_endpoint.v2_m_s,
            dtype=np.float64,
        ),
        receiver_v2_status=np.ascontiguousarray(
            weathering_model.receiver_endpoint.local_v2_status,
            dtype=_STATUS_DTYPE,
        ),
        receiver_weathering_replacement_shift_s=receiver_shift,
        receiver_static_status=receiver_status,
        trace_index_sorted=np.ascontiguousarray(
            weathering_model.trace_index_sorted,
            dtype=np.int64,
        ),
        source_endpoint_key_sorted=np.ascontiguousarray(
            weathering_model.source_endpoint_key_sorted
        ),
        receiver_endpoint_key_sorted=np.ascontiguousarray(
            weathering_model.receiver_endpoint_key_sorted
        ),
        source_node_id_sorted=np.ascontiguousarray(
            weathering_model.source_node_id_sorted,
            dtype=np.int64,
        ),
        receiver_node_id_sorted=np.ascontiguousarray(
            weathering_model.receiver_node_id_sorted,
            dtype=np.int64,
        ),
        source_weathering_thickness_m_sorted=np.ascontiguousarray(
            weathering_model.source_weathering_thickness_m_sorted,
            dtype=np.float64,
        ),
        receiver_weathering_thickness_m_sorted=np.ascontiguousarray(
            weathering_model.receiver_weathering_thickness_m_sorted,
            dtype=np.float64,
        ),
        source_weathering_replacement_shift_s_sorted=source_shift_sorted,
        receiver_weathering_replacement_shift_s_sorted=receiver_shift_sorted,
        weathering_replacement_trace_shift_s_sorted=trace_shift,
        source_static_status_sorted=source_status_sorted,
        receiver_static_status_sorted=receiver_status_sorted,
        trace_static_status_sorted=trace_status,
        trace_static_valid_mask_sorted=trace_valid,
        qc=qc,
    )


@dataclass(frozen=True)
class _VelocityContext:
    mode: Literal['solve_global', 'fixed_global', 'solve_cell']
    bedrock_slowness_s_per_m: float
    bedrock_velocity_m_s: float
    weathering_velocity_m_s: float
    replacement_slowness_delta_s_per_m: float


def _velocity_context(weathering_model: RefractionWeatheringModel) -> _VelocityContext:
    mode = _validate_velocity_mode(weathering_model.bedrock_velocity_mode)
    weathering_velocity = coerce_positive_finite_float(
        weathering_model.weathering_velocity_m_s,
        name='weathering_model.weathering_velocity_m_s',
        error_type=RefractionWeatheringReplacementError,
    )
    if mode == 'solve_cell':
        bedrock_velocity = _metadata_float(
            weathering_model.bedrock_velocity_m_s,
            name='weathering_model.bedrock_velocity_m_s',
        )
        bedrock_slowness = _metadata_float(
            weathering_model.bedrock_slowness_s_per_m,
            name='weathering_model.bedrock_slowness_s_per_m',
        )
        return _VelocityContext(
            mode=mode,
            bedrock_slowness_s_per_m=bedrock_slowness,
            bedrock_velocity_m_s=bedrock_velocity,
            weathering_velocity_m_s=weathering_velocity,
            replacement_slowness_delta_s_per_m=float('nan'),
        )

    bedrock_velocity = coerce_positive_finite_float(
        weathering_model.bedrock_velocity_m_s,
        name='weathering_model.bedrock_velocity_m_s',
        error_type=RefractionWeatheringReplacementError,
    )
    bedrock_slowness = coerce_positive_finite_float(
        weathering_model.bedrock_slowness_s_per_m,
        name='weathering_model.bedrock_slowness_s_per_m',
        error_type=RefractionWeatheringReplacementError,
    )
    if bedrock_velocity <= weathering_velocity:
        raise RefractionWeatheringReplacementError(
            'bedrock_velocity_m_s must be greater than weathering_velocity_m_s'
        )
    derived_slowness = 1.0 / bedrock_velocity
    slowness_tol = max(1.0e-12, abs(bedrock_slowness) * _SLOWNESS_RTOL)
    if abs(derived_slowness - bedrock_slowness) > slowness_tol:
        raise RefractionWeatheringReplacementError(
            'bedrock_velocity_m_s does not match bedrock_slowness_s_per_m'
        )
    return _VelocityContext(
        mode=mode,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_m_s=bedrock_velocity,
        weathering_velocity_m_s=weathering_velocity,
        replacement_slowness_delta_s_per_m=(
            1.0 / bedrock_velocity - 1.0 / weathering_velocity
        ),
    )


def _component_shift_and_status(
    *,
    weathering_thickness_m: np.ndarray,
    weathering_status: np.ndarray,
    local_v2_m_s: np.ndarray,
    local_v2_status: np.ndarray,
    weathering_velocity_m_s: float,
    max_abs_shift_ms: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_shift = _compute_weathering_replacement_shift_with_local_v2(
        weathering_thickness_m=weathering_thickness_m,
        weathering_velocity_m_s=weathering_velocity_m_s,
        bedrock_velocity_m_s=local_v2_m_s,
    )
    shift, status = _classify_component_status(
        weathering_status=weathering_status,
        weathering_thickness_m=weathering_thickness_m,
        raw_shift_s=raw_shift,
        max_abs_shift_ms=max_abs_shift_ms,
    )
    status = _overlay_local_v2_status(status, local_v2_status)
    shift = _nan_for_local_v2_status(shift, status)
    return shift, status


def _endpoint_shift_and_status(
    *,
    endpoint: RefractionWeatheringEndpointComponents,
    weathering_velocity_m_s: float,
    max_abs_shift_ms: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    return _component_shift_and_status(
        weathering_thickness_m=endpoint.weathering_thickness_m,
        weathering_status=endpoint.weathering_status,
        local_v2_m_s=endpoint.v2_m_s,
        local_v2_status=endpoint.local_v2_status,
        weathering_velocity_m_s=weathering_velocity_m_s,
        max_abs_shift_ms=max_abs_shift_ms,
    )


def _compute_weathering_replacement_shift_with_local_v2(
    *,
    weathering_thickness_m: np.ndarray,
    weathering_velocity_m_s: float,
    bedrock_velocity_m_s: np.ndarray,
) -> np.ndarray:
    thickness = np.asarray(weathering_thickness_m, dtype=np.float64)
    v2 = np.asarray(bedrock_velocity_m_s, dtype=np.float64)
    if thickness.shape != v2.shape:
        raise RefractionWeatheringReplacementError(
            'weathering_thickness_m and bedrock_velocity_m_s shape mismatch'
        )
    out = np.full(thickness.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(thickness) & np.isfinite(v2) & (v2 > weathering_velocity_m_s)
    if np.any(valid):
        out[valid] = compute_weathering_replacement_shift_s(
            weathering_thickness_m=thickness[valid],
            weathering_velocity_m_s=weathering_velocity_m_s,
            bedrock_velocity_m_s=v2[valid],
        )
    return np.ascontiguousarray(out, dtype=np.float64)


def _classify_component_status(
    *,
    weathering_status: np.ndarray,
    weathering_thickness_m: np.ndarray,
    raw_shift_s: np.ndarray,
    max_abs_shift_ms: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_items = int(weathering_thickness_m.shape[0])
    status = np.full(n_items, 'ok', dtype=_STATUS_DTYPE)
    shift = np.ascontiguousarray(raw_shift_s, dtype=np.float64).copy()
    inherited = np.asarray(weathering_status).astype(str, copy=False)
    thickness = np.asarray(weathering_thickness_m, dtype=np.float64)

    _assign(status, inherited == 'clipped_half_intercept_lower', 'clipped_half_intercept_lower')
    _assign(status, inherited == 'clipped_half_intercept_upper', 'clipped_half_intercept_upper')
    _assign(status, inherited == 'low_fold', 'low_fold')
    _assign(status, inherited == 'exceeds_max_thickness', 'exceeds_max_thickness')
    _assign(status, inherited == 'invalid_velocity', 'invalid_velocity')
    _assign(status, inherited == 'invalid_nonfinite_input', 'invalid_nonfinite_input')
    _assign(status, inherited == 'invalid_velocity_order', 'invalid_velocity_order')
    missing_thickness = ~np.isfinite(thickness)
    _assign(
        status,
        missing_thickness & (inherited == 'invalid_surface_elevation'),
        'invalid_surface_elevation',
    )
    _assign(
        status,
        missing_thickness & (inherited == 'invalid_refractor_elevation'),
        'invalid_refractor_elevation',
    )

    negative_thickness = np.isfinite(thickness) & (thickness < 0.0)
    negative_thickness |= (inherited == 'negative_thickness') | (
        inherited == 'negative_weathering_thickness'
    )
    invalid_thickness = (~np.isfinite(thickness)) & np.isin(
        inherited,
        ['ok', 'zero_thickness'],
    )
    invalid_thickness |= inherited == 'invalid_weathering_thickness'
    unknown_invalid = _unknown_invalid_weathering_status(inherited)
    invalid_shift = np.isfinite(thickness) & (thickness >= 0.0) & (~np.isfinite(shift))
    if max_abs_shift_ms is not None:
        exceeds_max_abs_shift = np.isfinite(shift) & (
            np.abs(shift) * 1000.0 > max_abs_shift_ms
        )
    else:
        exceeds_max_abs_shift = np.zeros(n_items, dtype=bool)

    _assign(status, exceeds_max_abs_shift, 'exceeds_max_abs_shift')
    _assign(status, invalid_shift, 'invalid_shift')
    _assign(status, negative_thickness, 'negative_weathering_thickness')
    _assign(status, invalid_thickness | unknown_invalid, 'invalid_weathering_thickness')
    for local_status in LOCAL_V2_STATUS_VALUES:
        _assign(status, inherited == local_status, local_status)
    _assign(status, inherited == 'inactive', 'inactive')
    _assign(status, inherited == 'missing_endpoint', 'missing_endpoint')
    _assign(status, inherited == 'missing_node', 'missing_node')

    invalid_output = np.isin(
        status.astype(str),
        [
            'missing_node',
            'missing_endpoint',
            'invalid_velocity',
            'outside_refractor_cell_grid',
            'inactive_v2_cell',
            'low_fold_v2_cell',
            'invalid_local_v2',
            'v2_not_greater_than_v1',
            'inactive',
            'invalid_surface_elevation',
            'invalid_refractor_elevation',
            'invalid_nonfinite_input',
            'invalid_velocity_order',
            'invalid_velocity',
            'invalid_weathering_thickness',
            'negative_weathering_thickness',
            'invalid_shift',
        ],
    )
    shift[invalid_output] = np.nan
    return (
        np.ascontiguousarray(shift, dtype=np.float64),
        np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )


def _overlay_local_v2_status(
    component_status: np.ndarray,
    local_v2_status: np.ndarray,
) -> np.ndarray:
    status = np.asarray(component_status).astype(_STATUS_DTYPE, copy=True)
    local = np.asarray(local_v2_status).astype(str, copy=False)
    invalid_local = np.isin(local, list(LOCAL_V2_STATUS_VALUES))
    status[invalid_local] = local[invalid_local]
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _nan_for_local_v2_status(values: np.ndarray, status: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(values, dtype=np.float64).copy()
    text = np.asarray(status).astype(str, copy=False)
    out[np.isin(text, list(LOCAL_V2_STATUS_VALUES))] = np.nan
    return out


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


def _classify_trace_status(
    *,
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    trace_shift_s: np.ndarray,
    max_abs_shift_ms: float | None,
) -> np.ndarray:
    n_traces = int(trace_shift_s.shape[0])
    status = np.full(n_traces, 'ok', dtype=_STATUS_DTYPE)
    source = np.asarray(source_status).astype(str, copy=False)
    receiver = np.asarray(receiver_status).astype(str, copy=False)
    for index in range(n_traces):
        status[index] = _highest_priority_status(source[index], receiver[index])
    otherwise_ok = (source == 'ok') & (receiver == 'ok')
    _assign(status, (~np.isfinite(trace_shift_s)) & otherwise_ok, 'invalid_shift')
    if max_abs_shift_ms is not None:
        exceeds = np.isfinite(trace_shift_s) & (
            np.abs(trace_shift_s) * 1000.0 > max_abs_shift_ms
        )
        _assign(status, exceeds, 'exceeds_max_abs_shift')
    return np.ascontiguousarray(status, dtype=_STATUS_DTYPE)


def _trace_valid_mask(
    *,
    trace_shift_s: np.ndarray,
    trace_static_status: np.ndarray,
    max_abs_shift_ms: float | None,
) -> np.ndarray:
    status = np.asarray(trace_static_status).astype(str, copy=False)
    valid = (status == 'ok') & np.isfinite(trace_shift_s)
    if max_abs_shift_ms is not None:
        valid &= np.abs(trace_shift_s) * 1000.0 <= max_abs_shift_ms
    return np.ascontiguousarray(valid, dtype=bool)


def _build_qc(
    *,
    weathering_model: RefractionWeatheringModel,
    velocity: _VelocityContext,
    node_shift_s: np.ndarray,
    node_static_status: np.ndarray,
    source_shift_s: np.ndarray,
    source_static_status: np.ndarray,
    receiver_shift_s: np.ndarray,
    receiver_static_status: np.ndarray,
    trace_shift_s: np.ndarray,
    trace_static_status: np.ndarray,
    trace_static_valid_mask: np.ndarray,
    max_abs_shift_ms: float | None,
) -> dict[str, Any]:
    valid_node_shift_ms = _valid_shift_ms(node_shift_s, node_static_status)
    valid_source_shift_ms = _valid_shift_ms(source_shift_s, source_static_status)
    valid_receiver_shift_ms = _valid_shift_ms(receiver_shift_s, receiver_static_status)
    valid_trace_shift_ms = trace_shift_s[trace_static_valid_mask] * 1000.0
    finite_trace_shift = trace_shift_s[np.isfinite(trace_shift_s)]
    qc = {
        'method': 'gli_variable_thickness',
        'static_component': 'weathering_replacement',
        'file_id': weathering_model.file_id,
        'bedrock_velocity_mode': velocity.mode,
        'bedrock_velocity_m_s': _json_finite_float(velocity.bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': _json_finite_float(
            velocity.bedrock_slowness_s_per_m
        ),
        'weathering_velocity_m_s': float(velocity.weathering_velocity_m_s),
        'replacement_slowness_delta_s_per_m': _json_finite_float(
            velocity.replacement_slowness_delta_s_per_m
        ),
        'n_traces': int(weathering_model.n_traces),
        'n_nodes': int(weathering_model.node_id.shape[0]),
        'n_source_endpoints': int(weathering_model.source_endpoint.endpoint_key.shape[0]),
        'n_receiver_endpoints': int(
            weathering_model.receiver_endpoint.endpoint_key.shape[0]
        ),
        'node_shift_min_ms': _json_stat(valid_node_shift_ms, 'min'),
        'node_shift_max_ms': _json_stat(valid_node_shift_ms, 'max'),
        'node_shift_median_ms': _json_stat(valid_node_shift_ms, 'median'),
        'node_shift_p95_abs_ms': _json_stat(np.abs(valid_node_shift_ms), 'p95'),
        'source_shift_min_ms': _json_stat(valid_source_shift_ms, 'min'),
        'source_shift_max_ms': _json_stat(valid_source_shift_ms, 'max'),
        'source_shift_median_ms': _json_stat(valid_source_shift_ms, 'median'),
        'receiver_shift_min_ms': _json_stat(valid_receiver_shift_ms, 'min'),
        'receiver_shift_max_ms': _json_stat(valid_receiver_shift_ms, 'max'),
        'receiver_shift_median_ms': _json_stat(valid_receiver_shift_ms, 'median'),
        'trace_shift_min_ms': _json_stat(valid_trace_shift_ms, 'min'),
        'trace_shift_max_ms': _json_stat(valid_trace_shift_ms, 'max'),
        'trace_shift_median_ms': _json_stat(valid_trace_shift_ms, 'median'),
        'trace_shift_p95_abs_ms': _json_stat(np.abs(valid_trace_shift_ms), 'p95'),
        'trace_shift_max_abs_ms': _json_stat(np.abs(valid_trace_shift_ms), 'max'),
        'negative_trace_shift_count': int(
            np.count_nonzero(valid_trace_shift_ms < -_ZERO_SHIFT_ATOL_S * 1000.0)
        ),
        'positive_trace_shift_count': int(
            np.count_nonzero(valid_trace_shift_ms > _ZERO_SHIFT_ATOL_S * 1000.0)
        ),
        'zero_trace_shift_count': int(
            np.count_nonzero(np.abs(valid_trace_shift_ms) <= _ZERO_SHIFT_ATOL_S * 1000.0)
        ),
        'invalid_trace_shift_count': int(np.count_nonzero(~np.isfinite(trace_shift_s))),
        'finite_trace_shift_count': int(finite_trace_shift.shape[0]),
        'max_abs_shift_ms': None if max_abs_shift_ms is None else float(max_abs_shift_ms),
        'exceeds_max_abs_shift_count': int(
            np.count_nonzero(trace_static_status == 'exceeds_max_abs_shift')
        ),
        'inactive_node_count': int(np.count_nonzero(node_static_status == 'inactive')),
        'low_fold_node_count': int(np.count_nonzero(node_static_status == 'low_fold')),
        'invalid_weathering_thickness_count': int(
            np.count_nonzero(node_static_status == 'invalid_weathering_thickness')
        ),
        'exceeds_max_thickness_count': int(
            np.count_nonzero(node_static_status == 'exceeds_max_thickness')
        ),
        'node_static_status_counts': _status_counts(node_static_status),
        'source_static_status_counts': _status_counts(source_static_status),
        'receiver_static_status_counts': _status_counts(receiver_static_status),
        'trace_static_status_counts': _status_counts(trace_static_status),
        'sign_convention': _SIGN_CONVENTION_TEXT,
        'formula': _FORMULA_TEXT,
    }
    _copy_cell_qc(qc, weathering_model.qc)
    return qc


def _copy_cell_qc(payload: dict[str, Any], upstream: dict[str, Any]) -> None:
    for key in (
        'min_observations_per_cell',
        'n_low_fold_cells',
        'n_observations_rejected_by_low_fold_cell',
        'low_fold_cell_rejection_reason',
        'low_fold_cell_id',
        'cell_observation_count',
        'layers',
    ):
        if key in upstream:
            payload[key] = upstream[key]


def _valid_shift_ms(shift_s: np.ndarray, status: np.ndarray) -> np.ndarray:
    arr = np.asarray(shift_s, dtype=np.float64)
    status_arr = np.asarray(status).astype(str, copy=False)
    return arr[(status_arr == 'ok') & np.isfinite(arr)] * 1000.0


def _json_stat(values: np.ndarray, stat: str) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if stat == 'min':
        return float(np.min(arr))
    if stat == 'max':
        return float(np.max(arr))
    if stat == 'median':
        return float(np.median(arr))
    if stat == 'p95':
        return float(np.percentile(arr, 95.0))
    raise RefractionWeatheringReplacementError(f'unsupported statistic: {stat}')


def _status_counts(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in values.tolist():
        key = str(raw)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _unknown_invalid_weathering_status(status: np.ndarray) -> np.ndarray:
    text = np.asarray(status).astype(str, copy=False)
    known = np.isin(
        text,
        [
            'ok',
            'zero_thickness',
            'inactive',
            'invalid_weathering_thickness',
            'negative_thickness',
            'negative_weathering_thickness',
            'low_fold',
            'clipped_half_intercept_lower',
            'clipped_half_intercept_upper',
            'exceeds_max_thickness',
            'invalid_velocity',
            'invalid_surface_elevation',
            'invalid_refractor_elevation',
            'invalid_nonfinite_input',
            'invalid_velocity_order',
            'missing_endpoint',
            'missing_node',
            'outside_refractor_cell_grid',
            'inactive_v2_cell',
            'low_fold_v2_cell',
            'invalid_local_v2',
            'v2_not_greater_than_v1',
        ],
    )
    return ~known


def _metadata_float(value: object, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionWeatheringReplacementError(f'{name} must be numeric') from exc


def _json_finite_float(value: float) -> float | None:
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def _assign(status: np.ndarray, mask: np.ndarray, value: str) -> None:
    current = np.asarray(status).astype(str, copy=False)
    priority = _STATUS_PRIORITY[value]
    should_assign = np.asarray(mask, dtype=bool) & np.asarray(
        [_STATUS_PRIORITY.get(str(item), -1) <= priority for item in current],
        dtype=bool,
    )
    status[should_assign] = value


def _highest_priority_status(left: str, right: str) -> str:
    if _STATUS_PRIORITY.get(left, -1) >= _STATUS_PRIORITY.get(right, -1):
        return left
    return right


def _validate_velocity_mode(
    value: object,
) -> Literal['solve_global', 'fixed_global', 'solve_cell']:
    if value == 'solve_global':
        return 'solve_global'
    if value == 'fixed_global':
        return 'fixed_global'
    if value == 'solve_cell':
        return 'solve_cell'
    raise RefractionWeatheringReplacementError(
        'bedrock_velocity_mode must be solve_global, fixed_global, or solve_cell'
    )


def _optional_positive_finite_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return coerce_positive_finite_float(
        value,
        name=name,
        error_type=RefractionWeatheringReplacementError,
    )


def _optional_int_array(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(values, dtype=np.int64)


__all__ = [
    'RefractionWeatheringReplacementError',
    'RefractionWeatheringReplacementResult',
    'build_refraction_weathering_replacement_statics',
    'compute_weathering_replacement_shift_s',
    'compute_weathering_replacement_shift_scalar_s',
]
