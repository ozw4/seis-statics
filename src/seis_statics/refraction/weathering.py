"""Pure weathering-thickness model assembly for refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seis_statics.refraction.cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_points,
)
from seis_statics.refraction.cell_grid import (
    assign_points_to_refraction_cells,
    build_refraction_cell_grid,
)
from seis_statics.refraction.first_layer import resolve_weathering_velocity_m_s
from seis_statics.refraction.half_intercept import (
    RefractionHalfInterceptEndpointResult,
    RefractionHalfInterceptResult,
)
from seis_statics.refraction.options import RefractionStaticModelOptions
from seis_statics.refraction.types import RefractionStaticInputModel


_STATUS_DTYPE = '<U32'


class RefractionWeatheringError(ValueError):
    """Raised when weathering-thickness inputs are inconsistent."""


@dataclass(frozen=True)
class RefractionWeatheringThicknessComputation:
    """Weathering thickness values and per-value conversion status."""

    weathering_thickness_m: np.ndarray
    refractor_elevation_m: np.ndarray
    weathering_status: np.ndarray


@dataclass(frozen=True)
class RefractionWeatheringEndpointComponents:
    """Endpoint-local weathering values derived from half-intercept time."""

    endpoint_key: np.ndarray
    endpoint_id: np.ndarray | None
    node_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    surface_elevation_m: np.ndarray
    half_intercept_time_s: np.ndarray
    v1_m_s: np.ndarray
    v2_m_s: np.ndarray
    weathering_thickness_m: np.ndarray
    refractor_elevation_m: np.ndarray
    solution_status: np.ndarray
    local_v2_status: np.ndarray
    weathering_status: np.ndarray
    pick_count: np.ndarray
    used_observation_count: np.ndarray
    rejected_observation_count: np.ndarray


@dataclass(frozen=True)
class RefractionWeatheringModel:
    """Full source, receiver, trace, and QC weathering-thickness model."""

    file_id: str
    n_traces: int
    bedrock_velocity_mode: str
    weathering_velocity_m_s: float
    bedrock_velocity_m_s: float
    bedrock_slowness_s_per_m: float
    bedrock_velocity_status: str
    v2_m_s: float

    node_id: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_surface_elevation_m: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_v1_m_s: np.ndarray
    node_v2_m_s: np.ndarray
    node_weathering_thickness_m: np.ndarray
    node_refractor_elevation_m: np.ndarray
    node_solution_status: np.ndarray
    node_local_v2_status: np.ndarray
    node_weathering_status: np.ndarray
    node_pick_count: np.ndarray
    node_used_observation_count: np.ndarray
    node_rejected_observation_count: np.ndarray

    source_endpoint: RefractionWeatheringEndpointComponents
    receiver_endpoint: RefractionWeatheringEndpointComponents

    trace_index_sorted: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_weathering_thickness_m_sorted: np.ndarray
    receiver_weathering_thickness_m_sorted: np.ndarray
    source_refractor_elevation_m_sorted: np.ndarray
    receiver_refractor_elevation_m_sorted: np.ndarray
    source_weathering_status_sorted: np.ndarray
    receiver_weathering_status_sorted: np.ndarray
    trace_weathering_thickness_m_sorted: np.ndarray
    trace_weathering_status_sorted: np.ndarray

    cell_id: np.ndarray
    cell_v2_m_s: np.ndarray
    cell_velocity_status: np.ndarray
    cell_observation_count: np.ndarray

    qc: dict[str, Any]


def compute_weathering_thickness_scalar_from_half_intercept_time(
    half_intercept_time_s: float,
    v1_m_s: float,
    v2_m_s: float,
) -> float:
    """Compute scalar one-layer weathering thickness from half-intercept time."""
    t1 = _finite_float(half_intercept_time_s, name='half_intercept_time_s')
    v1 = _positive_finite_float(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float(v2_m_s, name='v2_m_s')
    if v2 <= v1:
        raise RefractionWeatheringError('v2_m_s must be greater than v1_m_s')
    return float(t1 * v2 * v1 / np.sqrt(v2 * v2 - v1 * v1))


def compute_weathering_thickness_from_half_intercept_time(
    half_intercept_time_s: np.ndarray | float,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
) -> np.ndarray:
    """Compute one-layer weathering thickness from T1, V1, and V2."""
    t1 = _float_array(half_intercept_time_s, name='half_intercept_time_s')
    v1 = _float_array(v1_m_s, name='v1_m_s')
    v2 = _float_array(v2_m_s, name='v2_m_s')
    t1, v1, v2 = _broadcast((t1, v1, v2), names=('half_intercept_time_s', 'v1_m_s', 'v2_m_s'))
    if not (np.all(np.isfinite(t1)) and np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        raise RefractionWeatheringError('half_intercept_time_s, v1_m_s, and v2_m_s must be finite')
    if np.any(v1 <= 0.0) or np.any(v2 <= 0.0):
        raise RefractionWeatheringError('v1_m_s and v2_m_s must be positive')
    if np.any(v2 <= v1):
        raise RefractionWeatheringError('v2_m_s must be greater than v1_m_s')
    return np.ascontiguousarray(t1 * v2 * v1 / np.sqrt(v2 * v2 - v1 * v1), dtype=np.float64)


def compute_weathering_thickness_from_half_intercept_time_with_status(
    *,
    half_intercept_time_s: np.ndarray | float,
    surface_elevation_m: np.ndarray | float,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    solution_status: np.ndarray | str | None = None,
    local_v2_status: np.ndarray | str | None = None,
    max_weathering_thickness_m: float | None = None,
) -> RefractionWeatheringThicknessComputation:
    """Compute thickness, refractor elevation, and status without raising per value."""
    t1 = _float_array(half_intercept_time_s, name='half_intercept_time_s')
    surface = _float_array(surface_elevation_m, name='surface_elevation_m')
    v1 = _float_array(v1_m_s, name='v1_m_s')
    v2 = _float_array(v2_m_s, name='v2_m_s')
    t1, surface, v1, v2 = _broadcast(
        (t1, surface, v1, v2),
        names=('half_intercept_time_s', 'surface_elevation_m', 'v1_m_s', 'v2_m_s'),
    )
    shape = t1.shape
    solution = _status_array(solution_status, shape=shape, fill_value='solved')
    local_v2 = _status_array(local_v2_status, shape=shape, fill_value='ok')

    status = np.full(shape, 'ok', dtype=_STATUS_DTYPE)
    thickness = np.full(shape, np.nan, dtype=np.float64)
    refractor = np.full(shape, np.nan, dtype=np.float64)

    _overlay_non_ok_status(status, solution, allowed_ok={'ok', 'solved'})
    _overlay_non_ok_status(status, local_v2, allowed_ok={'ok', 'solved'})

    invalid_surface = ~np.isfinite(surface)
    status[invalid_surface & (status == 'ok')] = 'invalid_surface_elevation'

    invalid_nonfinite = ~(np.isfinite(t1) & np.isfinite(v1) & np.isfinite(v2))
    status[invalid_nonfinite & (status == 'ok')] = 'invalid_nonfinite_input'

    invalid_velocity = (v1 <= 0.0) | (v2 <= 0.0)
    status[invalid_velocity & (status == 'ok')] = 'invalid_velocity'
    invalid_order = (v2 <= v1) & ~(invalid_nonfinite | invalid_velocity)
    status[invalid_order & (status == 'ok')] = 'invalid_velocity_order'

    valid = status == 'ok'
    if np.any(valid):
        thickness[valid] = (
            t1[valid]
            * v2[valid]
            * v1[valid]
            / np.sqrt(v2[valid] * v2[valid] - v1[valid] * v1[valid])
        )
        refractor[valid] = surface[valid] - thickness[valid]

    invalid_thickness = valid & ~np.isfinite(thickness)
    status[invalid_thickness] = 'invalid_weathering_thickness'
    negative = np.isfinite(thickness) & (thickness < 0.0)
    status[negative] = 'negative_weathering_thickness'
    if max_weathering_thickness_m is not None:
        max_thickness = _positive_finite_float(
            max_weathering_thickness_m,
            name='max_weathering_thickness_m',
        )
        exceeds = np.isfinite(thickness) & (thickness > max_thickness)
        status[exceeds] = 'exceeds_max_thickness'

    zero = np.isfinite(thickness) & (thickness == 0.0) & (status == 'ok')
    status[zero] = 'zero_thickness'
    invalid_output = status != 'ok'
    invalid_output &= status != 'zero_thickness'
    thickness[invalid_output] = np.nan
    refractor[invalid_output] = np.nan
    invalid_refractor = (status == 'ok') & ~np.isfinite(refractor)
    status[invalid_refractor] = 'invalid_refractor_elevation'
    refractor[invalid_refractor] = np.nan
    thickness[invalid_refractor] = np.nan

    return RefractionWeatheringThicknessComputation(
        weathering_thickness_m=np.ascontiguousarray(thickness),
        refractor_elevation_m=np.ascontiguousarray(refractor),
        weathering_status=np.ascontiguousarray(status),
    )


def build_refraction_weathering_model_from_half_intercept_result(
    *,
    input_model: RefractionStaticInputModel,
    half_intercept_result: RefractionHalfInterceptResult,
    model: RefractionStaticModelOptions,
) -> RefractionWeatheringModel:
    """Build endpoint and trace weathering thickness from a half-intercept result."""
    _validate_weathering_inputs(
        input_model=input_model,
        half_intercept_result=half_intercept_result,
        model=model,
    )
    v1 = resolve_weathering_velocity_m_s(model=model, name='model.weathering_velocity_m_s')
    v1 = _positive_finite_float(v1, name='model.weathering_velocity_m_s')
    if model.max_weathering_thickness_m is not None:
        max_thickness = _positive_finite_float(
            model.max_weathering_thickness_m,
            name='model.max_weathering_thickness_m',
        )
    else:
        max_thickness = None

    node_local_v2, node_local_v2_status = _local_v2_for_nodes(
        input_model=input_model,
        half_intercept_result=half_intercept_result,
        model=model,
    )
    node_components = compute_weathering_thickness_from_half_intercept_time_with_status(
        half_intercept_time_s=half_intercept_result.node_half_intercept_time_s,
        surface_elevation_m=_node_elevation_from_input_model(
            input_model,
            half_intercept_result,
        ),
        v1_m_s=np.full(half_intercept_result.node_id.shape, v1, dtype=np.float64),
        v2_m_s=node_local_v2,
        solution_status=half_intercept_result.node_solution_status,
        local_v2_status=node_local_v2_status,
        max_weathering_thickness_m=max_thickness,
    )
    node_elevation = _node_elevation_from_input_model(input_model, half_intercept_result)

    source_endpoint = _build_endpoint_components(
        input_model=input_model,
        endpoint=half_intercept_result.source_endpoint,
        half_intercept_result=half_intercept_result,
        model=model,
        v1=v1,
        max_weathering_thickness_m=max_thickness,
        kind='source',
    )
    receiver_endpoint = _build_endpoint_components(
        input_model=input_model,
        endpoint=half_intercept_result.receiver_endpoint,
        half_intercept_result=half_intercept_result,
        model=model,
        v1=v1,
        max_weathering_thickness_m=max_thickness,
        kind='receiver',
    )

    source_thickness = _map_endpoint_values_to_trace(
        half_intercept_result.source_endpoint_key_sorted,
        source_endpoint.endpoint_key,
        source_endpoint.weathering_thickness_m,
        fill_value=np.nan,
    )
    receiver_thickness = _map_endpoint_values_to_trace(
        half_intercept_result.receiver_endpoint_key_sorted,
        receiver_endpoint.endpoint_key,
        receiver_endpoint.weathering_thickness_m,
        fill_value=np.nan,
    )
    source_refractor = _map_endpoint_values_to_trace(
        half_intercept_result.source_endpoint_key_sorted,
        source_endpoint.endpoint_key,
        source_endpoint.refractor_elevation_m,
        fill_value=np.nan,
    )
    receiver_refractor = _map_endpoint_values_to_trace(
        half_intercept_result.receiver_endpoint_key_sorted,
        receiver_endpoint.endpoint_key,
        receiver_endpoint.refractor_elevation_m,
        fill_value=np.nan,
    )
    source_status = _map_endpoint_status_to_trace(
        half_intercept_result.source_endpoint_key_sorted,
        source_endpoint.endpoint_key,
        source_endpoint.weathering_status,
        fill_value='missing_endpoint',
    )
    receiver_status = _map_endpoint_status_to_trace(
        half_intercept_result.receiver_endpoint_key_sorted,
        receiver_endpoint.endpoint_key,
        receiver_endpoint.weathering_status,
        fill_value='missing_endpoint',
    )
    trace_status = _trace_weathering_status(
        source_status=source_status,
        receiver_status=receiver_status,
        source_thickness=source_thickness,
        receiver_thickness=receiver_thickness,
    )
    qc = _build_weathering_qc(
        half_intercept_result=half_intercept_result,
        v1=v1,
        source_endpoint=source_endpoint,
        receiver_endpoint=receiver_endpoint,
        node_status=node_components.weathering_status,
        trace_status=trace_status,
        max_weathering_thickness_m=max_thickness,
    )

    return RefractionWeatheringModel(
        file_id=half_intercept_result.file_id,
        n_traces=half_intercept_result.n_traces,
        bedrock_velocity_mode=half_intercept_result.bedrock_velocity_mode,
        weathering_velocity_m_s=v1,
        bedrock_velocity_m_s=half_intercept_result.bedrock_velocity_m_s,
        bedrock_slowness_s_per_m=half_intercept_result.bedrock_slowness_s_per_m,
        bedrock_velocity_status=half_intercept_result.bedrock_velocity_status,
        v2_m_s=half_intercept_result.v2_m_s,
        node_id=np.ascontiguousarray(half_intercept_result.node_id, dtype=np.int64),
        node_x_m=_node_x_from_input_model(input_model, half_intercept_result),
        node_y_m=_node_y_from_input_model(input_model, half_intercept_result),
        node_surface_elevation_m=node_elevation,
        node_half_intercept_time_s=np.ascontiguousarray(
            half_intercept_result.node_half_intercept_time_s,
            dtype=np.float64,
        ),
        node_v1_m_s=np.full(half_intercept_result.node_id.shape, v1, dtype=np.float64),
        node_v2_m_s=node_local_v2,
        node_weathering_thickness_m=node_components.weathering_thickness_m,
        node_refractor_elevation_m=node_components.refractor_elevation_m,
        node_solution_status=np.ascontiguousarray(
            half_intercept_result.node_solution_status
        ),
        node_local_v2_status=node_local_v2_status,
        node_weathering_status=node_components.weathering_status,
        node_pick_count=np.ascontiguousarray(
            half_intercept_result.node_pick_count,
            dtype=np.int64,
        ),
        node_used_observation_count=np.ascontiguousarray(
            half_intercept_result.node_used_observation_count,
            dtype=np.int64,
        ),
        node_rejected_observation_count=np.ascontiguousarray(
            half_intercept_result.node_rejected_observation_count,
            dtype=np.int64,
        ),
        source_endpoint=source_endpoint,
        receiver_endpoint=receiver_endpoint,
        trace_index_sorted=np.ascontiguousarray(
            half_intercept_result.trace_index_sorted,
            dtype=np.int64,
        ),
        source_endpoint_key_sorted=np.ascontiguousarray(
            half_intercept_result.source_endpoint_key_sorted
        ),
        receiver_endpoint_key_sorted=np.ascontiguousarray(
            half_intercept_result.receiver_endpoint_key_sorted
        ),
        source_node_id_sorted=np.ascontiguousarray(
            half_intercept_result.source_node_id_sorted,
            dtype=np.int64,
        ),
        receiver_node_id_sorted=np.ascontiguousarray(
            half_intercept_result.receiver_node_id_sorted,
            dtype=np.int64,
        ),
        source_weathering_thickness_m_sorted=source_thickness,
        receiver_weathering_thickness_m_sorted=receiver_thickness,
        source_refractor_elevation_m_sorted=source_refractor,
        receiver_refractor_elevation_m_sorted=receiver_refractor,
        source_weathering_status_sorted=source_status,
        receiver_weathering_status_sorted=receiver_status,
        trace_weathering_thickness_m_sorted=np.ascontiguousarray(
            source_thickness + receiver_thickness,
            dtype=np.float64,
        ),
        trace_weathering_status_sorted=trace_status,
        cell_id=np.ascontiguousarray(half_intercept_result.cell_id, dtype=np.int64),
        cell_v2_m_s=np.ascontiguousarray(
            half_intercept_result.cell_v2_m_s,
            dtype=np.float64,
        ),
        cell_velocity_status=np.ascontiguousarray(
            half_intercept_result.cell_velocity_status
        ),
        cell_observation_count=np.ascontiguousarray(
            half_intercept_result.cell_observation_count,
            dtype=np.int64,
        ),
        qc=qc,
    )


def _validate_weathering_inputs(
    *,
    input_model: RefractionStaticInputModel,
    half_intercept_result: RefractionHalfInterceptResult,
    model: RefractionStaticModelOptions,
) -> None:
    if not isinstance(input_model, RefractionStaticInputModel):
        raise RefractionWeatheringError(
            'input_model must be a RefractionStaticInputModel instance'
        )
    if not isinstance(half_intercept_result, RefractionHalfInterceptResult):
        raise RefractionWeatheringError(
            'half_intercept_result must be a RefractionHalfInterceptResult instance'
        )
    if not isinstance(model, RefractionStaticModelOptions):
        raise RefractionWeatheringError(
            'model must be a RefractionStaticModelOptions instance'
        )
    if model.method != 'gli_variable_thickness':
        raise RefractionWeatheringError('model.method must be gli_variable_thickness')
    if model.bedrock_velocity_mode != half_intercept_result.bedrock_velocity_mode:
        raise RefractionWeatheringError(
            'model.bedrock_velocity_mode must match half_intercept_result'
        )
    if half_intercept_result.bedrock_velocity_mode == 'solve_cell' and model.refractor_cell is None:
        raise RefractionWeatheringError(
            'model.refractor_cell is required for solve_cell weathering'
        )
    if int(input_model.n_traces) != int(half_intercept_result.n_traces):
        raise RefractionWeatheringError(
            'input_model.n_traces must match half_intercept_result.n_traces'
        )
    if not np.array_equal(
        np.asarray(input_model.endpoint_table.node_id, dtype=np.int64),
        np.asarray(half_intercept_result.node_id, dtype=np.int64),
    ):
        raise RefractionWeatheringError(
            'input_model.endpoint_table.node_id must match half_intercept_result.node_id'
        )


def _build_endpoint_components(
    *,
    input_model: RefractionStaticInputModel,
    endpoint: RefractionHalfInterceptEndpointResult,
    half_intercept_result: RefractionHalfInterceptResult,
    model: RefractionStaticModelOptions,
    v1: float,
    max_weathering_thickness_m: float | None,
    kind: str,
) -> RefractionWeatheringEndpointComponents:
    x, y, elevation = _endpoint_geometry(
        endpoint.node_id,
        input_model=input_model,
        result=half_intercept_result,
    )
    local_v2, local_v2_status = _local_v2_for_points(
        x_m=x,
        y_m=y,
        half_intercept_result=half_intercept_result,
        model=model,
    )
    components = compute_weathering_thickness_from_half_intercept_time_with_status(
        half_intercept_time_s=endpoint.half_intercept_time_s,
        surface_elevation_m=elevation,
        v1_m_s=np.full(endpoint.node_id.shape, v1, dtype=np.float64),
        v2_m_s=local_v2,
        solution_status=endpoint.solution_status,
        local_v2_status=local_v2_status,
        max_weathering_thickness_m=max_weathering_thickness_m,
    )
    return RefractionWeatheringEndpointComponents(
        endpoint_key=np.ascontiguousarray(endpoint.endpoint_key),
        endpoint_id=None
        if endpoint.endpoint_id is None
        else np.ascontiguousarray(endpoint.endpoint_id, dtype=np.int64),
        node_id=np.ascontiguousarray(endpoint.node_id, dtype=np.int64),
        x_m=x,
        y_m=y,
        surface_elevation_m=elevation,
        half_intercept_time_s=np.ascontiguousarray(
            endpoint.half_intercept_time_s,
            dtype=np.float64,
        ),
        v1_m_s=np.full(endpoint.node_id.shape, v1, dtype=np.float64),
        v2_m_s=local_v2,
        weathering_thickness_m=components.weathering_thickness_m,
        refractor_elevation_m=components.refractor_elevation_m,
        solution_status=np.ascontiguousarray(endpoint.solution_status),
        local_v2_status=local_v2_status,
        weathering_status=components.weathering_status,
        pick_count=np.ascontiguousarray(endpoint.pick_count, dtype=np.int64),
        used_observation_count=np.ascontiguousarray(
            endpoint.used_observation_count,
            dtype=np.int64,
        ),
        rejected_observation_count=np.ascontiguousarray(
            endpoint.rejected_observation_count,
            dtype=np.int64,
        ),
    )


def _local_v2_for_nodes(
    *,
    input_model: RefractionStaticInputModel,
    half_intercept_result: RefractionHalfInterceptResult,
    model: RefractionStaticModelOptions,
) -> tuple[np.ndarray, np.ndarray]:
    return _local_v2_for_points(
        x_m=_node_x_from_input_model(input_model, half_intercept_result),
        y_m=_node_y_from_input_model(input_model, half_intercept_result),
        half_intercept_result=half_intercept_result,
        model=model,
    )


def _local_v2_for_points(
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    half_intercept_result: RefractionHalfInterceptResult,
    model: RefractionStaticModelOptions,
) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(x_m).shape
    if half_intercept_result.bedrock_velocity_mode != 'solve_cell':
        v2 = float(half_intercept_result.v2_m_s)
        status = 'ok' if np.isfinite(v2) and v2 > 0.0 else 'invalid_local_v2'
        return (
            np.full(shape, v2, dtype=np.float64),
            np.full(shape, status, dtype=_STATUS_DTYPE),
        )

    refractor_cell = model.refractor_cell
    if refractor_cell is None:
        raise RefractionWeatheringError(
            'model.refractor_cell is required for solve_cell weathering'
        )
    projected = project_refraction_cell_points(
        x_m=x_m,
        y_m=y_m,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    grid = build_refraction_cell_grid(effective_refraction_cell_grid_config(refractor_cell))
    assignment = assign_points_to_refraction_cells(
        grid,
        x_m=projected.x_m,
        y_m=projected.y_m,
    )
    cell_v2 = np.asarray(half_intercept_result.cell_v2_m_s, dtype=np.float64)
    cell_status = np.asarray(half_intercept_result.cell_velocity_status)
    v2 = np.full(shape, np.nan, dtype=np.float64)
    status = np.full(shape, 'outside_refractor_cell_grid', dtype=_STATUS_DTYPE)
    cell_id = np.asarray(assignment.cell_id, dtype=np.int64)
    inside = np.asarray(assignment.inside_grid_mask, dtype=bool)
    for index, raw_cell_id in enumerate(cell_id.tolist()):
        if not bool(inside[index]):
            continue
        current_cell = int(raw_cell_id)
        if current_cell < 0 or current_cell >= cell_v2.shape[0]:
            status[index] = 'outside_refractor_cell_grid'
            continue
        raw_status = str(cell_status[current_cell])
        v2[index] = float(cell_v2[current_cell])
        status[index] = _local_v2_status(raw_status, v2[index])
    return np.ascontiguousarray(v2), np.ascontiguousarray(status)


def _local_v2_status(raw_status: str, v2: float) -> str:
    if raw_status in {'solved', 'ok', 'clipped_lower', 'clipped_upper'}:
        return 'ok' if np.isfinite(v2) and v2 > 0.0 else 'invalid_local_v2'
    if raw_status == 'inactive':
        return 'inactive_v2_cell'
    if raw_status == 'low_fold':
        return 'low_fold_v2_cell'
    return 'invalid_local_v2'


def _node_x_from_input_model(
    input_model: RefractionStaticInputModel,
    result: RefractionHalfInterceptResult,
) -> np.ndarray:
    return _endpoint_table_values_by_node(input_model, result, 'x_m')


def _node_y_from_input_model(
    input_model: RefractionStaticInputModel,
    result: RefractionHalfInterceptResult,
) -> np.ndarray:
    return _endpoint_table_values_by_node(input_model, result, 'y_m')


def _node_elevation_from_input_model(
    input_model: RefractionStaticInputModel,
    result: RefractionHalfInterceptResult,
) -> np.ndarray:
    return _endpoint_table_values_by_node(input_model, result, 'elevation_m')


def _endpoint_table_values_by_node(
    input_model: RefractionStaticInputModel,
    result: RefractionHalfInterceptResult,
    attr: str,
) -> np.ndarray:
    table = input_model.endpoint_table
    table_node_id = np.asarray(table.node_id, dtype=np.int64)
    values = np.asarray(getattr(table, attr), dtype=np.float64)
    lookup = {
        int(node): float(value)
        for node, value in zip(table_node_id, values, strict=True)
    }
    nodes = np.asarray(result.node_id, dtype=np.int64)
    return np.ascontiguousarray(
        np.asarray([lookup.get(int(node), np.nan) for node in nodes], dtype=np.float64)
    )


def _endpoint_geometry(
    node_id: np.ndarray,
    *,
    input_model: RefractionStaticInputModel,
    result: RefractionHalfInterceptResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup_x = _lookup_by_node(
        result.node_id,
        _node_x_from_input_model(input_model, result),
    )
    lookup_y = _lookup_by_node(
        result.node_id,
        _node_y_from_input_model(input_model, result),
    )
    lookup_z = _lookup_by_node(
        result.node_id,
        _node_elevation_from_input_model(input_model, result),
    )
    nodes = np.asarray(node_id, dtype=np.int64)
    return (
        np.ascontiguousarray(
            np.asarray([lookup_x.get(int(node), np.nan) for node in nodes], dtype=np.float64)
        ),
        np.ascontiguousarray(
            np.asarray([lookup_y.get(int(node), np.nan) for node in nodes], dtype=np.float64)
        ),
        np.ascontiguousarray(
            np.asarray([lookup_z.get(int(node), np.nan) for node in nodes], dtype=np.float64)
        ),
    )


def _lookup_by_node(node_id: np.ndarray, values: np.ndarray) -> dict[int, float]:
    return {int(node): float(value) for node, value in zip(node_id, values, strict=True)}


def _map_endpoint_values_to_trace(
    trace_key: np.ndarray,
    endpoint_key: np.ndarray,
    values: np.ndarray,
    *,
    fill_value: float,
) -> np.ndarray:
    lookup = {str(key): float(value) for key, value in zip(endpoint_key, values, strict=True)}
    return np.ascontiguousarray(
        np.asarray([lookup.get(str(key), fill_value) for key in trace_key], dtype=np.float64)
    )


def _map_endpoint_status_to_trace(
    trace_key: np.ndarray,
    endpoint_key: np.ndarray,
    status: np.ndarray,
    *,
    fill_value: str,
) -> np.ndarray:
    lookup = {str(key): str(value) for key, value in zip(endpoint_key, status, strict=True)}
    return np.ascontiguousarray(
        np.asarray([lookup.get(str(key), fill_value) for key in trace_key], dtype=_STATUS_DTYPE)
    )


def _trace_weathering_status(
    *,
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    source_thickness: np.ndarray,
    receiver_thickness: np.ndarray,
) -> np.ndarray:
    status = np.full(source_status.shape, 'ok', dtype=_STATUS_DTYPE)
    for index, (src_status, rec_status) in enumerate(
        zip(source_status.tolist(), receiver_status.tolist(), strict=True)
    ):
        src = str(src_status)
        rec = str(rec_status)
        if src == rec:
            status[index] = src
        elif src == 'ok':
            status[index] = rec
        elif rec == 'ok':
            status[index] = src
        else:
            status[index] = 'mixed'
    invalid_value = ~(np.isfinite(source_thickness) & np.isfinite(receiver_thickness))
    status[invalid_value & (status == 'ok')] = 'invalid_weathering_thickness'
    return np.ascontiguousarray(status)


def _build_weathering_qc(
    *,
    half_intercept_result: RefractionHalfInterceptResult,
    v1: float,
    source_endpoint: RefractionWeatheringEndpointComponents,
    receiver_endpoint: RefractionWeatheringEndpointComponents,
    node_status: np.ndarray,
    trace_status: np.ndarray,
    max_weathering_thickness_m: float | None,
) -> dict[str, Any]:
    qc = {
        'file_id': half_intercept_result.file_id,
        'n_traces': int(half_intercept_result.n_traces),
        'bedrock_velocity_mode': half_intercept_result.bedrock_velocity_mode,
        'weathering_velocity_m_s': float(v1),
        'max_weathering_thickness_m': _json_optional_float(max_weathering_thickness_m),
        'source_weathering_status_counts': _status_counts(
            source_endpoint.weathering_status
        ),
        'receiver_weathering_status_counts': _status_counts(
            receiver_endpoint.weathering_status
        ),
        'node_weathering_status_counts': _status_counts(node_status),
        'trace_weathering_status_counts': _status_counts(trace_status),
    }
    _copy_cell_qc(qc, half_intercept_result.qc)
    design_qc = half_intercept_result.qc.get('design_matrix')
    if isinstance(design_qc, dict):
        _copy_cell_qc(qc, design_qc)
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


def _overlay_non_ok_status(
    status: np.ndarray,
    candidate: np.ndarray,
    *,
    allowed_ok: set[str],
) -> None:
    for index, raw in enumerate(candidate.tolist()):
        value = str(raw)
        if value not in allowed_ok and status[index] == 'ok':
            status[index] = value


def _status_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    return {
        str(status): int(count)
        for status, count in zip(unique.tolist(), counts.tolist(), strict=True)
    }


def _float_array(values: np.ndarray | float, *, name: str) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RefractionWeatheringError(f'{name} must be numeric') from exc
    return np.ascontiguousarray(out, dtype=np.float64)


def _status_array(
    values: np.ndarray | str | None,
    *,
    shape: tuple[int, ...],
    fill_value: str,
) -> np.ndarray:
    if values is None:
        return np.full(shape, fill_value, dtype=_STATUS_DTYPE)
    out = np.asarray(values)
    if out.shape == ():
        return np.full(shape, str(out.item()), dtype=_STATUS_DTYPE)
    if out.shape != shape:
        raise RefractionWeatheringError('status shape mismatch')
    return np.ascontiguousarray(out.astype(_STATUS_DTYPE, copy=False))


def _broadcast(
    arrays: tuple[np.ndarray, ...],
    *,
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    try:
        broadcasted = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        joined = ', '.join(names)
        raise RefractionWeatheringError(f'{joined} could not be broadcast together') from exc
    return tuple(np.ascontiguousarray(array, dtype=np.float64) for array in broadcasted)


def _finite_float(value: float, *, name: str) -> float:
    out = _float(value, name=name)
    if not np.isfinite(out):
        raise RefractionWeatheringError(f'{name} must be finite')
    return out


def _positive_finite_float(value: float, *, name: str) -> float:
    out = _finite_float(value, name=name)
    if out <= 0.0:
        raise RefractionWeatheringError(f'{name} must be positive')
    return out


def _float(value: float, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionWeatheringError(f'{name} must be numeric') from exc


def _json_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    'RefractionWeatheringEndpointComponents',
    'RefractionWeatheringError',
    'RefractionWeatheringModel',
    'RefractionWeatheringThicknessComputation',
    'build_refraction_weathering_model_from_half_intercept_result',
    'compute_weathering_thickness_from_half_intercept_time',
    'compute_weathering_thickness_from_half_intercept_time_with_status',
    'compute_weathering_thickness_scalar_from_half_intercept_time',
]
