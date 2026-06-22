"""Pure multi-layer T1LSST conversion and datum composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from seis_statics.refraction.cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_points,
)
from seis_statics.refraction.cell_grid import (
    assign_points_to_refraction_cells,
    build_refraction_cell_grid,
)
from seis_statics.refraction.datum import (
    RefractionDatumStaticsResult,
    ResolvedFloatingDatum,
    build_refraction_datum_statics,
    smooth_refraction_floating_datum_elevation,
)
from seis_statics.refraction.first_layer import resolve_weathering_velocity_m_s
from seis_statics.refraction.multilayer_solver import (
    RefractionMultilayerTimeTermSolveResult,
    solve_refraction_multilayer_time_terms,
)
from seis_statics.refraction.options import (
    RefractionStaticDatumOptions,
    RefractionStaticModelOptions,
    RefractionStaticSolverOptions,
)
from seis_statics.refraction.t1lsst import (
    RefractionT1LSST2LayerThicknessResult,
    RefractionT1LSST3LayerThicknessResult,
    compute_t1lsst_2layer_thicknesses_with_status,
    compute_t1lsst_3layer_thicknesses_with_status,
)
from seis_statics.refraction.types import (
    RefractionStaticInputModel,
    RefractionTraceFieldCorrectionResult,
    ResolvedRefractionFirstLayer,
)


_STATUS_DTYPE = '<U64'
_KEY_DTYPE = object
_OK_STATUSES = {'ok', 'solved', 'zero_thickness'}
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


class RefractionMultilayerConversionError(ValueError):
    """Raised when multi-layer conversion inputs are inconsistent."""


@dataclass(frozen=True)
class RefractionMultilayerEndpointConversion:
    """Endpoint-level T1LSST thickness and replacement outputs."""

    endpoint_kind: np.ndarray
    endpoint_key: np.ndarray
    endpoint_id: np.ndarray | None
    node_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    surface_elevation_m: np.ndarray
    t1_s: np.ndarray
    t2_s: np.ndarray
    t3_s: np.ndarray
    v1_m_s: np.ndarray
    v2_m_s: np.ndarray
    v3_m_s: np.ndarray
    vsub_m_s: np.ndarray
    sh1_m: np.ndarray
    sh2_m: np.ndarray
    sh3_m: np.ndarray
    refractor_elevation_m: np.ndarray
    weathering_replacement_shift_s: np.ndarray
    conversion_status: np.ndarray
    static_status: np.ndarray


@dataclass(frozen=True)
class RefractionMultilayerConversionResult:
    """Trace-order package result for 2/3-layer T1LSST conversion."""

    layer_count: Literal[2, 3]
    solve_result: RefractionMultilayerTimeTermSolveResult
    source_endpoint: RefractionMultilayerEndpointConversion
    receiver_endpoint: RefractionMultilayerEndpointConversion
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_weathering_replacement_shift_s_sorted: np.ndarray
    receiver_weathering_replacement_shift_s_sorted: np.ndarray
    weathering_replacement_trace_shift_s_sorted: np.ndarray
    source_static_status_sorted: np.ndarray
    receiver_static_status_sorted: np.ndarray
    trace_static_status_sorted: np.ndarray
    trace_static_valid_mask_sorted: np.ndarray
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionMultilayerDatumStaticsResult:
    """Combined multi-layer conversion and datum static composition."""

    conversion: RefractionMultilayerConversionResult
    datum: RefractionDatumStaticsResult
    qc: dict[str, Any]


def build_refraction_multilayer_conversion(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    solve_result: RefractionMultilayerTimeTermSolveResult,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    layer_count: int | None = None,
) -> RefractionMultilayerConversionResult:
    """Convert a multi-layer time-term solve to endpoint/trace static arrays."""
    _validate_inputs(input_model=input_model, model=model, solve_result=solve_result)
    count = _layer_count(solve_result=solve_result, layer_count=layer_count)
    v1 = resolve_weathering_velocity_m_s(
        model=model,
        resolved_first_layer=resolved_first_layer,
        name='model.resolved_weathering_velocity_m_s',
    )
    source_endpoint = _build_endpoint_conversion(
        endpoint_kind='source',
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        layer_count=count,
        v1_m_s=v1,
    )
    receiver_endpoint = _build_endpoint_conversion(
        endpoint_kind='receiver',
        input_model=input_model,
        model=model,
        solve_result=solve_result,
        layer_count=count,
        v1_m_s=v1,
    )
    source_keys_sorted = np.ascontiguousarray(
        np.asarray(input_model.source_endpoint_key_sorted, dtype=_KEY_DTYPE)
    )
    receiver_keys_sorted = np.ascontiguousarray(
        np.asarray(input_model.receiver_endpoint_key_sorted, dtype=_KEY_DTYPE)
    )
    source_shift_sorted = _map_endpoint_float_to_trace(
        source_keys_sorted,
        source_endpoint.endpoint_key,
        source_endpoint.weathering_replacement_shift_s,
    )
    receiver_shift_sorted = _map_endpoint_float_to_trace(
        receiver_keys_sorted,
        receiver_endpoint.endpoint_key,
        receiver_endpoint.weathering_replacement_shift_s,
    )
    source_status_sorted = _map_endpoint_status_to_trace(
        source_keys_sorted,
        source_endpoint.endpoint_key,
        source_endpoint.static_status,
    )
    receiver_status_sorted = _map_endpoint_status_to_trace(
        receiver_keys_sorted,
        receiver_endpoint.endpoint_key,
        receiver_endpoint.static_status,
    )
    trace_shift = _combine_trace_shifts(source_shift_sorted, receiver_shift_sorted)
    trace_status = _classify_trace_status(
        source_status_sorted,
        receiver_status_sorted,
        trace_shift,
    )
    trace_valid = (trace_status == 'ok') & np.isfinite(trace_shift)
    qc = {
        'method': 'multilayer_time_term',
        'static_component': 'multilayer_t1lsst_conversion',
        'file_id': input_model.file_id,
        'layer_count': count,
        'n_traces': int(input_model.n_traces),
        'n_source_endpoints': int(source_endpoint.endpoint_key.shape[0]),
        'n_receiver_endpoints': int(receiver_endpoint.endpoint_key.shape[0]),
        'n_valid_trace_statics': int(np.count_nonzero(trace_valid)),
        'solve': solve_result.qc,
    }
    return RefractionMultilayerConversionResult(
        layer_count=count,
        solve_result=solve_result,
        source_endpoint=source_endpoint,
        receiver_endpoint=receiver_endpoint,
        source_endpoint_key_sorted=source_keys_sorted,
        receiver_endpoint_key_sorted=receiver_keys_sorted,
        source_weathering_replacement_shift_s_sorted=source_shift_sorted,
        receiver_weathering_replacement_shift_s_sorted=receiver_shift_sorted,
        weathering_replacement_trace_shift_s_sorted=trace_shift,
        source_static_status_sorted=source_status_sorted,
        receiver_static_status_sorted=receiver_status_sorted,
        trace_static_status_sorted=trace_status,
        trace_static_valid_mask_sorted=np.ascontiguousarray(trace_valid, dtype=bool),
        qc=qc,
    )


def compute_refraction_multilayer_datum_statics_from_input_model(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    datum_options: RefractionStaticDatumOptions | None = None,
    solver_options: RefractionStaticSolverOptions | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    layer_count: int | None = None,
    floating_datum: ResolvedFloatingDatum | None = None,
    source_floating_datum_elevation_m: np.ndarray | float | None = None,
    receiver_floating_datum_elevation_m: np.ndarray | float | None = None,
    max_abs_datum_shift_ms: float | None = None,
    trace_field_correction: RefractionTraceFieldCorrectionResult | None = None,
    apply_field_correction_to_trace_shift: bool = False,
    include_diagnostics: bool = False,
) -> RefractionMultilayerDatumStaticsResult:
    """Solve multi-layer time terms, convert them, and call the datum API."""
    options = RefractionStaticDatumOptions() if datum_options is None else datum_options
    solve = solve_refraction_multilayer_time_terms(
        input_model=input_model,
        model=model,
        solver_options=solver_options,
        resolved_first_layer=resolved_first_layer,
        include_diagnostics=include_diagnostics,
    )
    conversion = build_refraction_multilayer_conversion(
        input_model=input_model,
        model=model,
        solve_result=solve,
        resolved_first_layer=resolved_first_layer,
        layer_count=layer_count,
    )
    resolved_floating = _resolve_floating_datum(
        options=options,
        conversion=conversion,
        floating_datum=floating_datum,
        source_floating_datum_elevation_m=source_floating_datum_elevation_m,
        receiver_floating_datum_elevation_m=receiver_floating_datum_elevation_m,
    )
    datum = build_refraction_datum_statics(
        source_endpoint_key=conversion.source_endpoint.endpoint_key,
        source_endpoint_id=conversion.source_endpoint.endpoint_id,
        source_node_id=conversion.source_endpoint.node_id,
        source_surface_elevation_m=conversion.source_endpoint.surface_elevation_m,
        source_refractor_elevation_m=conversion.source_endpoint.refractor_elevation_m,
        source_weathering_replacement_shift_s=(
            conversion.source_endpoint.weathering_replacement_shift_s
        ),
        source_weathering_replacement_status=conversion.source_endpoint.static_status,
        receiver_endpoint_key=conversion.receiver_endpoint.endpoint_key,
        receiver_endpoint_id=conversion.receiver_endpoint.endpoint_id,
        receiver_node_id=conversion.receiver_endpoint.node_id,
        receiver_surface_elevation_m=conversion.receiver_endpoint.surface_elevation_m,
        receiver_refractor_elevation_m=(
            conversion.receiver_endpoint.refractor_elevation_m
        ),
        receiver_weathering_replacement_shift_s=(
            conversion.receiver_endpoint.weathering_replacement_shift_s
        ),
        receiver_weathering_replacement_status=(
            conversion.receiver_endpoint.static_status
        ),
        source_endpoint_key_sorted=conversion.source_endpoint_key_sorted,
        receiver_endpoint_key_sorted=conversion.receiver_endpoint_key_sorted,
        mode=options.mode,
        source_replacement_velocity_m_s=_replacement_velocity(
            conversion.source_endpoint,
            layer_count=conversion.layer_count,
        ),
        receiver_replacement_velocity_m_s=_replacement_velocity(
            conversion.receiver_endpoint,
            layer_count=conversion.layer_count,
        ),
        floating_datum=resolved_floating,
        source_floating_datum_elevation_m=source_floating_datum_elevation_m,
        receiver_floating_datum_elevation_m=receiver_floating_datum_elevation_m,
        flat_datum_elevation_m=options.flat_datum_elevation_m,
        allow_flat_datum_above_topography=options.allow_flat_datum_above_topography,
        allow_flat_datum_below_refractor=options.allow_flat_datum_below_refractor,
        max_abs_datum_shift_ms=max_abs_datum_shift_ms,
        trace_field_correction=trace_field_correction,
        apply_field_correction_to_trace_shift=apply_field_correction_to_trace_shift,
    )
    return RefractionMultilayerDatumStaticsResult(
        conversion=conversion,
        datum=datum,
        qc={
            'method': 'multilayer_time_term',
            'static_component': 'multilayer_datum_statics',
            'file_id': input_model.file_id,
            'layer_count': conversion.layer_count,
            'conversion': conversion.qc,
            'datum': datum.qc,
        },
    )


def _validate_inputs(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    solve_result: RefractionMultilayerTimeTermSolveResult,
) -> None:
    if not isinstance(input_model, RefractionStaticInputModel):
        raise TypeError('input_model must be RefractionStaticInputModel')
    if not isinstance(model, RefractionStaticModelOptions):
        raise TypeError('model must be RefractionStaticModelOptions')
    if not isinstance(solve_result, RefractionMultilayerTimeTermSolveResult):
        raise TypeError('solve_result must be RefractionMultilayerTimeTermSolveResult')
    if model.method != 'multilayer_time_term':
        raise RefractionMultilayerConversionError(
            'model.method must be multilayer_time_term'
        )


def _layer_count(
    *,
    solve_result: RefractionMultilayerTimeTermSolveResult,
    layer_count: int | None,
) -> Literal[2, 3]:
    count = len(solve_result.layer_results) if layer_count is None else int(layer_count)
    if count not in {2, 3}:
        raise RefractionMultilayerConversionError('layer_count must be 2 or 3')
    required = ('v2_t1', 'v3_t2') if count == 2 else ('v2_t1', 'v3_t2', 'vsub_t3')
    missing = [kind for kind in required if kind not in solve_result.layer_result_by_kind]
    if missing:
        raise RefractionMultilayerConversionError(
            f'solve_result is missing required layers: {", ".join(missing)}'
        )
    return 2 if count == 2 else 3


def _build_endpoint_conversion(
    *,
    endpoint_kind: Literal['source', 'receiver'],
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    solve_result: RefractionMultilayerTimeTermSolveResult,
    layer_count: Literal[2, 3],
    v1_m_s: float,
) -> RefractionMultilayerEndpointConversion:
    endpoint = _endpoint_arrays(endpoint_kind=endpoint_kind, input_model=input_model)
    t1 = _node_values(
        endpoint.node_id,
        solve_result.layer_result_by_kind['v2_t1'].solve_result.node_id,
        solve_result.layer_result_by_kind[
            'v2_t1'
        ].solve_result.node_half_intercept_time_s,
    )
    t2 = _node_values(
        endpoint.node_id,
        solve_result.layer_result_by_kind['v3_t2'].solve_result.node_id,
        solve_result.layer_result_by_kind[
            'v3_t2'
        ].solve_result.node_half_intercept_time_s,
    )
    t3 = np.full(endpoint.node_id.shape, np.nan, dtype=np.float64)
    if layer_count == 3:
        t3 = _node_values(
            endpoint.node_id,
            solve_result.layer_result_by_kind['vsub_t3'].solve_result.node_id,
            solve_result.layer_result_by_kind[
                'vsub_t3'
            ].solve_result.node_half_intercept_time_s,
        )
    v1 = np.full(endpoint.node_id.shape, float(v1_m_s), dtype=np.float64)
    v2_projection = _endpoint_velocity(
        endpoint=endpoint,
        input_model=input_model,
        model=model,
        layer_kind='v2_t1',
        solve_result=solve_result,
        v1_m_s=v1,
    )
    v2 = v2_projection.velocity_m_s
    v3_projection = _endpoint_velocity(
        endpoint=endpoint,
        input_model=input_model,
        model=model,
        layer_kind='v3_t2',
        solve_result=solve_result,
    )
    v3 = v3_projection.velocity_m_s
    vsub = np.full(endpoint.node_id.shape, np.nan, dtype=np.float64)
    vsub_status = np.full(endpoint.node_id.shape, 'ok', dtype=_STATUS_DTYPE)
    if layer_count == 3:
        vsub_projection = _endpoint_velocity(
            endpoint=endpoint,
            input_model=input_model,
            model=model,
            layer_kind='vsub_t3',
            solve_result=solve_result,
        )
        vsub = vsub_projection.velocity_m_s
        vsub_status = vsub_projection.status

    if layer_count == 2:
        thickness = compute_t1lsst_2layer_thicknesses_with_status(
            t1_s=t1,
            t2_s=t2,
            v1_m_s=v1,
            v2_m_s=v2,
            v3_m_s=v3,
        )
        sh1, sh2, sh3 = (
            thickness.sh1_m,
            thickness.sh2_m,
            np.full(t1.shape, np.nan, dtype=np.float64),
        )
        replacement = _required_weathering_correction(thickness)
        conversion_status = thickness.status
    else:
        thickness3 = compute_t1lsst_3layer_thicknesses_with_status(
            t1_s=t1,
            t2_s=t2,
            t3_s=t3,
            v1_m_s=v1,
            v2_m_s=v2,
            v3_m_s=v3,
            vsub_m_s=vsub,
        )
        sh1, sh2, sh3 = thickness3.sh1_m, thickness3.sh2_m, thickness3.sh3_m
        replacement = _required_weathering_correction(thickness3)
        conversion_status = thickness3.status

    inherited_status = _endpoint_layer_status(endpoint.node_id, solve_result, layer_count)
    static_status = _compose_endpoint_status(
        conversion_status,
        inherited_status,
        v2_projection.status,
        v3_projection.status,
        vsub_status,
    )
    refractor = endpoint.surface_elevation_m - np.nansum(
        np.vstack((sh1, sh2, np.nan_to_num(sh3, nan=0.0))),
        axis=0,
    )
    invalid_refractor = static_status != 'ok'
    refractor[invalid_refractor] = np.nan
    replacement = replacement.copy()
    replacement[static_status != 'ok'] = np.nan

    return RefractionMultilayerEndpointConversion(
        endpoint_kind=np.full(endpoint.node_id.shape, endpoint_kind, dtype='<U16'),
        endpoint_key=endpoint.endpoint_key,
        endpoint_id=endpoint.endpoint_id,
        node_id=endpoint.node_id,
        x_m=endpoint.x_m,
        y_m=endpoint.y_m,
        surface_elevation_m=endpoint.surface_elevation_m,
        t1_s=np.ascontiguousarray(t1, dtype=np.float64),
        t2_s=np.ascontiguousarray(t2, dtype=np.float64),
        t3_s=np.ascontiguousarray(t3, dtype=np.float64),
        v1_m_s=np.ascontiguousarray(v1, dtype=np.float64),
        v2_m_s=np.ascontiguousarray(v2, dtype=np.float64),
        v3_m_s=np.ascontiguousarray(v3, dtype=np.float64),
        vsub_m_s=np.ascontiguousarray(vsub, dtype=np.float64),
        sh1_m=np.ascontiguousarray(sh1, dtype=np.float64),
        sh2_m=np.ascontiguousarray(sh2, dtype=np.float64),
        sh3_m=np.ascontiguousarray(sh3, dtype=np.float64),
        refractor_elevation_m=np.ascontiguousarray(refractor, dtype=np.float64),
        weathering_replacement_shift_s=np.ascontiguousarray(
            replacement,
            dtype=np.float64,
        ),
        conversion_status=np.ascontiguousarray(conversion_status, dtype=_STATUS_DTYPE),
        static_status=np.ascontiguousarray(static_status, dtype=_STATUS_DTYPE),
    )


@dataclass(frozen=True)
class _EndpointArrays:
    endpoint_key: np.ndarray
    endpoint_id: np.ndarray | None
    node_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    surface_elevation_m: np.ndarray


@dataclass(frozen=True)
class _EndpointVelocityProjection:
    velocity_m_s: np.ndarray
    status: np.ndarray


def _endpoint_arrays(
    *,
    endpoint_kind: Literal['source', 'receiver'],
    input_model: RefractionStaticInputModel,
) -> _EndpointArrays:
    if endpoint_kind == 'source':
        keys_sorted = input_model.source_endpoint_key_sorted
        ids_sorted = input_model.source_endpoint_id_sorted
        nodes_sorted = input_model.source_node_id_sorted
        x_sorted = input_model.source_x_m_sorted
        y_sorted = input_model.source_y_m_sorted
        elevation_sorted = input_model.source_elevation_m_sorted
    else:
        keys_sorted = input_model.receiver_endpoint_key_sorted
        ids_sorted = input_model.receiver_endpoint_id_sorted
        nodes_sorted = input_model.receiver_node_id_sorted
        x_sorted = input_model.receiver_x_m_sorted
        y_sorted = input_model.receiver_y_m_sorted
        elevation_sorted = input_model.receiver_elevation_m_sorted

    keys: list[object] = []
    ids: list[int] = []
    nodes: list[int] = []
    x_values: list[float] = []
    y_values: list[float] = []
    elevations: list[float] = []
    seen: set[str] = set()
    for index, raw_key in enumerate(keys_sorted.tolist()):
        key = str(raw_key)
        if key in seen:
            continue
        seen.add(key)
        node = int(nodes_sorted[index])
        keys.append(raw_key)
        nodes.append(node)
        if ids_sorted is not None:
            ids.append(int(ids_sorted[index]))
        else:
            ids.append(_endpoint_id_from_table(input_model, node_id=node))
        x_values.append(float(x_sorted[index]))
        y_values.append(float(y_sorted[index]))
        elevations.append(float(elevation_sorted[index]))
    endpoint_id = np.asarray(ids, dtype=np.int64) if ids else None
    return _EndpointArrays(
        endpoint_key=np.ascontiguousarray(np.asarray(keys, dtype=_KEY_DTYPE)),
        endpoint_id=endpoint_id,
        node_id=np.ascontiguousarray(np.asarray(nodes, dtype=np.int64)),
        x_m=np.ascontiguousarray(np.asarray(x_values, dtype=np.float64)),
        y_m=np.ascontiguousarray(np.asarray(y_values, dtype=np.float64)),
        surface_elevation_m=np.ascontiguousarray(
            np.asarray(elevations, dtype=np.float64)
        ),
    )


def _endpoint_id_from_table(
    input_model: RefractionStaticInputModel,
    *,
    node_id: int,
) -> int:
    table_nodes = np.asarray(input_model.endpoint_table.node_id, dtype=np.int64)
    matches = np.flatnonzero(table_nodes == node_id)
    if matches.size == 0:
        return node_id
    return int(np.asarray(input_model.endpoint_table.endpoint_id, dtype=np.int64)[matches[0]])


def _node_values(
    endpoint_node_id: np.ndarray,
    layer_node_id: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(node): float(value)
        for node, value in zip(layer_node_id.tolist(), values.tolist(), strict=True)
    }
    return np.ascontiguousarray(
        np.asarray(
            [lookup.get(int(node), np.nan) for node in endpoint_node_id.tolist()],
            dtype=np.float64,
        )
    )


def _endpoint_layer_status(
    endpoint_node_id: np.ndarray,
    solve_result: RefractionMultilayerTimeTermSolveResult,
    layer_count: Literal[2, 3],
) -> np.ndarray:
    statuses = []
    for kind in ('v2_t1', 'v3_t2', 'vsub_t3')[:layer_count]:
        layer = solve_result.layer_result_by_kind[kind].solve_result
        statuses.append(
            _node_values_as_status(
                endpoint_node_id,
                layer.node_id,
                layer.node_solution_status,
            )
        )
    out = np.full(endpoint_node_id.shape, 'ok', dtype=_STATUS_DTYPE)
    for index in range(endpoint_node_id.shape[0]):
        out[index] = _highest_priority_status(*(status[index] for status in statuses))
    return np.ascontiguousarray(out, dtype=_STATUS_DTYPE)


def _node_values_as_status(
    endpoint_node_id: np.ndarray,
    layer_node_id: np.ndarray,
    status: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(node): str(value)
        for node, value in zip(layer_node_id.tolist(), status.tolist(), strict=True)
    }
    return np.ascontiguousarray(
        np.asarray(
            [lookup.get(int(node), 'missing_node') for node in endpoint_node_id.tolist()],
            dtype=_STATUS_DTYPE,
        )
    )


def _endpoint_velocity(
    *,
    endpoint: _EndpointArrays,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    layer_kind: str,
    solve_result: RefractionMultilayerTimeTermSolveResult,
    v1_m_s: np.ndarray | None = None,
) -> _EndpointVelocityProjection:
    layer = solve_result.layer_result_by_kind[layer_kind]
    result = layer.solve_result
    if result.bedrock_velocity_mode in {'fixed_global', 'solve_global'}:
        velocity = np.full(
            endpoint.node_id.shape,
            result.bedrock_velocity_m_s,
            dtype=np.float64,
        )
        return _EndpointVelocityProjection(
            velocity_m_s=np.ascontiguousarray(velocity, dtype=np.float64),
            status=np.full(endpoint.node_id.shape, 'ok', dtype=_STATUS_DTYPE),
        )
    refractor_cell = model.refractor_cell
    if refractor_cell is None:
        raise RefractionMultilayerConversionError(
            'model.refractor_cell is required to project solve_cell endpoint velocity'
        )
    projected = project_refraction_cell_points(
        x_m=endpoint.x_m,
        y_m=endpoint.y_m,
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
    velocity_by_cell = {
        int(cell_id): float(velocity)
        for cell_id, velocity in zip(
            result.cell_id.tolist(),
            result.cell_bedrock_velocity_m_s.tolist(),
            strict=True,
        )
    }
    status_by_cell = {
        int(cell_id): str(status)
        for cell_id, status in zip(
            result.cell_id.tolist(),
            result.cell_velocity_status.tolist(),
            strict=True,
        )
    }
    velocity = np.full(endpoint.node_id.shape, np.nan, dtype=np.float64)
    status = np.full(
        endpoint.node_id.shape,
        'outside_refractor_cell_grid',
        dtype=_STATUS_DTYPE,
    )
    inside = np.asarray(assignment.inside_grid_mask, dtype=bool)
    for index, raw_cell_id in enumerate(assignment.cell_id.tolist()):
        if not bool(inside[index]):
            continue
        cell_id = int(raw_cell_id)
        velocity[index] = float(velocity_by_cell.get(cell_id, np.nan))
        status[index] = _local_endpoint_velocity_status(
            raw_status=status_by_cell.get(cell_id, 'inactive'),
            velocity_m_s=velocity[index],
        )
    if layer_kind == 'v2_t1' and v1_m_s is not None:
        v1 = np.asarray(v1_m_s, dtype=np.float64)
        invalid_order = (status == 'ok') & np.isfinite(velocity) & (velocity <= v1)
        status[invalid_order] = 'v2_not_greater_than_v1'
    return _EndpointVelocityProjection(
        velocity_m_s=np.ascontiguousarray(velocity, dtype=np.float64),
        status=np.ascontiguousarray(status, dtype=_STATUS_DTYPE),
    )


def _local_endpoint_velocity_status(*, raw_status: str, velocity_m_s: float) -> str:
    if raw_status in {'solved', 'ok', 'clipped_lower', 'clipped_upper'}:
        if np.isfinite(velocity_m_s) and velocity_m_s > 0.0:
            return 'ok'
        return 'invalid_local_v2'
    if raw_status == 'inactive':
        return 'inactive_v2_cell'
    if raw_status == 'low_fold':
        return 'low_fold_v2_cell'
    return 'invalid_local_v2'


def _required_weathering_correction(
    result: RefractionT1LSST2LayerThicknessResult | RefractionT1LSST3LayerThicknessResult,
) -> np.ndarray:
    if result.weathering_correction_s is None:
        raise RefractionMultilayerConversionError(
            'T1LSST thickness result did not include weathering correction'
        )
    return np.ascontiguousarray(result.weathering_correction_s, dtype=np.float64)


def _compose_endpoint_status(*status_arrays: np.ndarray) -> np.ndarray:
    if not status_arrays:
        raise RefractionMultilayerConversionError('at least one status array is required')
    conversion_status = status_arrays[0]
    out = np.full(conversion_status.shape, 'ok', dtype=_STATUS_DTYPE)
    normalized = [
        np.asarray(status).astype(str, copy=False)
        for status in status_arrays
    ]
    for index in range(out.shape[0]):
        out[index] = _highest_priority_status(
            *(status[index] for status in normalized)
        )
        if out[index] in _OK_STATUSES:
            out[index] = 'ok'
    return np.ascontiguousarray(out, dtype=_STATUS_DTYPE)


def _highest_priority_status(*statuses: str) -> str:
    best = 'ok'
    best_priority = -1
    for value in statuses:
        normalized = 'ok' if value in _OK_STATUSES else str(value)
        priority = _STATUS_PRIORITY.get(normalized, 100)
        if priority > best_priority:
            best = normalized
            best_priority = priority
    return best


def _map_endpoint_float_to_trace(
    trace_key: np.ndarray,
    endpoint_key: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    lookup = {
        str(key): float(value)
        for key, value in zip(endpoint_key.tolist(), values.tolist(), strict=True)
    }
    return np.ascontiguousarray(
        np.asarray([lookup.get(str(key), np.nan) for key in trace_key], dtype=np.float64)
    )


def _map_endpoint_status_to_trace(
    trace_key: np.ndarray,
    endpoint_key: np.ndarray,
    status: np.ndarray,
) -> np.ndarray:
    lookup = {
        str(key): str(value)
        for key, value in zip(endpoint_key.tolist(), status.tolist(), strict=True)
    }
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
    valid = np.isfinite(source_shift_s) & np.isfinite(receiver_shift_s)
    out[valid] = source_shift_s[valid] + receiver_shift_s[valid]
    return np.ascontiguousarray(out, dtype=np.float64)


def _classify_trace_status(
    source_status: np.ndarray,
    receiver_status: np.ndarray,
    trace_shift_s: np.ndarray,
) -> np.ndarray:
    out = np.full(trace_shift_s.shape, 'ok', dtype=_STATUS_DTYPE)
    source = np.asarray(source_status).astype(str, copy=False)
    receiver = np.asarray(receiver_status).astype(str, copy=False)
    for index in range(out.shape[0]):
        out[index] = _highest_priority_status(source[index], receiver[index])
        if out[index] in _OK_STATUSES:
            out[index] = 'ok'
    out[(out == 'ok') & ~np.isfinite(trace_shift_s)] = 'invalid_shift'
    return np.ascontiguousarray(out, dtype=_STATUS_DTYPE)


def _resolve_floating_datum(
    *,
    options: RefractionStaticDatumOptions,
    conversion: RefractionMultilayerConversionResult,
    floating_datum: ResolvedFloatingDatum | None,
    source_floating_datum_elevation_m: np.ndarray | float | None,
    receiver_floating_datum_elevation_m: np.ndarray | float | None,
) -> ResolvedFloatingDatum | None:
    if floating_datum is not None:
        return floating_datum
    if (
        source_floating_datum_elevation_m is not None
        or receiver_floating_datum_elevation_m is not None
    ):
        return None
    mode = options.floating_datum_mode
    if mode == 'provided':
        return None
    if mode == 'constant':
        return ResolvedFloatingDatum(
            source_elevation_m=float(options.floating_datum_elevation_m),
            receiver_elevation_m=float(options.floating_datum_elevation_m),
        )
    if mode == 'surface':
        return ResolvedFloatingDatum(
            source_elevation_m=conversion.source_endpoint.surface_elevation_m,
            receiver_elevation_m=conversion.receiver_endpoint.surface_elevation_m,
        )
    if mode == 'smoothed_topography':
        window = 1 if options.smoothing_window_nodes is None else options.smoothing_window_nodes
        return ResolvedFloatingDatum(
            source_elevation_m=smooth_refraction_floating_datum_elevation(
                conversion.source_endpoint.surface_elevation_m,
                window_nodes=window,
                method=options.smoothing_method,
            ),
            receiver_elevation_m=smooth_refraction_floating_datum_elevation(
                conversion.receiver_endpoint.surface_elevation_m,
                window_nodes=window,
                method=options.smoothing_method,
            ),
        )
    raise RefractionMultilayerConversionError(
        f'unsupported floating datum mode: {mode!r}'
    )


def _replacement_velocity(
    endpoint: RefractionMultilayerEndpointConversion,
    *,
    layer_count: Literal[2, 3],
) -> np.ndarray:
    values = endpoint.v3_m_s if layer_count == 2 else endpoint.vsub_m_s
    if not np.any(np.isfinite(values)):
        raise RefractionMultilayerConversionError(
            'replacement velocity could not be resolved from converted endpoints'
        )
    return np.ascontiguousarray(values, dtype=np.float64)


__all__ = [
    'RefractionMultilayerConversionError',
    'RefractionMultilayerConversionResult',
    'RefractionMultilayerDatumStaticsResult',
    'RefractionMultilayerEndpointConversion',
    'build_refraction_multilayer_conversion',
    'compute_refraction_multilayer_datum_statics_from_input_model',
]
