"""Pure multi-layer time-term orchestration for refraction statics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from seis_statics.refraction.cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_coordinates,
)
from seis_statics.refraction.cell_grid import (
    assign_observation_midpoint_cells,
    build_refraction_cell_grid,
)
from seis_statics.refraction.first_layer import resolve_weathering_velocity_m_s
from seis_statics.refraction.layer_config import (
    RefractionLayerConfig,
    RefractionLayerConfigLayer,
    normalize_refraction_layer_config,
)
from seis_statics.refraction.layer_observations import (
    INVALID_OFFSET_REJECTION_REASON,
    OUTSIDE_LAYER_GATE_REJECTION_REASON,
    build_refraction_layer_observation_masks,
    refraction_layer_observation_qc,
)
from seis_statics.refraction.options import (
    RefractionStaticFirstLayerOptions,
    RefractionStaticModelOptions,
    RefractionStaticRefractorCellOptions,
    RefractionStaticSolverOptions,
)
from seis_statics.refraction.solver import (
    RefractionStaticSolveResult,
    solve_refraction_static_least_squares,
)
from seis_statics.refraction.types import (
    RefractionLayerKind,
    RefractionLayerObservationMasks,
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)


ROBUST_REJECTION_REASON = 'robust_rejected'
VELOCITY_ORDER_REJECTION_REASON = 'invalid_velocity_order'

_LAYER_INDEX: dict[RefractionLayerKind, int] = {
    'v2_t1': 1,
    'v3_t2': 2,
    'vsub_t3': 3,
}


class RefractionMultilayerTimeTermSolverError(ValueError):
    """Raised when a multi-layer time-term solve is inconsistent."""


@dataclass(frozen=True)
class _RefractionLayerSolveContext:
    input_model: RefractionStaticInputModel
    model: RefractionStaticModelOptions
    solver_options: RefractionStaticSolverOptions | None
    include_diagnostics: bool
    layer: RefractionLayerConfigLayer


RefractionLayerSolver = Callable[
    [_RefractionLayerSolveContext],
    RefractionStaticSolveResult,
]


@dataclass(frozen=True)
class RefractionMultilayerTimeTermLayerResult:
    """Single enabled-layer solve and trace-order layer metadata."""

    layer_kind: RefractionLayerKind
    layer_index: int
    layer: RefractionLayerConfigLayer
    solve_result: RefractionStaticSolveResult
    velocity_m_s_sorted: np.ndarray
    rejection_reason_sorted: np.ndarray
    velocity_order_valid_mask_sorted: np.ndarray


@dataclass(frozen=True)
class RefractionMultilayerTimeTermSolveResult:
    """Combined trace-order result for enabled refraction time-term layers."""

    layer_results: tuple[RefractionMultilayerTimeTermLayerResult, ...]
    layer_result_by_kind: dict[str, RefractionMultilayerTimeTermLayerResult]
    layer_observation_masks: RefractionLayerObservationMasks
    modeled_pick_time_s_sorted: np.ndarray
    residual_s_sorted: np.ndarray
    residual_ms_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray
    rejected_observation_mask_sorted: np.ndarray
    layer_kind_sorted: np.ndarray
    rejection_reason_sorted: np.ndarray
    velocity_m_s_sorted: np.ndarray
    qc: dict[str, Any]


def solve_refraction_multilayer_time_terms(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    solver_options: RefractionStaticSolverOptions | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
) -> RefractionMultilayerTimeTermSolveResult:
    """Solve enabled ``v2_t1``, ``v3_t2``, and ``vsub_t3`` time-term layers.

    This function is intentionally package-local and pure: it consumes the
    already-normalized trace-order input model and refraction option dataclasses,
    then delegates each layer to the existing sparse time-term solver.
    """
    _validate_inputs(input_model=input_model, model=model)
    layer_config = normalize_refraction_layer_config(model)
    if model.method != 'multilayer_time_term':
        raise RefractionMultilayerTimeTermSolverError(
            'model.method must be multilayer_time_term'
        )
    layer_masks = input_model.layer_observation_masks
    if layer_masks is None:
        layer_masks = build_refraction_layer_observation_masks(
            layer_config=layer_config,
            offset_m_sorted=_input_offset_m_sorted(input_model),
            valid_observation_mask_sorted=input_model.valid_observation_mask_sorted,
            rejection_reason_sorted=input_model.rejection_reason_sorted,
        )
    else:
        _validate_layer_masks(layer_config=layer_config, layer_masks=layer_masks)

    weathering_velocity = resolve_weathering_velocity_m_s(
        model=model,
        resolved_first_layer=resolved_first_layer,
        name='model.resolved_weathering_velocity_m_s',
    )
    layer_results: list[RefractionMultilayerTimeTermLayerResult] = []
    previous_velocity_sorted: np.ndarray | None = None
    n_traces = int(input_model.n_traces)

    for layer in layer_config.layers:
        layer_input = _layer_input_model(
            input_model=input_model,
            layer_masks=layer_masks,
            layer=layer,
        )
        layer_model = _layer_static_model(
            source_model=model,
            layer=layer,
            weathering_velocity_m_s=weathering_velocity,
        )
        layer_solver = _solver_for_layer(layer)
        solve_result = layer_solver(
            _RefractionLayerSolveContext(
                input_model=layer_input,
                model=layer_model,
                solver_options=solver_options,
                include_diagnostics=include_diagnostics,
                layer=layer,
            )
        )
        velocity_sorted = _project_layer_velocity_to_sorted_traces(
            input_model=input_model,
            model=layer_model,
            solve_result=solve_result,
        )
        order_valid = _velocity_order_valid_mask(
            current_velocity_sorted=velocity_sorted,
            previous_velocity_sorted=previous_velocity_sorted,
            used_mask=solve_result.used_observation_mask_sorted,
        )
        rejection_reason = _layer_rejection_reason(
            layer_masks=layer_masks,
            layer=layer,
            solve_result=solve_result,
            velocity_order_valid_mask=order_valid,
        )
        solve_result = _apply_velocity_order_rejections(
            solve_result=solve_result,
            velocity_order_valid_mask=order_valid,
        )
        layer_result = RefractionMultilayerTimeTermLayerResult(
            layer_kind=layer.kind,
            layer_index=_LAYER_INDEX[layer.kind],
            layer=layer,
            solve_result=solve_result,
            velocity_m_s_sorted=velocity_sorted,
            rejection_reason_sorted=rejection_reason,
            velocity_order_valid_mask_sorted=order_valid,
        )
        layer_results.append(layer_result)
        previous_velocity_sorted = velocity_sorted

    return _combine_layer_results(
        n_traces=n_traces,
        layer_results=tuple(layer_results),
        layer_masks=layer_masks,
    )


def _solver_for_layer(
    layer: RefractionLayerConfigLayer,
) -> RefractionLayerSolver:
    dispatch: dict[
        tuple[RefractionLayerKind, str],
        RefractionLayerSolver,
    ] = {
        ('v2_t1', 'fixed_global'): _solve_existing_time_term_layer,
        ('v2_t1', 'solve_global'): _solve_existing_time_term_layer,
        ('v2_t1', 'solve_cell'): _solve_existing_time_term_layer,
        ('v3_t2', 'fixed_global'): _solve_existing_time_term_layer,
        ('v3_t2', 'solve_global'): _solve_existing_time_term_layer,
        ('v3_t2', 'solve_cell'): _solve_existing_time_term_layer,
        ('vsub_t3', 'fixed_global'): _solve_existing_time_term_layer,
        ('vsub_t3', 'solve_global'): _solve_existing_time_term_layer,
        ('vsub_t3', 'solve_cell'): _solve_existing_time_term_layer,
    }
    layer_solver = dispatch.get((layer.kind, layer.velocity_mode))
    if layer_solver is None:
        raise RefractionMultilayerTimeTermSolverError(
            f'refraction layer {layer.kind} with velocity_mode='
            f'{layer.velocity_mode} is not implemented'
        )
    return layer_solver


def _solve_existing_time_term_layer(
    context: _RefractionLayerSolveContext,
) -> RefractionStaticSolveResult:
    try:
        return solve_refraction_static_least_squares(
            input_model=context.input_model,
            model=context.model,
            solver_options=context.solver_options,
            resolved_first_layer=None,
            include_diagnostics=context.include_diagnostics,
        )
    except ValueError as exc:
        raise RefractionMultilayerTimeTermSolverError(
            f'refraction layer {context.layer.kind} solve failed: {exc}'
        ) from exc


def _validate_inputs(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
) -> None:
    if not isinstance(input_model, RefractionStaticInputModel):
        raise TypeError('input_model must be RefractionStaticInputModel')
    if not isinstance(model, RefractionStaticModelOptions):
        raise TypeError('model must be RefractionStaticModelOptions')


def _validate_layer_masks(
    *,
    layer_config: RefractionLayerConfig,
    layer_masks: RefractionLayerObservationMasks,
) -> None:
    expected = tuple(layer.kind for layer in layer_config.layers)
    actual = tuple(str(kind) for kind in layer_masks.layer_kind)
    if actual != expected:
        raise RefractionMultilayerTimeTermSolverError(
            'input_model.layer_observation_masks must match enabled model layers'
        )


def _input_offset_m_sorted(input_model: RefractionStaticInputModel) -> np.ndarray:
    if input_model.offset_m_sorted is not None:
        return input_model.offset_m_sorted
    return input_model.distance_m_sorted


def _layer_input_model(
    *,
    input_model: RefractionStaticInputModel,
    layer_masks: RefractionLayerObservationMasks,
    layer: RefractionLayerConfigLayer,
) -> RefractionStaticInputModel:
    return replace(
        input_model,
        valid_observation_mask_sorted=layer_masks.layer_used_mask_sorted[layer.kind],
        rejection_reason_sorted=layer_masks.layer_rejection_reason_sorted[layer.kind],
        layer_observation_masks=None,
    )


def _layer_static_model(
    *,
    source_model: RefractionStaticModelOptions,
    layer: RefractionLayerConfigLayer,
    weathering_velocity_m_s: float,
) -> RefractionStaticModelOptions:
    refractor_cell = None
    if layer.velocity_mode == 'solve_cell':
        refractor_cell = _layer_refractor_cell(
            source_model=source_model,
            layer=layer,
        )
    return RefractionStaticModelOptions(
        method='gli_variable_thickness',
        weathering_velocity_m_s=None,
        first_layer=RefractionStaticFirstLayerOptions(
            mode='constant',
            weathering_velocity_m_s=weathering_velocity_m_s,
        ),
        bedrock_velocity_mode=layer.velocity_mode,
        bedrock_velocity_m_s=layer.fixed_velocity_m_s,
        initial_bedrock_velocity_m_s=layer.initial_velocity_m_s,
        min_bedrock_velocity_m_s=(
            np.nextafter(weathering_velocity_m_s, np.inf)
            if layer.min_velocity_m_s is None
            else layer.min_velocity_m_s
        ),
        max_bedrock_velocity_m_s=(
            float(np.finfo(np.float64).max)
            if layer.max_velocity_m_s is None
            else layer.max_velocity_m_s
        ),
        refractor_cell=refractor_cell,
    )


def _layer_refractor_cell(
    *,
    source_model: RefractionStaticModelOptions,
    layer: RefractionLayerConfigLayer,
) -> RefractionStaticRefractorCellOptions:
    refractor_cell = source_model.refractor_cell
    if refractor_cell is None:
        raise RefractionMultilayerTimeTermSolverError(
            'model.refractor_cell is required for solve_cell layers'
        )
    updates: dict[str, object] = {}
    if layer.min_observations_per_cell is not None:
        updates['min_observations_per_cell'] = layer.min_observations_per_cell
    if layer.smoothing_weight is not None:
        updates['velocity_smoothing_weight'] = layer.smoothing_weight
    if not updates:
        return refractor_cell
    return replace(refractor_cell, **updates)


def _project_layer_velocity_to_sorted_traces(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    solve_result: RefractionStaticSolveResult,
) -> np.ndarray:
    n_traces = int(input_model.n_traces)
    if solve_result.bedrock_velocity_mode in {'solve_global', 'fixed_global'}:
        return np.full(n_traces, solve_result.bedrock_velocity_m_s, dtype=np.float64)
    refractor_cell = model.refractor_cell
    if refractor_cell is None:
        raise RefractionMultilayerTimeTermSolverError(
            'solve_cell velocity projection requires model.refractor_cell'
        )
    projected = project_refraction_cell_coordinates(
        source_x_m=input_model.source_x_m_sorted,
        source_y_m=input_model.source_y_m_sorted,
        receiver_x_m=input_model.receiver_x_m_sorted,
        receiver_y_m=input_model.receiver_y_m_sorted,
        mode=refractor_cell.coordinate_mode,
        line_origin_x_m=refractor_cell.line_origin_x_m,
        line_origin_y_m=refractor_cell.line_origin_y_m,
        line_azimuth_deg=refractor_cell.line_azimuth_deg,
    )
    grid = build_refraction_cell_grid(effective_refraction_cell_grid_config(refractor_cell))
    assignment = assign_observation_midpoint_cells(
        grid,
        source_x_m=projected.source_x_m,
        source_y_m=projected.source_y_m,
        receiver_x_m=projected.receiver_x_m,
        receiver_y_m=projected.receiver_y_m,
    )
    velocity_by_cell = {
        int(cell_id): float(velocity)
        for cell_id, velocity in zip(
            solve_result.cell_id.tolist(),
            solve_result.cell_bedrock_velocity_m_s.tolist(),
            strict=True,
        )
        if np.isfinite(velocity)
    }
    velocity = np.full(n_traces, np.nan, dtype=np.float64)
    for row_index, cell_id in enumerate(assignment.cell_id.tolist()):
        if row_index >= n_traces:
            break
        value = velocity_by_cell.get(int(cell_id))
        if value is not None:
            velocity[row_index] = value
    return np.ascontiguousarray(velocity, dtype=np.float64)


def _velocity_order_valid_mask(
    *,
    current_velocity_sorted: np.ndarray,
    previous_velocity_sorted: np.ndarray | None,
    used_mask: np.ndarray,
) -> np.ndarray:
    valid = np.ones(current_velocity_sorted.shape, dtype=bool)
    if previous_velocity_sorted is None:
        return valid
    comparable = (
        used_mask
        & np.isfinite(current_velocity_sorted)
        & np.isfinite(previous_velocity_sorted)
    )
    valid[comparable] = current_velocity_sorted[comparable] > previous_velocity_sorted[
        comparable
    ]
    return np.ascontiguousarray(valid, dtype=bool)


def _apply_velocity_order_rejections(
    *,
    solve_result: RefractionStaticSolveResult,
    velocity_order_valid_mask: np.ndarray,
) -> RefractionStaticSolveResult:
    invalid_velocity = (
        solve_result.used_observation_mask_sorted & ~velocity_order_valid_mask
    )
    if not np.any(invalid_velocity):
        return solve_result

    used = np.asarray(solve_result.used_observation_mask_sorted, dtype=bool).copy()
    rejected = np.asarray(
        solve_result.rejected_observation_mask_sorted,
        dtype=bool,
    ).copy()
    used[invalid_velocity] = False
    rejected[invalid_velocity] = True

    used_residual = solve_result.residual_s_sorted[
        used & np.isfinite(solve_result.residual_s_sorted)
    ]
    rms_s = (
        float(np.sqrt(np.mean(np.square(used_residual))))
        if used_residual.size > 0
        else float('nan')
    )
    qc = dict(solve_result.qc)
    qc['n_final_used_observations'] = int(np.count_nonzero(used))
    qc['n_rejected_observations'] = int(np.count_nonzero(rejected))
    qc['n_final_rejected_observations'] = int(np.count_nonzero(rejected))
    qc['rms_residual_ms'] = rms_s * 1000.0

    return replace(
        solve_result,
        used_observation_mask_sorted=np.ascontiguousarray(used),
        rejected_observation_mask_sorted=np.ascontiguousarray(rejected),
        rms_residual_s=rms_s,
        rms_residual_ms=float(rms_s * 1000.0),
        n_final_used_observations=int(np.count_nonzero(used)),
        n_rejected_observations=int(np.count_nonzero(rejected)),
        qc=qc,
    )


def _layer_rejection_reason(
    *,
    layer_masks: RefractionLayerObservationMasks,
    layer: RefractionLayerConfigLayer,
    solve_result: RefractionStaticSolveResult,
    velocity_order_valid_mask: np.ndarray,
) -> np.ndarray:
    reason = np.asarray(
        layer_masks.layer_rejection_reason_sorted[layer.kind],
        dtype='<U64',
    ).copy()
    design_reason = solve_result.design.rejection_reason_sorted
    if design_reason is not None:
        design_reason = np.asarray(design_reason, dtype='<U64')
        if design_reason.shape != reason.shape:
            raise RefractionMultilayerTimeTermSolverError(
                'solve_result.design.rejection_reason_sorted shape mismatch'
            )
        design_rejected = (design_reason != '') & (design_reason != 'ok')
        reason[design_rejected] = design_reason[design_rejected]
    robust_rejected = solve_result.rejected_observation_mask_sorted
    reason[robust_rejected] = ROBUST_REJECTION_REASON
    invalid_velocity = (
        solve_result.used_observation_mask_sorted & ~velocity_order_valid_mask
    )
    reason[invalid_velocity] = VELOCITY_ORDER_REJECTION_REASON
    solved = solve_result.used_observation_mask_sorted & velocity_order_valid_mask
    reason[solved] = ''
    return np.ascontiguousarray(reason)


def _combine_layer_results(
    *,
    n_traces: int,
    layer_results: tuple[RefractionMultilayerTimeTermLayerResult, ...],
    layer_masks: RefractionLayerObservationMasks,
) -> RefractionMultilayerTimeTermSolveResult:
    modeled = np.full(n_traces, np.nan, dtype=np.float64)
    residual = np.full(n_traces, np.nan, dtype=np.float64)
    velocity = np.full(n_traces, np.nan, dtype=np.float64)
    used = np.zeros(n_traces, dtype=bool)
    rejected = np.zeros(n_traces, dtype=bool)
    layer_kind = np.full(n_traces, '', dtype='<U16')
    reason = np.full(n_traces, '', dtype='<U64')
    final_membership_count = np.zeros(n_traces, dtype=np.int32)

    for layer_result in layer_results:
        solve_result = layer_result.solve_result
        layer_used = solve_result.used_observation_mask_sorted
        layer_reason = layer_result.rejection_reason_sorted
        layer_candidate = (
            (layer_reason != OUTSIDE_LAYER_GATE_REJECTION_REASON)
            & (layer_reason != INVALID_OFFSET_REJECTION_REASON)
        )
        final_membership_count += layer_used.astype(np.int32)
        fill_mask = layer_used & ~used
        modeled[fill_mask] = solve_result.modeled_pick_time_s_sorted[fill_mask]
        residual[fill_mask] = solve_result.residual_s_sorted[fill_mask]
        velocity[fill_mask] = layer_result.velocity_m_s_sorted[fill_mask]
        layer_kind[fill_mask] = layer_result.layer_kind
        used |= layer_used

        layer_rejected = layer_candidate & ~layer_used
        rejected |= layer_rejected
        reason_fill = layer_rejected & ~used & (reason == '')
        reason[reason_fill] = layer_result.rejection_reason_sorted[reason_fill]
        layer_kind[reason_fill] = layer_result.layer_kind

    rejected &= ~used
    reason[used] = ''
    qc = {
        'method': 'multilayer_time_term',
        'layer_count': len(layer_results),
        'layer_kind': [result.layer_kind for result in layer_results],
        'layer_observations': refraction_layer_observation_qc(layer_masks),
        'n_traces': n_traces,
        'n_used_observations': int(np.count_nonzero(used)),
        'n_rejected_observations': int(np.count_nonzero(rejected)),
        'n_final_multi_layer_membership': int(
            np.count_nonzero(final_membership_count > 1)
        ),
        'layer_solver_qc': {
            result.layer_kind: result.solve_result.qc for result in layer_results
        },
    }
    return RefractionMultilayerTimeTermSolveResult(
        layer_results=layer_results,
        layer_result_by_kind={result.layer_kind: result for result in layer_results},
        layer_observation_masks=layer_masks,
        modeled_pick_time_s_sorted=np.ascontiguousarray(modeled),
        residual_s_sorted=np.ascontiguousarray(residual),
        residual_ms_sorted=np.ascontiguousarray(residual * 1000.0),
        used_observation_mask_sorted=np.ascontiguousarray(used),
        rejected_observation_mask_sorted=np.ascontiguousarray(rejected),
        layer_kind_sorted=np.ascontiguousarray(layer_kind),
        rejection_reason_sorted=np.ascontiguousarray(reason),
        velocity_m_s_sorted=np.ascontiguousarray(velocity),
        qc=qc,
    )


__all__ = [
    'ROBUST_REJECTION_REASON',
    'VELOCITY_ORDER_REJECTION_REASON',
    'RefractionMultilayerTimeTermLayerResult',
    'RefractionMultilayerTimeTermSolveResult',
    'RefractionMultilayerTimeTermSolverError',
    'solve_refraction_multilayer_time_terms',
]
