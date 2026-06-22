"""Package facade for global bedrock slowness estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seis_statics.refraction.options import (
    RefractionStaticModelOptions,
    RefractionStaticSolverOptions,
)
from seis_statics.refraction.solver import (
    RefractionStaticRobustIterationSummary,
    RefractionStaticRobustStopReason,
    RefractionStaticSolveResult,
    RefractionStaticSolverError,
    solve_refraction_static_least_squares,
    validate_refraction_static_solver_options,
)
from seis_statics.refraction.types import (
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)


class RefractionBedrockEstimationError(ValueError):
    """Raised when global bedrock slowness estimation inputs are inconsistent."""


@dataclass(frozen=True)
class GlobalBedrockSlownessEstimateResult:
    """Pure numerical result for global bedrock slowness estimation."""

    file_id: str
    n_traces: int

    bedrock_velocity_mode: str
    bedrock_velocity_m_s: float
    bedrock_slowness_s_per_m: float
    bedrock_velocity_status: str
    v2_m_s: float

    node_id: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_solution_status: np.ndarray
    node_observation_count: np.ndarray

    modeled_pick_time_s_sorted: np.ndarray
    residual_s_sorted: np.ndarray
    residual_ms_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray
    rejected_observation_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    rms_residual_s: float
    rms_residual_ms: float
    residual_mean_s: float
    residual_median_s: float
    residual_mad_s: float
    residual_max_abs_s: float

    solver_success: bool
    solver_status: int
    solver_message: str
    solver_cost: float
    solver_optimality: float
    solver_iterations: int

    robust_enabled: bool
    robust_stop_reason: RefractionStaticRobustStopReason
    robust_iteration_summaries: tuple[RefractionStaticRobustIterationSummary, ...]
    n_initial_used_observations: int
    n_final_used_observations: int
    n_rejected_observations: int

    qc: dict[str, Any]
    debug_solve_result: RefractionStaticSolveResult | None = None


def estimate_global_bedrock_slowness_from_input_model(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    solver_options: RefractionStaticSolverOptions | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
    include_debug_objects: bool = False,
) -> GlobalBedrockSlownessEstimateResult:
    """Estimate one global refractor slowness from a package input model."""
    _validate_global_bedrock_inputs(input_model=input_model, model=model)
    options = validate_refraction_static_solver_options(solver_options)
    solve_result = solve_refraction_static_least_squares(
        input_model=input_model,
        model=model,
        solver_options=options,
        resolved_first_layer=resolved_first_layer,
        include_diagnostics=include_diagnostics,
    )
    return _public_result(
        input_model=input_model,
        solve_result=solve_result,
        include_debug_objects=include_debug_objects,
    )


def _validate_global_bedrock_inputs(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
) -> None:
    if not isinstance(input_model, RefractionStaticInputModel):
        raise RefractionBedrockEstimationError(
            'input_model must be a RefractionStaticInputModel instance'
        )
    if int(input_model.n_traces) <= 0:
        raise RefractionBedrockEstimationError('input_model.n_traces must be positive')
    if not isinstance(model, RefractionStaticModelOptions):
        raise RefractionBedrockEstimationError(
            'model must be a RefractionStaticModelOptions instance'
        )
    if model.method != 'gli_variable_thickness':
        raise RefractionBedrockEstimationError(
            'model.method must be gli_variable_thickness'
        )
    if model.bedrock_velocity_mode != 'solve_global':
        raise RefractionBedrockEstimationError(
            'model.bedrock_velocity_mode must be solve_global'
        )


def _public_result(
    *,
    input_model: RefractionStaticInputModel,
    solve_result: RefractionStaticSolveResult,
    include_debug_objects: bool,
) -> GlobalBedrockSlownessEstimateResult:
    if solve_result.bedrock_velocity_mode != 'solve_global':
        raise RefractionStaticSolverError(
            'global bedrock slowness result requires solve_global mode'
        )
    residual_stats = _residual_statistics(
        solve_result.residual_s_sorted,
        used_mask=solve_result.used_observation_mask_sorted,
    )
    qc = dict(solve_result.qc)
    qc['residual_statistics'] = dict(residual_stats)

    return GlobalBedrockSlownessEstimateResult(
        file_id=str(input_model.file_id),
        n_traces=int(input_model.n_traces),
        bedrock_velocity_mode=solve_result.bedrock_velocity_mode,
        bedrock_velocity_m_s=solve_result.bedrock_velocity_m_s,
        bedrock_slowness_s_per_m=solve_result.bedrock_slowness_s_per_m,
        bedrock_velocity_status=solve_result.bedrock_velocity_status,
        v2_m_s=solve_result.bedrock_velocity_m_s,
        node_id=solve_result.node_id,
        node_half_intercept_time_s=solve_result.node_half_intercept_time_s,
        node_solution_status=solve_result.node_solution_status,
        node_observation_count=solve_result.node_observation_count,
        modeled_pick_time_s_sorted=solve_result.modeled_pick_time_s_sorted,
        residual_s_sorted=solve_result.residual_s_sorted,
        residual_ms_sorted=solve_result.residual_ms_sorted,
        used_observation_mask_sorted=solve_result.used_observation_mask_sorted,
        rejected_observation_mask_sorted=(
            solve_result.rejected_observation_mask_sorted
        ),
        rejected_iteration_sorted=solve_result.rejected_iteration_sorted,
        rms_residual_s=solve_result.rms_residual_s,
        rms_residual_ms=solve_result.rms_residual_ms,
        residual_mean_s=residual_stats['mean_s'],
        residual_median_s=residual_stats['median_s'],
        residual_mad_s=residual_stats['mad_s'],
        residual_max_abs_s=residual_stats['max_abs_s'],
        solver_success=solve_result.solver_success,
        solver_status=solve_result.solver_status,
        solver_message=solve_result.solver_message,
        solver_cost=solve_result.solver_cost,
        solver_optimality=solve_result.solver_optimality,
        solver_iterations=solve_result.solver_iterations,
        robust_enabled=solve_result.robust_enabled,
        robust_stop_reason=solve_result.robust_stop_reason,
        robust_iteration_summaries=solve_result.robust_iteration_summaries,
        n_initial_used_observations=solve_result.n_initial_used_observations,
        n_final_used_observations=solve_result.n_final_used_observations,
        n_rejected_observations=solve_result.n_rejected_observations,
        qc=qc,
        debug_solve_result=solve_result if include_debug_objects else None,
    )


def _residual_statistics(
    residual_s_sorted: np.ndarray,
    *,
    used_mask: np.ndarray,
) -> dict[str, float]:
    residual = np.asarray(residual_s_sorted, dtype=np.float64)
    mask = np.asarray(used_mask, dtype=bool)
    values = residual[mask]
    if values.size == 0:
        return {
            'mean_s': float('nan'),
            'median_s': float('nan'),
            'mad_s': float('nan'),
            'max_abs_s': float('nan'),
        }
    median = float(np.median(values))
    return {
        'mean_s': float(np.mean(values)),
        'median_s': median,
        'mad_s': float(np.median(np.abs(values - median))),
        'max_abs_s': float(np.max(np.abs(values))),
    }


__all__ = [
    'GlobalBedrockSlownessEstimateResult',
    'RefractionBedrockEstimationError',
    'estimate_global_bedrock_slowness_from_input_model',
]
