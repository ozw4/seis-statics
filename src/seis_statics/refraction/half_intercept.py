"""Pure half-intercept result assembly for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seis_statics.refraction.bedrock import GlobalBedrockSlownessEstimateResult
from seis_statics.refraction.design_matrix import (
    RefractionStaticDesignMatrix,
    build_refraction_static_design_matrix,
)
from seis_statics.refraction.options import (
    RefractionStaticModelOptions,
    RefractionStaticSolverOptions,
)
from seis_statics.refraction.solver import (
    RefractionStaticRobustIterationSummary,
    RefractionStaticRobustStopReason,
    RefractionStaticSolveResult,
    solve_refraction_static_design_least_squares,
    solve_refraction_static_least_squares,
    validate_refraction_static_solver_options,
)
from seis_statics.refraction.types import (
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)


class RefractionHalfInterceptError(ValueError):
    """Raised when half-intercept result inputs are inconsistent."""


@dataclass(frozen=True)
class RefractionHalfInterceptEndpointResult:
    """Half-intercept values aggregated to unique endpoint keys."""

    endpoint_key: np.ndarray
    endpoint_id: np.ndarray | None
    node_id: np.ndarray
    half_intercept_time_s: np.ndarray
    solution_status: np.ndarray
    pick_count: np.ndarray
    used_observation_count: np.ndarray
    rejected_observation_count: np.ndarray


@dataclass(frozen=True)
class RefractionHalfInterceptResult:
    """Package half-intercept result without artifact or runtime dependencies."""

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
    node_pick_count: np.ndarray
    node_used_observation_count: np.ndarray
    node_rejected_observation_count: np.ndarray

    source_endpoint: RefractionHalfInterceptEndpointResult
    receiver_endpoint: RefractionHalfInterceptEndpointResult

    trace_index_sorted: np.ndarray
    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray
    source_endpoint_id_sorted: np.ndarray | None
    receiver_endpoint_id_sorted: np.ndarray | None
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    source_half_intercept_time_s_sorted: np.ndarray
    receiver_half_intercept_time_s_sorted: np.ndarray
    trace_half_intercept_time_s_sorted: np.ndarray
    trace_half_intercept_status_sorted: np.ndarray

    pick_time_s_sorted: np.ndarray
    modeled_pick_time_s_sorted: np.ndarray
    residual_s_sorted: np.ndarray
    residual_ms_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray
    rejected_observation_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    cell_id: np.ndarray
    cell_bedrock_slowness_s_per_m: np.ndarray
    cell_bedrock_velocity_m_s: np.ndarray
    cell_velocity_status: np.ndarray
    cell_observation_count: np.ndarray
    row_midpoint_cell_id: np.ndarray
    row_midpoint_bedrock_slowness_s_per_m: np.ndarray
    row_midpoint_bedrock_velocity_m_s: np.ndarray
    row_midpoint_v2_m_s: np.ndarray
    cell_v2_m_s: np.ndarray

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
    debug_design: RefractionStaticDesignMatrix | None = None
    debug_solve_result: RefractionStaticSolveResult | None = None


@dataclass(frozen=True)
class _HalfInterceptSolution:
    node_id: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_solution_status: np.ndarray
    node_observation_count: np.ndarray

    bedrock_velocity_mode: str
    bedrock_velocity_m_s: float
    bedrock_slowness_s_per_m: float
    bedrock_velocity_status: str

    cell_id: np.ndarray
    cell_bedrock_slowness_s_per_m: np.ndarray
    cell_bedrock_velocity_m_s: np.ndarray
    cell_velocity_status: np.ndarray
    cell_observation_count: np.ndarray
    row_midpoint_cell_id: np.ndarray
    row_midpoint_bedrock_slowness_s_per_m: np.ndarray
    row_midpoint_bedrock_velocity_m_s: np.ndarray
    row_midpoint_v2_m_s: np.ndarray
    cell_v2_m_s: np.ndarray

    modeled_pick_time_s_sorted: np.ndarray
    residual_s_sorted: np.ndarray
    residual_ms_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray
    rejected_observation_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    rms_residual_s: float
    rms_residual_ms: float

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
    trace_arrays_are_sorted: bool


def build_refraction_half_intercept_result(
    *,
    input_model: RefractionStaticInputModel,
    design: RefractionStaticDesignMatrix,
    solve_result: RefractionStaticSolveResult,
    include_debug_objects: bool = False,
) -> RefractionHalfInterceptResult:
    """Assemble node, endpoint, and trace-order half-intercept outputs."""
    _validate_result_inputs(
        input_model=input_model,
        design=design,
        solve_result=solve_result,
    )
    return _build_refraction_half_intercept_result_from_solution(
        input_model=input_model,
        solution=_solution_from_solve_result(solve_result),
        include_debug_objects=include_debug_objects,
        debug_design=design,
        debug_solve_result=solve_result,
    )


def _build_refraction_half_intercept_result_from_solution(
    *,
    input_model: RefractionStaticInputModel,
    solution: _HalfInterceptSolution,
    include_debug_objects: bool,
    debug_design: RefractionStaticDesignMatrix | None,
    debug_solve_result: RefractionStaticSolveResult | None,
) -> RefractionHalfInterceptResult:
    solve_result = solution
    trace_index = np.ascontiguousarray(input_model.sorted_trace_index, dtype=np.int64)
    (
        modeled_pick_time_s_sorted,
        residual_s_sorted,
        residual_ms_sorted,
        used_mask,
        rejected_mask,
        rejected_iteration_sorted,
    ) = _solution_trace_arrays_sorted(solve_result, trace_index)

    node_pick_count = _node_count_for_mask(
        input_model=input_model,
        mask=np.asarray(input_model.valid_pick_mask_sorted, dtype=bool),
    )
    node_rejected_count = _node_count_for_mask(
        input_model=input_model,
        mask=rejected_mask,
    )
    source_endpoint = _endpoint_result(
        input_model=input_model,
        kind='source',
        used_mask=used_mask,
        rejected_mask=rejected_mask,
        solve_result=solve_result,
    )
    receiver_endpoint = _endpoint_result(
        input_model=input_model,
        kind='receiver',
        used_mask=used_mask,
        rejected_mask=rejected_mask,
        solve_result=solve_result,
    )

    source_t1 = _map_nodes_to_values(
        input_model.source_node_id_sorted,
        node_id=solve_result.node_id,
        values=solve_result.node_half_intercept_time_s,
        fill_value=np.nan,
    )
    receiver_t1 = _map_nodes_to_values(
        input_model.receiver_node_id_sorted,
        node_id=solve_result.node_id,
        values=solve_result.node_half_intercept_time_s,
        fill_value=np.nan,
    )
    source_status = _map_nodes_to_strings(
        input_model.source_node_id_sorted,
        node_id=solve_result.node_id,
        values=solve_result.node_solution_status,
        fill_value='missing_node',
    )
    receiver_status = _map_nodes_to_strings(
        input_model.receiver_node_id_sorted,
        node_id=solve_result.node_id,
        values=solve_result.node_solution_status,
        fill_value='missing_node',
    )
    trace_half_intercept = np.ascontiguousarray(source_t1 + receiver_t1)
    trace_status = _trace_half_intercept_status(
        source_t1=source_t1,
        receiver_t1=receiver_t1,
        source_status=source_status,
        receiver_status=receiver_status,
    )

    residual_stats = _residual_statistics(
        residual_s_sorted,
        used_mask=used_mask,
    )
    qc = _build_qc(
        input_model=input_model,
        solve_result=solve_result,
        residual_stats=residual_stats,
        source_endpoint=source_endpoint,
        receiver_endpoint=receiver_endpoint,
        node_pick_count=node_pick_count,
        node_rejected_count=node_rejected_count,
    )

    return RefractionHalfInterceptResult(
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
        node_pick_count=node_pick_count,
        node_used_observation_count=solve_result.node_observation_count,
        node_rejected_observation_count=node_rejected_count,
        source_endpoint=source_endpoint,
        receiver_endpoint=receiver_endpoint,
        trace_index_sorted=trace_index,
        source_endpoint_key_sorted=np.ascontiguousarray(
            input_model.source_endpoint_key_sorted,
        ),
        receiver_endpoint_key_sorted=np.ascontiguousarray(
            input_model.receiver_endpoint_key_sorted,
        ),
        source_endpoint_id_sorted=_trace_endpoint_id(input_model, kind='source'),
        receiver_endpoint_id_sorted=_trace_endpoint_id(input_model, kind='receiver'),
        source_node_id_sorted=np.ascontiguousarray(input_model.source_node_id_sorted),
        receiver_node_id_sorted=np.ascontiguousarray(input_model.receiver_node_id_sorted),
        source_half_intercept_time_s_sorted=source_t1,
        receiver_half_intercept_time_s_sorted=receiver_t1,
        trace_half_intercept_time_s_sorted=trace_half_intercept,
        trace_half_intercept_status_sorted=trace_status,
        pick_time_s_sorted=np.ascontiguousarray(input_model.pick_time_s_sorted),
        modeled_pick_time_s_sorted=modeled_pick_time_s_sorted,
        residual_s_sorted=residual_s_sorted,
        residual_ms_sorted=residual_ms_sorted,
        used_observation_mask_sorted=used_mask,
        rejected_observation_mask_sorted=rejected_mask,
        rejected_iteration_sorted=rejected_iteration_sorted,
        cell_id=solve_result.cell_id,
        cell_bedrock_slowness_s_per_m=solve_result.cell_bedrock_slowness_s_per_m,
        cell_bedrock_velocity_m_s=solve_result.cell_bedrock_velocity_m_s,
        cell_velocity_status=solve_result.cell_velocity_status,
        cell_observation_count=solve_result.cell_observation_count,
        row_midpoint_cell_id=solve_result.row_midpoint_cell_id,
        row_midpoint_bedrock_slowness_s_per_m=(
            solve_result.row_midpoint_bedrock_slowness_s_per_m
        ),
        row_midpoint_bedrock_velocity_m_s=(
            solve_result.row_midpoint_bedrock_velocity_m_s
        ),
        row_midpoint_v2_m_s=solve_result.row_midpoint_v2_m_s,
        cell_v2_m_s=solve_result.cell_v2_m_s,
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
        debug_design=debug_design if include_debug_objects else None,
        debug_solve_result=debug_solve_result if include_debug_objects else None,
    )


def build_refraction_half_intercept_result_from_bedrock_result(
    *,
    input_model: RefractionStaticInputModel,
    bedrock_result: GlobalBedrockSlownessEstimateResult,
    include_debug_objects: bool = False,
) -> RefractionHalfInterceptResult:
    """Build a half-intercept result from a public bedrock facade result."""
    _validate_bedrock_result_inputs(
        input_model=input_model,
        bedrock_result=bedrock_result,
    )
    debug_solve_result = bedrock_result.debug_solve_result
    return _build_refraction_half_intercept_result_from_solution(
        input_model=input_model,
        solution=_solution_from_bedrock_result(bedrock_result),
        include_debug_objects=include_debug_objects,
        debug_design=None if debug_solve_result is None else debug_solve_result.design,
        debug_solve_result=debug_solve_result,
    )


def estimate_refraction_half_intercept_from_design(
    *,
    input_model: RefractionStaticInputModel,
    design: RefractionStaticDesignMatrix,
    model: RefractionStaticModelOptions,
    solver_options: RefractionStaticSolverOptions | None = None,
    include_debug_objects: bool = False,
) -> RefractionHalfInterceptResult:
    """Solve an existing pure design matrix and assemble half-intercept outputs."""
    options = validate_refraction_static_solver_options(solver_options)
    solve_result = solve_refraction_static_design_least_squares(
        design,
        model=model,
        solver_options=options,
    )
    return build_refraction_half_intercept_result(
        input_model=input_model,
        design=design,
        solve_result=solve_result,
        include_debug_objects=include_debug_objects,
    )


def estimate_refraction_half_intercept_from_input_model(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    solver_options: RefractionStaticSolverOptions | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
    include_debug_objects: bool = False,
) -> RefractionHalfInterceptResult:
    """Build the GLI design, solve it, and assemble half-intercept outputs."""
    options = validate_refraction_static_solver_options(solver_options)
    solve_result = solve_refraction_static_least_squares(
        input_model=input_model,
        model=model,
        solver_options=options,
        resolved_first_layer=resolved_first_layer,
        include_diagnostics=include_diagnostics,
    )
    return build_refraction_half_intercept_result(
        input_model=input_model,
        design=solve_result.design,
        solve_result=solve_result,
        include_debug_objects=include_debug_objects,
    )


def build_refraction_half_intercept_design(
    *,
    input_model: RefractionStaticInputModel,
    model: RefractionStaticModelOptions,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
) -> RefractionStaticDesignMatrix:
    """Build the package design matrix used by half-intercept solving."""
    return build_refraction_static_design_matrix(
        input_model=input_model,
        model=model,
        resolved_first_layer=resolved_first_layer,
        include_diagnostics=include_diagnostics,
    )


def _solution_from_solve_result(
    solve_result: RefractionStaticSolveResult,
) -> _HalfInterceptSolution:
    return _HalfInterceptSolution(
        node_id=solve_result.node_id,
        node_half_intercept_time_s=solve_result.node_half_intercept_time_s,
        node_solution_status=solve_result.node_solution_status,
        node_observation_count=solve_result.node_observation_count,
        bedrock_velocity_mode=solve_result.bedrock_velocity_mode,
        bedrock_velocity_m_s=solve_result.bedrock_velocity_m_s,
        bedrock_slowness_s_per_m=solve_result.bedrock_slowness_s_per_m,
        bedrock_velocity_status=solve_result.bedrock_velocity_status,
        cell_id=solve_result.cell_id,
        cell_bedrock_slowness_s_per_m=solve_result.cell_bedrock_slowness_s_per_m,
        cell_bedrock_velocity_m_s=solve_result.cell_bedrock_velocity_m_s,
        cell_velocity_status=solve_result.cell_velocity_status,
        cell_observation_count=solve_result.cell_observation_count,
        row_midpoint_cell_id=solve_result.row_midpoint_cell_id,
        row_midpoint_bedrock_slowness_s_per_m=(
            solve_result.row_midpoint_bedrock_slowness_s_per_m
        ),
        row_midpoint_bedrock_velocity_m_s=(
            solve_result.row_midpoint_bedrock_velocity_m_s
        ),
        row_midpoint_v2_m_s=solve_result.row_midpoint_v2_m_s,
        cell_v2_m_s=solve_result.cell_v2_m_s,
        modeled_pick_time_s_sorted=solve_result.modeled_pick_time_s_sorted,
        residual_s_sorted=solve_result.residual_s_sorted,
        residual_ms_sorted=solve_result.residual_ms_sorted,
        used_observation_mask_sorted=solve_result.used_observation_mask_sorted,
        rejected_observation_mask_sorted=solve_result.rejected_observation_mask_sorted,
        rejected_iteration_sorted=solve_result.rejected_iteration_sorted,
        rms_residual_s=solve_result.rms_residual_s,
        rms_residual_ms=solve_result.rms_residual_ms,
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
        qc=solve_result.qc,
        trace_arrays_are_sorted=False,
    )


def _solution_from_bedrock_result(
    bedrock_result: GlobalBedrockSlownessEstimateResult,
) -> _HalfInterceptSolution:
    empty_int = np.empty(0, dtype=np.int64)
    empty_float = np.empty(0, dtype=np.float64)
    empty_status = np.empty(0, dtype='<U32')
    return _HalfInterceptSolution(
        node_id=bedrock_result.node_id,
        node_half_intercept_time_s=bedrock_result.node_half_intercept_time_s,
        node_solution_status=bedrock_result.node_solution_status,
        node_observation_count=bedrock_result.node_observation_count,
        bedrock_velocity_mode=bedrock_result.bedrock_velocity_mode,
        bedrock_velocity_m_s=bedrock_result.bedrock_velocity_m_s,
        bedrock_slowness_s_per_m=bedrock_result.bedrock_slowness_s_per_m,
        bedrock_velocity_status=bedrock_result.bedrock_velocity_status,
        cell_id=empty_int,
        cell_bedrock_slowness_s_per_m=empty_float,
        cell_bedrock_velocity_m_s=empty_float,
        cell_velocity_status=empty_status,
        cell_observation_count=empty_int,
        row_midpoint_cell_id=empty_int,
        row_midpoint_bedrock_slowness_s_per_m=empty_float,
        row_midpoint_bedrock_velocity_m_s=empty_float,
        row_midpoint_v2_m_s=empty_float,
        cell_v2_m_s=empty_float,
        modeled_pick_time_s_sorted=bedrock_result.modeled_pick_time_s_sorted,
        residual_s_sorted=bedrock_result.residual_s_sorted,
        residual_ms_sorted=bedrock_result.residual_ms_sorted,
        used_observation_mask_sorted=bedrock_result.used_observation_mask_sorted,
        rejected_observation_mask_sorted=bedrock_result.rejected_observation_mask_sorted,
        rejected_iteration_sorted=bedrock_result.rejected_iteration_sorted,
        rms_residual_s=bedrock_result.rms_residual_s,
        rms_residual_ms=bedrock_result.rms_residual_ms,
        solver_success=bedrock_result.solver_success,
        solver_status=bedrock_result.solver_status,
        solver_message=bedrock_result.solver_message,
        solver_cost=bedrock_result.solver_cost,
        solver_optimality=bedrock_result.solver_optimality,
        solver_iterations=bedrock_result.solver_iterations,
        robust_enabled=bedrock_result.robust_enabled,
        robust_stop_reason=bedrock_result.robust_stop_reason,
        robust_iteration_summaries=bedrock_result.robust_iteration_summaries,
        n_initial_used_observations=bedrock_result.n_initial_used_observations,
        n_final_used_observations=bedrock_result.n_final_used_observations,
        n_rejected_observations=bedrock_result.n_rejected_observations,
        qc=bedrock_result.qc,
        trace_arrays_are_sorted=True,
    )


def _validate_bedrock_result_inputs(
    *,
    input_model: RefractionStaticInputModel,
    bedrock_result: GlobalBedrockSlownessEstimateResult,
) -> None:
    if not isinstance(input_model, RefractionStaticInputModel):
        raise RefractionHalfInterceptError(
            'input_model must be a RefractionStaticInputModel instance'
        )
    if not isinstance(bedrock_result, GlobalBedrockSlownessEstimateResult):
        raise RefractionHalfInterceptError(
            'bedrock_result must be a GlobalBedrockSlownessEstimateResult instance'
        )
    if int(input_model.n_traces) != int(bedrock_result.n_traces):
        raise RefractionHalfInterceptError(
            'input_model.n_traces must match bedrock_result.n_traces'
        )
    if bedrock_result.bedrock_velocity_mode != 'solve_global':
        raise RefractionHalfInterceptError(
            'bedrock_result.bedrock_velocity_mode must be solve_global'
        )
    expected_shape = (int(input_model.sorted_trace_index.shape[0]),)
    for name in (
        'pick_time_s_sorted',
        'valid_pick_mask_sorted',
        'source_endpoint_key_sorted',
        'receiver_endpoint_key_sorted',
        'source_node_id_sorted',
        'receiver_node_id_sorted',
    ):
        value = np.asarray(getattr(input_model, name))
        if value.shape != expected_shape:
            raise RefractionHalfInterceptError(f'input_model.{name} shape mismatch')
    trace_index = np.asarray(input_model.sorted_trace_index, dtype=np.int64)
    if trace_index.size and (
        np.min(trace_index) < 0
        or np.max(trace_index) >= bedrock_result.modeled_pick_time_s_sorted.shape[0]
    ):
        raise RefractionHalfInterceptError(
            'input_model.sorted_trace_index values are outside bedrock result trace range'
        )
    _validate_global_bedrock_consistency(_solution_from_bedrock_result(bedrock_result))


def _validate_result_inputs(
    *,
    input_model: RefractionStaticInputModel,
    design: RefractionStaticDesignMatrix,
    solve_result: RefractionStaticSolveResult,
) -> None:
    if not isinstance(input_model, RefractionStaticInputModel):
        raise RefractionHalfInterceptError(
            'input_model must be a RefractionStaticInputModel instance'
        )
    if not isinstance(design, RefractionStaticDesignMatrix):
        raise RefractionHalfInterceptError(
            'design must be a RefractionStaticDesignMatrix instance'
        )
    if not isinstance(solve_result, RefractionStaticSolveResult):
        raise RefractionHalfInterceptError(
            'solve_result must be a RefractionStaticSolveResult instance'
        )
    expected_shape = (int(input_model.sorted_trace_index.shape[0]),)
    for name in (
        'pick_time_s_sorted',
        'valid_pick_mask_sorted',
        'source_endpoint_key_sorted',
        'receiver_endpoint_key_sorted',
        'source_node_id_sorted',
        'receiver_node_id_sorted',
    ):
        value = np.asarray(getattr(input_model, name))
        if value.shape != expected_shape:
            raise RefractionHalfInterceptError(f'input_model.{name} shape mismatch')
    if int(input_model.n_traces) != int(design.qc['n_traces']):
        raise RefractionHalfInterceptError(
            'input_model.n_traces must match design.qc["n_traces"]'
        )
    if solve_result.design is not design:
        same_design = (
            int(solve_result.design.n_observations) == int(design.n_observations)
            and int(solve_result.design.n_parameters) == int(design.n_parameters)
            and solve_result.design.bedrock_velocity_mode
            == design.bedrock_velocity_mode
            and np.array_equal(
                solve_result.design.row_trace_index_sorted,
                design.row_trace_index_sorted,
            )
            and np.array_equal(
                solve_result.design.active_node_id,
                design.active_node_id,
            )
        )
        if not same_design:
            raise RefractionHalfInterceptError(
                'solve_result.design must match design'
            )
    trace_index = np.asarray(input_model.sorted_trace_index, dtype=np.int64)
    if trace_index.size and (
        np.min(trace_index) < 0
        or np.max(trace_index) >= solve_result.modeled_pick_time_s_sorted.shape[0]
    ):
        raise RefractionHalfInterceptError(
            'input_model.sorted_trace_index values are outside solve result trace range'
        )
    _validate_global_bedrock_consistency(solve_result)


def _validate_global_bedrock_consistency(
    solve_result: RefractionStaticSolveResult,
) -> None:
    if solve_result.bedrock_velocity_mode == 'solve_cell':
        return
    velocity = float(solve_result.bedrock_velocity_m_s)
    slowness = float(solve_result.bedrock_slowness_s_per_m)
    if not (np.isfinite(velocity) and velocity > 0.0):
        raise RefractionHalfInterceptError('bedrock_velocity_m_s must be positive')
    if not (np.isfinite(slowness) and slowness > 0.0):
        raise RefractionHalfInterceptError('bedrock_slowness_s_per_m must be positive')
    if not np.isclose(velocity * slowness, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise RefractionHalfInterceptError(
            'bedrock_velocity_m_s and bedrock_slowness_s_per_m are inconsistent'
        )


def _endpoint_result(
    *,
    input_model: RefractionStaticInputModel,
    kind: str,
    used_mask: np.ndarray,
    rejected_mask: np.ndarray,
    solve_result: RefractionStaticSolveResult,
) -> RefractionHalfInterceptEndpointResult:
    key_sorted = _endpoint_key_sorted(input_model, kind=kind)
    node_sorted = _endpoint_node_id_sorted(input_model, kind=kind)
    endpoint_id_sorted = _trace_endpoint_id(input_model, kind=kind)
    endpoint_key, first_index = _unique_first(key_sorted)
    node_id = np.ascontiguousarray(node_sorted[first_index], dtype=np.int64)
    endpoint_id = (
        None
        if endpoint_id_sorted is None
        else np.ascontiguousarray(endpoint_id_sorted[first_index], dtype=np.int64)
    )
    valid_pick_mask = np.asarray(input_model.valid_pick_mask_sorted, dtype=bool)
    return RefractionHalfInterceptEndpointResult(
        endpoint_key=endpoint_key,
        endpoint_id=endpoint_id,
        node_id=node_id,
        half_intercept_time_s=_map_nodes_to_values(
            node_id,
            node_id=solve_result.node_id,
            values=solve_result.node_half_intercept_time_s,
            fill_value=np.nan,
        ),
        solution_status=_map_nodes_to_strings(
            node_id,
            node_id=solve_result.node_id,
            values=solve_result.node_solution_status,
            fill_value='missing_node',
        ),
        pick_count=_endpoint_count(key_sorted, endpoint_key, mask=valid_pick_mask),
        used_observation_count=_endpoint_count(key_sorted, endpoint_key, mask=used_mask),
        rejected_observation_count=_endpoint_count(
            key_sorted,
            endpoint_key,
            mask=rejected_mask,
        ),
    )


def _endpoint_key_sorted(input_model: RefractionStaticInputModel, *, kind: str) -> np.ndarray:
    if kind == 'source':
        return np.ascontiguousarray(input_model.source_endpoint_key_sorted)
    if kind == 'receiver':
        return np.ascontiguousarray(input_model.receiver_endpoint_key_sorted)
    raise RefractionHalfInterceptError('endpoint kind must be source or receiver')


def _endpoint_node_id_sorted(
    input_model: RefractionStaticInputModel,
    *,
    kind: str,
) -> np.ndarray:
    if kind == 'source':
        return np.ascontiguousarray(input_model.source_node_id_sorted, dtype=np.int64)
    if kind == 'receiver':
        return np.ascontiguousarray(input_model.receiver_node_id_sorted, dtype=np.int64)
    raise RefractionHalfInterceptError('endpoint kind must be source or receiver')


def _trace_endpoint_id(
    input_model: RefractionStaticInputModel,
    *,
    kind: str,
) -> np.ndarray | None:
    explicit = (
        input_model.source_endpoint_id_sorted
        if kind == 'source'
        else input_model.receiver_endpoint_id_sorted
    )
    if explicit is not None:
        return np.ascontiguousarray(explicit, dtype=np.int64)
    lookup = _endpoint_table_id_lookup(input_model, kind=kind)
    node_id = _endpoint_node_id_sorted(input_model, kind=kind)
    if not lookup:
        return None
    values = np.asarray([lookup.get(int(node), -1) for node in node_id], dtype=np.int64)
    if np.any(values < 0):
        return None
    return np.ascontiguousarray(values)


def _endpoint_table_id_lookup(
    input_model: RefractionStaticInputModel,
    *,
    kind: str,
) -> dict[int, int]:
    table = input_model.endpoint_table
    lookup: dict[int, int] = {}
    for raw_node, raw_endpoint, raw_kind in zip(
        table.node_id.tolist(),
        table.endpoint_id.tolist(),
        table.kind.tolist(),
        strict=True,
    ):
        if str(raw_kind) == kind:
            lookup[int(raw_node)] = int(raw_endpoint)
    return lookup


def _unique_first(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen: set[str] = set()
    keys: list[str] = []
    indices: list[int] = []
    for index, raw_value in enumerate(values.tolist()):
        value = str(raw_value)
        if value in seen:
            continue
        seen.add(value)
        keys.append(value)
        indices.append(index)
    return (
        np.ascontiguousarray(np.asarray(keys, dtype=values.dtype)),
        np.ascontiguousarray(np.asarray(indices, dtype=np.int64)),
    )


def _endpoint_count(
    key_sorted: np.ndarray,
    endpoint_key: np.ndarray,
    *,
    mask: np.ndarray,
) -> np.ndarray:
    key_to_pos = {str(key): idx for idx, key in enumerate(endpoint_key.tolist())}
    count = np.zeros(endpoint_key.shape, dtype=np.int64)
    for key, selected in zip(key_sorted.tolist(), mask.tolist(), strict=True):
        if selected:
            count[key_to_pos[str(key)]] += 1
    return np.ascontiguousarray(count)


def _node_count_for_mask(
    *,
    input_model: RefractionStaticInputModel,
    mask: np.ndarray,
) -> np.ndarray:
    node_id = np.ascontiguousarray(input_model.endpoint_table.node_id, dtype=np.int64)
    node_to_pos = {int(node): idx for idx, node in enumerate(node_id.tolist())}
    count = np.zeros(node_id.shape, dtype=np.int64)
    for source, receiver, selected in zip(
        input_model.source_node_id_sorted.tolist(),
        input_model.receiver_node_id_sorted.tolist(),
        mask.tolist(),
        strict=True,
    ):
        if not selected:
            continue
        source_pos = node_to_pos.get(int(source))
        receiver_pos = node_to_pos.get(int(receiver))
        if source_pos is not None:
            count[source_pos] += 1
        if receiver_pos is not None and receiver_pos != source_pos:
            count[receiver_pos] += 1
    return np.ascontiguousarray(count)


def _map_nodes_to_values(
    nodes: np.ndarray,
    *,
    node_id: np.ndarray,
    values: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    lookup = {int(node): float(value) for node, value in zip(node_id, values, strict=True)}
    return np.ascontiguousarray(
        np.asarray([lookup.get(int(node), fill_value) for node in nodes], dtype=np.float64)
    )


def _map_nodes_to_strings(
    nodes: np.ndarray,
    *,
    node_id: np.ndarray,
    values: np.ndarray,
    fill_value: str,
) -> np.ndarray:
    lookup = {int(node): str(value) for node, value in zip(node_id, values, strict=True)}
    return np.ascontiguousarray(
        np.asarray([lookup.get(int(node), fill_value) for node in nodes])
    )


def _trace_half_intercept_status(
    *,
    source_t1: np.ndarray,
    receiver_t1: np.ndarray,
    source_status: np.ndarray,
    receiver_status: np.ndarray,
) -> np.ndarray:
    status = np.full(source_t1.shape, 'solved', dtype='<U32')
    source = np.asarray(source_status).astype(str, copy=False)
    receiver = np.asarray(receiver_status).astype(str, copy=False)
    missing_node = (source == 'missing_node') | (receiver == 'missing_node')
    low_fold = (source == 'low_fold') | (receiver == 'low_fold')
    same = source == receiver
    status[same] = source[same]
    mismatch = ~same
    source_solved = source == 'solved'
    receiver_solved = receiver == 'solved'
    status[mismatch & source_solved] = receiver[mismatch & source_solved]
    status[mismatch & receiver_solved] = source[mismatch & receiver_solved]
    status[mismatch & ~source_solved & ~receiver_solved] = 'mixed'
    status[low_fold] = 'low_fold'
    status[missing_node] = 'missing_node'
    invalid_value = ~np.isfinite(source_t1) | ~np.isfinite(receiver_t1)
    status[invalid_value & (status == 'solved')] = 'invalid_half_intercept'
    return np.ascontiguousarray(status)


def _solution_trace_arrays_sorted(
    solution: _HalfInterceptSolution,
    trace_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if solution.trace_arrays_are_sorted:
        return (
            np.ascontiguousarray(
                np.asarray(solution.modeled_pick_time_s_sorted, dtype=np.float64)
            ),
            np.ascontiguousarray(np.asarray(solution.residual_s_sorted, dtype=np.float64)),
            np.ascontiguousarray(
                np.asarray(solution.residual_ms_sorted, dtype=np.float64)
            ),
            np.ascontiguousarray(
                np.asarray(solution.used_observation_mask_sorted, dtype=bool)
            ),
            np.ascontiguousarray(
                np.asarray(solution.rejected_observation_mask_sorted, dtype=bool)
            ),
            np.ascontiguousarray(
                np.asarray(solution.rejected_iteration_sorted, dtype=np.int64)
            ),
        )
    return (
        _trace_indexed_to_sorted_float(
            solution.modeled_pick_time_s_sorted,
            trace_index,
        ),
        _trace_indexed_to_sorted_float(
            solution.residual_s_sorted,
            trace_index,
        ),
        _trace_indexed_to_sorted_float(
            solution.residual_ms_sorted,
            trace_index,
        ),
        _trace_indexed_to_sorted_bool(
            solution.used_observation_mask_sorted,
            trace_index,
        ),
        _trace_indexed_to_sorted_bool(
            solution.rejected_observation_mask_sorted,
            trace_index,
        ),
        _trace_indexed_to_sorted_int(
            solution.rejected_iteration_sorted,
            trace_index,
        ),
    )


def _trace_indexed_to_sorted_float(
    values: np.ndarray,
    trace_index: np.ndarray,
) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64)[trace_index])


def _trace_indexed_to_sorted_int(
    values: np.ndarray,
    trace_index: np.ndarray,
) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64)[trace_index])


def _trace_indexed_to_sorted_bool(
    values: np.ndarray,
    trace_index: np.ndarray,
) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=bool)[trace_index])


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


def _build_qc(
    *,
    input_model: RefractionStaticInputModel,
    solve_result: RefractionStaticSolveResult,
    residual_stats: dict[str, float],
    source_endpoint: RefractionHalfInterceptEndpointResult,
    receiver_endpoint: RefractionHalfInterceptEndpointResult,
    node_pick_count: np.ndarray,
    node_rejected_count: np.ndarray,
) -> dict[str, Any]:
    qc = dict(solve_result.qc)
    qc['residual_statistics'] = dict(residual_stats)
    qc['half_intercept'] = {
        'file_id': str(input_model.file_id),
        'n_traces': int(input_model.n_traces),
        'n_nodes': int(solve_result.node_id.shape[0]),
        'n_source_endpoints': int(source_endpoint.endpoint_key.shape[0]),
        'n_receiver_endpoints': int(receiver_endpoint.endpoint_key.shape[0]),
        'node_pick_count': [int(value) for value in node_pick_count.tolist()],
        'node_used_observation_count': [
            int(value) for value in solve_result.node_observation_count.tolist()
        ],
        'node_rejected_observation_count': [
            int(value) for value in node_rejected_count.tolist()
        ],
        'source_endpoint_key': [
            str(value) for value in source_endpoint.endpoint_key.tolist()
        ],
        'source_endpoint_pick_count': [
            int(value) for value in source_endpoint.pick_count.tolist()
        ],
        'source_endpoint_used_observation_count': [
            int(value) for value in source_endpoint.used_observation_count.tolist()
        ],
        'source_endpoint_rejected_observation_count': [
            int(value) for value in source_endpoint.rejected_observation_count.tolist()
        ],
        'receiver_endpoint_key': [
            str(value) for value in receiver_endpoint.endpoint_key.tolist()
        ],
        'receiver_endpoint_pick_count': [
            int(value) for value in receiver_endpoint.pick_count.tolist()
        ],
        'receiver_endpoint_used_observation_count': [
            int(value) for value in receiver_endpoint.used_observation_count.tolist()
        ],
        'receiver_endpoint_rejected_observation_count': [
            int(value) for value in receiver_endpoint.rejected_observation_count.tolist()
        ],
    }
    return qc


__all__ = [
    'RefractionHalfInterceptEndpointResult',
    'RefractionHalfInterceptError',
    'RefractionHalfInterceptResult',
    'build_refraction_half_intercept_design',
    'build_refraction_half_intercept_result',
    'build_refraction_half_intercept_result_from_bedrock_result',
    'estimate_refraction_half_intercept_from_design',
    'estimate_refraction_half_intercept_from_input_model',
]
