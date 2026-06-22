"""Bounded least-squares solver for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import optimize, sparse
from scipy.sparse import linalg as sparse_linalg

from seis_statics._validation import (
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics._endpoint_sum_graph import (
    EndpointSumGraphSummary,
    analyze_endpoint_sum_graph,
)
from seis_statics.refraction.cell_regularization import (
    CellSlownessSmoothingRows,
    build_cell_slowness_smoothing_rows,
)
from seis_statics.refraction.cell_velocity_status import (
    LOW_FOLD_CELL_VELOCITY_STATUS,
)
from seis_statics.refraction.design_matrix import (
    LOW_FOLD_NODE_STATUS,
    RefractionStaticDesignMatrix,
    build_refraction_static_design_matrix,
)
from seis_statics.refraction.options import (
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
)
from seis_statics.refraction.types import (
    RefractionStaticInputModel,
    ResolvedRefractionFirstLayer,
)


class RefractionStaticSolverError(ValueError):
    """Raised when refraction static solver inputs are inconsistent."""


_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    allow_integer_like_float=False,
    error_type=RefractionStaticSolverError,
)


@dataclass(frozen=True)
class _NumericalRankDiagnostic:
    """Column-scaled numerical rank summary for an identifiability check."""

    method: str
    n_rows: int
    n_columns: int
    expected_rank: int
    estimated_rank: int
    expected_nullity: int
    gauge_nullity: int
    threshold: float
    critical_singular_value: float
    largest_singular_value: float
    rtol: float
    sparse_solver_name: str = ''
    certification_status: str = 'not_applicable'
    requested_smallest_count: int = 0
    returned_smallest_count: int = 0
    max_singular_triplet_residual: float = 0.0
    failure_reason: str = ''


@dataclass(frozen=True)
class RefractionStaticSolveSystem:
    """Augmented bounded linear system used by the refraction solver."""

    augmented_matrix: sparse.csr_matrix
    augmented_rhs_s: np.ndarray
    observation_matrix: sparse.csr_matrix
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    initial_parameter_vector: np.ndarray

    n_observation_rows: int
    n_smoothing_rows: int
    n_damping_rows: int
    n_gauge_rows: int
    n_augmented_rows: int
    n_parameters: int
    component_id_by_node: np.ndarray
    n_node_components: int
    is_bipartite_by_component: np.ndarray
    signed_partition_by_node: np.ndarray
    gauge_required_by_component: np.ndarray
    n_bipartite_node_components: int
    gauge_resolution: str

    half_intercept_damping_lambda: float
    regularized_parameter_group: str
    regularization_row_count: int
    node_lower_bound_s: float
    node_upper_bound_s: float
    slowness_lower_bound_s_per_m: float | None
    slowness_upper_bound_s_per_m: float | None
    initial_bedrock_slowness_s_per_m: float | None
    smoothing_rows: CellSlownessSmoothingRows | None
    identifiability: _NumericalRankDiagnostic


RefractionStaticRobustStopReason = Literal[
    'disabled',
    'converged',
    'max_iterations',
    'zero_scale',
    'safe_rejection',
]


@dataclass(frozen=True)
class RefractionStaticRobustIterationSummary:
    """Numerical summary for one robust refraction rejection pass."""

    iteration_index: int
    method: str

    n_used_before: int
    n_rejected_this_iteration: int
    n_used_after: int

    residual_center_s: float
    residual_scale_s: float
    residual_scale_floor_s: float
    residual_cutoff_s: float
    max_abs_centered_residual_s: float

    converged: bool
    stop_reason: RefractionStaticRobustStopReason | None


@dataclass(frozen=True)
class RefractionStaticSolveResult:
    """GLI refraction static solution."""

    parameter_vector: np.ndarray

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

    row_modeled_pick_time_s: np.ndarray
    row_residual_s: np.ndarray
    rms_residual_s: float
    rms_residual_ms: float

    solver_success: bool
    solver_status: int
    solver_message: str
    solver_cost: float
    solver_optimality: float
    solver_iterations: int
    solver_active_mask: np.ndarray

    robust_enabled: bool
    robust_stop_reason: RefractionStaticRobustStopReason
    robust_iteration_summaries: tuple[RefractionStaticRobustIterationSummary, ...]
    n_initial_used_observations: int
    n_final_used_observations: int
    n_rejected_observations: int

    design: RefractionStaticDesignMatrix
    system: RefractionStaticSolveSystem
    solver_options: RefractionStaticSolverOptions
    qc: dict[str, Any]


@dataclass(frozen=True)
class _RefractionStaticRobustRunResult:
    raw_result: optimize.OptimizeResult
    system: RefractionStaticSolveSystem
    row_used_mask: np.ndarray
    rejected_iteration_sorted: np.ndarray
    iteration_summaries: tuple[RefractionStaticRobustIterationSummary, ...]
    stop_reason: RefractionStaticRobustStopReason


@dataclass(frozen=True)
class _LsqLinearQualityDiagnostic:
    """Post-solve quality checks in the original unscaled LSQ system."""

    verified: bool
    stage: str
    failure_reason: str
    solve_scale: float
    scipy_success: bool
    scipy_status: int
    scipy_optimality: float
    scipy_iterations: int
    unscaled_augmented_residual_norm: float
    unscaled_objective: float
    projected_gradient_inf_norm: float
    kkt_tolerance: float
    max_bound_violation: float
    bound_tolerance: float


def build_refraction_static_solver_system(
    design: RefractionStaticDesignMatrix,
    *,
    model: Any,
    solver_options: RefractionStaticSolverOptions | None = None,
    row_used_mask: np.ndarray | None = None,
) -> RefractionStaticSolveSystem:
    """Build observation, damping, bounds, and initial vector."""
    options = validate_refraction_static_solver_options(solver_options)
    _validate_design(design)
    _validate_model_for_design(model=model, design=design)

    n_parameters = int(design.n_parameters)
    lower_bounds = np.zeros(n_parameters, dtype=np.float64)
    upper_bounds = np.full(n_parameters, np.inf, dtype=np.float64)
    initial = np.zeros(n_parameters, dtype=np.float64)

    node_upper = options.max_abs_half_intercept_time_ms / 1000.0
    lower_bounds[: design.n_active_nodes] = 0.0
    upper_bounds[: design.n_active_nodes] = node_upper

    slowness_lower: float | None = None
    slowness_upper: float | None = None
    initial_slowness: float | None = None
    if design.bedrock_velocity_mode == 'solve_global':
        slowness_col = _global_slowness_col(design)
        min_velocity, max_velocity, initial_velocity = _model_velocity_bounds(model)
        slowness_lower = float(1.0 / max_velocity)
        slowness_upper = float(1.0 / min_velocity)
        initial_slowness = float(1.0 / initial_velocity)
        lower_bounds[slowness_col] = slowness_lower
        upper_bounds[slowness_col] = slowness_upper
        initial[slowness_col] = initial_slowness
    elif design.bedrock_velocity_mode == 'solve_cell':
        min_velocity, max_velocity, initial_velocity = _model_velocity_bounds(model)
        slowness_lower = float(1.0 / max_velocity)
        slowness_upper = float(1.0 / min_velocity)
        initial_slowness = float(1.0 / initial_velocity)
        for col in _cell_slowness_cols(design):
            lower_bounds[col] = slowness_lower
            upper_bounds[col] = slowness_upper
            initial[col] = initial_slowness

    smoothing_rows = _build_cell_smoothing_rows_for_design(design, model=model)

    damping_matrix, damping_rhs = _build_damping_system(
        n_parameters=n_parameters,
        n_active_nodes=design.n_active_nodes,
        half_intercept_damping_lambda=options.half_intercept_damping_lambda,
    )
    return _build_refraction_static_solver_system_from_parts(
        design=design,
        row_used_mask=row_used_mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        initial_parameter_vector=initial,
        smoothing_rows=smoothing_rows,
        damping_matrix=damping_matrix,
        damping_rhs=damping_rhs,
        half_intercept_damping_lambda=options.half_intercept_damping_lambda,
        identifiability_rtol=options.identifiability_rtol,
        node_lower_bound_s=0.0,
        node_upper_bound_s=float(node_upper),
        slowness_lower_bound_s_per_m=slowness_lower,
        slowness_upper_bound_s_per_m=slowness_upper,
        initial_bedrock_slowness_s_per_m=initial_slowness,
    )


def _build_refraction_static_solver_system_from_parts(
    *,
    design: RefractionStaticDesignMatrix,
    row_used_mask: np.ndarray | None,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    initial_parameter_vector: np.ndarray,
    smoothing_rows: CellSlownessSmoothingRows | None,
    damping_matrix: sparse.csr_matrix,
    damping_rhs: np.ndarray,
    half_intercept_damping_lambda: float,
    identifiability_rtol: float,
    node_lower_bound_s: float,
    node_upper_bound_s: float,
    slowness_lower_bound_s_per_m: float | None,
    slowness_upper_bound_s_per_m: float | None,
    initial_bedrock_slowness_s_per_m: float | None,
) -> RefractionStaticSolveSystem:
    n_parameters = int(design.n_parameters)
    mask = (
        np.ones(design.n_observations, dtype=bool)
        if row_used_mask is None
        else _coerce_design_row_mask(design, row_used_mask=row_used_mask)
    )
    used_rows = np.flatnonzero(mask).astype(np.int64, copy=False)
    if used_rows.size == 0:
        raise RefractionStaticSolverError(
            'refraction solver system requires at least one observation row'
        )

    observation_matrix = design.matrix[used_rows].tocsr()
    observation_rhs = np.ascontiguousarray(design.rhs_s[used_rows], dtype=np.float64)
    graph = _analyze_design_endpoint_graph(design, row_used_mask=mask)
    n_gauge_required_components = int(
        np.count_nonzero(graph.gauge_required_by_component)
    )
    gauge_resolution = _gauge_resolution_for_system(
        half_intercept_damping_lambda=half_intercept_damping_lambda,
        n_gauge_required_components=n_gauge_required_components,
    )
    smoothing_matrix = (
        smoothing_rows.matrix if smoothing_rows is not None else _empty_rows(n_parameters)
    )
    smoothing_rhs = (
        smoothing_rows.rhs_s if smoothing_rows is not None else np.empty(0)
    )
    physical_matrix = sparse.vstack(
        [observation_matrix, smoothing_matrix],
        format='csr',
        dtype=np.float64,
    )
    physical_matrix.sort_indices()
    identifiability = _validate_physical_identifiability(
        physical_matrix,
        mode=design.bedrock_velocity_mode,
        n_parameters=n_parameters,
        gauge_nullity=n_gauge_required_components,
        rtol=float(identifiability_rtol),
    )

    augmented_matrix = sparse.vstack(
        [
            observation_matrix,
            smoothing_matrix,
            damping_matrix,
        ],
        format='csr',
        dtype=np.float64,
    )
    augmented_matrix.sort_indices()
    augmented_rhs = np.ascontiguousarray(
        np.concatenate(
            [
                observation_rhs,
                smoothing_rhs,
                damping_rhs,
            ]
        ),
        dtype=np.float64,
    )
    n_augmented_rows = int(augmented_matrix.shape[0])
    if augmented_matrix.shape != (n_augmented_rows, n_parameters):
        raise RefractionStaticSolverError('augmented_matrix shape mismatch')
    if augmented_rhs.shape != (n_augmented_rows,):
        raise RefractionStaticSolverError('augmented_rhs_s shape mismatch')
    _validate_finite(augmented_matrix.data, name='augmented_matrix.data')
    _validate_finite(augmented_rhs, name='augmented_rhs_s')
    _validate_bounds(lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    return RefractionStaticSolveSystem(
        augmented_matrix=augmented_matrix,
        augmented_rhs_s=augmented_rhs,
        observation_matrix=observation_matrix,
        lower_bounds=np.ascontiguousarray(lower_bounds, dtype=np.float64),
        upper_bounds=np.ascontiguousarray(upper_bounds, dtype=np.float64),
        initial_parameter_vector=np.ascontiguousarray(
            initial_parameter_vector,
            dtype=np.float64,
        ),
        n_observation_rows=int(used_rows.size),
        n_smoothing_rows=0 if smoothing_rows is None else smoothing_rows.n_rows,
        n_damping_rows=int(damping_matrix.shape[0]),
        n_gauge_rows=0,
        n_augmented_rows=n_augmented_rows,
        n_parameters=n_parameters,
        component_id_by_node=graph.component_id_by_node,
        n_node_components=graph.n_components,
        is_bipartite_by_component=graph.is_bipartite_by_component,
        signed_partition_by_node=graph.signed_partition_by_node,
        gauge_required_by_component=graph.gauge_required_by_component,
        n_bipartite_node_components=int(
            np.count_nonzero(graph.is_bipartite_by_component)
        ),
        gauge_resolution=gauge_resolution,
        half_intercept_damping_lambda=float(half_intercept_damping_lambda),
        regularized_parameter_group='node_half_intercept_time_s',
        regularization_row_count=int(damping_matrix.shape[0]),
        node_lower_bound_s=float(node_lower_bound_s),
        node_upper_bound_s=float(node_upper_bound_s),
        slowness_lower_bound_s_per_m=slowness_lower_bound_s_per_m,
        slowness_upper_bound_s_per_m=slowness_upper_bound_s_per_m,
        initial_bedrock_slowness_s_per_m=initial_bedrock_slowness_s_per_m,
        smoothing_rows=smoothing_rows,
        identifiability=identifiability,
    )


def _rebuild_refraction_static_solver_system_for_row_mask(
    *,
    design: RefractionStaticDesignMatrix,
    system: RefractionStaticSolveSystem,
    row_used_mask: np.ndarray,
) -> RefractionStaticSolveSystem:
    damping_matrix, damping_rhs = _build_damping_system(
        n_parameters=system.n_parameters,
        n_active_nodes=design.n_active_nodes,
        half_intercept_damping_lambda=system.half_intercept_damping_lambda,
    )
    return _build_refraction_static_solver_system_from_parts(
        design=design,
        row_used_mask=row_used_mask,
        lower_bounds=system.lower_bounds,
        upper_bounds=system.upper_bounds,
        initial_parameter_vector=system.initial_parameter_vector,
        smoothing_rows=system.smoothing_rows,
        damping_matrix=damping_matrix,
        damping_rhs=damping_rhs,
        half_intercept_damping_lambda=system.half_intercept_damping_lambda,
        identifiability_rtol=system.identifiability.rtol,
        node_lower_bound_s=system.node_lower_bound_s,
        node_upper_bound_s=system.node_upper_bound_s,
        slowness_lower_bound_s_per_m=system.slowness_lower_bound_s_per_m,
        slowness_upper_bound_s_per_m=system.slowness_upper_bound_s_per_m,
        initial_bedrock_slowness_s_per_m=system.initial_bedrock_slowness_s_per_m,
    )


def solve_refraction_static_least_squares(
    *,
    input_model: RefractionStaticInputModel,
    model: Any,
    solver_options: RefractionStaticSolverOptions | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
) -> RefractionStaticSolveResult:
    """Build and solve a non-robust GLI refraction system."""
    options = validate_refraction_static_solver_options(solver_options)
    design = build_refraction_static_design_matrix(
        input_model=input_model,
        model=model,
        resolved_first_layer=resolved_first_layer,
        min_observations_per_node=options.min_picks_per_node,
        include_diagnostics=include_diagnostics,
    )
    return solve_refraction_static_design_least_squares(
        design,
        model=model,
        solver_options=options,
    )


def solve_refraction_static_design_least_squares(
    design: RefractionStaticDesignMatrix,
    *,
    model: Any,
    solver_options: RefractionStaticSolverOptions | None = None,
) -> RefractionStaticSolveResult:
    """Solve a pre-built refraction design matrix."""
    options = validate_refraction_static_solver_options(solver_options)
    system = build_refraction_static_solver_system(
        design,
        model=model,
        solver_options=options,
    )
    robust_result = _run_refraction_static_solver(
        design=design,
        system=system,
        solver_options=options,
    )
    raw = robust_result.raw_result
    final_system = robust_result.system
    row_used_mask = robust_result.row_used_mask
    raw_parameter_vector = np.ascontiguousarray(raw.x, dtype=np.float64)
    parameter_vector = _canonicalize_refraction_parameter_vector(
        raw_parameter_vector,
        system=final_system,
        design=design,
    )
    _validate_canonicalized_lsq_solution(
        raw_parameter_vector,
        parameter_vector,
        system=final_system,
    )
    _validate_finite(parameter_vector, name='parameter_vector')
    active_mask = _active_mask_for_parameter_vector(
        parameter_vector,
        lower_bounds=final_system.lower_bounds,
        upper_bounds=final_system.upper_bounds,
    )

    row_modeled = _row_modeled_pick_time(
        design,
        parameter_vector=parameter_vector,
    )
    row_residual = np.ascontiguousarray(
        design.observed_pick_time_s - row_modeled,
        dtype=np.float64,
    )
    _validate_finite(row_modeled, name='row_modeled_pick_time_s')
    _validate_finite(row_residual, name='row_residual_s')

    full_modeled = np.full(design.qc['n_traces'], np.nan, dtype=np.float64)
    full_residual = np.full(design.qc['n_traces'], np.nan, dtype=np.float64)
    full_modeled[design.row_trace_index_sorted] = row_modeled
    full_residual[design.row_trace_index_sorted] = row_residual
    used_mask = np.zeros(design.qc['n_traces'], dtype=bool)
    used_mask[design.row_trace_index_sorted[row_used_mask]] = True
    initial_used_mask = np.zeros(design.qc['n_traces'], dtype=bool)
    initial_used_mask[design.row_trace_index_sorted] = True
    rejected_mask = np.ascontiguousarray(initial_used_mask & ~used_mask, dtype=bool)

    node_id, node_t1, node_status = _assemble_node_solution(
        design,
        parameter_vector=parameter_vector,
        active_mask=active_mask,
        system=final_system,
    )
    bedrock_slowness, bedrock_velocity, bedrock_status = _bedrock_solution(
        design,
        parameter_vector=parameter_vector,
        active_mask=active_mask,
    )
    node_observation_count = _node_observation_count_for_row_mask(
        design,
        row_used_mask=row_used_mask,
    )
    cell_observation_count = _cell_observation_count_for_row_mask(
        design,
        row_used_mask=row_used_mask,
    )
    (
        cell_id,
        cell_slowness,
        cell_velocity,
        cell_status,
        cell_result_observation_count,
        row_midpoint_cell_id,
        row_midpoint_slowness,
        row_midpoint_velocity,
    ) = _cell_solution(
        design,
        parameter_vector=parameter_vector,
        active_mask=active_mask,
        cell_observation_count=cell_observation_count,
    )
    rms_s = _rms(row_residual[row_used_mask])
    qc = _build_solver_qc(
        design=design,
        system=final_system,
        result=raw,
        rms_residual_s=rms_s,
        bedrock_velocity_m_s=bedrock_velocity,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_status=bedrock_status,
        node_solution_status=node_status,
        cell_velocity_status=cell_status,
        robust_result=robust_result,
        node_observation_count=node_observation_count,
        cell_observation_count=cell_observation_count,
    )

    return RefractionStaticSolveResult(
        parameter_vector=parameter_vector,
        node_id=node_id,
        node_half_intercept_time_s=node_t1,
        node_solution_status=node_status,
        node_observation_count=node_observation_count,
        bedrock_velocity_mode=design.bedrock_velocity_mode,
        bedrock_velocity_m_s=bedrock_velocity,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_status=bedrock_status,
        cell_id=cell_id,
        cell_bedrock_slowness_s_per_m=cell_slowness,
        cell_bedrock_velocity_m_s=cell_velocity,
        cell_velocity_status=cell_status,
        cell_observation_count=cell_result_observation_count,
        row_midpoint_cell_id=row_midpoint_cell_id,
        row_midpoint_bedrock_slowness_s_per_m=row_midpoint_slowness,
        row_midpoint_bedrock_velocity_m_s=row_midpoint_velocity,
        row_midpoint_v2_m_s=row_midpoint_velocity,
        cell_v2_m_s=cell_velocity,
        modeled_pick_time_s_sorted=full_modeled,
        residual_s_sorted=full_residual,
        residual_ms_sorted=np.ascontiguousarray(full_residual * 1000.0),
        used_observation_mask_sorted=used_mask,
        rejected_observation_mask_sorted=rejected_mask,
        rejected_iteration_sorted=robust_result.rejected_iteration_sorted,
        row_modeled_pick_time_s=row_modeled,
        row_residual_s=row_residual,
        rms_residual_s=rms_s,
        rms_residual_ms=float(rms_s * 1000.0),
        solver_success=bool(raw.success),
        solver_status=int(raw.status),
        solver_message=str(raw.message),
        solver_cost=float(raw.cost),
        solver_optimality=float(raw.optimality),
        solver_iterations=int(raw.nit),
        solver_active_mask=active_mask,
        robust_enabled=bool(options.robust.enabled),
        robust_stop_reason=robust_result.stop_reason,
        robust_iteration_summaries=robust_result.iteration_summaries,
        n_initial_used_observations=int(design.n_observations),
        n_final_used_observations=int(np.count_nonzero(row_used_mask)),
        n_rejected_observations=int(np.count_nonzero(rejected_mask)),
        design=design,
        system=final_system,
        solver_options=options,
        qc=qc,
    )


def summarize_refraction_static_solve_result(
    result: RefractionStaticSolveResult,
) -> dict[str, Any]:
    """Return JSON-safe solver QC without artifact I/O."""
    return dict(result.qc)


def validate_refraction_static_solver_options(
    options: RefractionStaticSolverOptions | None,
) -> RefractionStaticSolverOptions:
    """Validate and normalize refraction solver options for this implementation."""
    opts = RefractionStaticSolverOptions() if options is None else options
    if not isinstance(opts, RefractionStaticSolverOptions):
        raise RefractionStaticSolverError(
            'solver_options must be a RefractionStaticSolverOptions instance'
        )
    robust = opts.robust
    if not isinstance(robust, RefractionStaticRobustOptions):
        raise RefractionStaticSolverError(
            'solver.robust must be a RefractionStaticRobustOptions instance'
        )
    robust_options = RefractionStaticRobustOptions(
        enabled=bool(robust.enabled),
        method=_validate_robust_method(robust.method),
        threshold=_coerce_positive_finite_float(
            robust.threshold,
            name='solver.robust.threshold',
            error_type=RefractionStaticSolverError,
        ),
        scale_floor_ms=_coerce_nonnegative_finite_float(
            robust.scale_floor_ms,
            name='solver.robust.scale_floor_ms',
            error_type=RefractionStaticSolverError,
        ),
        max_iterations=_coerce_positive_int(
            robust.max_iterations,
            name='solver.robust.max_iterations',
            error_type=RefractionStaticSolverError,
        ),
        min_used_fraction=_coerce_positive_finite_float(
            robust.min_used_fraction,
            name='solver.robust.min_used_fraction',
            error_type=RefractionStaticSolverError,
        ),
        min_used_observations=_coerce_positive_int(
            robust.min_used_observations,
            name='solver.robust.min_used_observations',
            error_type=RefractionStaticSolverError,
        ),
    )
    if robust_options.min_used_fraction > 1.0:
        raise RefractionStaticSolverError(
            'solver.robust.min_used_fraction must be <= 1'
        )
    return RefractionStaticSolverOptions(
        half_intercept_damping_lambda=_coerce_nonnegative_finite_float(
            opts.half_intercept_damping_lambda,
            name='solver.half_intercept_damping_lambda',
            error_type=RefractionStaticSolverError,
        ),
        min_picks_per_node=_coerce_positive_int(
            opts.min_picks_per_node,
            name='solver.min_picks_per_node',
            error_type=RefractionStaticSolverError,
        ),
        max_abs_half_intercept_time_ms=_coerce_positive_finite_float(
            opts.max_abs_half_intercept_time_ms,
            name='solver.max_abs_half_intercept_time_ms',
            error_type=RefractionStaticSolverError,
        ),
        identifiability_rtol=_coerce_positive_finite_float(
            opts.identifiability_rtol,
            name='solver.identifiability_rtol',
            error_type=RefractionStaticSolverError,
        ),
        robust=robust_options,
    )


def _validate_design(design: RefractionStaticDesignMatrix) -> None:
    if not isinstance(design, RefractionStaticDesignMatrix):
        raise RefractionStaticSolverError(
            'design must be a RefractionStaticDesignMatrix instance'
        )
    if design.bedrock_velocity_mode not in {
        'solve_global',
        'fixed_global',
        'solve_cell',
    }:
        raise RefractionStaticSolverError(
            'design.bedrock_velocity_mode must be solve_global, fixed_global, '
            'or solve_cell'
        )
    if design.matrix.shape != (design.n_observations, design.n_parameters):
        raise RefractionStaticSolverError('design.matrix shape mismatch')
    if not sparse.isspmatrix_csr(design.matrix):
        raise RefractionStaticSolverError('design.matrix must be CSR')
    _validate_finite(design.matrix.data, name='design.matrix.data')
    _validate_finite(design.rhs_s, name='design.rhs_s')
    if design.row_trace_index_sorted.shape != (design.n_observations,):
        raise RefractionStaticSolverError('design.row_trace_index_sorted shape mismatch')
    try:
        n_traces = int(design.qc['n_traces'])
    except (KeyError, TypeError, ValueError) as exc:
        raise RefractionStaticSolverError(
            'design.qc["n_traces"] must be an integer'
        ) from exc
    if n_traces <= 0:
        raise RefractionStaticSolverError(
            'design.qc["n_traces"] must be greater than 0'
        )
    invalid_trace_mask = (design.row_trace_index_sorted < 0) | (
        design.row_trace_index_sorted >= n_traces
    )
    if np.any(invalid_trace_mask):
        invalid = int(design.row_trace_index_sorted[np.flatnonzero(invalid_trace_mask)[0]])
        raise RefractionStaticSolverError(
            'design.row_trace_index_sorted values must be in [0, n_traces); '
            f'got {invalid} with n_traces={n_traces}'
        )


def _validate_model_for_design(
    *,
    model: Any,
    design: RefractionStaticDesignMatrix,
) -> None:
    method = getattr(model, 'method', None)
    if method != 'gli_variable_thickness':
        raise RefractionStaticSolverError(
            'model.method must be gli_variable_thickness'
        )
    mode = getattr(model, 'bedrock_velocity_mode', None)
    if mode != design.bedrock_velocity_mode:
        raise RefractionStaticSolverError(
            'model.bedrock_velocity_mode must match design.bedrock_velocity_mode'
        )
    if mode == 'solve_cell':
        if getattr(model, 'bedrock_velocity_m_s', None) is not None:
            raise RefractionStaticSolverError(
                'model.bedrock_velocity_m_s is only allowed when '
                'model.bedrock_velocity_mode is fixed_global'
            )
        if getattr(model, 'refractor_cell', None) is None:
            raise RefractionStaticSolverError(
                'model.refractor_cell is required when '
                'model.bedrock_velocity_mode is solve_cell'
            )
        _validate_cell_design(design)
    if mode == 'fixed_global':
        fixed_velocity = _positive_model_float(
            model,
            'bedrock_velocity_m_s',
            name='model.bedrock_velocity_m_s',
        )
        if design.fixed_bedrock_velocity_m_s is None or not np.isclose(
            fixed_velocity,
            design.fixed_bedrock_velocity_m_s,
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            raise RefractionStaticSolverError(
                'model.bedrock_velocity_m_s must match design fixed velocity'
            )


def _validate_cell_design(design: RefractionStaticDesignMatrix) -> None:
    if design.active_cell_id is None or design.inactive_cell_id is None:
        raise RefractionStaticSolverError('solve_cell design requires cell IDs')
    if design.cell_id_to_col is None:
        raise RefractionStaticSolverError('solve_cell design requires cell columns')
    if design.row_midpoint_cell_id is None or design.row_midpoint_cell_col is None:
        raise RefractionStaticSolverError(
            'solve_cell design requires row midpoint cell columns'
        )
    if design.bedrock_slowness_cell_col_start is None:
        raise RefractionStaticSolverError(
            'solve_cell design requires bedrock slowness cell column start'
        )
    if design.n_total_cells is None or design.number_of_cell_x is None:
        raise RefractionStaticSolverError('solve_cell design requires cell grid shape')


def _model_velocity_bounds(model: Any) -> tuple[float, float, float]:
    min_velocity = _positive_model_float(
        model,
        'min_bedrock_velocity_m_s',
        name='model.min_bedrock_velocity_m_s',
    )
    max_velocity = _positive_model_float(
        model,
        'max_bedrock_velocity_m_s',
        name='model.max_bedrock_velocity_m_s',
    )
    if min_velocity >= max_velocity:
        raise RefractionStaticSolverError(
            'model.min_bedrock_velocity_m_s must be less than '
            'model.max_bedrock_velocity_m_s'
        )
    initial_velocity = _positive_model_float(
        model,
        'initial_bedrock_velocity_m_s',
        name='model.initial_bedrock_velocity_m_s',
    )
    if not (min_velocity <= initial_velocity <= max_velocity):
        raise RefractionStaticSolverError(
            'model.initial_bedrock_velocity_m_s must be within bedrock '
            'velocity bounds'
        )
    return min_velocity, max_velocity, initial_velocity


def _cell_slowness_cols(design: RefractionStaticDesignMatrix) -> np.ndarray:
    _validate_cell_design(design)
    if design.active_cell_id is None or design.cell_id_to_col is None:
        raise RefractionStaticSolverError('solve_cell design requires active cells')
    return np.asarray(
        [design.cell_id_to_col[int(cell_id)] for cell_id in design.active_cell_id],
        dtype=np.int64,
    )


def _build_cell_smoothing_rows_for_design(
    design: RefractionStaticDesignMatrix,
    *,
    model: Any,
) -> CellSlownessSmoothingRows | None:
    if design.bedrock_velocity_mode != 'solve_cell':
        return None
    _validate_cell_design(design)
    refractor_cell = getattr(model, 'refractor_cell', None)
    if refractor_cell is None:
        raise RefractionStaticSolverError(
            'model.refractor_cell is required when model.bedrock_velocity_mode '
            'is solve_cell'
        )
    if (
        design.active_cell_id is None
        or design.cell_id_to_col is None
        or design.n_total_cells is None
        or design.number_of_cell_x is None
    ):
        raise RefractionStaticSolverError('solve_cell design requires cell metadata')
    return build_cell_slowness_smoothing_rows(
        active_cell_id=design.active_cell_id,
        velocity_smoothing_weight=getattr(
            refractor_cell,
            'velocity_smoothing_weight',
            0.0,
        ),
        smoothing_reference_distance_m=getattr(
            refractor_cell,
            'smoothing_reference_distance_m',
            None,
        ),
        row_distance_m=design.row_distance_m,
        n_total_cells=design.n_total_cells,
        number_of_cell_x=design.number_of_cell_x,
        number_of_cell_y=design.number_of_cell_y,
        cell_id_to_col=design.cell_id_to_col,
        n_parameters=design.n_parameters,
    )


def _empty_rows(n_parameters: int) -> sparse.csr_matrix:
    return sparse.csr_matrix((0, n_parameters), dtype=np.float64)


def _gauge_resolution_for_system(
    *,
    half_intercept_damping_lambda: float,
    n_gauge_required_components: int,
) -> str:
    if n_gauge_required_components <= 0:
        return 'not_required'
    if float(half_intercept_damping_lambda) > 0.0:
        return 'node_damping'
    return 'postsolve_minimum_norm'


def _canonicalize_refraction_parameter_vector(
    parameter_vector: np.ndarray,
    *,
    system: RefractionStaticSolveSystem,
    design: RefractionStaticDesignMatrix,
) -> np.ndarray:
    """Resolve node gauge without changing physical fit or slowness parameters.

    For zero damping, each bipartite node component is shifted only along its
    known endpoint-sum null direction. The selected shift is the component
    minimum-L2 solution clipped to the feasible node-bound interval, so
    observation predictions, smoothing residuals, and slowness values are
    invariant to numerical tolerance.
    """
    vector = np.ascontiguousarray(parameter_vector, dtype=np.float64)
    if vector.shape != (system.n_parameters,):
        raise RefractionStaticSolverError('parameter_vector shape mismatch')
    if system.half_intercept_damping_lambda > 0.0 or not np.any(
        system.gauge_required_by_component
    ):
        _validate_parameter_bounds(vector, system=system)
        return vector

    before_observation = _observation_prediction(system, parameter_vector=vector)
    before_smoothing = _smoothing_prediction(system, parameter_vector=vector)
    before_slowness = np.ascontiguousarray(
        vector[design.n_active_nodes :],
        dtype=np.float64,
    )

    out = vector.copy()
    component_id = np.asarray(system.component_id_by_node, dtype=np.int64)
    signed_partition = np.asarray(system.signed_partition_by_node, dtype=np.float64)
    lower = np.asarray(system.lower_bounds[: design.n_active_nodes], dtype=np.float64)
    upper = np.asarray(system.upper_bounds[: design.n_active_nodes], dtype=np.float64)
    for component in np.flatnonzero(system.gauge_required_by_component).tolist():
        nodes = np.flatnonzero(component_id == int(component)).astype(
            np.int64,
            copy=False,
        )
        if nodes.size == 0:
            raise RefractionStaticSolverError('gauge component contains no nodes')
        g = signed_partition[nodes]
        if np.any(g == 0.0):
            raise RefractionStaticSolverError('gauge partition contains zero entries')
        t = out[nodes]
        denominator = float(g @ g)
        if denominator <= 0.0:
            raise RefractionStaticSolverError('gauge partition has zero norm')
        unconstrained = -float(g @ t) / denominator
        c_lower = -np.inf
        c_upper = np.inf
        for value, sign, lo, hi in zip(t, g, lower[nodes], upper[nodes], strict=True):
            if sign > 0.0:
                node_lower = float(lo - value)
                node_upper = float(hi - value)
            else:
                node_lower = float(value - hi)
                node_upper = float(value - lo)
            c_lower = max(c_lower, node_lower)
            c_upper = min(c_upper, node_upper)
        if c_lower > c_upper + 1.0e-10:
            raise RefractionStaticSolverError(
                'gauge canonicalization feasible interval is empty'
            )
        if c_lower > c_upper:
            chosen = 0.5 * (c_lower + c_upper)
        else:
            chosen = min(max(unconstrained, c_lower), c_upper)
        out[nodes] = t + chosen * g

    _validate_parameter_bounds(out, system=system)
    if not np.allclose(
        _observation_prediction(system, parameter_vector=out),
        before_observation,
        rtol=1.0e-9,
        atol=1.0e-10,
    ):
        raise RefractionStaticSolverError(
            'gauge canonicalization changed observation prediction'
        )
    if not np.allclose(
        _smoothing_prediction(system, parameter_vector=out),
        before_smoothing,
        rtol=1.0e-9,
        atol=1.0e-10,
    ):
        raise RefractionStaticSolverError(
            'gauge canonicalization changed smoothing prediction'
        )
    if not np.array_equal(out[design.n_active_nodes :], before_slowness):
        raise RefractionStaticSolverError(
            'gauge canonicalization changed slowness parameters'
        )
    return np.ascontiguousarray(out, dtype=np.float64)


def _observation_prediction(
    system: RefractionStaticSolveSystem,
    *,
    parameter_vector: np.ndarray,
) -> np.ndarray:
    return np.ascontiguousarray(
        system.observation_matrix @ parameter_vector,
        dtype=np.float64,
    )


def _smoothing_prediction(
    system: RefractionStaticSolveSystem,
    *,
    parameter_vector: np.ndarray,
) -> np.ndarray:
    if system.smoothing_rows is None:
        return np.empty(0, dtype=np.float64)
    return np.ascontiguousarray(
        system.smoothing_rows.matrix @ parameter_vector,
        dtype=np.float64,
    )


def _validate_parameter_bounds(
    parameter_vector: np.ndarray,
    *,
    system: RefractionStaticSolveSystem,
) -> None:
    lower_violation = parameter_vector < system.lower_bounds - 1.0e-9
    finite_upper = np.isfinite(system.upper_bounds)
    upper_violation = finite_upper & (
        parameter_vector > system.upper_bounds + 1.0e-9
    )
    if np.any(lower_violation) or np.any(upper_violation):
        raise RefractionStaticSolverError(
            'canonical parameter vector violates solver bounds'
        )


def _validate_canonicalized_lsq_solution(
    raw_parameter_vector: np.ndarray,
    canonical_parameter_vector: np.ndarray,
    *,
    system: RefractionStaticSolveSystem,
) -> None:
    _validate_parameter_bounds(canonical_parameter_vector, system=system)
    _, _, raw_objective = _lsq_residual_gradient_objective(
        system,
        np.ascontiguousarray(raw_parameter_vector, dtype=np.float64),
    )
    _, _, canonical_objective = _lsq_residual_gradient_objective(
        system,
        np.ascontiguousarray(canonical_parameter_vector, dtype=np.float64),
    )
    if not np.isclose(
        canonical_objective,
        raw_objective,
        rtol=1.0e-9,
        atol=max(1.0e-20, 1.0e-12 * max(1.0, raw_objective)),
    ):
        raise RefractionStaticSolverError(
            'gauge canonicalization changed augmented objective'
        )


_DENSE_IDENTIFIABILITY_MAX_ELEMENTS = 1_000_000


def _validate_physical_identifiability(
    matrix: sparse.csr_matrix,
    *,
    mode: str,
    n_parameters: int,
    gauge_nullity: int,
    rtol: float,
) -> _NumericalRankDiagnostic:
    if matrix.shape[1] != n_parameters:
        raise RefractionStaticSolverError('physical identifiability matrix shape mismatch')
    expected_rank = int(n_parameters) - int(gauge_nullity)
    if expected_rank < 0:
        raise RefractionStaticSolverError(
            f'{mode} physical system has invalid gauge nullity: '
            f'n_parameters={n_parameters}, gauge_nullity={gauge_nullity}'
        )
    diagnostic = _column_scaled_numerical_rank(
        matrix,
        expected_rank=expected_rank,
        expected_nullity=int(gauge_nullity),
        rtol=rtol,
    )
    if diagnostic.estimated_rank != diagnostic.expected_rank:
        raise RefractionStaticSolverError(
            f'{mode} physical system is not identifiable: '
            f'expected_rank={diagnostic.expected_rank}, '
            f'actual_rank={diagnostic.estimated_rank}, '
            f'gauge_nullity={diagnostic.gauge_nullity}, '
            f'expected_nullity={diagnostic.expected_nullity}, '
            f'threshold={diagnostic.threshold:.6g}, '
            f'critical_singular_value={diagnostic.critical_singular_value:.6g}, '
            f'certification_status={diagnostic.certification_status}, '
            f'failure_reason={diagnostic.failure_reason}'
        )
    return diagnostic


def _column_scaled_numerical_rank(
    matrix: sparse.spmatrix,
    *,
    expected_rank: int,
    expected_nullity: int,
    rtol: float,
) -> _NumericalRankDiagnostic:
    if expected_rank < 0:
        raise RefractionStaticSolverError('expected_rank must be nonnegative')
    if expected_nullity < 0:
        raise RefractionStaticSolverError('expected_nullity must be nonnegative')
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise RefractionStaticSolverError('solver.identifiability_rtol must be positive finite')
    csr = matrix.tocsr().astype(np.float64, copy=False)
    _validate_finite(csr.data, name='physical_matrix.data')
    n_rows, n_columns = map(int, csr.shape)
    if expected_rank > min(n_rows, n_columns):
        return _NumericalRankDiagnostic(
            method='row_count',
            n_rows=n_rows,
            n_columns=n_columns,
            expected_rank=int(expected_rank),
            estimated_rank=int(min(n_rows, n_columns)),
            expected_nullity=int(expected_nullity),
            gauge_nullity=int(expected_nullity),
            threshold=0.0,
            critical_singular_value=0.0,
            largest_singular_value=0.0,
            rtol=float(rtol),
        )
    scaled = _column_l2_scaled_matrix(csr)
    if n_rows * n_columns <= _DENSE_IDENTIFIABILITY_MAX_ELEMENTS:
        return _dense_column_scaled_numerical_rank(
            scaled,
            expected_rank=expected_rank,
            expected_nullity=expected_nullity,
            rtol=rtol,
        )
    return _sparse_column_scaled_numerical_rank(
        scaled,
        expected_rank=expected_rank,
        expected_nullity=expected_nullity,
        rtol=rtol,
    )


def _column_l2_scaled_matrix(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    column_norm = np.sqrt(
        np.asarray(matrix.power(2).sum(axis=0), dtype=np.float64).ravel()
    )
    _validate_finite(column_norm, name='physical_matrix column norms')
    scale = np.zeros(column_norm.shape, dtype=np.float64)
    nonzero = column_norm > 0.0
    scale[nonzero] = 1.0 / column_norm[nonzero]
    scaled = matrix @ sparse.diags(scale, offsets=0, format='csr')
    scaled.sort_indices()
    return scaled


def _dense_column_scaled_numerical_rank(
    scaled_matrix: sparse.csr_matrix,
    *,
    expected_rank: int,
    expected_nullity: int,
    rtol: float,
) -> _NumericalRankDiagnostic:
    dense = scaled_matrix.toarray()
    singular_values = np.linalg.svd(dense, compute_uv=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = float(rtol) * largest
    estimated_rank = int(np.count_nonzero(singular_values > threshold))
    critical = (
        float(singular_values[expected_rank - 1])
        if expected_rank > 0 and singular_values.size >= expected_rank
        else 0.0
    )
    return _NumericalRankDiagnostic(
        method='dense_svd',
        n_rows=int(scaled_matrix.shape[0]),
        n_columns=int(scaled_matrix.shape[1]),
        expected_rank=int(expected_rank),
        estimated_rank=estimated_rank,
        expected_nullity=int(expected_nullity),
        gauge_nullity=int(expected_nullity),
        threshold=threshold,
        critical_singular_value=critical,
        largest_singular_value=largest,
        rtol=float(rtol),
    )


def _sparse_column_scaled_numerical_rank(
    scaled_matrix: sparse.csr_matrix,
    *,
    expected_rank: int,
    expected_nullity: int,
    rtol: float,
) -> _NumericalRankDiagnostic:
    n_rows, n_columns = map(int, scaled_matrix.shape)
    min_dim = min(n_rows, n_columns)
    requested_smallest_count = 0
    returned_smallest_count = 0
    max_residual = 0.0
    failure_reason = ''
    if min_dim == 0:
        largest = 0.0
        threshold = 0.0
        estimated_rank = 0
        critical = 0.0
        certification_status = 'certified'
    elif min_dim == 1:
        largest = float(np.sqrt(float(scaled_matrix.power(2).sum())))
        _validate_finite(
            np.asarray([largest], dtype=np.float64),
            name='physical_matrix sparse singular values',
        )
        threshold = float(rtol) * largest
        critical = largest if expected_rank > 0 else 0.0
        estimated_rank = int(largest > threshold)
        certification_status = (
            'certified'
            if estimated_rank >= int(expected_rank)
            else 'rank_deficient'
        )
        if certification_status != 'certified':
            failure_reason = 'critical singular value is not above threshold'
    else:
        largest_triplets = _sparse_normal_singular_triplets(
            scaled_matrix,
            k=1,
            which='LA',
            name='physical_matrix largest sparse singular triplet',
        )
        largest = float(largest_triplets.singular_values[-1])
        max_residual = max(max_residual, largest_triplets.max_residual)
        threshold = float(rtol) * largest
        residual_tolerance = _sparse_singular_triplet_residual_tolerance(
            scaled_matrix,
            largest=largest,
        )
        if largest_triplets.max_residual > residual_tolerance:
            raise RefractionStaticSolverError(
                'sparse physical identifiability largest singular triplet '
                'residual is too large'
            )
        if expected_rank == 0:
            critical = 0.0
            estimated_rank = int(largest > threshold)
            certification_status = (
                'certified' if estimated_rank == 0 else 'rank_deficient'
            )
            if certification_status != 'certified':
                failure_reason = 'matrix has singular values above threshold'
        elif expected_rank == 1:
            critical = largest
            if _critical_singular_value_is_ambiguous(
                critical,
                threshold=threshold,
                scaled_matrix=scaled_matrix,
                largest=largest,
            ):
                raise RefractionStaticSolverError(
                    'sparse physical identifiability critical singular value '
                    'is too close to the rank threshold to certify'
                )
            if critical > threshold:
                estimated_rank = int(expected_rank)
                certification_status = 'certified'
            else:
                estimated_rank = 0
                certification_status = 'rank_deficient'
                failure_reason = 'critical singular value is not above threshold'
        else:
            allowed_small_count = min_dim - int(expected_rank)
            if allowed_small_count < 0:
                critical = 0.0
                estimated_rank = min_dim
                certification_status = 'rank_deficient'
                failure_reason = 'expected rank exceeds matrix smaller dimension'
            else:
                requested_smallest_count = allowed_small_count + 1
                boundary_triplets = _sparse_svds_singular_triplets(
                    scaled_matrix,
                    k=requested_smallest_count,
                    name='physical_matrix smallest sparse singular triplets',
                )
                corroborating_triplets = _sparse_normal_singular_triplets(
                    scaled_matrix,
                    k=requested_smallest_count,
                    which='SA',
                    name='physical_matrix corroborating sparse singular values',
                )
                corroborating_values = corroborating_triplets.singular_values
                returned_smallest_count = int(boundary_triplets.singular_values.size)
                if returned_smallest_count < requested_smallest_count:
                    raise RefractionStaticSolverError(
                        'sparse physical identifiability returned too few '
                        'smallest singular triplets'
                    )
                critical = min(
                    float(boundary_triplets.singular_values[allowed_small_count]),
                    float(corroborating_values[allowed_small_count]),
                )
                if _critical_singular_value_is_ambiguous(
                    critical,
                    threshold=threshold,
                    scaled_matrix=scaled_matrix,
                    largest=largest,
                ):
                    raise RefractionStaticSolverError(
                        'sparse physical identifiability critical singular value '
                        'is too close to the rank threshold to certify'
                    )
                if critical > threshold:
                    certification_residual = _sparse_triplet_residual_at(
                        boundary_triplets,
                        allowed_small_count,
                    )
                    if allowed_small_count > 0:
                        certification_residual = max(
                            certification_residual,
                            float(
                                np.max(
                                    _sparse_triplet_residuals(
                                        corroborating_triplets,
                                    )[:allowed_small_count]
                                )
                            ),
                        )
                    max_residual = max(max_residual, certification_residual)
                    if certification_residual > residual_tolerance:
                        raise RefractionStaticSolverError(
                            'sparse physical identifiability smallest singular '
                            'triplet residual is too large'
                        )
                    estimated_rank = int(expected_rank)
                    certification_status = 'certified'
                else:
                    n_small = int(
                        np.count_nonzero(
                            np.minimum(
                                boundary_triplets.singular_values,
                                corroborating_values,
                            )
                            <= threshold
                        )
                    )
                    estimated_rank = max(0, min_dim - n_small)
                    certification_status = 'rank_deficient'
                    max_residual = max(
                        max_residual,
                        boundary_triplets.max_residual,
                        corroborating_triplets.max_residual,
                    )
                    failure_reason = (
                        'critical singular value is not above threshold'
                    )
    return _NumericalRankDiagnostic(
        method='sparse_normal_eigsh',
        n_rows=n_rows,
        n_columns=n_columns,
        expected_rank=int(expected_rank),
        estimated_rank=int(estimated_rank),
        expected_nullity=int(expected_nullity),
        gauge_nullity=int(expected_nullity),
        threshold=float(threshold),
        critical_singular_value=float(critical),
        largest_singular_value=float(largest),
        rtol=float(rtol),
        sparse_solver_name='propack_svds+eigsh_normal',
        certification_status=certification_status,
        requested_smallest_count=int(requested_smallest_count),
        returned_smallest_count=int(returned_smallest_count),
        max_singular_triplet_residual=float(max_residual),
        failure_reason=failure_reason,
    )


@dataclass(frozen=True)
class _SparseSingularTripletDiagnostic:
    singular_values: np.ndarray
    max_residual: float
    residuals: np.ndarray | None = None


def _sparse_triplet_residuals(
    diagnostic: _SparseSingularTripletDiagnostic,
) -> np.ndarray:
    if diagnostic.residuals is None:
        return np.full(
            diagnostic.singular_values.shape,
            float(diagnostic.max_residual),
            dtype=np.float64,
        )
    return np.asarray(diagnostic.residuals, dtype=np.float64)


def _sparse_triplet_residual_at(
    diagnostic: _SparseSingularTripletDiagnostic,
    index: int,
) -> float:
    residuals = _sparse_triplet_residuals(diagnostic)
    return float(residuals[int(index)])


def _sparse_normal_singular_triplets(
    scaled_matrix: sparse.csr_matrix,
    *,
    k: int,
    which: Literal['SA', 'LA'],
    name: str,
) -> _SparseSingularTripletDiagnostic:
    n_rows, n_columns = map(int, scaled_matrix.shape)
    min_dim = min(n_rows, n_columns)
    if k <= 0:
        return _SparseSingularTripletDiagnostic(
            singular_values=np.empty(0, dtype=np.float64),
            max_residual=0.0,
        )
    if k >= min_dim:
        raise RefractionStaticSolverError(
            'sparse physical identifiability requires too many singular '
            'triplets for sparse certification'
        )
    use_right_operator = n_columns <= n_rows
    operator_shape = n_columns if use_right_operator else n_rows

    def matvec(vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=np.float64)
        if use_right_operator:
            return scaled_matrix.T @ (scaled_matrix @ x)
        return scaled_matrix @ (scaled_matrix.T @ x)

    normal_operator = sparse_linalg.LinearOperator(
        (operator_shape, operator_shape),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )
    v0 = np.linspace(1.0, 2.0, operator_shape, dtype=np.float64)
    try:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(
            normal_operator,
            k=int(k),
            which=which,
            return_eigenvectors=True,
            tol=0.0,
            v0=v0,
        )
    except Exception as exc:
        raise RefractionStaticSolverError(
            'sparse physical identifiability singular triplets did not converge'
        ) from exc
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)
    _validate_finite(eigenvalues, name=f'{name} eigenvalues')
    _validate_finite(eigenvectors, name=f'{name} eigenvectors')
    if eigenvalues.size != int(k) or eigenvectors.shape != (operator_shape, int(k)):
        raise RefractionStaticSolverError(
            'sparse physical identifiability returned incomplete singular triplets'
        )
    order = np.argsort(eigenvalues)
    if which == 'LA':
        order = order[-int(k) :]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    singular_values = np.asarray(
        [
            _sparse_singular_value_from_normal_vector(
                scaled_matrix,
                vector=eigenvectors[:, index],
                use_right_operator=use_right_operator,
            )
            for index in range(int(k))
        ],
        dtype=np.float64,
    )
    residuals = [
        _sparse_singular_triplet_residual(
            scaled_matrix,
            singular_value=float(singular_values[index]),
            vector=eigenvectors[:, index],
            use_right_operator=use_right_operator,
        )
        for index in range(int(k))
    ]
    residual_array = np.asarray(residuals, dtype=np.float64)
    sort_order = np.argsort(singular_values)
    out = np.asarray(singular_values[sort_order], dtype=np.float64)
    residual_array = np.asarray(residual_array[sort_order], dtype=np.float64)
    _validate_finite(out, name=name)
    _validate_finite(residual_array, name=f'{name} residuals')
    return _SparseSingularTripletDiagnostic(
        singular_values=out,
        max_residual=float(np.max(residual_array)) if residual_array.size else 0.0,
        residuals=residual_array,
    )


def _sparse_svds_singular_triplets(
    scaled_matrix: sparse.csr_matrix,
    *,
    k: int,
    name: str,
) -> _SparseSingularTripletDiagnostic:
    try:
        left_vectors, values, right_vectors_t = sparse_linalg.svds(
            scaled_matrix,
            k=int(k),
            which='SM',
            return_singular_vectors=True,
            solver='propack',
            tol=0.0,
        )
    except Exception as exc:
        raise RefractionStaticSolverError(
            'sparse physical identifiability singular triplets did not converge'
        ) from exc
    values = np.asarray(values, dtype=np.float64)
    left_vectors = np.asarray(left_vectors, dtype=np.float64)
    right_vectors_t = np.asarray(right_vectors_t, dtype=np.float64)
    _validate_finite(values, name=name)
    _validate_finite(left_vectors, name=f'{name} left vectors')
    _validate_finite(right_vectors_t, name=f'{name} right vectors')
    if (
        values.size != int(k)
        or left_vectors.shape != (int(scaled_matrix.shape[0]), int(k))
        or right_vectors_t.shape != (int(k), int(scaled_matrix.shape[1]))
    ):
        raise RefractionStaticSolverError(
            'sparse physical identifiability returned incomplete singular triplets'
        )
    order = np.argsort(values)
    values = values[order]
    left_vectors = left_vectors[:, order]
    right_vectors_t = right_vectors_t[order, :]
    residuals = [
        max(
            np.linalg.norm(
                scaled_matrix @ right_vectors_t[index, :]
                - values[index] * left_vectors[:, index]
            ),
            np.linalg.norm(
                scaled_matrix.T @ left_vectors[:, index]
                - values[index] * right_vectors_t[index, :]
            ),
        )
        for index in range(int(k))
    ]
    residual_array = np.asarray(residuals, dtype=np.float64)
    _validate_finite(residual_array, name=f'{name} residuals')
    return _SparseSingularTripletDiagnostic(
        singular_values=values,
        max_residual=float(np.max(residual_array)) if residual_array.size else 0.0,
        residuals=residual_array,
    )


def _sparse_singular_value_from_normal_vector(
    scaled_matrix: sparse.csr_matrix,
    *,
    vector: np.ndarray,
    use_right_operator: bool,
) -> float:
    if use_right_operator:
        return float(np.linalg.norm(scaled_matrix @ vector))
    return float(np.linalg.norm(scaled_matrix.T @ vector))


def _sparse_singular_triplet_residual(
    scaled_matrix: sparse.csr_matrix,
    *,
    singular_value: float,
    vector: np.ndarray,
    use_right_operator: bool,
) -> float:
    sigma = float(singular_value)
    eps_scale = np.finfo(np.float64).eps * max(
        1.0,
        float(np.sqrt(float(scaled_matrix.power(2).sum()))),
    )
    if use_right_operator:
        v = np.asarray(vector, dtype=np.float64)
        mv = scaled_matrix @ v
        if sigma <= eps_scale:
            return float(np.linalg.norm(mv))
        u = mv / sigma
        return float(
            max(
                np.linalg.norm(mv - sigma * u),
                np.linalg.norm(scaled_matrix.T @ u - sigma * v),
            )
        )
    u = np.asarray(vector, dtype=np.float64)
    mtu = scaled_matrix.T @ u
    if sigma <= eps_scale:
        return float(np.linalg.norm(mtu))
    v = mtu / sigma
    return float(
        max(
            np.linalg.norm(scaled_matrix @ v - sigma * u),
            np.linalg.norm(mtu - sigma * v),
        )
    )


def _sparse_singular_triplet_residual_tolerance(
    scaled_matrix: sparse.csr_matrix,
    *,
    largest: float,
) -> float:
    dimension = max(1, max(map(int, scaled_matrix.shape)))
    return float(1.0e3 * np.finfo(np.float64).eps * max(1.0, largest) * dimension)


def _critical_singular_value_is_ambiguous(
    critical: float,
    *,
    threshold: float,
    scaled_matrix: sparse.csr_matrix,
    largest: float,
) -> bool:
    dimension = max(1, max(map(int, scaled_matrix.shape)))
    margin = float(
        1.0e2 * np.finfo(np.float64).eps * max(1.0, largest) * dimension
    )
    return abs(float(critical) - float(threshold)) <= margin


def _build_damping_system(
    *,
    n_parameters: int,
    n_active_nodes: int,
    half_intercept_damping_lambda: float,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    if half_intercept_damping_lambda == 0.0:
        return (
            sparse.csr_matrix((0, n_parameters), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    damping_weight = float(np.sqrt(half_intercept_damping_lambda))
    row_index = np.arange(n_active_nodes, dtype=np.int64)
    matrix = sparse.csr_matrix(
        (
            np.full(n_active_nodes, damping_weight, dtype=np.float64),
            (row_index, row_index),
        ),
        shape=(n_active_nodes, n_parameters),
        dtype=np.float64,
    )
    rhs = np.zeros(n_active_nodes, dtype=np.float64)
    return matrix, rhs


def _analyze_design_endpoint_graph(
    design: RefractionStaticDesignMatrix,
    *,
    row_used_mask: np.ndarray | None = None,
) -> EndpointSumGraphSummary:
    row_index = (
        np.arange(design.n_observations, dtype=np.int64)
        if row_used_mask is None
        else np.flatnonzero(
            _coerce_design_row_mask(design, row_used_mask=row_used_mask)
        )
    )
    return analyze_endpoint_sum_graph(
        n_nodes=int(design.n_active_nodes),
        row_source_node_id=design.source_node_col[row_index],
        row_receiver_node_id=design.receiver_node_col[row_index],
    )


def _run_refraction_static_solver(
    *,
    design: RefractionStaticDesignMatrix,
    system: RefractionStaticSolveSystem,
    solver_options: RefractionStaticSolverOptions,
) -> _RefractionStaticRobustRunResult:
    robust = solver_options.robust
    n_traces = int(design.qc['n_traces'])
    rejected_iteration_sorted = np.full(n_traces, -1, dtype=np.int64)
    initial_row_mask = np.ones(design.n_observations, dtype=bool)
    if not robust.enabled:
        raw = _run_lsq_linear(system)
        return _RefractionStaticRobustRunResult(
            raw_result=raw,
            system=system,
            row_used_mask=initial_row_mask,
            rejected_iteration_sorted=rejected_iteration_sorted,
            iteration_summaries=(),
            stop_reason='disabled',
        )

    current_row_mask = initial_row_mask.copy()
    final_raw: optimize.OptimizeResult | None = None
    final_system: RefractionStaticSolveSystem | None = None
    stop_reason: RefractionStaticRobustStopReason | None = None
    summaries: list[RefractionStaticRobustIterationSummary] = []

    for iteration_index in range(robust.max_iterations):
        current_system = _rebuild_refraction_static_solver_system_for_row_mask(
            design=design,
            system=system,
            row_used_mask=current_row_mask,
        )
        raw = _run_lsq_linear(current_system)
        parameter_vector = np.ascontiguousarray(raw.x, dtype=np.float64)
        row_residual = np.ascontiguousarray(
            design.observed_pick_time_s
            - _row_modeled_pick_time(design, parameter_vector=parameter_vector),
            dtype=np.float64,
        )
        _validate_finite(row_residual, name='row_residual_s')
        current_residual = row_residual[current_row_mask]
        center_s, raw_scale_s = _robust_center_scale(
            current_residual,
            method=robust.method,
        )
        scale_floor_s = float(robust.scale_floor_ms) / 1000.0
        scale_s = max(raw_scale_s, scale_floor_s)
        cutoff_s = float(robust.threshold) * scale_s
        max_abs_centered_residual_s = _max_abs_centered_residual(
            current_residual,
            center_s=center_s,
        )
        n_used_before = int(np.count_nonzero(current_row_mask))

        if scale_s <= 0.0:
            stop_reason = 'zero_scale'
            final_raw = raw
            final_system = current_system
            summaries.append(
                RefractionStaticRobustIterationSummary(
                    iteration_index=iteration_index,
                    method=robust.method,
                    n_used_before=n_used_before,
                    n_rejected_this_iteration=0,
                    n_used_after=n_used_before,
                    residual_center_s=center_s,
                    residual_scale_s=raw_scale_s,
                    residual_scale_floor_s=scale_floor_s,
                    residual_cutoff_s=cutoff_s,
                    max_abs_centered_residual_s=max_abs_centered_residual_s,
                    converged=False,
                    stop_reason=stop_reason,
                )
            )
            break

        current_row_index = np.flatnonzero(current_row_mask)
        centered_abs = np.abs(current_residual - center_s)
        outlier_local = centered_abs > cutoff_s
        if not np.any(outlier_local):
            stop_reason = 'converged'
            final_raw = raw
            final_system = current_system
            summaries.append(
                RefractionStaticRobustIterationSummary(
                    iteration_index=iteration_index,
                    method=robust.method,
                    n_used_before=n_used_before,
                    n_rejected_this_iteration=0,
                    n_used_after=n_used_before,
                    residual_center_s=center_s,
                    residual_scale_s=raw_scale_s,
                    residual_scale_floor_s=scale_floor_s,
                    residual_cutoff_s=cutoff_s,
                    max_abs_centered_residual_s=max_abs_centered_residual_s,
                    converged=True,
                    stop_reason=stop_reason,
                )
            )
            break

        candidate_rows = current_row_index[outlier_local]
        candidate_score = np.abs(row_residual[candidate_rows] - center_s)
        candidate_rows = candidate_rows[np.argsort(-candidate_score, kind='stable')]
        rejected_rows = _safe_rejection_rows(
            design=design,
            system=system,
            initial_row_mask=initial_row_mask,
            current_row_mask=current_row_mask,
            candidate_rows=candidate_rows,
            min_used_fraction=robust.min_used_fraction,
            min_used_observations=robust.min_used_observations,
        )
        if rejected_rows.size == 0:
            stop_reason = 'safe_rejection'
            final_raw = raw
            final_system = current_system
            summaries.append(
                RefractionStaticRobustIterationSummary(
                    iteration_index=iteration_index,
                    method=robust.method,
                    n_used_before=n_used_before,
                    n_rejected_this_iteration=0,
                    n_used_after=n_used_before,
                    residual_center_s=center_s,
                    residual_scale_s=raw_scale_s,
                    residual_scale_floor_s=scale_floor_s,
                    residual_cutoff_s=cutoff_s,
                    max_abs_centered_residual_s=max_abs_centered_residual_s,
                    converged=False,
                    stop_reason=stop_reason,
                )
            )
            break

        proposed_row_mask = current_row_mask.copy()
        proposed_row_mask[rejected_rows] = False
        rejected_trace_index = design.row_trace_index_sorted[rejected_rows]
        rejected_iteration_sorted[rejected_trace_index] = iteration_index
        summary_stop_reason: RefractionStaticRobustStopReason | None = None
        if iteration_index == robust.max_iterations - 1:
            summary_stop_reason = 'max_iterations'
        summaries.append(
            RefractionStaticRobustIterationSummary(
                iteration_index=iteration_index,
                method=robust.method,
                n_used_before=n_used_before,
                n_rejected_this_iteration=int(rejected_rows.shape[0]),
                n_used_after=int(np.count_nonzero(proposed_row_mask)),
                residual_center_s=center_s,
                residual_scale_s=raw_scale_s,
                residual_scale_floor_s=scale_floor_s,
                residual_cutoff_s=cutoff_s,
                max_abs_centered_residual_s=max_abs_centered_residual_s,
                converged=False,
                stop_reason=summary_stop_reason,
            )
        )
        current_row_mask = proposed_row_mask
    else:
        stop_reason = 'max_iterations'
        final_system = _rebuild_refraction_static_solver_system_for_row_mask(
            design=design,
            system=system,
            row_used_mask=current_row_mask,
        )
        final_raw = _run_lsq_linear(final_system)

    if final_raw is None or final_system is None or stop_reason is None:
        raise RefractionStaticSolverError('robust refraction solver produced no result')

    if int(np.count_nonzero(current_row_mask)) < design.n_observations:
        final_system = _rebuild_refraction_static_solver_system_for_row_mask(
            design=design,
            system=system,
            row_used_mask=current_row_mask,
        )
        final_raw = _run_lsq_linear(final_system)

    return _RefractionStaticRobustRunResult(
        raw_result=final_raw,
        system=final_system,
        row_used_mask=np.ascontiguousarray(current_row_mask, dtype=bool),
        rejected_iteration_sorted=np.ascontiguousarray(
            rejected_iteration_sorted,
            dtype=np.int64,
        ),
        iteration_summaries=tuple(summaries),
        stop_reason=stop_reason,
    )

def _safe_rejection_rows(
    *,
    design: RefractionStaticDesignMatrix,
    system: RefractionStaticSolveSystem,
    initial_row_mask: np.ndarray,
    current_row_mask: np.ndarray,
    candidate_rows: np.ndarray,
    min_used_fraction: float,
    min_used_observations: int,
) -> np.ndarray:
    proposed = np.ascontiguousarray(current_row_mask.copy(), dtype=bool)
    accepted: list[int] = []
    for raw_row in candidate_rows.tolist():
        row = int(raw_row)
        trial = proposed.copy()
        trial[row] = False
        if not _robust_row_mask_is_safe(
            design=design,
            system=system,
            initial_row_mask=initial_row_mask,
            row_used_mask=trial,
            min_used_fraction=min_used_fraction,
            min_used_observations=min_used_observations,
        ):
            continue
        proposed = trial
        accepted.append(row)
    return np.ascontiguousarray(accepted, dtype=np.int64)


def _robust_row_mask_is_safe(
    *,
    design: RefractionStaticDesignMatrix,
    system: RefractionStaticSolveSystem,
    initial_row_mask: np.ndarray,
    row_used_mask: np.ndarray,
    min_used_fraction: float,
    min_used_observations: int,
) -> bool:
    n_used = int(np.count_nonzero(row_used_mask))
    min_fraction_count = int(
        np.ceil(float(np.count_nonzero(initial_row_mask)) * min_used_fraction)
    )
    if n_used < max(min_fraction_count, int(min_used_observations)):
        return False
    if not _node_coverage_is_safe(design, row_used_mask=row_used_mask):
        return False
    if design.bedrock_velocity_mode == 'solve_cell':
        if not _cell_coverage_is_safe(
            design,
            row_used_mask=row_used_mask,
        ):
            return False
    try:
        candidate_system = _rebuild_refraction_static_solver_system_for_row_mask(
            design=design,
            system=system,
            row_used_mask=row_used_mask,
        )
    except RefractionStaticSolverError:
        return False
    return bool(
        candidate_system.identifiability.estimated_rank
        == candidate_system.identifiability.expected_rank
    )


def _node_coverage_is_safe(
    design: RefractionStaticDesignMatrix,
    *,
    row_used_mask: np.ndarray,
) -> bool:
    row_indices = np.flatnonzero(row_used_mask)
    if row_indices.size == 0:
        return False
    node_pos = np.concatenate(
        (
            design.source_node_col[row_indices],
            design.receiver_node_col[row_indices],
        )
    )
    count_row_index = np.concatenate((row_indices, row_indices))
    unique_pair_key = np.unique(node_pos * int(design.n_observations) + count_row_index)
    unique_node_pos = unique_pair_key // int(design.n_observations)
    counts = np.bincount(
        unique_node_pos,
        minlength=int(design.n_active_nodes),
    ).astype(np.int64, copy=False)
    return bool(np.all(counts >= int(design.min_observations_per_node)))


def _cell_coverage_is_safe(
    design: RefractionStaticDesignMatrix,
    *,
    row_used_mask: np.ndarray,
) -> bool:
    _validate_cell_design(design)
    if design.active_cell_id is None or design.row_midpoint_cell_id is None:
        raise RefractionStaticSolverError('solve_cell design requires cell metadata')
    if 'min_observations_per_cell' not in design.qc:
        raise RefractionStaticSolverError(
            'solve_cell design requires min_observations_per_cell QC'
        )
    min_observations = int(design.qc['min_observations_per_cell'])
    if design.n_total_cells is None:
        raise RefractionStaticSolverError('solve_cell design requires n_total_cells')
    used_cell_id = design.row_midpoint_cell_id[row_used_mask]
    counts = np.bincount(
        used_cell_id,
        minlength=int(design.n_total_cells),
    ).astype(np.int64, copy=False)
    return bool(np.all(counts[design.active_cell_id] >= min_observations))


def _source_receiver_graph_component_count(
    design: RefractionStaticDesignMatrix,
    *,
    row_used_mask: np.ndarray,
) -> int:
    mask = _coerce_design_row_mask(design, row_used_mask=row_used_mask)
    n_nodes = int(design.n_active_nodes)
    if n_nodes <= 0:
        return 0

    row_indices = np.flatnonzero(mask)
    graph = analyze_endpoint_sum_graph(
        n_nodes=n_nodes,
        row_source_node_id=design.source_node_col[row_indices],
        row_receiver_node_id=design.receiver_node_col[row_indices],
    )
    return int(graph.n_components)


def _robust_center_scale(
    residual_s: np.ndarray,
    *,
    method: str,
) -> tuple[float, float]:
    if residual_s.size == 0:
        raise RefractionStaticSolverError('robust residual_s must be non-empty')
    _validate_finite(residual_s, name='robust residual_s')
    robust_method = _validate_robust_method(method)
    if robust_method == 'mad':
        center = float(np.median(residual_s))
        raw_mad = float(np.median(np.abs(residual_s - center)))
        scale = 1.4826 * raw_mad
    else:
        center = float(np.mean(residual_s))
        scale = float(np.std(residual_s, ddof=0))
    return center, scale


def _max_abs_centered_residual(
    residual_s: np.ndarray,
    *,
    center_s: float,
) -> float:
    if residual_s.size == 0:
        return 0.0
    return float(np.max(np.abs(residual_s - center_s)))


def _validate_robust_method(value: object) -> str:
    if value == 'mad':
        return 'mad'
    if value == 'sigma':
        return 'sigma'
    raise RefractionStaticSolverError('solver.robust.method must be mad or sigma')


_LSQ_SOLVE_SCALE_MIN = 1.0e-150
_LSQ_SOLVE_SCALE_MAX = 1.0e150
_LSQ_FIRST_TOL = 1.0e-12
_LSQ_FIRST_LSMR_TOL = 1.0e-12
_LSQ_RETRY_TOL = 1.0e-13
_LSQ_RETRY_LSMR_TOL = 1.0e-14
_LSQ_DENSE_RETRY_MAX_ELEMENTS = 200_000
_LSQ_ACTIVE_SET_MAX_ITERATIONS = 20


def _run_lsq_linear(system: RefractionStaticSolveSystem) -> optimize.OptimizeResult:
    scale = _common_lsq_solve_scale(
        system.augmented_matrix,
        system.augmented_rhs_s,
    )
    attempts: list[tuple[str, optimize.OptimizeResult, _LsqLinearQualityDiagnostic]] = []
    first = _run_scaled_lsq_linear_attempt(
        system,
        solve_scale=scale,
        stage='first_attempt',
        tol=_LSQ_FIRST_TOL,
        lsmr_tol=_LSQ_FIRST_LSMR_TOL,
        max_iter=max(100, 20 * int(system.n_parameters)),
        dense=False,
    )
    first_x, first_quality = _verify_lsq_linear_solution(
        system,
        first.x,
        solve_scale=scale,
        stage='first_attempt',
        scipy_result=first,
    )
    first.x = first_x
    first.success = bool(first_quality.verified)
    first.cost = first_quality.unscaled_objective
    first.optimality = first_quality.projected_gradient_inf_norm
    first.refraction_solver_quality = _lsq_quality_json(first_quality)
    attempts.append(('first_attempt', first, first_quality))
    if first_quality.verified:
        polished = _active_set_polish_lsq_solution(
            system,
            first.x,
            solve_scale=scale,
            scipy_result=first,
        )
        if polished is not None:
            polished_quality = polished.refraction_solver_quality_diagnostic
            if (
                polished_quality.verified
                and polished_quality.unscaled_objective
                <= first_quality.unscaled_objective
                + max(1.0e-24, 1.0e-12 * max(1.0, first_quality.unscaled_objective))
            ):
                return polished
        return first

    retry = _run_scaled_lsq_linear_attempt(
        system,
        solve_scale=scale,
        stage='retry_strict_lsmr',
        tol=_LSQ_RETRY_TOL,
        lsmr_tol=_LSQ_RETRY_LSMR_TOL,
        max_iter=max(1000, 100 * int(system.n_parameters)),
        dense=False,
    )
    retry_x, retry_quality = _verify_lsq_linear_solution(
        system,
        retry.x,
        solve_scale=scale,
        stage='retry_strict_lsmr',
        scipy_result=retry,
    )
    retry.x = retry_x
    retry.success = bool(retry_quality.verified)
    retry.cost = retry_quality.unscaled_objective
    retry.optimality = retry_quality.projected_gradient_inf_norm
    retry.refraction_solver_quality = _lsq_quality_json(retry_quality)
    attempts.append(('retry_strict_lsmr', retry, retry_quality))
    if retry_quality.verified:
        return retry

    polished = _active_set_polish_lsq_solution(
        system,
        retry.x,
        solve_scale=scale,
        scipy_result=retry,
    )
    if polished is not None:
        polished_quality = polished.refraction_solver_quality_diagnostic
        attempts.append(('active_set_polish', polished, polished_quality))
        if polished_quality.verified:
            return polished

    if _can_use_dense_lsq_retry(system.augmented_matrix):
        dense = _run_scaled_lsq_linear_attempt(
            system,
            solve_scale=scale,
            stage='dense_bvls_retry',
            tol=_LSQ_RETRY_TOL,
            lsmr_tol=_LSQ_RETRY_LSMR_TOL,
            max_iter=max(1000, 100 * int(system.n_parameters)),
            dense=True,
        )
        dense_x, dense_quality = _verify_lsq_linear_solution(
            system,
            dense.x,
            solve_scale=scale,
            stage='dense_bvls_retry',
            scipy_result=dense,
        )
        dense.x = dense_x
        dense.success = bool(dense_quality.verified)
        dense.cost = dense_quality.unscaled_objective
        dense.optimality = dense_quality.projected_gradient_inf_norm
        dense.refraction_solver_quality = _lsq_quality_json(dense_quality)
        attempts.append(('dense_bvls_retry', dense, dense_quality))
        if dense_quality.verified:
            return dense

    best_stage, best_result, best_quality = min(
        attempts,
        key=lambda item: (
            item[2].projected_gradient_inf_norm,
            item[2].unscaled_objective,
        ),
    )
    best_result.refraction_solver_quality = _lsq_quality_json(best_quality)
    raise RefractionStaticSolverError(
        'refraction static lsq_linear solve failed quality verification '
        f'at {best_stage}: {best_quality.failure_reason}'
    )


def _run_scaled_lsq_linear_attempt(
    system: RefractionStaticSolveSystem,
    *,
    solve_scale: float,
    stage: str,
    tol: float,
    lsmr_tol: float,
    max_iter: int,
    dense: bool,
) -> optimize.OptimizeResult:
    matrix = system.augmented_matrix * float(solve_scale)
    rhs = np.ascontiguousarray(system.augmented_rhs_s * float(solve_scale))
    solver_matrix: sparse.csr_matrix | np.ndarray
    method = 'trf'
    lsq_solver: str | None = 'lsmr'
    if dense:
        solver_matrix = matrix.toarray()
        method = 'bvls'
        lsq_solver = None
    else:
        solver_matrix = matrix
    try:
        result = optimize.lsq_linear(
            solver_matrix,
            rhs,
            bounds=(system.lower_bounds, system.upper_bounds),
            method=method,
            tol=float(tol),
            lsq_solver=lsq_solver,
            lsmr_tol=None if dense else float(lsmr_tol),
            max_iter=int(max_iter),
            verbose=0,
        )
    except Exception as exc:
        raise RefractionStaticSolverError(
            f'refraction static lsq_linear solve failed during {stage}'
        ) from exc
    if result.x.shape != (system.n_parameters,):
        raise RefractionStaticSolverError('solver x shape mismatch')
    result.refraction_solver_stage = stage
    result.refraction_solver_scale = float(solve_scale)
    return result


def _common_lsq_solve_scale(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
) -> float:
    matrix_abs_max = 0.0
    if matrix.nnz:
        matrix_abs_max = float(np.max(np.abs(matrix.data)))
    rhs_abs_max = 0.0 if rhs.size == 0 else float(np.max(np.abs(rhs)))
    if rhs_abs_max > 0.0:
        reference = rhs_abs_max
    else:
        reference = max(matrix_abs_max, 1.0)
    if not np.isfinite(reference) or reference <= 0.0:
        reference = 1.0
    scale = 1.0 / reference
    return float(np.clip(scale, _LSQ_SOLVE_SCALE_MIN, _LSQ_SOLVE_SCALE_MAX))


def _can_use_dense_lsq_retry(matrix: sparse.csr_matrix) -> bool:
    return int(matrix.shape[0]) * int(matrix.shape[1]) <= _LSQ_DENSE_RETRY_MAX_ELEMENTS


def _verify_lsq_linear_solution(
    system: RefractionStaticSolveSystem,
    parameter_vector: np.ndarray,
    *,
    solve_scale: float,
    stage: str,
    scipy_result: optimize.OptimizeResult | None = None,
) -> tuple[np.ndarray, _LsqLinearQualityDiagnostic]:
    x = np.ascontiguousarray(parameter_vector, dtype=np.float64)
    scipy_success = bool(getattr(scipy_result, 'success', False))
    scipy_status = int(getattr(scipy_result, 'status', 0))
    scipy_optimality = float(getattr(scipy_result, 'optimality', np.nan))
    scipy_iterations = int(getattr(scipy_result, 'nit', -1))
    base_kwargs = {
        'stage': stage,
        'solve_scale': float(solve_scale),
        'scipy_success': scipy_success,
        'scipy_status': scipy_status,
        'scipy_optimality': scipy_optimality,
        'scipy_iterations': scipy_iterations,
    }
    if x.shape != (system.n_parameters,):
        return x, _failed_lsq_quality(
            **base_kwargs,
            failure_reason='solver x shape mismatch',
        )
    if not np.all(np.isfinite(x)):
        return x, _failed_lsq_quality(
            **base_kwargs,
            failure_reason='solver x contains non-finite values',
        )

    bound_tolerance = _lsq_bound_tolerance(system, x)
    lower_violation = np.maximum(system.lower_bounds - x, 0.0)
    finite_upper = np.isfinite(system.upper_bounds)
    upper_violation = np.zeros_like(x)
    upper_violation[finite_upper] = np.maximum(
        x[finite_upper] - system.upper_bounds[finite_upper],
        0.0,
    )
    max_bound_violation = float(
        max(
            float(np.max(lower_violation)) if lower_violation.size else 0.0,
            float(np.max(upper_violation)) if upper_violation.size else 0.0,
        )
    )
    if max_bound_violation > bound_tolerance:
        residual, gradient, objective = _lsq_residual_gradient_objective(system, x)
        pg_norm, kkt_tolerance = _projected_gradient_quality(
            system,
            x,
            residual,
            gradient,
            bound_tolerance=bound_tolerance,
        )
        return x, _LsqLinearQualityDiagnostic(
            verified=False,
            failure_reason='solver x violates bounds',
            unscaled_augmented_residual_norm=float(np.linalg.norm(residual)),
            unscaled_objective=objective,
            projected_gradient_inf_norm=pg_norm,
            kkt_tolerance=kkt_tolerance,
            max_bound_violation=max_bound_violation,
            bound_tolerance=bound_tolerance,
            **base_kwargs,
        )

    if max_bound_violation > 0.0:
        x = np.maximum(x, system.lower_bounds)
        x[finite_upper] = np.minimum(x[finite_upper], system.upper_bounds[finite_upper])
    residual, gradient, objective = _lsq_residual_gradient_objective(system, x)
    residual_norm = float(np.linalg.norm(residual))
    pg_norm, kkt_tolerance = _projected_gradient_quality(
        system,
        x,
        residual,
        gradient,
        bound_tolerance=bound_tolerance,
    )
    verified = bool(pg_norm <= kkt_tolerance)
    reason = '' if verified else 'projected gradient violates KKT tolerance'
    return x, _LsqLinearQualityDiagnostic(
        verified=verified,
        failure_reason=reason,
        unscaled_augmented_residual_norm=residual_norm,
        unscaled_objective=objective,
        projected_gradient_inf_norm=pg_norm,
        kkt_tolerance=kkt_tolerance,
        max_bound_violation=max_bound_violation,
        bound_tolerance=bound_tolerance,
        **base_kwargs,
    )


def _failed_lsq_quality(
    *,
    stage: str,
    solve_scale: float,
    scipy_success: bool,
    scipy_status: int,
    scipy_optimality: float,
    scipy_iterations: int,
    failure_reason: str,
) -> _LsqLinearQualityDiagnostic:
    return _LsqLinearQualityDiagnostic(
        verified=False,
        stage=stage,
        failure_reason=failure_reason,
        solve_scale=float(solve_scale),
        scipy_success=bool(scipy_success),
        scipy_status=int(scipy_status),
        scipy_optimality=float(scipy_optimality),
        scipy_iterations=int(scipy_iterations),
        unscaled_augmented_residual_norm=np.inf,
        unscaled_objective=np.inf,
        projected_gradient_inf_norm=np.inf,
        kkt_tolerance=0.0,
        max_bound_violation=np.inf,
        bound_tolerance=0.0,
    )


def _lsq_residual_gradient_objective(
    system: RefractionStaticSolveSystem,
    parameter_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    residual = np.ascontiguousarray(
        system.augmented_matrix @ parameter_vector - system.augmented_rhs_s,
        dtype=np.float64,
    )
    gradient = np.ascontiguousarray(system.augmented_matrix.T @ residual, dtype=np.float64)
    objective = float(0.5 * np.dot(residual, residual))
    return residual, gradient, objective


def _lsq_bound_tolerance(
    system: RefractionStaticSolveSystem,
    parameter_vector: np.ndarray,
) -> float:
    finite_upper = system.upper_bounds[np.isfinite(system.upper_bounds)]
    bound_scale = 1.0
    if finite_upper.size:
        bound_scale = max(bound_scale, float(np.max(np.abs(finite_upper))))
    if system.lower_bounds.size:
        bound_scale = max(bound_scale, float(np.max(np.abs(system.lower_bounds))))
    if parameter_vector.size:
        bound_scale = max(bound_scale, float(np.max(np.abs(parameter_vector))))
    return float(max(1.0e3 * np.finfo(np.float64).eps * bound_scale, 1.0e-15 * bound_scale))


def _projected_gradient_quality(
    system: RefractionStaticSolveSystem,
    parameter_vector: np.ndarray,
    residual: np.ndarray,
    gradient: np.ndarray,
    *,
    bound_tolerance: float,
) -> tuple[float, float]:
    lower_active = parameter_vector <= system.lower_bounds + bound_tolerance
    finite_upper = np.isfinite(system.upper_bounds)
    upper_active = finite_upper & (
        parameter_vector >= system.upper_bounds - bound_tolerance
    )
    fixed = lower_active & upper_active
    free = ~(lower_active | upper_active)
    violation = np.zeros_like(gradient)
    violation[free] = np.abs(gradient[free])
    lower_only = lower_active & ~fixed
    upper_only = upper_active & ~fixed
    violation[lower_only] = np.maximum(-gradient[lower_only], 0.0)
    violation[upper_only] = np.maximum(gradient[upper_only], 0.0)
    pg_norm = float(np.max(violation)) if violation.size else 0.0

    col_norm = np.sqrt(
        np.asarray(system.augmented_matrix.power(2).sum(axis=0)).ravel()
    )
    max_col_norm = float(np.max(col_norm)) if col_norm.size else 0.0
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(system.augmented_rhs_s))
    matrix_fro_norm = float(np.linalg.norm(system.augmented_matrix.data))
    x_norm = float(np.linalg.norm(parameter_vector, ord=np.inf)) if parameter_vector.size else 0.0
    gradient_reference = max(
        max_col_norm * residual_norm,
        matrix_fro_norm * matrix_fro_norm * x_norm + max_col_norm * rhs_norm,
        np.finfo(np.float64).tiny,
    )
    tolerance = max(
        1.0e-10 * gradient_reference,
        1.0e4 * np.finfo(np.float64).eps * gradient_reference,
        np.finfo(np.float64).tiny,
    )
    return pg_norm, float(tolerance)


def _active_set_polish_lsq_solution(
    system: RefractionStaticSolveSystem,
    parameter_vector: np.ndarray,
    *,
    solve_scale: float,
    scipy_result: optimize.OptimizeResult,
) -> optimize.OptimizeResult | None:
    if not _can_use_dense_lsq_retry(system.augmented_matrix):
        return None
    x = np.ascontiguousarray(parameter_vector, dtype=np.float64)
    x = np.maximum(x, system.lower_bounds)
    finite_upper = np.isfinite(system.upper_bounds)
    x[finite_upper] = np.minimum(x[finite_upper], system.upper_bounds[finite_upper])
    matrix = system.augmented_matrix.toarray()
    rhs = system.augmented_rhs_s
    _, _, best_objective = _lsq_residual_gradient_objective(system, x)
    for _ in range(_LSQ_ACTIVE_SET_MAX_ITERATIONS):
        bound_tolerance = _lsq_bound_tolerance(system, x)
        lower_active = x <= system.lower_bounds + bound_tolerance
        upper_active = finite_upper & (x >= system.upper_bounds - bound_tolerance)
        active = lower_active | upper_active
        free = ~active
        candidate = x.copy()
        if np.any(free):
            fixed_prediction = matrix[:, active] @ candidate[active]
            free_solution, *_ = np.linalg.lstsq(
                matrix[:, free],
                rhs - fixed_prediction,
                rcond=None,
            )
            candidate[free] = free_solution
        candidate = np.maximum(candidate, system.lower_bounds)
        candidate[finite_upper] = np.minimum(
            candidate[finite_upper],
            system.upper_bounds[finite_upper],
        )
        _, _, objective = _lsq_residual_gradient_objective(system, candidate)
        if objective > best_objective + max(1.0e-24, 1.0e-12 * max(1.0, best_objective)):
            break
        x = candidate
        best_objective = objective
        verified_x, quality = _verify_lsq_linear_solution(
            system,
            x,
            solve_scale=solve_scale,
            stage='active_set_polish',
            scipy_result=scipy_result,
        )
        x = verified_x
        if quality.verified:
            result = optimize.OptimizeResult()
            result.x = x
            result.success = bool(quality.verified)
            result.status = int(getattr(scipy_result, 'status', 0))
            result.message = 'active-set polished after lsq_linear retry'
            result.cost = quality.unscaled_objective
            result.optimality = quality.projected_gradient_inf_norm
            result.active_mask = _active_mask_for_parameter_vector(
                x,
                lower_bounds=system.lower_bounds,
                upper_bounds=system.upper_bounds,
            )
            result.nit = int(getattr(scipy_result, 'nit', 0))
            result.refraction_solver_stage = 'active_set_polish'
            result.refraction_solver_scale = float(solve_scale)
            result.refraction_solver_quality_diagnostic = quality
            result.refraction_solver_quality = _lsq_quality_json(quality)
            return result
    return None


def _lsq_quality_json(diagnostic: _LsqLinearQualityDiagnostic) -> dict[str, Any]:
    return {
        'verified': bool(diagnostic.verified),
        'stage': diagnostic.stage,
        'failure_reason': diagnostic.failure_reason,
        'solve_scale': _json_finite_float(diagnostic.solve_scale),
        'scipy_success': bool(diagnostic.scipy_success),
        'scipy_status': int(diagnostic.scipy_status),
        'scipy_optimality': _json_finite_float(diagnostic.scipy_optimality),
        'scipy_iterations': int(diagnostic.scipy_iterations),
        'unscaled_augmented_residual_norm': _json_finite_float(
            diagnostic.unscaled_augmented_residual_norm
        ),
        'unscaled_objective': _json_finite_float(diagnostic.unscaled_objective),
        'projected_gradient_inf_norm': _json_finite_float(
            diagnostic.projected_gradient_inf_norm
        ),
        'kkt_tolerance': _json_finite_float(diagnostic.kkt_tolerance),
        'max_bound_violation': _json_finite_float(diagnostic.max_bound_violation),
        'bound_tolerance': _json_finite_float(diagnostic.bound_tolerance),
    }


def _row_modeled_pick_time(
    design: RefractionStaticDesignMatrix,
    *,
    parameter_vector: np.ndarray,
) -> np.ndarray:
    modeled = np.ascontiguousarray(design.matrix @ parameter_vector, dtype=np.float64)
    if design.bedrock_velocity_mode == 'fixed_global':
        fixed_slowness = design.fixed_bedrock_slowness_s_per_m
        if fixed_slowness is None:
            raise RefractionStaticSolverError(
                'fixed_global design requires fixed bedrock slowness'
            )
        modeled = np.ascontiguousarray(
            modeled + design.row_distance_m * fixed_slowness,
            dtype=np.float64,
        )
    return modeled


def _assemble_node_solution(
    design: RefractionStaticDesignMatrix,
    *,
    parameter_vector: np.ndarray,
    active_mask: np.ndarray,
    system: RefractionStaticSolveSystem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_id = np.ascontiguousarray(design.diagnostics_context.node_id, dtype=np.int64)
    node_t1 = np.full(node_id.shape, np.nan, dtype=np.float64)
    status = np.full(node_id.shape, 'inactive', dtype='<U32')
    low_fold_mask = np.isin(node_id, design.low_fold_node_id)
    status[low_fold_mask] = LOW_FOLD_NODE_STATUS

    active_by_id = {int(node): col for node, col in design.node_id_to_col.items()}
    active_solver_mask = _coerce_active_mask(
        active_mask,
        expected_shape=(design.n_parameters,),
    )
    for index, current_node_id in enumerate(node_id.tolist()):
        col = active_by_id.get(int(current_node_id))
        if col is None:
            continue
        value = float(parameter_vector[col])
        node_t1[index] = value
        if int(active_solver_mask[col]) < 0 or np.isclose(
            value,
            system.node_lower_bound_s,
            rtol=0.0,
            atol=1.0e-10,
        ):
            status[index] = 'clipped_half_intercept_lower'
        elif int(active_solver_mask[col]) > 0 or np.isclose(
            value,
            system.node_upper_bound_s,
            rtol=0.0,
            atol=1.0e-10,
        ):
            status[index] = 'clipped_half_intercept_upper'
        else:
            status[index] = 'solved'
    return node_id, node_t1, np.ascontiguousarray(status)


def _bedrock_solution(
    design: RefractionStaticDesignMatrix,
    *,
    parameter_vector: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[float, float, str]:
    if design.bedrock_velocity_mode == 'fixed_global':
        fixed_slowness = design.fixed_bedrock_slowness_s_per_m
        fixed_velocity = design.fixed_bedrock_velocity_m_s
        if fixed_slowness is None or fixed_velocity is None:
            raise RefractionStaticSolverError(
                'fixed_global design requires fixed bedrock velocity'
            )
        return float(fixed_slowness), float(fixed_velocity), 'fixed'
    if design.bedrock_velocity_mode == 'solve_cell':
        return float('nan'), float('nan'), 'cell'

    col = _global_slowness_col(design)
    slowness = float(parameter_vector[col])
    if slowness <= 0.0 or not np.isfinite(slowness):
        raise RefractionStaticSolverError('estimated bedrock slowness must be positive')
    velocity = float(1.0 / slowness)
    mask_value = int(active_mask[col])
    if mask_value < 0:
        status = 'clipped_upper'
    elif mask_value > 0:
        status = 'clipped_lower'
    else:
        status = 'solved'
    return slowness, velocity, status


def _node_observation_count_for_row_mask(
    design: RefractionStaticDesignMatrix,
    *,
    row_used_mask: np.ndarray,
) -> np.ndarray:
    mask = _coerce_design_row_mask(design, row_used_mask=row_used_mask)
    node_id = np.ascontiguousarray(design.diagnostics_context.node_id, dtype=np.int64)
    if node_id.size == 0:
        return np.empty(0, dtype=np.int64)
    row_index = np.flatnonzero(mask).astype(np.int64, copy=False)
    if row_index.size == 0:
        return np.zeros(node_id.shape, dtype=np.int64)

    node_position_by_id = {
        int(raw_node_id): idx for idx, raw_node_id in enumerate(node_id.tolist())
    }
    source_pos = np.asarray(
        [
            node_position_by_id.get(int(raw_node_id), -1)
            for raw_node_id in design.row_source_node_id.tolist()
        ],
        dtype=np.int64,
    )
    receiver_pos = np.asarray(
        [
            node_position_by_id.get(int(raw_node_id), -1)
            for raw_node_id in design.row_receiver_node_id.tolist()
        ],
        dtype=np.int64,
    )
    selected_source_pos = source_pos[row_index]
    selected_receiver_pos = receiver_pos[row_index]
    valid_source = selected_source_pos >= 0
    valid_receiver = selected_receiver_pos >= 0
    node_pos = np.concatenate(
        (
            selected_source_pos[valid_source],
            selected_receiver_pos[valid_receiver],
        )
    )
    count_row_index = np.concatenate(
        (
            row_index[valid_source],
            row_index[valid_receiver],
        )
    )
    if node_pos.size == 0:
        return np.zeros(node_id.shape, dtype=np.int64)

    unique_pair_key = np.unique(node_pos * int(design.n_observations) + count_row_index)
    unique_node_pos = unique_pair_key // int(design.n_observations)
    return np.ascontiguousarray(
        np.bincount(unique_node_pos, minlength=int(node_id.shape[0])).astype(
            np.int64,
            copy=False,
        ),
        dtype=np.int64,
    )


def _cell_observation_count_for_row_mask(
    design: RefractionStaticDesignMatrix,
    *,
    row_used_mask: np.ndarray,
) -> np.ndarray:
    if design.bedrock_velocity_mode != 'solve_cell':
        return np.empty(0, dtype=np.int64)
    _validate_cell_design(design)
    if design.n_total_cells is None or design.active_cell_id is None:
        raise RefractionStaticSolverError('solve_cell design requires cell metadata')
    if design.row_midpoint_cell_id is None:
        raise RefractionStaticSolverError('solve_cell design requires row cell IDs')

    mask = _coerce_design_row_mask(design, row_used_mask=row_used_mask)
    base_counts = np.asarray(
        design.qc.get(
            'cell_observation_count',
            np.zeros(design.n_total_cells, dtype=np.int64),
        ),
        dtype=np.int64,
    )
    if base_counts.shape != (design.n_total_cells,):
        raise RefractionStaticSolverError('cell_observation_count shape mismatch')
    counts = np.ascontiguousarray(base_counts.copy(), dtype=np.int64)
    used_cell_id = design.row_midpoint_cell_id[mask]
    active_counts = np.bincount(
        used_cell_id,
        minlength=int(design.n_total_cells),
    ).astype(np.int64, copy=False)
    counts[design.active_cell_id] = active_counts[design.active_cell_id]
    return counts


def _coerce_design_row_mask(
    design: RefractionStaticDesignMatrix,
    *,
    row_used_mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(row_used_mask)
    if mask.shape != (design.n_observations,):
        raise RefractionStaticSolverError('row_used_mask shape mismatch')
    if not np.issubdtype(mask.dtype, np.bool_):
        raise RefractionStaticSolverError('row_used_mask must have bool dtype')
    return np.ascontiguousarray(mask, dtype=bool)


def _cell_solution(
    design: RefractionStaticDesignMatrix,
    *,
    parameter_vector: np.ndarray,
    active_mask: np.ndarray,
    cell_observation_count: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if design.bedrock_velocity_mode != 'solve_cell':
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype='<U32'),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    _validate_cell_design(design)
    if (
        design.n_total_cells is None
        or design.active_cell_id is None
        or design.cell_id_to_col is None
        or design.row_midpoint_cell_id is None
        or design.row_midpoint_cell_col is None
    ):
        raise RefractionStaticSolverError('solve_cell design requires cell metadata')

    cell_id = np.arange(design.n_total_cells, dtype=np.int64)
    cell_slowness = np.full(cell_id.shape, np.nan, dtype=np.float64)
    cell_velocity = np.full(cell_id.shape, np.nan, dtype=np.float64)
    cell_status = np.full(cell_id.shape, 'inactive', dtype='<U32')

    low_fold_cell_id = np.asarray(
        design.qc.get('low_fold_cell_id', []),
        dtype=np.int64,
    )
    if low_fold_cell_id.size:
        cell_status[low_fold_cell_id] = LOW_FOLD_CELL_VELOCITY_STATUS

    solver_active_mask = _coerce_active_mask(
        active_mask,
        expected_shape=(design.n_parameters,),
    )
    for raw_cell_id in design.active_cell_id.tolist():
        current_cell = int(raw_cell_id)
        col = int(design.cell_id_to_col[current_cell])
        slowness = float(parameter_vector[col])
        if slowness <= 0.0 or not np.isfinite(slowness):
            raise RefractionStaticSolverError(
                'estimated cell bedrock slowness must be positive'
            )
        cell_slowness[current_cell] = slowness
        cell_velocity[current_cell] = float(1.0 / slowness)
        mask_value = int(solver_active_mask[col])
        if mask_value < 0:
            cell_status[current_cell] = 'clipped_upper'
        elif mask_value > 0:
            cell_status[current_cell] = 'clipped_lower'
        else:
            cell_status[current_cell] = 'solved'

    row_midpoint_cell_id = np.ascontiguousarray(
        design.row_midpoint_cell_id,
        dtype=np.int64,
    )
    row_midpoint_slowness = np.ascontiguousarray(
        cell_slowness[row_midpoint_cell_id],
        dtype=np.float64,
    )
    row_midpoint_velocity = np.ascontiguousarray(
        cell_velocity[row_midpoint_cell_id],
        dtype=np.float64,
    )
    _validate_finite(row_midpoint_slowness, name='row_midpoint_bedrock_slowness')
    _validate_finite(row_midpoint_velocity, name='row_midpoint_bedrock_velocity')

    cell_observation_count = np.asarray(cell_observation_count, dtype=np.int64)
    if cell_observation_count.shape != cell_id.shape:
        raise RefractionStaticSolverError('cell_observation_count shape mismatch')
    return (
        np.ascontiguousarray(cell_id, dtype=np.int64),
        np.ascontiguousarray(cell_slowness, dtype=np.float64),
        np.ascontiguousarray(cell_velocity, dtype=np.float64),
        np.ascontiguousarray(cell_status),
        np.ascontiguousarray(cell_observation_count, dtype=np.int64),
        row_midpoint_cell_id,
        row_midpoint_slowness,
        row_midpoint_velocity,
    )


def _coerce_active_mask(
    active_mask: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(active_mask)
    if values.shape != expected_shape:
        raise RefractionStaticSolverError('active_mask shape mismatch')
    try:
        out = np.ascontiguousarray(values, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise RefractionStaticSolverError(
            'active_mask must contain integer values'
        ) from exc
    if not np.array_equal(values, out):
        raise RefractionStaticSolverError('active_mask must contain integer values')
    return out


def _active_mask_for_parameter_vector(
    parameter_vector: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    active = np.zeros(parameter_vector.shape, dtype=np.int64)
    active[np.isclose(parameter_vector, lower_bounds, rtol=0.0, atol=1.0e-10)] = -1
    finite_upper = np.isfinite(upper_bounds)
    active[
        finite_upper
        & np.isclose(parameter_vector, upper_bounds, rtol=0.0, atol=1.0e-10)
    ] = 1
    return np.ascontiguousarray(active, dtype=np.int64)


def _global_slowness_col(design: RefractionStaticDesignMatrix) -> int:
    col = design.bedrock_slowness_col
    if col is None:
        raise RefractionStaticSolverError(
            'solve_global design requires a bedrock slowness column'
        )
    return int(col)


def _positive_model_float(model: Any, attr: str, *, name: str) -> float:
    return _coerce_positive_finite_float(
        getattr(model, attr, None),
        name=name,
        error_type=RefractionStaticSolverError,
    )


def _validate_bounds(*, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
    _validate_finite(lower_bounds, name='lower_bounds')
    if np.any(~np.isfinite(upper_bounds) & ~np.isposinf(upper_bounds)):
        raise RefractionStaticSolverError('upper_bounds must be finite or +inf')
    if lower_bounds.shape != upper_bounds.shape:
        raise RefractionStaticSolverError('bounds shape mismatch')
    if np.any(lower_bounds > upper_bounds):
        raise RefractionStaticSolverError('lower bounds must not exceed upper bounds')


def _validate_finite(values: np.ndarray, *, name: str) -> None:
    if np.any(~np.isfinite(values)):
        raise RefractionStaticSolverError(f'{name} must contain only finite values')


def _rms(values: np.ndarray) -> float:
    _validate_finite(values, name='rms values')
    return float(np.sqrt(np.mean(np.square(values))))


def _build_solver_qc(
    *,
    design: RefractionStaticDesignMatrix,
    system: RefractionStaticSolveSystem,
    result: optimize.OptimizeResult,
    rms_residual_s: float,
    bedrock_velocity_m_s: float,
    bedrock_slowness_s_per_m: float,
    bedrock_velocity_status: str,
    node_solution_status: np.ndarray,
    cell_velocity_status: np.ndarray,
    robust_result: _RefractionStaticRobustRunResult,
    node_observation_count: np.ndarray,
    cell_observation_count: np.ndarray,
) -> dict[str, Any]:
    unique_status, status_counts = np.unique(node_solution_status, return_counts=True)
    unique_cell_status, cell_status_counts = np.unique(
        cell_velocity_status,
        return_counts=True,
    )
    design_qc = _solver_design_qc(
        design,
        node_observation_count=node_observation_count,
        cell_observation_count=cell_observation_count,
        row_used_mask=robust_result.row_used_mask,
    )
    initial_graph = _analyze_design_endpoint_graph(design)
    initial_gauge_required_components = int(
        np.count_nonzero(initial_graph.gauge_required_by_component)
    )
    qc = {
        'method': 'gli_variable_thickness',
        'bedrock_velocity_mode': design.bedrock_velocity_mode,
        'bedrock_velocity_m_s': _json_finite_float(bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': _json_finite_float(bedrock_slowness_s_per_m),
        'bedrock_velocity_status': bedrock_velocity_status,
        'n_observations': int(design.n_observations),
        'n_observation_rows': int(system.n_observation_rows),
        'n_initial_observation_rows': int(design.n_observations),
        'n_final_observation_rows': int(system.n_observation_rows),
        'n_initial_used_observations': int(design.n_observations),
        'n_final_used_observations': int(
            np.count_nonzero(robust_result.row_used_mask)
        ),
        'n_rejected_observations': int(
            design.n_observations - np.count_nonzero(robust_result.row_used_mask)
        ),
        'n_initial_rejected_observations': 0,
        'n_final_rejected_observations': int(
            design.n_observations - np.count_nonzero(robust_result.row_used_mask)
        ),
        'n_parameters': int(design.n_parameters),
        'n_augmented_rows': int(system.n_augmented_rows),
        'n_smoothing_rows': int(system.n_smoothing_rows),
        'n_damping_rows': int(system.n_damping_rows),
        'n_gauge_rows': int(system.n_gauge_rows),
        'n_initial_gauge_rows': 0,
        'n_final_gauge_rows': int(system.n_gauge_rows),
        'n_node_components': int(system.n_node_components),
        'n_initial_node_components': int(initial_graph.n_components),
        'n_final_node_components': int(system.n_node_components),
        'n_bipartite_node_components': int(system.n_bipartite_node_components),
        'n_initial_bipartite_node_components': int(
            np.count_nonzero(initial_graph.is_bipartite_by_component)
        ),
        'n_final_bipartite_node_components': int(
            system.n_bipartite_node_components
        ),
        'n_gauge_required_node_components': int(
            np.count_nonzero(system.gauge_required_by_component)
        ),
        'n_initial_gauge_required_node_components': initial_gauge_required_components,
        'n_final_gauge_required_node_components': int(
            np.count_nonzero(system.gauge_required_by_component)
        ),
        'gauge_resolution': system.gauge_resolution,
        'half_intercept_damping_lambda': float(
            system.half_intercept_damping_lambda
        ),
        'regularized_parameter_group': system.regularized_parameter_group,
        'regularization_row_count': int(system.regularization_row_count),
        'node_lower_bound_s': float(system.node_lower_bound_s),
        'node_upper_bound_s': float(system.node_upper_bound_s),
        'slowness_lower_bound_s_per_m': _optional_float(
            system.slowness_lower_bound_s_per_m
        ),
        'slowness_upper_bound_s_per_m': _optional_float(
            system.slowness_upper_bound_s_per_m
        ),
        'initial_bedrock_slowness_s_per_m': _optional_float(
            system.initial_bedrock_slowness_s_per_m
        ),
        'solver_name': 'lsq_linear',
        'solver_success': bool(result.success),
        'solver_status': int(result.status),
        'solver_message': str(result.message),
        'solver_cost': float(result.cost),
        'solver_optimality': float(result.optimality),
        'solver_iterations': int(result.nit),
        'solver_quality': dict(
            getattr(result, 'refraction_solver_quality', {})
        ),
        'rms_residual_ms': float(rms_residual_s * 1000.0),
        'physical_identifiability': _identifiability_diagnostic_json(
            system.identifiability
        ),
        'robust_enabled': bool(
            robust_result.stop_reason != 'disabled'
            or len(robust_result.iteration_summaries) > 0
        ),
        'robust_stop_reason': robust_result.stop_reason,
        'robust_iteration_count': len(robust_result.iteration_summaries),
        'robust_iterations': [
            _robust_iteration_summary_json(summary)
            for summary in robust_result.iteration_summaries
        ],
        'node_solution_status_counts': {
            str(key): int(value)
            for key, value in zip(
                unique_status.tolist(),
                status_counts.tolist(),
                strict=True,
            )
        },
        'design_matrix': design_qc,
    }
    if system.smoothing_rows is not None:
        qc['cell_smoothing'] = dict(system.smoothing_rows.qc)
    if cell_velocity_status.size:
        qc['cell_velocity_status_counts'] = {
            str(key): int(value)
            for key, value in zip(
                unique_cell_status.tolist(),
                cell_status_counts.tolist(),
                strict=True,
            )
        }
    return qc


def _identifiability_diagnostic_json(
    diagnostic: _NumericalRankDiagnostic,
) -> dict[str, Any]:
    return {
        'method': diagnostic.method,
        'n_rows': int(diagnostic.n_rows),
        'n_columns': int(diagnostic.n_columns),
        'expected_rank': int(diagnostic.expected_rank),
        'estimated_numerical_rank': int(diagnostic.estimated_rank),
        'expected_nullity': int(diagnostic.expected_nullity),
        'gauge_nullity': int(diagnostic.gauge_nullity),
        'threshold': float(diagnostic.threshold),
        'critical_singular_value': float(diagnostic.critical_singular_value),
        'largest_singular_value': float(diagnostic.largest_singular_value),
        'rtol': float(diagnostic.rtol),
        'sparse_solver_name': diagnostic.sparse_solver_name,
        'certification_status': diagnostic.certification_status,
        'requested_smallest_count': int(diagnostic.requested_smallest_count),
        'returned_smallest_count': int(diagnostic.returned_smallest_count),
        'max_singular_triplet_residual': float(
            diagnostic.max_singular_triplet_residual
        ),
        'failure_reason': diagnostic.failure_reason,
    }


def _solver_design_qc(
    design: RefractionStaticDesignMatrix,
    *,
    node_observation_count: np.ndarray,
    cell_observation_count: np.ndarray,
    row_used_mask: np.ndarray,
) -> dict[str, Any]:
    qc = dict(design.qc)
    qc['node_observation_count'] = [
        int(value) for value in node_observation_count.tolist()
    ]
    if design.bedrock_velocity_mode != 'solve_cell':
        return qc
    _validate_cell_design(design)
    if design.active_cell_id is None:
        raise RefractionStaticSolverError('solve_cell design requires active cells')
    qc['cell_observation_count'] = [
        int(value) for value in cell_observation_count.tolist()
    ]
    qc['n_observations_used'] = int(np.count_nonzero(row_used_mask))
    active_counts = cell_observation_count[design.active_cell_id]
    if active_counts.size:
        qc['min_observations_per_active_cell'] = int(np.min(active_counts))
        qc['median_observations_per_active_cell'] = float(np.median(active_counts))
        qc['max_observations_per_active_cell'] = int(np.max(active_counts))
    else:
        qc['min_observations_per_active_cell'] = None
        qc['median_observations_per_active_cell'] = None
        qc['max_observations_per_active_cell'] = None
    return qc


def _robust_iteration_summary_json(
    summary: RefractionStaticRobustIterationSummary,
) -> dict[str, Any]:
    return {
        'iteration_index': int(summary.iteration_index),
        'method': str(summary.method),
        'n_used_before': int(summary.n_used_before),
        'n_rejected_this_iteration': int(summary.n_rejected_this_iteration),
        'n_used_after': int(summary.n_used_after),
        'residual_center_s': float(summary.residual_center_s),
        'residual_scale_s': float(summary.residual_scale_s),
        'residual_scale_floor_s': float(summary.residual_scale_floor_s),
        'residual_cutoff_s': float(summary.residual_cutoff_s),
        'max_abs_centered_residual_s': float(
            summary.max_abs_centered_residual_s
        ),
        'converged': bool(summary.converged),
        'stop_reason': summary.stop_reason,
    }


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _json_finite_float(value: float) -> float | None:
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


__all__ = [
    'RefractionStaticRobustIterationSummary',
    'RefractionStaticRobustStopReason',
    'RefractionStaticSolveResult',
    'RefractionStaticSolveSystem',
    'RefractionStaticSolverError',
    'build_refraction_static_solver_system',
    'solve_refraction_static_design_least_squares',
    'solve_refraction_static_least_squares',
    'summarize_refraction_static_solve_result',
    'validate_refraction_static_solver_options',
]
