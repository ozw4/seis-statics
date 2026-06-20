"""Bounded least-squares solver for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
from scipy import optimize, sparse

from seis_statics._validation import (
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
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
class RefractionStaticSolveSystem:
    """Augmented bounded linear system used by the refraction solver."""

    augmented_matrix: sparse.csr_matrix
    augmented_rhs_s: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    initial_parameter_vector: np.ndarray

    n_observation_rows: int
    n_smoothing_rows: int
    n_damping_rows: int
    n_gauge_rows: int
    n_augmented_rows: int
    n_parameters: int

    damping: float
    node_lower_bound_s: float
    node_upper_bound_s: float
    slowness_lower_bound_s_per_m: float | None
    slowness_upper_bound_s_per_m: float | None
    initial_bedrock_slowness_s_per_m: float | None
    smoothing_rows: CellSlownessSmoothingRows | None


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


def build_refraction_static_solver_system(
    design: RefractionStaticDesignMatrix,
    *,
    model: Any,
    solver_options: RefractionStaticSolverOptions | None = None,
) -> RefractionStaticSolveSystem:
    """Build observation, damping, gauge, bounds, and initial vector."""
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
        damping=options.damping,
        initial_parameter_vector=initial,
    )
    gauge_matrix = _build_source_receiver_gauge_matrix(design)
    gauge_rhs = np.zeros(gauge_matrix.shape[0], dtype=np.float64)

    augmented_matrix = sparse.vstack(
        [
            design.matrix,
            (
                smoothing_rows.matrix
                if smoothing_rows is not None
                else _empty_rows(n_parameters)
            ),
            damping_matrix,
            gauge_matrix,
        ],
        format='csr',
        dtype=np.float64,
    )
    augmented_matrix.sort_indices()
    augmented_rhs = np.ascontiguousarray(
        np.concatenate(
            [
                design.rhs_s,
                smoothing_rows.rhs_s if smoothing_rows is not None else np.empty(0),
                damping_rhs,
                gauge_rhs,
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
        lower_bounds=np.ascontiguousarray(lower_bounds, dtype=np.float64),
        upper_bounds=np.ascontiguousarray(upper_bounds, dtype=np.float64),
        initial_parameter_vector=np.ascontiguousarray(initial, dtype=np.float64),
        n_observation_rows=int(design.n_observations),
        n_smoothing_rows=0 if smoothing_rows is None else smoothing_rows.n_rows,
        n_damping_rows=int(damping_matrix.shape[0]),
        n_gauge_rows=int(gauge_matrix.shape[0]),
        n_augmented_rows=n_augmented_rows,
        n_parameters=n_parameters,
        damping=options.damping,
        node_lower_bound_s=0.0,
        node_upper_bound_s=float(node_upper),
        slowness_lower_bound_s_per_m=slowness_lower,
        slowness_upper_bound_s_per_m=slowness_upper,
        initial_bedrock_slowness_s_per_m=initial_slowness,
        smoothing_rows=smoothing_rows,
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
        model=model,
        solver_options=options,
    )
    raw = robust_result.raw_result
    final_system = robust_result.system
    row_used_mask = robust_result.row_used_mask
    parameter_vector = np.ascontiguousarray(raw.x, dtype=np.float64)
    _validate_finite(parameter_vector, name='parameter_vector')

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
        active_mask=raw.active_mask,
        system=final_system,
    )
    bedrock_slowness, bedrock_velocity, bedrock_status = _bedrock_solution(
        design,
        parameter_vector=parameter_vector,
        active_mask=raw.active_mask,
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
        active_mask=raw.active_mask,
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
        solver_active_mask=np.ascontiguousarray(raw.active_mask, dtype=np.int64),
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
        damping=_coerce_nonnegative_finite_float(
            opts.damping,
            name='solver.damping',
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


def _build_damping_system(
    *,
    n_parameters: int,
    damping: float,
    initial_parameter_vector: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    if damping == 0.0:
        return (
            sparse.csr_matrix((0, n_parameters), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    matrix = sparse.eye(n_parameters, format='csr', dtype=np.float64) * damping
    rhs = np.ascontiguousarray(damping * initial_parameter_vector, dtype=np.float64)
    return matrix, rhs


def _build_source_receiver_gauge_matrix(
    design: RefractionStaticDesignMatrix,
) -> sparse.csr_matrix:
    n_parameters = int(design.n_parameters)
    n_active_nodes = int(design.n_active_nodes)
    if n_active_nodes <= 0:
        return sparse.csr_matrix((0, n_parameters), dtype=np.float64)

    source_counts = _node_role_counts(
        design.row_source_node_id,
        node_id_to_col=design.node_id_to_col,
        n_active_nodes=n_active_nodes,
    )
    receiver_counts = _node_role_counts(
        design.row_receiver_node_id,
        node_id_to_col=design.node_id_to_col,
        n_active_nodes=n_active_nodes,
    )
    source_cols = np.flatnonzero(source_counts > 0).astype(np.int64, copy=False)
    receiver_cols = np.flatnonzero(receiver_counts > 0).astype(np.int64, copy=False)
    if source_cols.size == 0 or receiver_cols.size == 0:
        return sparse.csr_matrix((0, n_parameters), dtype=np.float64)

    coeff = np.zeros(n_parameters, dtype=np.float64)
    coeff[source_cols] += 1.0 / float(source_cols.size)
    coeff[receiver_cols] -= 1.0 / float(receiver_cols.size)
    nonzero = np.flatnonzero(coeff != 0.0).astype(np.int64, copy=False)
    if nonzero.size == 0:
        return sparse.csr_matrix((0, n_parameters), dtype=np.float64)
    matrix = sparse.coo_matrix(
        (
            coeff[nonzero],
            (np.zeros(nonzero.size, dtype=np.int64), nonzero),
        ),
        shape=(1, n_parameters),
        dtype=np.float64,
    ).tocsr()
    matrix.sort_indices()
    return matrix


def _node_role_counts(
    node_ids: np.ndarray,
    *,
    node_id_to_col: dict[int, int],
    n_active_nodes: int,
) -> np.ndarray:
    values = _coerce_1d_integer_int64(node_ids, name='node_ids')
    cols = np.asarray([node_id_to_col[int(value)] for value in values], dtype=np.int64)
    return np.bincount(cols, minlength=n_active_nodes).astype(np.int64, copy=False)


def _run_refraction_static_solver(
    *,
    design: RefractionStaticDesignMatrix,
    system: RefractionStaticSolveSystem,
    model: Any,
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
        current_system = _system_with_observation_row_mask(
            system,
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
        outlier_local = np.abs(current_residual - center_s) > cutoff_s
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
            model=model,
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
        final_system = _system_with_observation_row_mask(
            system,
            row_used_mask=current_row_mask,
        )
        final_raw = _run_lsq_linear(final_system)

    if final_raw is None or final_system is None or stop_reason is None:
        raise RefractionStaticSolverError('robust refraction solver produced no result')

    if int(np.count_nonzero(current_row_mask)) < design.n_observations:
        final_system = _system_with_observation_row_mask(
            system,
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


def _system_with_observation_row_mask(
    system: RefractionStaticSolveSystem,
    *,
    row_used_mask: np.ndarray,
) -> RefractionStaticSolveSystem:
    mask = np.asarray(row_used_mask)
    if mask.shape != (system.n_observation_rows,):
        raise RefractionStaticSolverError('robust row mask shape mismatch')
    if not np.issubdtype(mask.dtype, np.bool_):
        raise RefractionStaticSolverError('robust row mask must have bool dtype')
    used_observation_rows = np.flatnonzero(mask).astype(np.int64, copy=False)
    if used_observation_rows.size == system.n_observation_rows:
        return system
    if used_observation_rows.size == 0:
        raise RefractionStaticSolverError(
            'robust rejection would drop all observations'
        )
    tail_rows = np.arange(
        system.n_observation_rows,
        system.n_augmented_rows,
        dtype=np.int64,
    )
    selected_rows = np.ascontiguousarray(
        np.concatenate((used_observation_rows, tail_rows)),
        dtype=np.int64,
    )
    augmented_matrix = system.augmented_matrix[selected_rows].tocsr()
    augmented_matrix.sort_indices()
    augmented_rhs = np.ascontiguousarray(
        system.augmented_rhs_s[selected_rows],
        dtype=np.float64,
    )
    return RefractionStaticSolveSystem(
        augmented_matrix=augmented_matrix,
        augmented_rhs_s=augmented_rhs,
        lower_bounds=system.lower_bounds,
        upper_bounds=system.upper_bounds,
        initial_parameter_vector=system.initial_parameter_vector,
        n_observation_rows=int(used_observation_rows.size),
        n_smoothing_rows=system.n_smoothing_rows,
        n_damping_rows=system.n_damping_rows,
        n_gauge_rows=system.n_gauge_rows,
        n_augmented_rows=int(augmented_matrix.shape[0]),
        n_parameters=system.n_parameters,
        damping=system.damping,
        node_lower_bound_s=system.node_lower_bound_s,
        node_upper_bound_s=system.node_upper_bound_s,
        slowness_lower_bound_s_per_m=system.slowness_lower_bound_s_per_m,
        slowness_upper_bound_s_per_m=system.slowness_upper_bound_s_per_m,
        initial_bedrock_slowness_s_per_m=(
            system.initial_bedrock_slowness_s_per_m
        ),
        smoothing_rows=system.smoothing_rows,
    )


def _safe_rejection_rows(
    *,
    design: RefractionStaticDesignMatrix,
    system: RefractionStaticSolveSystem,
    model: Any,
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
            model=model,
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
    model: Any,
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
    if (
        n_used
        + system.n_smoothing_rows
        + system.n_damping_rows
        + system.n_gauge_rows
        < system.n_parameters
    ):
        return False
    if not _node_coverage_is_safe(design, row_used_mask=row_used_mask):
        return False
    if design.bedrock_velocity_mode == 'solve_cell':
        return _cell_coverage_is_safe(
            design,
            model=model,
            row_used_mask=row_used_mask,
        )
    return True


def _node_coverage_is_safe(
    design: RefractionStaticDesignMatrix,
    *,
    row_used_mask: np.ndarray,
) -> bool:
    counts = np.zeros(design.n_active_nodes, dtype=np.int64)
    row_indices = np.flatnonzero(row_used_mask)
    if row_indices.size == 0:
        return False
    np.add.at(counts, design.source_node_col[row_indices], 1)
    np.add.at(counts, design.receiver_node_col[row_indices], 1)
    return bool(np.all(counts >= int(design.min_observations_per_node)))


def _cell_coverage_is_safe(
    design: RefractionStaticDesignMatrix,
    *,
    model: Any,
    row_used_mask: np.ndarray,
) -> bool:
    _validate_cell_design(design)
    if design.active_cell_id is None or design.row_midpoint_cell_id is None:
        raise RefractionStaticSolverError('solve_cell design requires cell metadata')
    refractor_cell = getattr(model, 'refractor_cell', None)
    min_observations = int(
        getattr(
            refractor_cell,
            'min_observations_per_cell',
            design.qc.get('min_observations_per_cell', 1),
        )
    )
    if design.n_total_cells is None:
        raise RefractionStaticSolverError('solve_cell design requires n_total_cells')
    used_cell_id = design.row_midpoint_cell_id[row_used_mask]
    counts = np.bincount(
        used_cell_id,
        minlength=int(design.n_total_cells),
    ).astype(np.int64, copy=False)
    return bool(np.all(counts[design.active_cell_id] >= min_observations))


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


def _run_lsq_linear(system: RefractionStaticSolveSystem) -> optimize.OptimizeResult:
    try:
        result = optimize.lsq_linear(
            system.augmented_matrix,
            system.augmented_rhs_s,
            bounds=(system.lower_bounds, system.upper_bounds),
            method='trf',
            tol=1.0e-10,
            lsq_solver='lsmr',
            lsmr_tol='auto',
            max_iter=None,
            verbose=0,
        )
    except Exception as exc:
        raise RuntimeError('refraction static lsq_linear solve failed') from exc
    if result.x.shape != (system.n_parameters,):
        raise RefractionStaticSolverError('solver x shape mismatch')
    return result


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
    qc = {
        'method': 'gli_variable_thickness',
        'bedrock_velocity_mode': design.bedrock_velocity_mode,
        'bedrock_velocity_m_s': _json_finite_float(bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': _json_finite_float(bedrock_slowness_s_per_m),
        'bedrock_velocity_status': bedrock_velocity_status,
        'n_observations': int(design.n_observations),
        'n_initial_used_observations': int(design.n_observations),
        'n_final_used_observations': int(
            np.count_nonzero(robust_result.row_used_mask)
        ),
        'n_rejected_observations': int(
            design.n_observations - np.count_nonzero(robust_result.row_used_mask)
        ),
        'n_parameters': int(design.n_parameters),
        'n_augmented_rows': int(system.n_augmented_rows),
        'n_smoothing_rows': int(system.n_smoothing_rows),
        'n_damping_rows': int(system.n_damping_rows),
        'n_gauge_rows': int(system.n_gauge_rows),
        'damping': float(system.damping),
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
        'rms_residual_ms': float(rms_residual_s * 1000.0),
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
