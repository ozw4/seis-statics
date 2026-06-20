"""Bounded least-squares solver for GLI refraction statics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy import optimize, sparse

from seis_statics._validation import (
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
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


@dataclass(frozen=True)
class RefractionStaticSolveResult:
    """Non-robust global/fixed GLI refraction static solution."""

    parameter_vector: np.ndarray

    node_id: np.ndarray
    node_half_intercept_time_s: np.ndarray
    node_solution_status: np.ndarray
    node_observation_count: np.ndarray

    bedrock_velocity_mode: str
    bedrock_velocity_m_s: float
    bedrock_slowness_s_per_m: float
    bedrock_velocity_status: str

    modeled_pick_time_s_sorted: np.ndarray
    residual_s_sorted: np.ndarray
    residual_ms_sorted: np.ndarray
    used_observation_mask_sorted: np.ndarray

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

    design: RefractionStaticDesignMatrix
    system: RefractionStaticSolveSystem
    solver_options: RefractionStaticSolverOptions
    qc: dict[str, Any]


def build_refraction_static_solver_system(
    design: RefractionStaticDesignMatrix,
    *,
    model: Any,
    solver_options: RefractionStaticSolverOptions | None = None,
) -> RefractionStaticSolveSystem:
    """Build observation, damping, gauge, bounds, and initial vector."""
    options = validate_refraction_static_solver_options(solver_options)
    _validate_global_design(design)
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
        slowness_lower = float(1.0 / max_velocity)
        slowness_upper = float(1.0 / min_velocity)
        initial_slowness = float(1.0 / initial_velocity)
        lower_bounds[slowness_col] = slowness_lower
        upper_bounds[slowness_col] = slowness_upper
        initial[slowness_col] = initial_slowness

    damping_matrix, damping_rhs = _build_damping_system(
        n_parameters=n_parameters,
        damping=options.damping,
        initial_parameter_vector=initial,
    )
    gauge_matrix = _build_source_receiver_gauge_matrix(design)
    gauge_rhs = np.zeros(gauge_matrix.shape[0], dtype=np.float64)

    augmented_matrix = sparse.vstack(
        [design.matrix, damping_matrix, gauge_matrix],
        format='csr',
        dtype=np.float64,
    )
    augmented_matrix.sort_indices()
    augmented_rhs = np.ascontiguousarray(
        np.concatenate([design.rhs_s, damping_rhs, gauge_rhs]),
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
    )


def solve_refraction_static_least_squares(
    *,
    input_model: RefractionStaticInputModel,
    model: Any,
    solver_options: RefractionStaticSolverOptions | None = None,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    include_diagnostics: bool = False,
) -> RefractionStaticSolveResult:
    """Build and solve a non-robust global/fixed GLI refraction system."""
    options = validate_refraction_static_solver_options(solver_options)
    if getattr(model, 'bedrock_velocity_mode', None) == 'solve_cell':
        raise RefractionStaticSolverError(
            'solve_cell mode is not supported by this solver; use the cell '
            'solver implementation from the follow-up issue'
        )
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
    """Solve a pre-built non-robust global/fixed refraction design matrix."""
    options = validate_refraction_static_solver_options(solver_options)
    system = build_refraction_static_solver_system(
        design,
        model=model,
        solver_options=options,
    )
    raw = _run_lsq_linear(system)
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
    used_mask[design.row_trace_index_sorted] = True

    node_id, node_t1, node_status = _assemble_node_solution(
        design,
        parameter_vector=parameter_vector,
        active_mask=raw.active_mask,
        system=system,
    )
    bedrock_slowness, bedrock_velocity, bedrock_status = _bedrock_solution(
        design,
        parameter_vector=parameter_vector,
        active_mask=raw.active_mask,
    )
    rms_s = _rms(row_residual)
    qc = _build_solver_qc(
        design=design,
        system=system,
        result=raw,
        rms_residual_s=rms_s,
        bedrock_velocity_m_s=bedrock_velocity,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_status=bedrock_status,
        node_solution_status=node_status,
    )

    return RefractionStaticSolveResult(
        parameter_vector=parameter_vector,
        node_id=node_id,
        node_half_intercept_time_s=node_t1,
        node_solution_status=node_status,
        node_observation_count=np.ascontiguousarray(
            design.node_observation_count,
            dtype=np.int64,
        ),
        bedrock_velocity_mode=design.bedrock_velocity_mode,
        bedrock_velocity_m_s=bedrock_velocity,
        bedrock_slowness_s_per_m=bedrock_slowness,
        bedrock_velocity_status=bedrock_status,
        modeled_pick_time_s_sorted=full_modeled,
        residual_s_sorted=full_residual,
        residual_ms_sorted=np.ascontiguousarray(full_residual * 1000.0),
        used_observation_mask_sorted=used_mask,
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
        design=design,
        system=system,
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
    if robust.enabled:
        raise RefractionStaticSolverError(
            'robust refraction solving is not supported in this issue; set '
            'solver.robust.enabled=False'
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
        robust=robust,
    )


def _validate_global_design(design: RefractionStaticDesignMatrix) -> None:
    if not isinstance(design, RefractionStaticDesignMatrix):
        raise RefractionStaticSolverError(
            'design must be a RefractionStaticDesignMatrix instance'
        )
    if design.bedrock_velocity_mode == 'solve_cell':
        raise RefractionStaticSolverError(
            'solve_cell mode is not supported by this solver'
        )
    if design.bedrock_velocity_mode not in {'solve_global', 'fixed_global'}:
        raise RefractionStaticSolverError(
            'design.bedrock_velocity_mode must be solve_global or fixed_global'
        )
    if design.matrix.shape != (design.n_observations, design.n_parameters):
        raise RefractionStaticSolverError('design.matrix shape mismatch')
    if not sparse.isspmatrix_csr(design.matrix):
        raise RefractionStaticSolverError('design.matrix must be CSR')
    _validate_finite(design.matrix.data, name='design.matrix.data')
    _validate_finite(design.rhs_s, name='design.rhs_s')


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
) -> dict[str, Any]:
    unique_status, status_counts = np.unique(node_solution_status, return_counts=True)
    return {
        'method': 'gli_variable_thickness',
        'bedrock_velocity_mode': design.bedrock_velocity_mode,
        'bedrock_velocity_m_s': float(bedrock_velocity_m_s),
        'bedrock_slowness_s_per_m': float(bedrock_slowness_s_per_m),
        'bedrock_velocity_status': bedrock_velocity_status,
        'n_observations': int(design.n_observations),
        'n_parameters': int(design.n_parameters),
        'n_augmented_rows': int(system.n_augmented_rows),
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
        'node_solution_status_counts': {
            str(key): int(value)
            for key, value in zip(
                unique_status.tolist(),
                status_counts.tolist(),
                strict=True,
            )
        },
        'design_matrix': dict(design.qc),
    }


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    'RefractionStaticSolveResult',
    'RefractionStaticSolveSystem',
    'RefractionStaticSolverError',
    'build_refraction_static_solver_system',
    'solve_refraction_static_design_least_squares',
    'solve_refraction_static_least_squares',
    'summarize_refraction_static_solve_result',
    'validate_refraction_static_solver_options',
]
