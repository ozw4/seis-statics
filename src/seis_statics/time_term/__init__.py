"""Time-term statics input, moveout, design-matrix, and sparse-solver APIs."""

from __future__ import annotations

from seis_statics.time_term.design_matrix import (
    TimeTermDesignMatrix,
    TimeTermDesignMatrixOptions,
    build_time_term_design_matrix,
    summarize_time_term_design_matrix,
)
from seis_statics.time_term.moveout import (
    MoveoutDistanceSource,
    TimeTermMoveoutConfig,
    TimeTermMoveoutModel,
    TimeTermMoveoutResult,
    build_reciprocal_pair_index,
    compute_geometry_distance_m,
    compute_time_term_moveout,
    summarize_time_term_moveout,
)
from seis_statics.time_term.apply_shift import (
    TimeTermAppliedShiftResult,
    compose_time_term_applied_shifts,
    delay_to_applied_shift,
)
from seis_statics.time_term.robust_solver import (
    ROBUST_SCALE_FLOOR_S,
    TimeTermRobustIterationSummary,
    TimeTermRobustMethod,
    TimeTermRobustOptions,
    TimeTermRobustSolveResult,
    TimeTermRobustStopReason,
    build_time_term_outlier_mask,
    compute_time_term_robust_center_scale,
    solve_time_term_robust_least_squares,
    validate_time_term_robust_options,
)
from seis_statics.time_term.types import (
    ORDER,
    SIGN_CONVENTION,
    TimeTermInversionInputs,
)
from seis_statics.time_term.sparse_solver import (
    TimeTermGaugeMode,
    TimeTermSolverSystem,
    TimeTermSparseSolverName,
    TimeTermSparseSolverOptions,
    TimeTermSparseSolverResult,
    build_gauge_matrix,
    build_node_components,
    build_time_term_solver_system,
    solve_time_term_sparse_least_squares,
    summarize_time_term_sparse_solver_result,
)

__all__ = [
    'MoveoutDistanceSource',
    'ORDER',
    'SIGN_CONVENTION',
    'ROBUST_SCALE_FLOOR_S',
    'TimeTermDesignMatrix',
    'TimeTermDesignMatrixOptions',
    'TimeTermGaugeMode',
    'TimeTermAppliedShiftResult',
    'TimeTermInversionInputs',
    'TimeTermMoveoutConfig',
    'TimeTermMoveoutModel',
    'TimeTermMoveoutResult',
    'TimeTermRobustIterationSummary',
    'TimeTermRobustMethod',
    'TimeTermRobustOptions',
    'TimeTermRobustSolveResult',
    'TimeTermRobustStopReason',
    'TimeTermSolverSystem',
    'TimeTermSparseSolverName',
    'TimeTermSparseSolverOptions',
    'TimeTermSparseSolverResult',
    'build_gauge_matrix',
    'build_node_components',
    'build_reciprocal_pair_index',
    'build_time_term_outlier_mask',
    'build_time_term_design_matrix',
    'build_time_term_solver_system',
    'compose_time_term_applied_shifts',
    'compute_geometry_distance_m',
    'compute_time_term_moveout',
    'compute_time_term_robust_center_scale',
    'delay_to_applied_shift',
    'solve_time_term_sparse_least_squares',
    'solve_time_term_robust_least_squares',
    'summarize_time_term_design_matrix',
    'summarize_time_term_moveout',
    'summarize_time_term_sparse_solver_result',
    'validate_time_term_robust_options',
]
