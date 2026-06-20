"""Robust outlier rejection for node time-term estimation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from seis_statics._validation import (
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics.time_term.design_matrix import TimeTermDesignMatrix
from seis_statics.time_term.sparse_solver import (
    TimeTermSparseSolverOptions,
    TimeTermSparseSolverResult,
    solve_time_term_sparse_least_squares,
)

TimeTermRobustMethod = Literal['mad', 'sigma']
TimeTermRobustStopReason = Literal[
    'disabled',
    'converged',
    'max_iterations',
    'zero_scale',
]

ROBUST_SCALE_FLOOR_S = 1.0e-12


@dataclass(frozen=True)
class TimeTermRobustOptions:
    enabled: bool = True
    method: TimeTermRobustMethod = 'mad'
    max_iterations: int = 3
    threshold: float = 4.0
    min_used_fraction: float = 0.5
    min_used_count: int = 1


@dataclass(frozen=True)
class TimeTermRobustIterationSummary:
    iteration_index: int
    method: TimeTermRobustMethod

    n_used_before: int
    n_rejected_this_iteration: int
    n_used_after: int

    residual_center_s: float
    residual_scale_s: float
    residual_cutoff_s: float
    max_abs_centered_residual_s: float

    converged: bool
    stop_reason: TimeTermRobustStopReason | None


@dataclass(frozen=True)
class TimeTermRobustSolveResult:
    initial_solver_result: TimeTermSparseSolverResult
    final_solver_result: TimeTermSparseSolverResult

    robust_options: TimeTermRobustOptions
    sparse_solver_options: TimeTermSparseSolverOptions | None

    initial_used_trace_mask_sorted: np.ndarray
    final_used_trace_mask_sorted: np.ndarray
    rejected_trace_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    iteration_summaries: tuple[TimeTermRobustIterationSummary, ...]
    stop_reason: TimeTermRobustStopReason

    n_initial_used_traces: int
    n_final_used_traces: int
    n_rejected_total: int


def validate_time_term_robust_options(
    options: TimeTermRobustOptions,
) -> TimeTermRobustOptions:
    """Validate and normalize robust outlier-rejection options."""
    if not isinstance(options, TimeTermRobustOptions):
        raise ValueError('options must be a TimeTermRobustOptions instance')
    if not isinstance(options.enabled, bool):
        raise ValueError('enabled must be a bool')
    return TimeTermRobustOptions(
        enabled=options.enabled,
        method=_validate_robust_method(options.method),
        max_iterations=_coerce_positive_int(
            options.max_iterations,
            name='max_iterations',
        ),
        threshold=_coerce_positive_finite_float(
            options.threshold,
            name='threshold',
        ),
        min_used_fraction=_coerce_min_used_fraction(options.min_used_fraction),
        min_used_count=_coerce_positive_int(
            options.min_used_count,
            name='min_used_count',
        ),
    )


def compute_time_term_robust_center_scale(
    residual_s: np.ndarray,
    *,
    method: TimeTermRobustMethod,
) -> tuple[float, float]:
    """Compute residual center and scale for robust outlier rejection."""
    residual = _coerce_1d_real_numeric_float64(residual_s, name='residual_s')
    if residual.size == 0:
        raise ValueError('residual_s must be non-empty')
    _validate_all_finite(residual, name='residual_s')

    robust_method = _validate_robust_method(method)
    if robust_method == 'mad':
        center = float(np.median(residual))
        raw_mad = float(np.median(np.abs(residual - center)))
        scale = 1.4826 * raw_mad
    else:
        center = float(np.mean(residual))
        scale = float(np.std(residual, ddof=0))
    return center, scale


def build_time_term_outlier_mask(
    residual_s: np.ndarray,
    *,
    method: TimeTermRobustMethod,
    threshold: float,
) -> tuple[np.ndarray, float, float, float]:
    """Return the residual outlier mask plus center, scale, and cutoff."""
    cutoff_threshold = _coerce_positive_finite_float(threshold, name='threshold')
    residual = _coerce_1d_real_numeric_float64(residual_s, name='residual_s')
    center_s, scale_s = compute_time_term_robust_center_scale(
        residual,
        method=method,
    )
    cutoff_s = cutoff_threshold * scale_s
    if scale_s <= ROBUST_SCALE_FLOOR_S:
        outlier_mask = np.zeros(residual.shape, dtype=bool)
    else:
        outlier_mask = np.abs(residual - center_s) > cutoff_s
    return (
        np.ascontiguousarray(outlier_mask, dtype=bool),
        center_s,
        scale_s,
        cutoff_s,
    )


def solve_time_term_robust_least_squares(
    design: TimeTermDesignMatrix,
    *,
    sparse_solver_options: TimeTermSparseSolverOptions | None = None,
    robust_options: TimeTermRobustOptions | None = None,
) -> TimeTermRobustSolveResult:
    """Iteratively reject time-term residual outliers and rerun the solver."""
    validated_robust_options = validate_time_term_robust_options(
        robust_options or TimeTermRobustOptions()
    )
    n_traces = _coerce_positive_int(design.n_traces, name='design.n_traces')
    rejected_iteration_sorted = np.full(n_traces, -1, dtype=np.int64)

    if not validated_robust_options.enabled:
        solver_result = solve_time_term_sparse_least_squares(
            design,
            options=sparse_solver_options,
        )
        return _build_robust_result(
            initial_solver_result=solver_result,
            final_solver_result=solver_result,
            robust_options=validated_robust_options,
            sparse_solver_options=sparse_solver_options,
            initial_used_mask=solver_result.used_trace_mask_sorted,
            rejected_iteration_sorted=rejected_iteration_sorted,
            iteration_summaries=(),
            stop_reason='disabled',
        )

    current_design = design
    initial_used_mask: np.ndarray | None = None
    initial_solver_result: TimeTermSparseSolverResult | None = None
    final_solver_result: TimeTermSparseSolverResult | None = None
    stop_reason: TimeTermRobustStopReason | None = None
    iteration_summaries: list[TimeTermRobustIterationSummary] = []

    for iteration_index in range(validated_robust_options.max_iterations):
        solver_result = solve_time_term_sparse_least_squares(
            current_design,
            options=sparse_solver_options,
        )
        if initial_solver_result is None:
            initial_solver_result = solver_result
            initial_used_mask = np.ascontiguousarray(
                solver_result.used_trace_mask_sorted,
                dtype=bool,
            )

        residual_s = solver_result.row_residual_after_s
        (
            outlier_local,
            center_s,
            scale_s,
            cutoff_s,
        ) = build_time_term_outlier_mask(
            residual_s,
            method=validated_robust_options.method,
            threshold=validated_robust_options.threshold,
        )
        n_used_before = int(np.count_nonzero(solver_result.used_trace_mask_sorted))
        max_abs_centered_residual_s = _max_abs_centered_residual(
            residual_s,
            center_s=center_s,
        )

        if scale_s <= ROBUST_SCALE_FLOOR_S:
            stop_reason = 'zero_scale'
            final_solver_result = solver_result
            iteration_summaries.append(
                TimeTermRobustIterationSummary(
                    iteration_index=iteration_index,
                    method=validated_robust_options.method,
                    n_used_before=n_used_before,
                    n_rejected_this_iteration=0,
                    n_used_after=n_used_before,
                    residual_center_s=center_s,
                    residual_scale_s=scale_s,
                    residual_cutoff_s=cutoff_s,
                    max_abs_centered_residual_s=max_abs_centered_residual_s,
                    converged=False,
                    stop_reason=stop_reason,
                )
            )
            break

        if not np.any(outlier_local):
            stop_reason = 'converged'
            final_solver_result = solver_result
            iteration_summaries.append(
                TimeTermRobustIterationSummary(
                    iteration_index=iteration_index,
                    method=validated_robust_options.method,
                    n_used_before=n_used_before,
                    n_rejected_this_iteration=0,
                    n_used_after=n_used_before,
                    residual_center_s=center_s,
                    residual_scale_s=scale_s,
                    residual_cutoff_s=cutoff_s,
                    max_abs_centered_residual_s=max_abs_centered_residual_s,
                    converged=True,
                    stop_reason=stop_reason,
                )
            )
            break

        newly_rejected_indices = solver_result.row_trace_index_sorted[outlier_local]
        proposed_used_mask = solver_result.used_trace_mask_sorted.copy()
        proposed_used_mask[newly_rejected_indices] = False
        if initial_used_mask is None:
            raise RuntimeError('robust time-term solver did not record initial mask')
        _validate_min_used_fraction(
            initial_used_mask,
            proposed_used_mask,
            min_used_fraction=validated_robust_options.min_used_fraction,
        )
        _validate_min_used_count(
            proposed_used_mask,
            min_used_count=validated_robust_options.min_used_count,
        )

        rejected_iteration_sorted[newly_rejected_indices] = iteration_index
        n_rejected = int(newly_rejected_indices.shape[0])
        summary_stop_reason: TimeTermRobustStopReason | None = None
        if iteration_index == validated_robust_options.max_iterations - 1:
            summary_stop_reason = 'max_iterations'
        iteration_summaries.append(
            TimeTermRobustIterationSummary(
                iteration_index=iteration_index,
                method=validated_robust_options.method,
                n_used_before=n_used_before,
                n_rejected_this_iteration=n_rejected,
                n_used_after=int(np.count_nonzero(proposed_used_mask)),
                residual_center_s=center_s,
                residual_scale_s=scale_s,
                residual_cutoff_s=cutoff_s,
                max_abs_centered_residual_s=max_abs_centered_residual_s,
                converged=False,
                stop_reason=summary_stop_reason,
            )
        )
        current_design = _design_with_used_mask(current_design, proposed_used_mask)
    else:
        stop_reason = 'max_iterations'
        final_solver_result = solve_time_term_sparse_least_squares(
            current_design,
            options=sparse_solver_options,
        )

    if (
        initial_solver_result is None
        or final_solver_result is None
        or initial_used_mask is None
    ):
        raise RuntimeError('robust time-term solver did not produce a result')
    if stop_reason is None:
        raise RuntimeError('robust time-term solver did not set a stop reason')

    return _build_robust_result(
        initial_solver_result=initial_solver_result,
        final_solver_result=final_solver_result,
        robust_options=validated_robust_options,
        sparse_solver_options=sparse_solver_options,
        initial_used_mask=initial_used_mask,
        rejected_iteration_sorted=rejected_iteration_sorted,
        iteration_summaries=tuple(iteration_summaries),
        stop_reason=stop_reason,
    )


def _build_robust_result(
    *,
    initial_solver_result: TimeTermSparseSolverResult,
    final_solver_result: TimeTermSparseSolverResult,
    robust_options: TimeTermRobustOptions,
    sparse_solver_options: TimeTermSparseSolverOptions | None,
    initial_used_mask: np.ndarray,
    rejected_iteration_sorted: np.ndarray,
    iteration_summaries: tuple[TimeTermRobustIterationSummary, ...],
    stop_reason: TimeTermRobustStopReason,
) -> TimeTermRobustSolveResult:
    final_used_mask = np.ascontiguousarray(
        final_solver_result.used_trace_mask_sorted,
        dtype=bool,
    )
    masked_final_solver_result = _mask_unused_trace_delays(
        final_solver_result,
        used_trace_mask=final_used_mask,
    )
    rejected_mask = np.ascontiguousarray(initial_used_mask & ~final_used_mask, dtype=bool)
    return TimeTermRobustSolveResult(
        initial_solver_result=initial_solver_result,
        final_solver_result=masked_final_solver_result,
        robust_options=robust_options,
        sparse_solver_options=sparse_solver_options,
        initial_used_trace_mask_sorted=np.ascontiguousarray(
            initial_used_mask,
            dtype=bool,
        ),
        final_used_trace_mask_sorted=final_used_mask,
        rejected_trace_mask_sorted=rejected_mask,
        rejected_iteration_sorted=np.ascontiguousarray(
            rejected_iteration_sorted,
            dtype=np.int64,
        ),
        iteration_summaries=iteration_summaries,
        stop_reason=stop_reason,
        n_initial_used_traces=int(np.count_nonzero(initial_used_mask)),
        n_final_used_traces=int(np.count_nonzero(final_used_mask)),
        n_rejected_total=int(np.count_nonzero(rejected_mask)),
    )


def _mask_unused_trace_delays(
    solver_result: TimeTermSparseSolverResult,
    *,
    used_trace_mask: np.ndarray,
) -> TimeTermSparseSolverResult:
    if np.all(used_trace_mask):
        return solver_result
    estimated_delay = _coerce_1d_real_numeric_float64(
        solver_result.estimated_trace_time_term_delay_s_sorted,
        name='estimated_trace_time_term_delay_s_sorted',
        expected_shape=used_trace_mask.shape,
    ).copy()
    estimated_delay[~used_trace_mask] = np.nan
    return replace(
        solver_result,
        estimated_trace_time_term_delay_s_sorted=np.ascontiguousarray(
            estimated_delay,
            dtype=np.float64,
        ),
    )


def _design_with_used_mask(
    design: TimeTermDesignMatrix,
    used_trace_mask: np.ndarray,
) -> TimeTermDesignMatrix:
    mask = _validate_used_trace_mask(used_trace_mask, n_traces=design.n_traces)
    row_trace_index = np.ascontiguousarray(np.flatnonzero(mask), dtype=np.int64)
    n_observations = int(row_trace_index.shape[0])
    if n_observations <= 0:
        raise ValueError('at least one usable time-term observation is required')

    trace_to_row_index = np.full(int(design.n_traces), -1, dtype=np.int64)
    trace_to_row_index[row_trace_index] = np.arange(n_observations, dtype=np.int64)
    source_node_id = np.asarray(design.source_node_id_sorted, dtype=np.int64)
    receiver_node_id = np.asarray(design.receiver_node_id_sorted, dtype=np.int64)
    row_source_node_id = np.ascontiguousarray(
        source_node_id[row_trace_index],
        dtype=np.int64,
    )
    row_receiver_node_id = np.ascontiguousarray(
        receiver_node_id[row_trace_index],
        dtype=np.int64,
    )
    row_data_s = _rows_from_original_design(
        design.row_data_s,
        design.row_trace_index_sorted,
        row_trace_index,
        name='row_data_s',
    )
    row_pick_time_after_static_s = _rows_from_original_design(
        design.row_pick_time_after_static_s,
        design.row_trace_index_sorted,
        row_trace_index,
        name='row_pick_time_after_static_s',
    )
    row_moveout_time_s = _rows_from_original_design(
        design.row_moveout_time_s,
        design.row_trace_index_sorted,
        row_trace_index,
        name='row_moveout_time_s',
    )
    source_count = np.bincount(row_source_node_id, minlength=design.n_nodes).astype(
        np.int64,
        copy=False,
    )
    receiver_count = np.bincount(row_receiver_node_id, minlength=design.n_nodes).astype(
        np.int64,
        copy=False,
    )

    return TimeTermDesignMatrix(
        matrix=design.matrix[design.trace_to_row_index_sorted[row_trace_index]],
        data_s=row_data_s,
        n_traces=int(design.n_traces),
        n_observations=n_observations,
        n_nodes=int(design.n_nodes),
        used_trace_mask_sorted=mask,
        row_trace_index_sorted=row_trace_index,
        trace_to_row_index_sorted=trace_to_row_index,
        source_node_id_sorted=np.ascontiguousarray(source_node_id, dtype=np.int64),
        receiver_node_id_sorted=np.ascontiguousarray(receiver_node_id, dtype=np.int64),
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
        row_pick_time_after_static_s=row_pick_time_after_static_s,
        row_moveout_time_s=row_moveout_time_s,
        row_data_s=row_data_s,
        source_observation_count_by_node=np.ascontiguousarray(
            source_count,
            dtype=np.int64,
        ),
        receiver_observation_count_by_node=np.ascontiguousarray(
            receiver_count,
            dtype=np.int64,
        ),
        total_observation_count_by_node=np.ascontiguousarray(
            source_count + receiver_count,
            dtype=np.int64,
        ),
    )


def _rows_from_original_design(
    values: np.ndarray,
    old_row_trace_index: np.ndarray,
    new_row_trace_index: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    old_trace_to_row = {
        int(trace_index): row_index
        for row_index, trace_index in enumerate(old_row_trace_index)
    }
    row_indices = np.asarray(
        [old_trace_to_row[int(trace_index)] for trace_index in new_row_trace_index],
        dtype=np.int64,
    )
    arr = _coerce_1d_real_numeric_float64(values, name=name)
    return np.ascontiguousarray(arr[row_indices], dtype=np.float64)


def _validate_used_trace_mask(used_trace_mask: np.ndarray, *, n_traces: int) -> np.ndarray:
    mask = np.asarray(used_trace_mask)
    if mask.shape != (int(n_traces),):
        raise ValueError('used_trace_mask_sorted shape mismatch')
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError('used_trace_mask_sorted must have bool dtype')
    return np.ascontiguousarray(mask, dtype=bool)


def _validate_min_used_fraction(
    initial_used_mask: np.ndarray,
    proposed_used_mask: np.ndarray,
    *,
    min_used_fraction: float,
) -> None:
    min_allowed_used = int(
        np.ceil(float(np.count_nonzero(initial_used_mask)) * min_used_fraction)
    )
    if int(np.count_nonzero(proposed_used_mask)) < min_allowed_used:
        raise ValueError(
            'robust outlier rejection would drop used traces below min_used_fraction'
        )


def _validate_min_used_count(
    proposed_used_mask: np.ndarray,
    *,
    min_used_count: int,
) -> None:
    proposed_count = int(np.count_nonzero(proposed_used_mask))
    if proposed_count >= min_used_count:
        return
    if min_used_count <= 1:
        raise ValueError('robust outlier rejection would drop all used traces')
    raise ValueError(
        'robust outlier rejection would drop used traces below min_used_count'
    )


def _max_abs_centered_residual(
    residual_s: np.ndarray,
    *,
    center_s: float,
) -> float:
    if residual_s.size == 0:
        return 0.0
    return float(np.max(np.abs(residual_s - center_s)))


def _validate_robust_method(value: object) -> TimeTermRobustMethod:
    if value == 'mad':
        return 'mad'
    if value == 'sigma':
        return 'sigma'
    raise ValueError('method must be mad or sigma')


def _coerce_min_used_fraction(value: object) -> float:
    out = _coerce_positive_finite_float(value, name='min_used_fraction')
    if out > 1.0:
        raise ValueError('min_used_fraction must be less than or equal to 1')
    return out


def _validate_all_finite(values: np.ndarray, *, name: str) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f'{name} must contain only finite values')


__all__ = [
    'ROBUST_SCALE_FLOOR_S',
    'TimeTermRobustIterationSummary',
    'TimeTermRobustMethod',
    'TimeTermRobustOptions',
    'TimeTermRobustSolveResult',
    'TimeTermRobustStopReason',
    'build_time_term_outlier_mask',
    'compute_time_term_robust_center_scale',
    'solve_time_term_robust_least_squares',
    'validate_time_term_robust_options',
]
