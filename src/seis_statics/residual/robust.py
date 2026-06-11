"""Robust outlier rejection for residual static estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from seis_statics._validation import (
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
)
from seis_statics.residual.solver import (
    ResidualStaticLsmrOptions,
    ResidualStaticStabilizationOptions,
    ResidualStaticStabilizedLeastSquaresResult,
    solve_residual_static_stabilized_least_squares,
    validate_residual_static_stabilization_options,
    validate_residual_static_used_mask,
)
from seis_statics.residual.types import ResidualStaticSolverInputs

ResidualStaticRobustMethod = Literal['mad', 'sigma']
ResidualStaticRobustStopReason = Literal[
    'disabled',
    'converged',
    'max_iterations',
    'zero_scale',
]

ROBUST_SCALE_FLOOR_S = 1.0e-12


@dataclass(frozen=True)
class ResidualStaticRobustOptions:
    enabled: bool = True
    method: ResidualStaticRobustMethod = 'mad'
    max_iterations: int = 3
    threshold: float = 4.0
    min_used_fraction: float = 0.5


@dataclass(frozen=True)
class ResidualStaticRobustIterationSummary:
    iteration_index: int
    method: ResidualStaticRobustMethod

    n_used_before: int
    n_rejected_this_iteration: int
    n_used_after: int

    residual_center_s: float
    residual_scale_s: float
    residual_cutoff_s: float
    max_abs_centered_residual_s: float

    converged: bool
    stop_reason: ResidualStaticRobustStopReason | None


@dataclass(frozen=True)
class ResidualStaticRobustSolveResult:
    initial_solver_result: ResidualStaticStabilizedLeastSquaresResult
    final_solver_result: ResidualStaticStabilizedLeastSquaresResult

    robust_options: ResidualStaticRobustOptions
    stabilization_options: ResidualStaticStabilizationOptions

    initial_used_mask_sorted: np.ndarray
    final_used_mask_sorted: np.ndarray
    rejected_mask_sorted: np.ndarray
    rejected_iteration_sorted: np.ndarray

    iteration_summaries: tuple[ResidualStaticRobustIterationSummary, ...]
    stop_reason: ResidualStaticRobustStopReason

    n_initial_used_picks: int
    n_final_used_picks: int
    n_rejected_total: int


def validate_residual_static_robust_options(
    options: ResidualStaticRobustOptions,
) -> ResidualStaticRobustOptions:
    """Validate and normalize robust outlier-rejection options."""
    if not isinstance(options, ResidualStaticRobustOptions):
        raise ValueError('options must be a ResidualStaticRobustOptions instance')
    if not isinstance(options.enabled, bool):
        raise ValueError('enabled must be a bool')
    method = _validate_robust_method(options.method)
    return ResidualStaticRobustOptions(
        enabled=options.enabled,
        method=method,
        max_iterations=_coerce_positive_int(
            options.max_iterations,
            name='max_iterations',
        ),
        threshold=_coerce_positive_finite_float(
            options.threshold,
            name='threshold',
        ),
        min_used_fraction=_coerce_min_used_fraction(options.min_used_fraction),
    )


def compute_residual_static_robust_center_scale(
    residual_s: np.ndarray,
    *,
    method: ResidualStaticRobustMethod,
) -> tuple[float, float]:
    """Compute residual center and scale for robust outlier rejection."""
    robust_method = _validate_robust_method(method)
    residual = _coerce_1d_real_numeric_float64(residual_s, name='residual_s')
    if residual.size == 0:
        raise ValueError('residual_s must be non-empty')
    _validate_all_finite(residual, name='residual_s')

    if robust_method == 'mad':
        center = float(np.median(residual))
        raw_mad = float(np.median(np.abs(residual - center)))
        scale = 1.4826 * raw_mad
    else:
        center = float(np.mean(residual))
        scale = float(np.std(residual, ddof=0))
    return center, scale


def build_residual_static_outlier_mask(
    residual_s: np.ndarray,
    *,
    method: ResidualStaticRobustMethod,
    threshold: float,
) -> tuple[np.ndarray, float, float, float]:
    """Return the residual outlier mask plus center, scale, and cutoff."""
    cutoff_threshold = _coerce_positive_finite_float(threshold, name='threshold')
    residual = _coerce_1d_real_numeric_float64(residual_s, name='residual_s')
    center_s, scale_s = compute_residual_static_robust_center_scale(
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


def solve_residual_static_robust_least_squares(
    inputs: ResidualStaticSolverInputs,
    *,
    used_mask_sorted: np.ndarray | None = None,
    stabilization_options: ResidualStaticStabilizationOptions | None = None,
    robust_options: ResidualStaticRobustOptions | None = None,
    lsmr_options: ResidualStaticLsmrOptions | None = None,
) -> ResidualStaticRobustSolveResult:
    """Iteratively reject residual outliers and rerun the stabilized solver."""
    initial_used_mask = validate_residual_static_used_mask(inputs, used_mask_sorted)
    validated_robust_options = validate_residual_static_robust_options(
        robust_options or ResidualStaticRobustOptions()
    )
    validated_stabilization_options = validate_residual_static_stabilization_options(
        stabilization_options or ResidualStaticStabilizationOptions()
    )
    n_traces = int(initial_used_mask.shape[0])
    rejected_iteration_sorted = np.full(n_traces, -1, dtype=np.int64)

    if not validated_robust_options.enabled:
        solver_result = solve_residual_static_stabilized_least_squares(
            inputs,
            used_mask_sorted=initial_used_mask,
            stabilization_options=validated_stabilization_options,
            lsmr_options=lsmr_options,
        )
        return _build_robust_result(
            initial_solver_result=solver_result,
            final_solver_result=solver_result,
            robust_options=validated_robust_options,
            stabilization_options=validated_stabilization_options,
            initial_used_mask=initial_used_mask,
            rejected_iteration_sorted=rejected_iteration_sorted,
            iteration_summaries=(),
            stop_reason='disabled',
        )

    current_used_mask = initial_used_mask.copy()
    initial_solver_result: ResidualStaticStabilizedLeastSquaresResult | None = None
    final_solver_result: ResidualStaticStabilizedLeastSquaresResult | None = None
    stop_reason: ResidualStaticRobustStopReason | None = None
    iteration_summaries: list[ResidualStaticRobustIterationSummary] = []

    for iteration_index in range(validated_robust_options.max_iterations):
        solver_result = solve_residual_static_stabilized_least_squares(
            inputs,
            used_mask_sorted=current_used_mask,
            stabilization_options=validated_stabilization_options,
            lsmr_options=lsmr_options,
        )
        if initial_solver_result is None:
            initial_solver_result = solver_result

        residual_s = solver_result.model_evaluation.residual_s_sorted[
            current_used_mask
        ]
        (
            outlier_local,
            center_s,
            scale_s,
            cutoff_s,
        ) = build_residual_static_outlier_mask(
            residual_s,
            method=validated_robust_options.method,
            threshold=validated_robust_options.threshold,
        )
        n_used_before = int(np.count_nonzero(current_used_mask))
        max_abs_centered_residual_s = _max_abs_centered_residual(
            residual_s,
            center_s=center_s,
        )

        if scale_s <= ROBUST_SCALE_FLOOR_S:
            stop_reason = 'zero_scale'
            final_solver_result = solver_result
            iteration_summaries.append(
                ResidualStaticRobustIterationSummary(
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
                ResidualStaticRobustIterationSummary(
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

        current_used_indices = np.flatnonzero(current_used_mask)
        newly_rejected_indices = current_used_indices[outlier_local]
        proposed_used_mask = current_used_mask.copy()
        proposed_used_mask[newly_rejected_indices] = False
        _validate_min_used_fraction(
            initial_used_mask,
            proposed_used_mask,
            min_used_fraction=validated_robust_options.min_used_fraction,
        )

        current_used_mask = proposed_used_mask
        rejected_iteration_sorted[newly_rejected_indices] = iteration_index
        n_rejected = int(newly_rejected_indices.shape[0])
        summary_stop_reason: ResidualStaticRobustStopReason | None = None
        if iteration_index == validated_robust_options.max_iterations - 1:
            summary_stop_reason = 'max_iterations'
        iteration_summaries.append(
            ResidualStaticRobustIterationSummary(
                iteration_index=iteration_index,
                method=validated_robust_options.method,
                n_used_before=n_used_before,
                n_rejected_this_iteration=n_rejected,
                n_used_after=int(np.count_nonzero(current_used_mask)),
                residual_center_s=center_s,
                residual_scale_s=scale_s,
                residual_cutoff_s=cutoff_s,
                max_abs_centered_residual_s=max_abs_centered_residual_s,
                converged=False,
                stop_reason=summary_stop_reason,
            )
        )
    else:
        stop_reason = 'max_iterations'
        final_solver_result = solve_residual_static_stabilized_least_squares(
            inputs,
            used_mask_sorted=current_used_mask,
            stabilization_options=validated_stabilization_options,
            lsmr_options=lsmr_options,
        )

    if initial_solver_result is None or final_solver_result is None:
        raise RuntimeError('robust residual static solver did not produce a result')
    if stop_reason is None:
        raise RuntimeError('robust residual static solver did not set a stop reason')

    return _build_robust_result(
        initial_solver_result=initial_solver_result,
        final_solver_result=final_solver_result,
        robust_options=validated_robust_options,
        stabilization_options=validated_stabilization_options,
        initial_used_mask=initial_used_mask,
        rejected_iteration_sorted=rejected_iteration_sorted,
        iteration_summaries=tuple(iteration_summaries),
        stop_reason=stop_reason,
    )


def _build_robust_result(
    *,
    initial_solver_result: ResidualStaticStabilizedLeastSquaresResult,
    final_solver_result: ResidualStaticStabilizedLeastSquaresResult,
    robust_options: ResidualStaticRobustOptions,
    stabilization_options: ResidualStaticStabilizationOptions,
    initial_used_mask: np.ndarray,
    rejected_iteration_sorted: np.ndarray,
    iteration_summaries: tuple[ResidualStaticRobustIterationSummary, ...],
    stop_reason: ResidualStaticRobustStopReason,
) -> ResidualStaticRobustSolveResult:
    final_used_mask = np.ascontiguousarray(
        final_solver_result.used_mask_sorted,
        dtype=bool,
    )
    rejected_mask = np.ascontiguousarray(initial_used_mask & ~final_used_mask, dtype=bool)
    return ResidualStaticRobustSolveResult(
        initial_solver_result=initial_solver_result,
        final_solver_result=final_solver_result,
        robust_options=robust_options,
        stabilization_options=stabilization_options,
        initial_used_mask_sorted=np.ascontiguousarray(initial_used_mask, dtype=bool),
        final_used_mask_sorted=final_used_mask,
        rejected_mask_sorted=rejected_mask,
        rejected_iteration_sorted=np.ascontiguousarray(
            rejected_iteration_sorted,
            dtype=np.int64,
        ),
        iteration_summaries=iteration_summaries,
        stop_reason=stop_reason,
        n_initial_used_picks=int(np.count_nonzero(initial_used_mask)),
        n_final_used_picks=int(np.count_nonzero(final_used_mask)),
        n_rejected_total=int(np.count_nonzero(rejected_mask)),
    )


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
            'robust outlier rejection would drop used picks below min_used_fraction'
        )


def _max_abs_centered_residual(
    residual_s: np.ndarray,
    *,
    center_s: float,
) -> float:
    if residual_s.size == 0:
        return 0.0
    return float(np.max(np.abs(residual_s - center_s)))


def _validate_robust_method(value: object) -> ResidualStaticRobustMethod:
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
    'ResidualStaticRobustIterationSummary',
    'ResidualStaticRobustMethod',
    'ResidualStaticRobustOptions',
    'ResidualStaticRobustSolveResult',
    'ResidualStaticRobustStopReason',
    'build_residual_static_outlier_mask',
    'compute_residual_static_robust_center_scale',
    'solve_residual_static_robust_least_squares',
    'validate_residual_static_robust_options',
]
