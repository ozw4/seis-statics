"""Public result objects for first-break residual statics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seis_statics.residual.robust import (
    ResidualStaticRobustIterationSummary,
    ResidualStaticRobustOptions,
    ResidualStaticRobustSolveResult,
    ResidualStaticRobustStopReason,
)
from seis_statics.residual.solver import (
    ResidualStaticLsmrDiagnostics,
    ResidualStaticMinimumDataSummary,
    ResidualStaticObservationGraphSummary,
    ResidualStaticStabilizationOptions,
)
from seis_statics.residual.types import MoveoutModel


@dataclass(frozen=True)
class FirstBreakResidualStaticsResult:
    """Stable package-level result for first-break residual statics."""

    moveout_model: MoveoutModel
    source_id: np.ndarray
    receiver_id: np.ndarray

    intercept_s: float
    slowness_s_per_offset_unit: float | None
    source_delay_s: np.ndarray
    receiver_delay_s: np.ndarray

    moveout_model_time_s: np.ndarray
    estimated_trace_delay_s: np.ndarray
    modeled_pick_time_s: np.ndarray
    residual_s: np.ndarray
    residual_valid_mask: np.ndarray

    initial_used_pick_mask: np.ndarray
    used_pick_mask: np.ndarray
    rejected_pick_mask: np.ndarray
    rejected_iteration: np.ndarray

    diagnostics: ResidualStaticLsmrDiagnostics
    minimum_data: ResidualStaticMinimumDataSummary
    graph: ResidualStaticObservationGraphSummary
    stabilization_options: ResidualStaticStabilizationOptions
    robust_options: ResidualStaticRobustOptions
    robust_iteration_summaries: tuple[ResidualStaticRobustIterationSummary, ...]
    robust_stop_reason: ResidualStaticRobustStopReason

    n_initial_used_picks: int
    n_final_used_picks: int
    n_rejected_total: int
    n_observations: int
    n_model_parameters: int
    n_gauge_rows: int
    n_damping_rows: int
    max_abs_estimated_delay_s: float

    @property
    def robust_solve_result(self) -> ResidualStaticRobustSolveResult:
        """Return the package solver result retained for artifact adapters."""
        result = getattr(self, '_robust_solve_result', None)
        if result is None:
            raise AttributeError('robust_solve_result is not attached')
        return result


@dataclass(frozen=True)
class SourceReceiverStaticsMinimumDataSummary:
    """Minimum-data diagnostics for source/receiver lag decomposition."""

    n_used_observations: int
    n_sources: int
    n_receivers: int
    n_model_parameters: int
    n_effective_parameters: int

    source_used_observation_counts: np.ndarray
    receiver_used_observation_counts: np.ndarray
    underconstrained_source_ids: np.ndarray
    underconstrained_receiver_ids: np.ndarray

    n_zero_weight_observations: int
    rank_deficient_possible: bool


@dataclass(frozen=True)
class SourceReceiverStaticsResult:
    """Public result for source/receiver decomposition of lag observations."""

    source_unique_ids: np.ndarray
    receiver_unique_ids: np.ndarray
    source_delay_s: np.ndarray
    receiver_delay_s: np.ndarray

    trace_delay_s: np.ndarray
    applied_shift_s: np.ndarray
    residual_s: np.ndarray

    initial_used_mask: np.ndarray
    used_mask: np.ndarray
    rejected_mask: np.ndarray
    rejected_iteration: np.ndarray
    weight: np.ndarray

    diagnostics: ResidualStaticLsmrDiagnostics
    minimum_data: SourceReceiverStaticsMinimumDataSummary
    graph: ResidualStaticObservationGraphSummary
    stabilization_options: ResidualStaticStabilizationOptions
    robust_options: ResidualStaticRobustOptions
    robust_iteration_summaries: tuple[ResidualStaticRobustIterationSummary, ...]
    robust_stop_reason: ResidualStaticRobustStopReason

    n_initial_used_observations: int
    n_final_used_observations: int
    n_rejected_total: int
    n_observations: int
    n_model_parameters: int
    n_gauge_rows: int
    n_damping_rows: int
    max_abs_trace_delay_s: float


__all__ = [
    'FirstBreakResidualStaticsResult',
    'SourceReceiverStaticsMinimumDataSummary',
    'SourceReceiverStaticsResult',
]
