"""Pure data types for time-term static inversion services."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ORDER = 'trace_store_sorted'
SIGN_CONVENTION = (
    'pick_time_after_static_s = pick_time_raw_s '
    '+ datum_trace_shift_s + residual_applied_shift_s; '
    'residual_applied_shift_s is an applied event-time shift, not an estimated delay'
)


@dataclass(frozen=True)
class TimeTermInversionInputs:
    """Trace-level arrays aligned to TraceStore sorted trace order."""

    n_traces: int
    n_samples: int
    dt: float
    key1_byte: int
    key2_byte: int

    pick_time_raw_s_sorted: np.ndarray
    valid_pick_mask_sorted: np.ndarray

    datum_trace_shift_s_sorted: np.ndarray
    residual_applied_shift_s_sorted: np.ndarray
    pick_time_after_static_s_sorted: np.ndarray

    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    n_nodes: int

    source_id_sorted: np.ndarray
    receiver_id_sorted: np.ndarray
    offset_sorted: np.ndarray | None

    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    source_depth_m_sorted: np.ndarray

    input_file_id: str
    pick_source_description: str
    datum_solution_path: Path | None
    residual_solution_path: Path | None
    linkage_artifact_path: Path | None

    order: str = ORDER
    sign_convention: str = SIGN_CONVENTION
    metadata: dict[str, object] = field(default_factory=dict)


__all__ = ['ORDER', 'SIGN_CONVENTION', 'TimeTermInversionInputs']
