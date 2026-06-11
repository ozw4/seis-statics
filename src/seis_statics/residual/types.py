"""Lightweight shared types for residual static solver stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

MoveoutModel = Literal['linear_abs_offset', 'none']


@dataclass(frozen=True)
class ResidualStaticSolverInputs:
    picks_time_s_sorted: np.ndarray
    valid_pick_mask_sorted: np.ndarray
    pick_time_after_datum_s_sorted: np.ndarray
    datum_trace_shift_s_sorted: np.ndarray

    source_id_sorted: np.ndarray
    receiver_id_sorted: np.ndarray
    source_unique_ids: np.ndarray
    receiver_unique_ids: np.ndarray
    source_index_sorted: np.ndarray
    receiver_index_sorted: np.ndarray
    source_valid_pick_counts: np.ndarray
    receiver_valid_pick_counts: np.ndarray

    offset_sorted: np.ndarray | None
    abs_offset_sorted: np.ndarray | None

    key1_sorted: np.ndarray
    key2_sorted: np.ndarray
    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray

    dt: float
    n_traces: int
    n_samples: int
    key1_byte: int
    key2_byte: int
    source_id_byte: int
    receiver_id_byte: int
    offset_byte: int | None
    moveout_model: MoveoutModel

    input_file_id: str
    datum_source_file_id: str
    datum_job_id: str
    pick_source_kind: str
    metadata: dict[str, Any]


__all__ = ['MoveoutModel', 'ResidualStaticSolverInputs']
