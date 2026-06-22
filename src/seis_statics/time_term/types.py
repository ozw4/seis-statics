"""Pure data types for time-term static inversion services."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ORDER = 'trace_store_sorted'
SIGN_CONVENTION = (
    'pick_time_after_static_s_sorted is a resolved event time after any '
    'consumer-owned datum and residual shifts; time-term delays are estimated '
    'from pick_time_after_static_s_sorted minus positive moveout time'
)


@dataclass(frozen=True)
class TimeTermInversionInputs:
    """Resolved numeric arrays consumed by time-term moveout and inversion.

    Provenance, artifact locations, job identity, raw pick copies, and SEG-Y
    key metadata are owned by caller applications before they build this
    numerical contract.
    """

    n_traces: int
    dt: float

    valid_pick_mask_sorted: np.ndarray
    pick_time_after_static_s_sorted: np.ndarray

    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    n_nodes: int

    offset_sorted: np.ndarray | None

    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray

    order: str = ORDER
    sign_convention: str = SIGN_CONVENTION


__all__ = ['ORDER', 'SIGN_CONVENTION', 'TimeTermInversionInputs']
