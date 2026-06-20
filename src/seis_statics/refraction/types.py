"""Dependency-light types for refraction static workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np


BedrockVelocityMode = Literal['solve_global', 'fixed_global', 'solve_cell']
RefractionFirstLayerMode = Literal['constant', 'estimate_direct_arrival']
RefractionLayerKind = Literal['v2_t1', 'v3_t2', 'vsub_t3']
RefractionLayerVelocityMode = BedrockVelocityMode
RefractionSourceDepthMode = Literal['none', 'weathering_velocity_time']
RefractionSourceDepthStatus = Literal[
    'ok',
    'missing_source_depth',
    'invalid_source_depth',
    'inconsistent_source_depth',
    'exceeds_max_abs_source_depth',
    'inactive_source_endpoint',
]
RefractionUpholeStatus = Literal[
    'ok',
    'missing_uphole_time',
    'invalid_uphole_time',
    'inconsistent_uphole_time',
    'exceeds_max_abs_uphole_time',
    'inactive_source_endpoint',
]
RefractionFieldCorrectionComponentName = Literal[
    'source_depth_shift_s',
    'uphole_shift_s',
    'manual_static_shift_s',
]
REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES: Final[
    tuple[RefractionFieldCorrectionComponentName, ...]
] = (
    'source_depth_shift_s',
    'uphole_shift_s',
    'manual_static_shift_s',
)


@dataclass(frozen=True)
class ResolvedRefractionFirstLayer:
    """Resolved V1/first-layer velocity used by downstream refraction statics."""

    mode: RefractionFirstLayerMode
    weathering_velocity_m_s: float
    status: str
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionEndpointTable:
    node_id: np.ndarray
    endpoint_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    elevation_m: np.ndarray
    kind: np.ndarray
    pick_count: np.ndarray


@dataclass(frozen=True)
class RefractionLayerObservationMasks:
    """Per-layer sorted-observation masks for multi-layer refraction branches."""

    layer_kind: np.ndarray
    layer_enabled: np.ndarray
    layer_min_offset_m: np.ndarray
    layer_max_offset_m: np.ndarray
    layer_used_mask_sorted: dict[str, np.ndarray]
    layer_rejection_reason_sorted: dict[str, np.ndarray]
    layer_candidate_count: dict[str, int]
    layer_observation_count: dict[str, int]


@dataclass(frozen=True)
class RefractionSourceDepthResult:
    """Resolved source-depth values aggregated to source endpoints."""

    source_endpoint_key: np.ndarray
    source_endpoint_id: np.ndarray
    source_node_id: np.ndarray
    source_depth_m: np.ndarray
    source_depth_status: np.ndarray
    source_depth_pick_count: np.ndarray
    source_depth_trace_count: np.ndarray
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionUpholeResult:
    """Resolved uphole-time values aggregated to source endpoints."""

    source_endpoint_key: np.ndarray
    source_endpoint_id: np.ndarray
    source_node_id: np.ndarray
    uphole_time_s: np.ndarray
    uphole_status: np.ndarray
    uphole_pick_count: np.ndarray
    uphole_trace_count: np.ndarray
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionEndpointFieldCorrectionResult:
    """Endpoint-level source-depth, uphole, and manual static corrections."""

    endpoint_kind: np.ndarray
    endpoint_key: np.ndarray
    endpoint_id: np.ndarray
    node_id: np.ndarray
    component_shift_s: dict[RefractionFieldCorrectionComponentName, np.ndarray]
    component_status: dict[RefractionFieldCorrectionComponentName, np.ndarray]
    total_field_shift_s: np.ndarray
    field_static_status: np.ndarray
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionStaticInputModel:
    file_id: str
    n_traces: int

    sorted_trace_index: np.ndarray
    pick_time_s_sorted: np.ndarray
    valid_pick_mask_sorted: np.ndarray
    valid_observation_mask_sorted: np.ndarray

    source_id_sorted: np.ndarray
    receiver_id_sorted: np.ndarray

    source_x_m_sorted: np.ndarray
    source_y_m_sorted: np.ndarray
    receiver_x_m_sorted: np.ndarray
    receiver_y_m_sorted: np.ndarray

    source_elevation_m_sorted: np.ndarray
    receiver_elevation_m_sorted: np.ndarray
    source_depth_m_sorted: np.ndarray | None

    geometry_distance_m_sorted: np.ndarray
    offset_m_sorted: np.ndarray | None
    distance_m_sorted: np.ndarray

    source_endpoint_key_sorted: np.ndarray
    receiver_endpoint_key_sorted: np.ndarray

    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    node_x_m: np.ndarray
    node_y_m: np.ndarray
    node_elevation_m: np.ndarray
    node_kind: np.ndarray

    rejection_reason_sorted: np.ndarray
    qc: dict[str, Any]
    endpoint_table: RefractionEndpointTable
    metadata: dict[str, Any]
    layer_observation_masks: RefractionLayerObservationMasks | None = None
    source_endpoint_id_sorted: np.ndarray | None = None
    receiver_endpoint_id_sorted: np.ndarray | None = None


__all__ = [
    'BedrockVelocityMode',
    'REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES',
    'RefractionEndpointTable',
    'RefractionEndpointFieldCorrectionResult',
    'RefractionFieldCorrectionComponentName',
    'RefractionFirstLayerMode',
    'RefractionLayerKind',
    'RefractionLayerObservationMasks',
    'RefractionLayerVelocityMode',
    'RefractionSourceDepthResult',
    'RefractionSourceDepthMode',
    'RefractionSourceDepthStatus',
    'RefractionStaticInputModel',
    'RefractionUpholeResult',
    'RefractionUpholeStatus',
    'ResolvedRefractionFirstLayer',
]
