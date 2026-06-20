"""Observation gate assignment for refraction layer configs."""

from __future__ import annotations

import numpy as np

from seis_statics.refraction.layer_config import (
    RefractionLayerConfig,
    RefractionLayerConfigLayer,
)
from seis_statics.refraction.types import RefractionLayerObservationMasks
from seis_statics.validation import (
    coerce_1d_bool_array,
    coerce_1d_real_numeric_float64,
    coerce_1d_string_array,
)


OK_REJECTION_REASON = ''
INVALID_OFFSET_REJECTION_REASON = 'invalid_offset'
INVALID_OBSERVATION_REJECTION_REASON = 'invalid_observation'
OUTSIDE_LAYER_GATE_REJECTION_REASON = 'outside_layer_offset_gate'
ALREADY_ASSIGNED_REJECTION_REASON = 'already_assigned_to_shallower_layer'


def build_refraction_layer_observation_masks(
    *,
    layer_config: RefractionLayerConfig,
    offset_m_sorted: object,
    valid_observation_mask_sorted: object | None = None,
    rejection_reason_sorted: object | None = None,
) -> RefractionLayerObservationMasks:
    """Assign sorted observations to refraction layers by absolute offset gates."""
    if not isinstance(layer_config, RefractionLayerConfig):
        raise TypeError('layer_config must be RefractionLayerConfig')
    offset = coerce_1d_real_numeric_float64(offset_m_sorted, name='offset_m_sorted')
    valid_observation_mask = _valid_observation_mask(
        valid_observation_mask_sorted,
        shape=offset.shape,
    )
    base_rejection_reason = _base_rejection_reason(
        rejection_reason_sorted,
        shape=offset.shape,
    )

    offset_abs = np.abs(offset)
    finite_offset_mask = np.isfinite(offset_abs)
    assigned_mask = np.zeros(offset.shape, dtype=bool)
    used_masks: dict[str, np.ndarray] = {}
    rejection_reasons: dict[str, np.ndarray] = {}
    candidate_counts: dict[str, int] = {}
    observation_counts: dict[str, int] = {}

    for layer in layer_config.layers:
        candidate_mask = finite_offset_mask & _layer_candidate_mask(layer, offset_abs)
        used_mask = candidate_mask & valid_observation_mask & ~assigned_mask
        assigned_mask |= used_mask

        kind = layer.kind
        used_masks[kind] = np.ascontiguousarray(used_mask, dtype=bool)
        rejection_reasons[kind] = _layer_rejection_reason(
            candidate_mask=candidate_mask,
            used_mask=used_mask,
            finite_offset_mask=finite_offset_mask,
            valid_observation_mask=valid_observation_mask,
            base_rejection_reason=base_rejection_reason,
        )
        candidate_counts[kind] = int(np.count_nonzero(candidate_mask))
        observation_counts[kind] = int(np.count_nonzero(used_mask))

    return RefractionLayerObservationMasks(
        layer_kind=np.asarray([layer.kind for layer in layer_config.layers], dtype=str),
        layer_enabled=np.ones(len(layer_config.layers), dtype=bool),
        layer_min_offset_m=_layer_bound_array(
            [layer.min_offset_m for layer in layer_config.layers],
            none_value=-np.inf,
        ),
        layer_max_offset_m=_layer_bound_array(
            [layer.max_offset_m for layer in layer_config.layers],
            none_value=np.inf,
        ),
        layer_used_mask_sorted=used_masks,
        layer_rejection_reason_sorted=rejection_reasons,
        layer_candidate_count=candidate_counts,
        layer_observation_count=observation_counts,
    )


def refraction_layer_observation_qc(
    layer_masks: RefractionLayerObservationMasks,
) -> dict[str, object]:
    """Return stable QC counts for layer observation masks."""
    candidate_count = {
        str(kind): int(layer_masks.layer_candidate_count[str(kind)])
        for kind in layer_masks.layer_kind
    }
    observation_count = {
        str(kind): int(layer_masks.layer_observation_count[str(kind)])
        for kind in layer_masks.layer_kind
    }
    total_candidate_count = sum(candidate_count.values())
    total_observation_count = sum(observation_count.values())
    return {
        'layer_count': int(len(layer_masks.layer_kind)),
        'layer_candidate_count': candidate_count,
        'layer_observation_count': observation_count,
        'total_layer_candidate_count': int(total_candidate_count),
        'total_layer_observation_count': int(total_observation_count),
    }


def _valid_observation_mask(values: object | None, *, shape: tuple[int, ...]) -> np.ndarray:
    if values is None:
        return np.ones(shape, dtype=bool)
    return coerce_1d_bool_array(
        values,
        name='valid_observation_mask_sorted',
        expected_shape=shape,
    )


def _base_rejection_reason(values: object | None, *, shape: tuple[int, ...]) -> np.ndarray:
    if values is None:
        return np.full(shape, INVALID_OBSERVATION_REJECTION_REASON, dtype=str)
    return coerce_1d_string_array(
        values,
        name='rejection_reason_sorted',
        expected_shape=shape,
        reject_object_dtype=False,
    )


def _layer_candidate_mask(
    layer: RefractionLayerConfigLayer,
    offset_abs_m_sorted: np.ndarray,
) -> np.ndarray:
    mask = np.ones(offset_abs_m_sorted.shape, dtype=bool)
    if layer.min_offset_m is not None:
        mask &= offset_abs_m_sorted >= layer.min_offset_m
    if layer.max_offset_m is not None:
        mask &= offset_abs_m_sorted < layer.max_offset_m
    return mask


def _layer_rejection_reason(
    *,
    candidate_mask: np.ndarray,
    used_mask: np.ndarray,
    finite_offset_mask: np.ndarray,
    valid_observation_mask: np.ndarray,
    base_rejection_reason: np.ndarray,
) -> np.ndarray:
    reason = np.full(candidate_mask.shape, OK_REJECTION_REASON, dtype='<U64')
    reason[~finite_offset_mask] = INVALID_OFFSET_REJECTION_REASON
    reason[finite_offset_mask & ~candidate_mask] = OUTSIDE_LAYER_GATE_REJECTION_REASON
    invalid_candidate_mask = candidate_mask & ~valid_observation_mask
    reason[invalid_candidate_mask] = _candidate_invalid_reason(
        base_rejection_reason[invalid_candidate_mask],
    )
    assigned_elsewhere_mask = candidate_mask & valid_observation_mask & ~used_mask
    reason[assigned_elsewhere_mask] = ALREADY_ASSIGNED_REJECTION_REASON
    reason[used_mask] = OK_REJECTION_REASON
    return np.ascontiguousarray(reason)


def _candidate_invalid_reason(base_reason: np.ndarray) -> np.ndarray:
    reason = np.asarray(base_reason, dtype='<U64')
    empty_mask = reason == OK_REJECTION_REASON
    if np.any(empty_mask):
        reason = reason.copy()
        reason[empty_mask] = INVALID_OBSERVATION_REJECTION_REASON
    return reason


def _layer_bound_array(values: list[float | None], *, none_value: float) -> np.ndarray:
    return np.asarray(
        [none_value if value is None else value for value in values],
        dtype=np.float64,
    )


__all__ = [
    'ALREADY_ASSIGNED_REJECTION_REASON',
    'INVALID_OBSERVATION_REJECTION_REASON',
    'INVALID_OFFSET_REJECTION_REASON',
    'OK_REJECTION_REASON',
    'OUTSIDE_LAYER_GATE_REJECTION_REASON',
    'build_refraction_layer_observation_masks',
    'refraction_layer_observation_qc',
]
