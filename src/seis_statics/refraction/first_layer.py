"""Resolved first-layer/V1 helpers for refraction statics."""

from __future__ import annotations

from typing import Any

import numpy as np

from seis_statics.refraction.types import ResolvedRefractionFirstLayer


def resolve_weathering_velocity_m_s(
    *,
    model: Any,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    name: str = 'model.resolved_weathering_velocity_m_s',
) -> float:
    """Return the V1 value that downstream refraction statics must use."""
    if resolved_first_layer is None:
        return _positive_finite(getattr(model, 'resolved_weathering_velocity_m_s', None), name=name)

    expected_mode = getattr(model, 'first_layer_mode', None)
    if expected_mode is not None and resolved_first_layer.mode != expected_mode:
        raise ValueError('resolved first-layer mode does not match model')

    velocity = resolved_first_layer_weathering_velocity_m_s(resolved_first_layer)
    model_velocity = _model_resolved_weathering_velocity_or_none(model)
    if model_velocity is not None and not _velocities_close(velocity, model_velocity):
        raise ValueError('resolved first-layer weathering velocity does not match model')
    return velocity


def normalize_refraction_first_layer_request(model: Any) -> ResolvedRefractionFirstLayer:
    """Resolve a schema-like first-layer/V1 block to the downstream V1."""
    mode = getattr(model, 'first_layer_mode')
    velocity = float(getattr(model, 'resolved_weathering_velocity_m_s'))
    status = 'estimated' if mode == 'estimate_direct_arrival' else 'resolved_constant'
    return ResolvedRefractionFirstLayer(
        mode=mode,
        weathering_velocity_m_s=velocity,
        status=status,
        qc={
            'v1_mode': mode,
            'weathering_velocity_m_s': velocity,
            'resolved_weathering_velocity_m_s': velocity,
            'v1_status': status,
        },
    )


def validate_resolved_first_layer_velocity_match(
    *,
    weathering_velocity_m_s: Any,
    resolved_first_layer: ResolvedRefractionFirstLayer | None = None,
    name: str,
) -> float:
    """Validate a downstream result velocity against the resolved V1, if supplied."""
    velocity = _positive_finite(weathering_velocity_m_s, name=name)
    if resolved_first_layer is None:
        return velocity

    resolved_velocity = resolved_first_layer_weathering_velocity_m_s(resolved_first_layer)
    if not _velocities_close(resolved_velocity, velocity):
        raise ValueError('resolved first-layer weathering velocity does not match ' f'{name}')
    return resolved_velocity


def resolved_first_layer_weathering_velocity_m_s(
    resolved_first_layer: ResolvedRefractionFirstLayer,
) -> float:
    """Coerce a resolved first-layer dataclass velocity."""
    return _positive_finite(
        resolved_first_layer.weathering_velocity_m_s,
        name='resolved_first_layer.weathering_velocity_m_s',
    )


def _model_resolved_weathering_velocity_or_none(model: Any) -> float | None:
    try:
        value = getattr(model, 'resolved_weathering_velocity_m_s', None)
    except ValueError:
        return None
    if value is None:
        return None
    return _positive_finite(value, name='model.resolved_weathering_velocity_m_s')


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite and positive')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and positive') from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return number


def _velocities_close(left: float, right: float) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=1.0e-6, atol=1.0e-6))


__all__ = [
    'normalize_refraction_first_layer_request',
    'resolve_weathering_velocity_m_s',
    'resolved_first_layer_weathering_velocity_m_s',
    'validate_resolved_first_layer_velocity_match',
]
