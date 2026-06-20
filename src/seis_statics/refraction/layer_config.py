"""Normalized refraction layer configuration."""

from __future__ import annotations

from dataclasses import dataclass

from seis_statics.refraction.options import (
    RefractionStaticLayerOptions,
    RefractionStaticModelOptions,
)
from seis_statics.refraction.types import RefractionLayerKind, RefractionLayerVelocityMode
from seis_statics.validation import (
    coerce_nonnegative_finite_float,
    coerce_positive_finite_float,
    coerce_positive_int,
)


_LAYER_ORDER: dict[RefractionLayerKind, int] = {
    'v2_t1': 0,
    'v3_t2': 1,
    'vsub_t3': 2,
}


@dataclass(frozen=True)
class RefractionLayerConfigLayer:
    """Resolved package-native layer options used by downstream solvers."""

    kind: RefractionLayerKind
    min_offset_m: float | None
    max_offset_m: float | None
    velocity_mode: RefractionLayerVelocityMode
    initial_velocity_m_s: float | None = None
    fixed_velocity_m_s: float | None = None
    min_velocity_m_s: float | None = None
    max_velocity_m_s: float | None = None
    min_observations_per_cell: int | None = None
    smoothing_weight: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _LAYER_ORDER:
            raise ValueError('layer.kind must be v2_t1, v3_t2, or vsub_t3')
        if self.velocity_mode not in {'fixed_global', 'solve_global', 'solve_cell'}:
            raise ValueError('layer.velocity_mode must be fixed_global, solve_global, or solve_cell')
        _set_optional_nonnegative_float(self, 'min_offset_m', 'layer.min_offset_m')
        _set_optional_nonnegative_float(self, 'max_offset_m', 'layer.max_offset_m')
        _set_optional_positive_float(self, 'initial_velocity_m_s', 'layer.initial_velocity_m_s')
        _set_optional_positive_float(self, 'fixed_velocity_m_s', 'layer.fixed_velocity_m_s')
        _set_optional_positive_float(self, 'min_velocity_m_s', 'layer.min_velocity_m_s')
        _set_optional_positive_float(self, 'max_velocity_m_s', 'layer.max_velocity_m_s')
        _set_optional_positive_int(
            self,
            'min_observations_per_cell',
            'layer.min_observations_per_cell',
        )
        _set_optional_nonnegative_float(self, 'smoothing_weight', 'layer.smoothing_weight')
        if self.min_offset_m is not None and self.max_offset_m is not None:
            if self.min_offset_m >= self.max_offset_m:
                raise ValueError('layer.min_offset_m must be less than layer.max_offset_m')
        if self.min_velocity_m_s is not None and self.max_velocity_m_s is not None:
            if self.min_velocity_m_s >= self.max_velocity_m_s:
                raise ValueError(
                    'layer.min_velocity_m_s must be less than layer.max_velocity_m_s'
                )


@dataclass(frozen=True)
class RefractionLayerConfig:
    """Validated enabled refraction layers with observation gates."""

    layers: tuple[RefractionLayerConfigLayer, ...]
    allow_overlapping_layer_gates: bool = False

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        object.__setattr__(self, 'layers', layers)
        if not layers:
            raise ValueError('layer config must contain at least one layer')
        seen_kinds: set[RefractionLayerKind] = set()
        previous_order = -1
        for layer in layers:
            if layer.kind in seen_kinds:
                raise ValueError('layer config must not contain duplicate layer kinds')
            order = _LAYER_ORDER[layer.kind]
            if order < previous_order:
                raise ValueError('layer config layers must be ordered v2_t1, v3_t2, vsub_t3')
            seen_kinds.add(layer.kind)
            previous_order = order
        if 'v2_t1' not in seen_kinds:
            raise ValueError('layer config must include v2_t1')
        if 'vsub_t3' in seen_kinds and 'v3_t2' not in seen_kinds:
            raise ValueError('layer config cannot include vsub_t3 unless v3_t2 is included')
        if not self.allow_overlapping_layer_gates:
            _validate_non_overlapping_gates(layers)

    @property
    def layer_count(self) -> int:
        return len(self.layers)


def normalize_refraction_layer_config(
    model: RefractionStaticModelOptions,
) -> RefractionLayerConfig:
    """Normalize package model options to enabled refraction layer config."""
    if not isinstance(model, RefractionStaticModelOptions):
        raise TypeError('model must be RefractionStaticModelOptions')
    if model.method == 'multilayer_time_term':
        layers = tuple(
            _normalize_multilayer_layer(model, layer)
            for layer in model.layers or ()
            if layer.enabled
        )
        return RefractionLayerConfig(
            layers=layers,
            allow_overlapping_layer_gates=model.allow_overlapping_layer_gates,
        )
    return RefractionLayerConfig(
        layers=(
            RefractionLayerConfigLayer(
                kind='v2_t1',
                min_offset_m=None,
                max_offset_m=None,
                velocity_mode=model.bedrock_velocity_mode,
                initial_velocity_m_s=model.initial_bedrock_velocity_m_s,
                fixed_velocity_m_s=model.bedrock_velocity_m_s,
                min_velocity_m_s=model.min_bedrock_velocity_m_s,
                max_velocity_m_s=model.max_bedrock_velocity_m_s,
            ),
        ),
    )


def layer_offset_gate_contains(
    layer: RefractionLayerConfigLayer,
    offset_abs_m: float,
) -> bool:
    """Return true when an absolute offset is inside a layer gate."""
    if layer.min_offset_m is not None and offset_abs_m < layer.min_offset_m:
        return False
    if layer.max_offset_m is not None and offset_abs_m >= layer.max_offset_m:
        return False
    return True


def _normalize_multilayer_layer(
    model: RefractionStaticModelOptions,
    layer: RefractionStaticLayerOptions,
) -> RefractionLayerConfigLayer:
    return RefractionLayerConfigLayer(
        kind=layer.kind,
        min_offset_m=layer.min_offset_m,
        max_offset_m=layer.max_offset_m,
        velocity_mode=layer.velocity_mode,
        initial_velocity_m_s=_layer_initial_velocity_m_s(model, layer),
        fixed_velocity_m_s=_layer_fixed_velocity_m_s(model, layer),
        min_velocity_m_s=_layer_min_velocity_m_s(model, layer),
        max_velocity_m_s=_layer_max_velocity_m_s(model, layer),
        min_observations_per_cell=layer.min_observations_per_cell,
        smoothing_weight=layer.smoothing_weight,
    )


def _layer_initial_velocity_m_s(
    model: RefractionStaticModelOptions,
    layer: RefractionStaticLayerOptions,
) -> float | None:
    if layer.initial_velocity_m_s is not None:
        return layer.initial_velocity_m_s
    if layer.kind == 'v2_t1':
        return model.initial_bedrock_velocity_m_s
    return None


def _layer_fixed_velocity_m_s(
    model: RefractionStaticModelOptions,
    layer: RefractionStaticLayerOptions,
) -> float | None:
    if layer.fixed_velocity_m_s is not None:
        return layer.fixed_velocity_m_s
    if layer.kind == 'v2_t1':
        return model.bedrock_velocity_m_s
    return None


def _layer_min_velocity_m_s(
    model: RefractionStaticModelOptions,
    layer: RefractionStaticLayerOptions,
) -> float | None:
    if layer.min_velocity_m_s is not None:
        return layer.min_velocity_m_s
    if layer.kind == 'v2_t1':
        return model.min_bedrock_velocity_m_s
    return None


def _layer_max_velocity_m_s(
    model: RefractionStaticModelOptions,
    layer: RefractionStaticLayerOptions,
) -> float | None:
    if layer.max_velocity_m_s is not None:
        return layer.max_velocity_m_s
    if layer.kind == 'v2_t1':
        return model.max_bedrock_velocity_m_s
    return None


def _validate_non_overlapping_gates(
    layers: tuple[RefractionLayerConfigLayer, ...],
) -> None:
    for index, layer in enumerate(layers):
        layer_min = float('-inf') if layer.min_offset_m is None else layer.min_offset_m
        layer_max = float('inf') if layer.max_offset_m is None else layer.max_offset_m
        for other in layers[index + 1 :]:
            other_min = float('-inf') if other.min_offset_m is None else other.min_offset_m
            other_max = float('inf') if other.max_offset_m is None else other.max_offset_m
            if max(layer_min, other_min) < min(layer_max, other_max):
                raise ValueError(
                    'layer config offset gates must not overlap unless '
                    'allow_overlapping_layer_gates is true'
                )


def _set_optional_positive_float(object_: object, name: str, validation_name: str) -> None:
    value = getattr(object_, name)
    if value is not None:
        object.__setattr__(
            object_,
            name,
            coerce_positive_finite_float(value, name=validation_name),
        )


def _set_optional_nonnegative_float(object_: object, name: str, validation_name: str) -> None:
    value = getattr(object_, name)
    if value is not None:
        object.__setattr__(
            object_,
            name,
            coerce_nonnegative_finite_float(value, name=validation_name),
        )


def _set_optional_positive_int(object_: object, name: str, validation_name: str) -> None:
    value = getattr(object_, name)
    if value is not None:
        object.__setattr__(object_, name, coerce_positive_int(value, name=validation_name))


__all__ = [
    'RefractionLayerConfig',
    'RefractionLayerConfigLayer',
    'layer_offset_gate_contains',
    'normalize_refraction_layer_config',
]
