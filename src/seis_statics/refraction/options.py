"""Frozen option dataclasses for refraction static workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeVar

import numpy as np

from seis_statics.validation import (
    coerce_finite_float,
    coerce_header_byte,
    coerce_nonnegative_finite_float,
    coerce_positive_finite_float,
    coerce_positive_int,
)

from seis_statics.refraction.types import (
    BedrockVelocityMode,
    RefractionFirstLayerMode,
    RefractionLayerKind,
    RefractionLayerVelocityMode,
)


RefractionStaticMethod = Literal['gli_variable_thickness', 'multilayer_time_term']
RefractionStaticMoveoutModel = Literal['head_wave_linear_offset']
RefractionStaticDistanceSource = Literal['geometry', 'offset_header', 'auto']
RefractionStaticLayerKind = RefractionLayerKind
RefractionStaticLayerVelocityMode = RefractionLayerVelocityMode
RefractionStaticRefractorCellAssignmentMode = Literal['midpoint']
RefractionStaticRefractorCellOutsideGridPolicy = Literal['reject']
RefractionStaticRefractorCellCoordinateMode = Literal['grid_3d', 'line_2d_projected']
RefractionStaticRobustMethod = Literal['mad', 'sigma']
RefractionStaticDatumMode = Literal['floating_and_flat', 'floating_only', 'flat_only', 'none']
RefractionStaticFloatingDatumMode = Literal['smoothed_topography', 'constant', 'surface', 'from_artifact']
RefractionStaticDatumSmoothingMethod = Literal['moving_average', 'median']
RefractionStaticConversionMode = Literal['existing', 't1lsst_1layer', 't1lsst_multilayer']
RefractionStaticReducedTimeQcVelocityMode = Literal['layer_velocity', 'fixed', 'initial_velocity']

_T = TypeVar('_T', bound=str)
_REFRACTION_STATIC_LAYER_ORDER: dict[RefractionLayerKind, int] = {
    'v2_t1': 0,
    'v3_t2': 1,
    'vsub_t3': 2,
}


@dataclass(frozen=True)
class RefractionStaticFirstLayerOptions:
    """First-layer / V1 configuration for refraction static inversion."""

    mode: RefractionFirstLayerMode = 'constant'
    weathering_velocity_m_s: float | None = None
    min_weathering_velocity_m_s: float = 250.0
    max_weathering_velocity_m_s: float = 1800.0
    min_direct_offset_m: float | None = None
    max_direct_offset_m: float | None = None
    min_picks_per_fit: int = 5
    min_groups: int = 3
    robust_enabled: bool = True
    robust_threshold: float = 3.5

    def __post_init__(self) -> None:
        _set(self, 'mode', _literal(self.mode, {'constant', 'estimate_direct_arrival'}, 'model.first_layer.mode'))
        _set_optional_positive_float(self, 'weathering_velocity_m_s', 'model.first_layer.weathering_velocity_m_s')
        _set_positive_float(self, 'min_weathering_velocity_m_s', 'model.first_layer.min_weathering_velocity_m_s')
        _set_positive_float(self, 'max_weathering_velocity_m_s', 'model.first_layer.max_weathering_velocity_m_s')
        _set_optional_nonnegative_float(self, 'min_direct_offset_m', 'model.first_layer.min_direct_offset_m')
        _set_optional_nonnegative_float(self, 'max_direct_offset_m', 'model.first_layer.max_direct_offset_m')
        _set_positive_int(self, 'min_picks_per_fit', 'model.first_layer.min_picks_per_fit')
        _set_positive_int(self, 'min_groups', 'model.first_layer.min_groups')
        _set(self, 'robust_enabled', _bool(self.robust_enabled, 'model.first_layer.robust_enabled'))
        _set_positive_float(self, 'robust_threshold', 'model.first_layer.robust_threshold')
        if self.min_weathering_velocity_m_s >= self.max_weathering_velocity_m_s:
            raise ValueError('model.first_layer.min_weathering_velocity_m_s must be less than model.first_layer.max_weathering_velocity_m_s')
        if self.mode == 'estimate_direct_arrival' and self.weathering_velocity_m_s is not None:
            raise ValueError('model.first_layer.weathering_velocity_m_s must be omitted when model.first_layer.mode is estimate_direct_arrival')
        if self.mode == 'estimate_direct_arrival' and (self.min_direct_offset_m is None or self.max_direct_offset_m is None):
            raise ValueError('model.first_layer.min_direct_offset_m and model.first_layer.max_direct_offset_m are required when model.first_layer.mode is estimate_direct_arrival')
        if self.min_direct_offset_m is not None and self.max_direct_offset_m is not None and self.min_direct_offset_m >= self.max_direct_offset_m:
            raise ValueError('model.first_layer.min_direct_offset_m must be less than model.first_layer.max_direct_offset_m')


@dataclass(frozen=True)
class RefractionStaticRefractorCellOptions:
    """Spatial refractor V2 cell configuration."""

    number_of_cell_x: int
    size_of_cell_x_m: float
    x_coordinate_origin_m: float
    number_of_cell_y: int = 1
    size_of_cell_y_m: float | None = None
    y_coordinate_origin_m: float = 0.0
    assignment_mode: RefractionStaticRefractorCellAssignmentMode = 'midpoint'
    outside_grid_policy: RefractionStaticRefractorCellOutsideGridPolicy = 'reject'
    coordinate_mode: RefractionStaticRefractorCellCoordinateMode = 'grid_3d'
    line_origin_x_m: float | None = None
    line_origin_y_m: float | None = None
    line_azimuth_deg: float | None = None
    min_observations_per_cell: int = 5
    velocity_smoothing_weight: float = 0.0
    smoothing_reference_distance_m: float | None = None

    def __post_init__(self) -> None:
        _set_positive_int(self, 'number_of_cell_x', 'model.refractor_cell.number_of_cell_x')
        _set_positive_float(self, 'size_of_cell_x_m', 'model.refractor_cell.size_of_cell_x_m')
        _set_finite_float(self, 'x_coordinate_origin_m', 'model.refractor_cell.x_coordinate_origin_m')
        _set_positive_int(self, 'number_of_cell_y', 'model.refractor_cell.number_of_cell_y')
        _set_optional_positive_float(self, 'size_of_cell_y_m', 'model.refractor_cell.size_of_cell_y_m')
        _set_finite_float(self, 'y_coordinate_origin_m', 'model.refractor_cell.y_coordinate_origin_m')
        _set(self, 'assignment_mode', _literal(self.assignment_mode, {'midpoint'}, 'model.refractor_cell.assignment_mode'))
        _set(self, 'outside_grid_policy', _literal(self.outside_grid_policy, {'reject'}, 'model.refractor_cell.outside_grid_policy'))
        _set(self, 'coordinate_mode', _literal(self.coordinate_mode, {'grid_3d', 'line_2d_projected'}, 'model.refractor_cell.coordinate_mode'))
        _set_optional_finite_float(self, 'line_origin_x_m', 'model.refractor_cell.line_origin_x_m')
        _set_optional_finite_float(self, 'line_origin_y_m', 'model.refractor_cell.line_origin_y_m')
        _set_optional_finite_float(self, 'line_azimuth_deg', 'model.refractor_cell.line_azimuth_deg')
        _set_positive_int(self, 'min_observations_per_cell', 'model.refractor_cell.min_observations_per_cell')
        _set_nonnegative_float(self, 'velocity_smoothing_weight', 'model.refractor_cell.velocity_smoothing_weight')
        _set_optional_positive_float(self, 'smoothing_reference_distance_m', 'model.refractor_cell.smoothing_reference_distance_m')
        if self.number_of_cell_y > 1 and self.size_of_cell_y_m is None:
            raise ValueError('model.refractor_cell.size_of_cell_y_m is required when model.refractor_cell.number_of_cell_y > 1')
        if self.coordinate_mode == 'line_2d_projected':
            if self.line_origin_x_m is None or self.line_origin_y_m is None or self.line_azimuth_deg is None:
                raise ValueError('model.refractor_cell.line_origin_x_m, model.refractor_cell.line_origin_y_m, and model.refractor_cell.line_azimuth_deg are required when model.refractor_cell.coordinate_mode is line_2d_projected')
            if self.number_of_cell_y != 1:
                raise ValueError('model.refractor_cell.number_of_cell_y must be 1 when model.refractor_cell.coordinate_mode is line_2d_projected')


@dataclass(frozen=True)
class RefractionStaticLayerOptions:
    """Layer-specific time-term configuration for multi-layer refraction statics."""

    kind: RefractionLayerKind
    enabled: bool = True
    min_offset_m: float | None = None
    max_offset_m: float | None = None
    velocity_mode: RefractionLayerVelocityMode = 'solve_global'
    initial_velocity_m_s: float | None = None
    fixed_velocity_m_s: float | None = None
    min_velocity_m_s: float | None = None
    max_velocity_m_s: float | None = None
    min_observations_per_cell: int | None = None
    smoothing_weight: float | None = None

    def __post_init__(self) -> None:
        _set(self, 'kind', _literal(self.kind, {'v2_t1', 'v3_t2', 'vsub_t3'}, 'model.layers.kind'))
        _set(self, 'enabled', _bool(self.enabled, 'model.layers.enabled'))
        _set_optional_nonnegative_float(self, 'min_offset_m', 'model.layers.min_offset_m')
        _set_optional_nonnegative_float(self, 'max_offset_m', 'model.layers.max_offset_m')
        _set(self, 'velocity_mode', _literal(self.velocity_mode, {'fixed_global', 'solve_global', 'solve_cell'}, 'model.layers.velocity_mode'))
        _set_optional_positive_float(self, 'initial_velocity_m_s', 'model.layers.initial_velocity_m_s')
        _set_optional_positive_float(self, 'fixed_velocity_m_s', 'model.layers.fixed_velocity_m_s')
        _set_optional_positive_float(self, 'min_velocity_m_s', 'model.layers.min_velocity_m_s')
        _set_optional_positive_float(self, 'max_velocity_m_s', 'model.layers.max_velocity_m_s')
        _set_optional_positive_int(self, 'min_observations_per_cell', 'model.layers.min_observations_per_cell')
        _set_optional_nonnegative_float(self, 'smoothing_weight', 'model.layers.smoothing_weight')
        if self.min_offset_m is not None and self.max_offset_m is not None and self.min_offset_m >= self.max_offset_m:
            raise ValueError('model.layers.min_offset_m must be less than model.layers.max_offset_m')
        if self.min_velocity_m_s is not None and self.max_velocity_m_s is not None and self.min_velocity_m_s >= self.max_velocity_m_s:
            raise ValueError('model.layers.min_velocity_m_s must be less than model.layers.max_velocity_m_s')


@dataclass(frozen=True)
class RefractionStaticModelOptions:
    """Near-surface model options for refraction static inversion."""

    method: RefractionStaticMethod = 'gli_variable_thickness'
    weathering_velocity_m_s: float | None = None
    first_layer: RefractionStaticFirstLayerOptions | None = None
    bedrock_velocity_mode: BedrockVelocityMode = 'solve_global'
    bedrock_velocity_m_s: float | None = None
    initial_bedrock_velocity_m_s: float | None = None
    min_bedrock_velocity_m_s: float = 1200.0
    max_bedrock_velocity_m_s: float = 6000.0
    max_weathering_thickness_m: float | None = None
    refractor_cell: RefractionStaticRefractorCellOptions | None = None
    layers: tuple[RefractionStaticLayerOptions, ...] | None = None
    allow_overlapping_layer_gates: bool = False

    def __post_init__(self) -> None:
        _set(self, 'method', _literal(self.method, {'gli_variable_thickness', 'multilayer_time_term'}, 'model.method'))
        _set_optional_positive_float(self, 'weathering_velocity_m_s', 'model.weathering_velocity_m_s')
        _set(self, 'bedrock_velocity_mode', _literal(self.bedrock_velocity_mode, {'solve_global', 'fixed_global', 'solve_cell'}, 'model.bedrock_velocity_mode'))
        _set_optional_positive_float(self, 'bedrock_velocity_m_s', 'model.bedrock_velocity_m_s')
        _set_optional_positive_float(self, 'initial_bedrock_velocity_m_s', 'model.initial_bedrock_velocity_m_s')
        _set_positive_float(self, 'min_bedrock_velocity_m_s', 'model.min_bedrock_velocity_m_s')
        _set_positive_float(self, 'max_bedrock_velocity_m_s', 'model.max_bedrock_velocity_m_s')
        _set_optional_positive_float(self, 'max_weathering_thickness_m', 'model.max_weathering_thickness_m')
        _set(self, 'allow_overlapping_layer_gates', _bool(self.allow_overlapping_layer_gates, 'model.allow_overlapping_layer_gates'))
        if self.layers is not None:
            _set(self, 'layers', tuple(self.layers))

        resolved_weathering_velocity = self._constant_weathering_velocity_or_none()
        if self.method == 'multilayer_time_term':
            self._check_multilayer_values(resolved_weathering_velocity)
            return
        if self.layers is not None:
            raise ValueError('model.layers is only allowed when model.method is multilayer_time_term')
        if self.min_bedrock_velocity_m_s >= self.max_bedrock_velocity_m_s:
            raise ValueError('model.min_bedrock_velocity_m_s must be less than model.max_bedrock_velocity_m_s')
        self._check_velocity_greater_than_weathering(resolved_weathering_velocity)
        if self.initial_bedrock_velocity_m_s is not None and not (self.min_bedrock_velocity_m_s <= self.initial_bedrock_velocity_m_s <= self.max_bedrock_velocity_m_s):
            raise ValueError('model.initial_bedrock_velocity_m_s must be within bedrock velocity bounds')
        if self.bedrock_velocity_mode == 'fixed_global':
            if self.bedrock_velocity_m_s is None:
                raise ValueError('model.bedrock_velocity_m_s is required when model.bedrock_velocity_mode is fixed_global')
            if not (self.min_bedrock_velocity_m_s <= self.bedrock_velocity_m_s <= self.max_bedrock_velocity_m_s):
                raise ValueError('model.bedrock_velocity_m_s must be within bedrock velocity bounds')
        elif self.bedrock_velocity_m_s is not None:
            raise ValueError('model.bedrock_velocity_m_s is only allowed when model.bedrock_velocity_mode is fixed_global')
        if self.bedrock_velocity_mode == 'solve_cell':
            if self.refractor_cell is None:
                raise ValueError('model.refractor_cell is required when model.bedrock_velocity_mode is solve_cell')
        elif self.refractor_cell is not None:
            raise ValueError('model.refractor_cell is only allowed when model.bedrock_velocity_mode is solve_cell')
        if self.bedrock_velocity_mode in {'solve_global', 'solve_cell'} and self.initial_bedrock_velocity_m_s is None:
            raise ValueError(
                'model.initial_bedrock_velocity_m_s is required when '
                'model.bedrock_velocity_mode is solve_global or solve_cell'
            )

    @property
    def enabled_refraction_layer_count(self) -> int:
        if self.layers is None:
            return 1
        return sum(1 for layer in self.layers if layer.enabled)

    @property
    def first_layer_mode(self) -> RefractionFirstLayerMode:
        first_layer = self.first_layer
        if first_layer is None:
            return 'constant'
        return first_layer.mode

    @property
    def resolved_weathering_velocity_m_s(self) -> float:
        value = self._constant_weathering_velocity_or_none()
        if value is None:
            raise ValueError('model.first_layer.mode="estimate_direct_arrival" requires a resolved weathering velocity before downstream processing')
        return value

    def _constant_weathering_velocity_or_none(self) -> float | None:
        legacy_velocity = self.weathering_velocity_m_s
        first_layer = self.first_layer
        if first_layer is None:
            if legacy_velocity is None:
                raise ValueError('model.weathering_velocity_m_s is required when model.first_layer is omitted')
            return legacy_velocity
        first_layer_velocity = first_layer.weathering_velocity_m_s
        if first_layer.mode == 'estimate_direct_arrival':
            if legacy_velocity is not None:
                raise ValueError('model.weathering_velocity_m_s must be omitted when model.first_layer.mode is estimate_direct_arrival')
            if first_layer_velocity is not None:
                raise ValueError('model.first_layer.weathering_velocity_m_s must be omitted when model.first_layer.mode is estimate_direct_arrival')
            return None
        if legacy_velocity is not None and first_layer_velocity is not None and not _values_close(legacy_velocity, first_layer_velocity):
            raise ValueError('model.weathering_velocity_m_s and model.first_layer.weathering_velocity_m_s must match when both are specified')
        if first_layer_velocity is None:
            raise ValueError('model.first_layer.weathering_velocity_m_s is required when model.first_layer.mode is constant')
        return first_layer_velocity

    def _check_velocity_greater_than_weathering(self, weathering_velocity: float | None) -> None:
        if weathering_velocity is None:
            return
        for name in ('min_bedrock_velocity_m_s', 'max_bedrock_velocity_m_s', 'initial_bedrock_velocity_m_s', 'bedrock_velocity_m_s'):
            value = getattr(self, name)
            if value is not None and value <= weathering_velocity:
                raise ValueError(f'model.{name} must be greater than model.resolved_weathering_velocity_m_s')

    def _check_multilayer_values(self, resolved_weathering_velocity: float | None) -> None:
        layers = self.layers
        if not layers:
            raise ValueError('model.layers must include enabled v2_t1 when model.method is multilayer_time_term')
        seen_kinds: set[RefractionLayerKind] = set()
        previous_order = -1
        for layer in layers:
            if layer.kind in seen_kinds:
                raise ValueError('model.layers must not contain duplicate layer kinds')
            order = _REFRACTION_STATIC_LAYER_ORDER[layer.kind]
            if order < previous_order:
                raise ValueError('model.layers must be ordered v2_t1, v3_t2, vsub_t3')
            seen_kinds.add(layer.kind)
            previous_order = order
        enabled_layers = tuple(layer for layer in layers if layer.enabled)
        enabled_kinds = {layer.kind for layer in enabled_layers}
        if 'v2_t1' not in enabled_kinds:
            raise ValueError('model.layers must include an enabled v2_t1 layer when model.method is multilayer_time_term')
        if 'vsub_t3' in enabled_kinds and 'v3_t2' not in enabled_kinds:
            raise ValueError('model.layers cannot enable vsub_t3 unless v3_t2 is enabled')
        deepest_enabled_order = max(_REFRACTION_STATIC_LAYER_ORDER[layer.kind] for layer in enabled_layers)
        for layer in enabled_layers:
            if layer.min_offset_m is None and layer.max_offset_m is None:
                raise ValueError('model.layers.min_offset_m or model.layers.max_offset_m is required for each enabled layer')
            if layer.max_offset_m is None and _REFRACTION_STATIC_LAYER_ORDER[layer.kind] != deepest_enabled_order:
                raise ValueError('model.layers.max_offset_m may be null only for the deepest enabled layer')
            self._check_multilayer_velocity_layer(layer, resolved_weathering_velocity=resolved_weathering_velocity)
        if not self.allow_overlapping_layer_gates:
            self._check_multilayer_layer_gate_overlap(enabled_layers)
        has_enabled_solve_cell_layer = any(layer.enabled and layer.velocity_mode == 'solve_cell' for layer in layers)
        if has_enabled_solve_cell_layer and self.refractor_cell is None:
            raise ValueError('model.refractor_cell is required when an enabled multi-layer refraction layer uses solve_cell')
        if self.refractor_cell is not None and not has_enabled_solve_cell_layer:
            raise ValueError('model.refractor_cell is only allowed when an enabled multi-layer refraction layer uses solve_cell')

    def _check_multilayer_velocity_layer(self, layer: RefractionStaticLayerOptions, *, resolved_weathering_velocity: float | None) -> None:
        min_velocity = self._layer_min_velocity_m_s(layer)
        max_velocity = self._layer_max_velocity_m_s(layer)
        if min_velocity is not None and max_velocity is not None and min_velocity >= max_velocity:
            raise ValueError('model.layers.min_velocity_m_s must be less than model.layers.max_velocity_m_s')
        if resolved_weathering_velocity is not None:
            if min_velocity is not None and min_velocity <= resolved_weathering_velocity:
                raise ValueError('model.layers.min_velocity_m_s must be greater than model.resolved_weathering_velocity_m_s')
            if max_velocity is not None and max_velocity <= resolved_weathering_velocity:
                raise ValueError('model.layers.max_velocity_m_s must be greater than model.resolved_weathering_velocity_m_s')
        if layer.velocity_mode == 'fixed_global':
            fixed_velocity = self._layer_fixed_velocity_m_s(layer)
            if fixed_velocity is None:
                raise ValueError('model.layers.fixed_velocity_m_s is required when model.layers.velocity_mode is fixed_global')
            self._check_layer_velocity_in_bounds(fixed_velocity, min_velocity=min_velocity, max_velocity=max_velocity, field_name='fixed_velocity_m_s')
            if resolved_weathering_velocity is not None and fixed_velocity <= resolved_weathering_velocity:
                raise ValueError('model.layers.fixed_velocity_m_s must be greater than model.resolved_weathering_velocity_m_s')
            return
        initial_velocity = self._layer_initial_velocity_m_s(layer)
        if initial_velocity is None:
            raise ValueError('model.layers.initial_velocity_m_s or model.initial_bedrock_velocity_m_s is required when model.layers.velocity_mode is solve_global or solve_cell')
        self._check_layer_velocity_in_bounds(initial_velocity, min_velocity=min_velocity, max_velocity=max_velocity, field_name='initial_velocity_m_s')
        if resolved_weathering_velocity is not None and initial_velocity <= resolved_weathering_velocity:
            raise ValueError('model.layers.initial_velocity_m_s must be greater than model.resolved_weathering_velocity_m_s')

    def _check_layer_velocity_in_bounds(self, velocity: float, *, min_velocity: float | None, max_velocity: float | None, field_name: str) -> None:
        if min_velocity is not None and velocity < min_velocity:
            raise ValueError(f'model.layers.{field_name} must be within velocity bounds')
        if max_velocity is not None and velocity > max_velocity:
            raise ValueError(f'model.layers.{field_name} must be within velocity bounds')

    def _check_multilayer_layer_gate_overlap(self, enabled_layers: tuple[RefractionStaticLayerOptions, ...]) -> None:
        for index, layer in enumerate(enabled_layers):
            layer_min = float('-inf') if layer.min_offset_m is None else layer.min_offset_m
            layer_max = float('inf') if layer.max_offset_m is None else layer.max_offset_m
            for other in enabled_layers[index + 1 :]:
                other_min = float('-inf') if other.min_offset_m is None else other.min_offset_m
                other_max = float('inf') if other.max_offset_m is None else other.max_offset_m
                if max(layer_min, other_min) < min(layer_max, other_max):
                    raise ValueError('model.layers offset gates must not overlap unless model.allow_overlapping_layer_gates is true')

    def _layer_initial_velocity_m_s(self, layer: RefractionStaticLayerOptions) -> float | None:
        if layer.initial_velocity_m_s is not None:
            return layer.initial_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.initial_bedrock_velocity_m_s
        return None

    def _layer_fixed_velocity_m_s(self, layer: RefractionStaticLayerOptions) -> float | None:
        if layer.fixed_velocity_m_s is not None:
            return layer.fixed_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.bedrock_velocity_m_s
        return None

    def _layer_min_velocity_m_s(self, layer: RefractionStaticLayerOptions) -> float | None:
        if layer.min_velocity_m_s is not None:
            return layer.min_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.min_bedrock_velocity_m_s
        return None

    def _layer_max_velocity_m_s(self, layer: RefractionStaticLayerOptions) -> float | None:
        if layer.max_velocity_m_s is not None:
            return layer.max_velocity_m_s
        if layer.kind == 'v2_t1':
            return self.max_bedrock_velocity_m_s
        return None


@dataclass(frozen=True)
class RefractionStaticMoveoutOptions:
    """Moveout distance source and filtering options for refraction statics."""

    model: RefractionStaticMoveoutModel = 'head_wave_linear_offset'
    distance_source: RefractionStaticDistanceSource = 'geometry'
    offset_byte: int | None = 37
    min_offset_m: float | None = None
    max_offset_m: float | None = None
    allow_missing_offset: bool = False
    max_geometry_offset_mismatch_m: float | None = None

    def __post_init__(self) -> None:
        _set(self, 'model', _literal(self.model, {'head_wave_linear_offset'}, 'moveout.model'))
        _set(self, 'distance_source', _literal(self.distance_source, {'geometry', 'offset_header', 'auto'}, 'moveout.distance_source'))
        if self.offset_byte is not None:
            _set(self, 'offset_byte', coerce_header_byte(self.offset_byte, name='moveout.offset_byte'))
        _set_optional_nonnegative_float(self, 'min_offset_m', 'moveout.min_offset_m')
        _set_optional_nonnegative_float(self, 'max_offset_m', 'moveout.max_offset_m')
        _set(self, 'allow_missing_offset', _bool(self.allow_missing_offset, 'moveout.allow_missing_offset'))
        _set_optional_nonnegative_float(self, 'max_geometry_offset_mismatch_m', 'moveout.max_geometry_offset_mismatch_m')
        if self.distance_source == 'offset_header' and self.offset_byte is None:
            raise ValueError('moveout.offset_byte is required when moveout.distance_source is offset_header')
        if self.min_offset_m is not None and self.max_offset_m is not None and self.min_offset_m >= self.max_offset_m:
            raise ValueError('moveout.min_offset_m must be less than moveout.max_offset_m')


@dataclass(frozen=True)
class RefractionStaticRobustOptions:
    """Robust outlier-rejection options for refraction inversion."""

    enabled: bool = True
    method: RefractionStaticRobustMethod = 'mad'
    threshold: float = 3.5
    scale_floor_ms: float = 0.05
    max_iterations: int = 5
    min_used_fraction: float = 0.5
    min_used_observations: int = 1

    def __post_init__(self) -> None:
        _set(self, 'enabled', _bool(self.enabled, 'solver.robust.enabled'))
        _set(self, 'method', _literal(self.method, {'mad', 'sigma'}, 'solver.robust.method'))
        _set_positive_float(self, 'threshold', 'solver.robust.threshold')
        _set_nonnegative_float(self, 'scale_floor_ms', 'solver.robust.scale_floor_ms')
        _set_positive_int(self, 'max_iterations', 'solver.robust.max_iterations')
        _set_positive_float(self, 'min_used_fraction', 'solver.robust.min_used_fraction')
        if self.min_used_fraction > 1.0:
            raise ValueError('solver.robust.min_used_fraction must be <= 1')
        _set_positive_int(self, 'min_used_observations', 'solver.robust.min_used_observations')


@dataclass(frozen=True)
class RefractionStaticSolverOptions:
    """Solver options for refraction static inversion."""

    damping: float = 0.01
    min_picks_per_node: int = 1
    max_abs_half_intercept_time_ms: float = 500.0
    robust: RefractionStaticRobustOptions = field(default_factory=RefractionStaticRobustOptions)

    def __post_init__(self) -> None:
        _set_nonnegative_float(self, 'damping', 'solver.damping')
        _set_positive_int(self, 'min_picks_per_node', 'solver.min_picks_per_node')
        _set_positive_float(self, 'max_abs_half_intercept_time_ms', 'solver.max_abs_half_intercept_time_ms')


@dataclass(frozen=True)
class RefractionStaticDatumOptions:
    """Datum options for refraction static composition."""

    mode: RefractionStaticDatumMode = 'none'
    floating_datum_mode: RefractionStaticFloatingDatumMode = 'smoothed_topography'
    flat_datum_elevation_m: float | None = None
    floating_datum_elevation_m: float | None = None
    smoothing_radius_m: float | None = None
    smoothing_window_nodes: int | None = 11
    smoothing_method: RefractionStaticDatumSmoothingMethod = 'moving_average'
    floating_datum_job_id: str | None = None
    floating_datum_artifact_name: str | None = None
    allow_flat_datum_above_topography: bool = True
    allow_flat_datum_below_refractor: bool = False

    def __post_init__(self) -> None:
        _set(self, 'mode', _literal(self.mode, {'floating_and_flat', 'floating_only', 'flat_only', 'none'}, 'datum.mode'))
        _set(self, 'floating_datum_mode', _literal(self.floating_datum_mode, {'smoothed_topography', 'constant', 'surface', 'from_artifact'}, 'datum.floating_datum_mode'))
        _set_optional_finite_float(self, 'flat_datum_elevation_m', 'datum.flat_datum_elevation_m')
        _set_optional_finite_float(self, 'floating_datum_elevation_m', 'datum.floating_datum_elevation_m')
        _set_optional_positive_float(self, 'smoothing_radius_m', 'datum.smoothing_radius_m')
        _set_optional_positive_int(self, 'smoothing_window_nodes', 'datum.smoothing_window_nodes')
        if self.smoothing_window_nodes is not None and self.smoothing_window_nodes % 2 == 0:
            raise ValueError('datum.smoothing_window_nodes must be odd')
        _set(self, 'smoothing_method', _literal(self.smoothing_method, {'moving_average', 'median'}, 'datum.smoothing_method'))
        if self.floating_datum_artifact_name is not None:
            _set(self, 'floating_datum_artifact_name', _artifact_basename(self.floating_datum_artifact_name, 'datum.floating_datum_artifact_name'))
        _set(self, 'allow_flat_datum_above_topography', _bool(self.allow_flat_datum_above_topography, 'datum.allow_flat_datum_above_topography'))
        _set(self, 'allow_flat_datum_below_refractor', _bool(self.allow_flat_datum_below_refractor, 'datum.allow_flat_datum_below_refractor'))
        if self.mode in {'flat_only', 'floating_and_flat'} and self.flat_datum_elevation_m is None:
            raise ValueError('datum.flat_datum_elevation_m is required for flat datum modes')
        if self.floating_datum_mode == 'constant' and self.floating_datum_elevation_m is None:
            raise ValueError('datum.floating_datum_elevation_m is required when floating_datum_mode is constant')
        if self.floating_datum_mode == 'from_artifact':
            if not self.floating_datum_job_id:
                raise ValueError('datum.floating_datum_job_id is required when floating_datum_mode is from_artifact')
            if not self.floating_datum_artifact_name:
                raise ValueError('datum.floating_datum_artifact_name is required when floating_datum_mode is from_artifact')
        elif self.floating_datum_job_id is not None:
            raise ValueError('datum.floating_datum_job_id is only allowed when floating_datum_mode is from_artifact')
        elif self.floating_datum_artifact_name is not None:
            raise ValueError('datum.floating_datum_artifact_name is only allowed when floating_datum_mode is from_artifact')


@dataclass(frozen=True)
class RefractionStaticConversionOptions:
    """Conversion/output mode for refraction static component artifacts."""

    mode: RefractionStaticConversionMode = 'existing'
    layer_count: int | None = None

    def __post_init__(self) -> None:
        _set(self, 'mode', _literal(self.mode, {'existing', 't1lsst_1layer', 't1lsst_multilayer'}, 'conversion.mode'))
        _set_optional_positive_int(self, 'layer_count', 'conversion.layer_count')
        if self.layer_count is not None and self.layer_count > 3:
            raise ValueError('conversion.layer_count must be 1, 2, or 3')
        if self.mode == 't1lsst_multilayer':
            if self.layer_count is None:
                raise ValueError('conversion.layer_count is required when conversion.mode is t1lsst_multilayer')
        elif self.layer_count is not None:
            raise ValueError('conversion.layer_count is only allowed when conversion.mode is t1lsst_multilayer')


@dataclass(frozen=True)
class RefractionStaticReducedTimeQcOptions:
    """Reduced-time QC velocity selection for refraction first-break artifacts."""

    reduction_velocity_mode: RefractionStaticReducedTimeQcVelocityMode = 'layer_velocity'
    fixed_velocity_m_s: float | None = None

    def __post_init__(self) -> None:
        _set(self, 'reduction_velocity_mode', _literal(self.reduction_velocity_mode, {'layer_velocity', 'fixed', 'initial_velocity'}, 'reduced_time_qc.reduction_velocity_mode'))
        _set_optional_positive_float(self, 'fixed_velocity_m_s', 'reduced_time_qc.fixed_velocity_m_s')
        if self.reduction_velocity_mode == 'fixed':
            if self.fixed_velocity_m_s is None:
                raise ValueError('reduced_time_qc.fixed_velocity_m_s is required when reduced_time_qc.reduction_velocity_mode is fixed')
        elif self.fixed_velocity_m_s is not None:
            raise ValueError('reduced_time_qc.fixed_velocity_m_s is only allowed when reduced_time_qc.reduction_velocity_mode is fixed')


def _set(instance: object, name: str, value: object) -> None:
    object.__setattr__(instance, name, value)


def _literal(value: object, allowed: set[_T], name: str) -> _T:
    if value not in allowed:
        allowed_text = ', '.join(sorted(allowed))
        raise ValueError(f'{name} must be one of: {allowed_text}')
    return value  # type: ignore[return-value]


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a boolean')
    return bool(value)


def _values_close(left: float, right: float) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=1.0e-6, atol=1.0e-6))


def _artifact_basename(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a plain file name')
    if value in {'', '.', '..'} or '/' in value or '\\' in value:
        raise ValueError(f'{name} must be a plain file name')
    return value


def _set_finite_float(instance: object, attr: str, name: str) -> None:
    _set(instance, attr, coerce_finite_float(getattr(instance, attr), name=name))


def _set_optional_finite_float(instance: object, attr: str, name: str) -> None:
    value = getattr(instance, attr)
    if value is not None:
        _set(instance, attr, coerce_finite_float(value, name=name))


def _set_positive_float(instance: object, attr: str, name: str) -> None:
    _set(instance, attr, coerce_positive_finite_float(getattr(instance, attr), name=name))


def _set_optional_positive_float(instance: object, attr: str, name: str) -> None:
    value = getattr(instance, attr)
    if value is not None:
        _set(instance, attr, coerce_positive_finite_float(value, name=name))


def _set_nonnegative_float(instance: object, attr: str, name: str) -> None:
    _set(instance, attr, coerce_nonnegative_finite_float(getattr(instance, attr), name=name))


def _set_optional_nonnegative_float(instance: object, attr: str, name: str) -> None:
    value = getattr(instance, attr)
    if value is not None:
        _set(instance, attr, coerce_nonnegative_finite_float(value, name=name))


def _set_positive_int(instance: object, attr: str, name: str) -> None:
    _set(instance, attr, coerce_positive_int(getattr(instance, attr), name=name))


def _set_optional_positive_int(instance: object, attr: str, name: str) -> None:
    value = getattr(instance, attr)
    if value is not None:
        _set(instance, attr, coerce_positive_int(value, name=name))


__all__ = [
    'RefractionStaticConversionMode',
    'RefractionStaticConversionOptions',
    'RefractionStaticDatumMode',
    'RefractionStaticDatumOptions',
    'RefractionStaticDatumSmoothingMethod',
    'RefractionStaticDistanceSource',
    'RefractionStaticFirstLayerOptions',
    'RefractionStaticFloatingDatumMode',
    'RefractionStaticLayerKind',
    'RefractionStaticMethod',
    'RefractionStaticModelOptions',
    'RefractionStaticMoveoutModel',
    'RefractionStaticMoveoutOptions',
    'RefractionStaticReducedTimeQcOptions',
    'RefractionStaticReducedTimeQcVelocityMode',
    'RefractionStaticRefractorCellAssignmentMode',
    'RefractionStaticRefractorCellCoordinateMode',
    'RefractionStaticRefractorCellOptions',
    'RefractionStaticRefractorCellOutsideGridPolicy',
    'RefractionStaticRobustMethod',
    'RefractionStaticRobustOptions',
    'RefractionStaticSolverOptions',
    'RefractionStaticLayerOptions',
    'RefractionStaticLayerVelocityMode',
]
