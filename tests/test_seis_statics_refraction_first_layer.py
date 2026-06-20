from __future__ import annotations

import pytest

from seis_statics.refraction import (
    RefractionStaticFirstLayerOptions,
    RefractionStaticModelOptions,
    ResolvedRefractionFirstLayer,
    normalize_refraction_first_layer_request,
    resolve_weathering_velocity_m_s,
    resolved_first_layer_weathering_velocity_m_s,
    validate_resolved_first_layer_velocity_match,
)


def test_refraction_first_layer_normalize_constant_model() -> None:
    model = RefractionStaticModelOptions(
        first_layer=RefractionStaticFirstLayerOptions(weathering_velocity_m_s=600.0),
        initial_bedrock_velocity_m_s=1700.0,
    )

    resolved = normalize_refraction_first_layer_request(model)

    assert resolved.mode == 'constant'
    assert resolved.weathering_velocity_m_s == 600.0
    assert resolved.status == 'resolved_constant'
    assert resolved.qc['v1_status'] == 'resolved_constant'


def test_refraction_first_layer_resolve_and_validate_match() -> None:
    model = RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        initial_bedrock_velocity_m_s=1700.0,
    )
    resolved = ResolvedRefractionFirstLayer(
        mode='constant',
        weathering_velocity_m_s=500.0,
        status='resolved_constant',
        qc={},
    )

    assert resolve_weathering_velocity_m_s(model=model) == 500.0
    assert resolve_weathering_velocity_m_s(model=model, resolved_first_layer=resolved) == 500.0
    assert validate_resolved_first_layer_velocity_match(
        weathering_velocity_m_s=500.0000001,
        resolved_first_layer=resolved,
        name='weathering_velocity_m_s',
    ) == 500.0
    assert resolved_first_layer_weathering_velocity_m_s(resolved) == 500.0


def test_refraction_first_layer_rejects_mode_and_velocity_mismatch() -> None:
    model = RefractionStaticModelOptions(
        first_layer=RefractionStaticFirstLayerOptions(
            mode='estimate_direct_arrival',
            min_direct_offset_m=100.0,
            max_direct_offset_m=500.0,
        ),
        initial_bedrock_velocity_m_s=1700.0,
    )
    resolved = ResolvedRefractionFirstLayer(
        mode='constant',
        weathering_velocity_m_s=500.0,
        status='resolved_constant',
        qc={},
    )

    with pytest.raises(ValueError, match='mode'):
        resolve_weathering_velocity_m_s(model=model, resolved_first_layer=resolved)

    with pytest.raises(ValueError, match='does not match'):
        validate_resolved_first_layer_velocity_match(
            weathering_velocity_m_s=600.0,
            resolved_first_layer=resolved,
            name='weathering_velocity_m_s',
        )
