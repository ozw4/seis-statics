from __future__ import annotations

import pytest

from seis_statics.refraction import (
    RefractionLayerConfig,
    RefractionLayerConfigLayer,
    RefractionStaticFirstLayerOptions,
    RefractionStaticLayerOptions,
    RefractionStaticModelOptions,
    normalize_refraction_layer_config,
)


def test_refraction_layer_config_normalizes_legacy_one_layer_v2_model() -> None:
    model = RefractionStaticModelOptions(
        first_layer=RefractionStaticFirstLayerOptions(weathering_velocity_m_s=500.0),
        bedrock_velocity_mode='fixed_global',
        bedrock_velocity_m_s=1800.0,
        initial_bedrock_velocity_m_s=1700.0,
        min_bedrock_velocity_m_s=1200.0,
        max_bedrock_velocity_m_s=3000.0,
    )

    config = normalize_refraction_layer_config(model)

    assert config.layer_count == 1
    assert config.layers == (
        RefractionLayerConfigLayer(
            kind='v2_t1',
            min_offset_m=None,
            max_offset_m=None,
            velocity_mode='fixed_global',
            initial_velocity_m_s=1700.0,
            fixed_velocity_m_s=1800.0,
            min_velocity_m_s=1200.0,
            max_velocity_m_s=3000.0,
        ),
    )


def test_refraction_layer_config_normalizes_two_and_three_layer_models() -> None:
    first_layer = RefractionStaticFirstLayerOptions(weathering_velocity_m_s=500.0)
    two_layer_model = RefractionStaticModelOptions(
        method='multilayer_time_term',
        first_layer=first_layer,
        initial_bedrock_velocity_m_s=1800.0,
        layers=(
            RefractionStaticLayerOptions(
                kind='v2_t1',
                min_offset_m=0.0,
                max_offset_m=1000.0,
            ),
            RefractionStaticLayerOptions(
                kind='v3_t2',
                min_offset_m=1000.0,
                max_offset_m=None,
                velocity_mode='fixed_global',
                fixed_velocity_m_s=3200.0,
                min_velocity_m_s=2500.0,
                max_velocity_m_s=4200.0,
            ),
        ),
    )
    three_layer_model = RefractionStaticModelOptions(
        method='multilayer_time_term',
        first_layer=first_layer,
        initial_bedrock_velocity_m_s=1800.0,
        layers=(
            RefractionStaticLayerOptions(
                kind='v2_t1',
                min_offset_m=0.0,
                max_offset_m=700.0,
            ),
            RefractionStaticLayerOptions(
                kind='v3_t2',
                min_offset_m=700.0,
                max_offset_m=1400.0,
                initial_velocity_m_s=2600.0,
            ),
            RefractionStaticLayerOptions(
                kind='vsub_t3',
                min_offset_m=1400.0,
                max_offset_m=None,
                initial_velocity_m_s=4200.0,
            ),
        ),
    )

    two_layer_config = normalize_refraction_layer_config(two_layer_model)
    three_layer_config = normalize_refraction_layer_config(three_layer_model)

    assert [layer.kind for layer in two_layer_config.layers] == ['v2_t1', 'v3_t2']
    assert two_layer_config.layers[0].initial_velocity_m_s == 1800.0
    assert two_layer_config.layers[0].min_velocity_m_s == 1200.0
    assert two_layer_config.layers[0].max_velocity_m_s == 6000.0
    assert two_layer_config.layers[1].fixed_velocity_m_s == 3200.0
    assert two_layer_config.layers[1].min_velocity_m_s == 2500.0
    assert two_layer_config.layers[1].max_velocity_m_s == 4200.0
    assert [layer.kind for layer in three_layer_config.layers] == [
        'v2_t1',
        'v3_t2',
        'vsub_t3',
    ]
    assert three_layer_config.layers[2].min_offset_m == 1400.0
    assert three_layer_config.layers[2].max_offset_m is None


def test_refraction_layer_config_rejects_invalid_bounds_and_overlap() -> None:
    with pytest.raises(ValueError, match='min_offset_m'):
        RefractionLayerConfigLayer(
            kind='v2_t1',
            min_offset_m=500.0,
            max_offset_m=500.0,
            velocity_mode='solve_global',
        )

    with pytest.raises(ValueError, match='offset gates must not overlap'):
        RefractionLayerConfig(
            layers=(
                RefractionLayerConfigLayer(
                    kind='v2_t1',
                    min_offset_m=0.0,
                    max_offset_m=500.0,
                    velocity_mode='solve_global',
                ),
                RefractionLayerConfigLayer(
                    kind='v3_t2',
                    min_offset_m=499.9,
                    max_offset_m=None,
                    velocity_mode='solve_global',
                ),
            ),
        )

    config = RefractionLayerConfig(
        layers=(
            RefractionLayerConfigLayer(
                kind='v2_t1',
                min_offset_m=0.0,
                max_offset_m=500.0,
                velocity_mode='solve_global',
            ),
            RefractionLayerConfigLayer(
                kind='v3_t2',
                min_offset_m=500.0,
                max_offset_m=None,
                velocity_mode='solve_global',
            ),
        ),
    )

    assert config.layer_count == 2
