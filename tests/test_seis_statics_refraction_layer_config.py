from __future__ import annotations

import pytest

from seis_statics.refraction import (
    RefractionLayerConfig,
    RefractionLayerConfigLayer,
    RefractionStaticFirstLayerOptions,
    RefractionStaticLayerOptions,
    RefractionStaticModelOptions,
    RefractionStaticRefractorCellOptions,
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


def test_refraction_layer_config_preserves_legacy_one_layer_cell_controls() -> None:
    model = RefractionStaticModelOptions(
        first_layer=RefractionStaticFirstLayerOptions(weathering_velocity_m_s=500.0),
        bedrock_velocity_mode='solve_cell',
        initial_bedrock_velocity_m_s=1800.0,
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=4,
            size_of_cell_x_m=100.0,
            x_coordinate_origin_m=0.0,
            min_observations_per_cell=7,
            velocity_smoothing_weight=0.25,
        ),
    )

    config = normalize_refraction_layer_config(model)

    assert config.layers == (
        RefractionLayerConfigLayer(
            kind='v2_t1',
            min_offset_m=None,
            max_offset_m=None,
            velocity_mode='solve_cell',
            initial_velocity_m_s=1800.0,
            fixed_velocity_m_s=None,
            min_velocity_m_s=1200.0,
            max_velocity_m_s=6000.0,
            min_observations_per_cell=7,
            smoothing_weight=0.25,
        ),
    )


@pytest.mark.parametrize('bedrock_velocity_mode', ('solve_global', 'solve_cell'))
def test_refraction_layer_config_rejects_legacy_solve_model_without_initial_velocity(
    bedrock_velocity_mode: str,
) -> None:
    refractor_cell = None
    if bedrock_velocity_mode == 'solve_cell':
        refractor_cell = RefractionStaticRefractorCellOptions(
            number_of_cell_x=4,
            size_of_cell_x_m=100.0,
            x_coordinate_origin_m=0.0,
        )

    with pytest.raises(ValueError, match='initial_bedrock_velocity_m_s'):
        RefractionStaticModelOptions(
            first_layer=RefractionStaticFirstLayerOptions(weathering_velocity_m_s=500.0),
            bedrock_velocity_mode=bedrock_velocity_mode,
            refractor_cell=refractor_cell,
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
                min_offset_m=499.9,
                max_offset_m=None,
                velocity_mode='solve_global',
            ),
        ),
        assignment_policy='exclusive_shallowest',
    )

    assert config.assignment_policy == 'exclusive_shallowest'

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


def test_refraction_layer_config_allows_half_open_contact_gates() -> None:
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

    assert config.assignment_policy == 'reject_overlap'


@pytest.mark.parametrize('policy', ('reject_overlap', 'exclusive_shallowest', 'independent'))
def test_refraction_layer_config_normalizes_assignment_policy(policy: str) -> None:
    if policy == 'reject_overlap':
        with pytest.raises(ValueError, match='layer_assignment_policy is reject_overlap'):
            RefractionStaticModelOptions(
                method='multilayer_time_term',
                first_layer=RefractionStaticFirstLayerOptions(
                    weathering_velocity_m_s=500.0
                ),
                initial_bedrock_velocity_m_s=1800.0,
                layer_assignment_policy=policy,
                layers=(
                    RefractionStaticLayerOptions(
                        kind='v2_t1',
                        min_offset_m=0.0,
                        max_offset_m=600.0,
                    ),
                    RefractionStaticLayerOptions(
                        kind='v3_t2',
                        min_offset_m=500.0,
                        max_offset_m=None,
                        initial_velocity_m_s=2600.0,
                    ),
                ),
            )
        return

    model = RefractionStaticModelOptions(
        method='multilayer_time_term',
        first_layer=RefractionStaticFirstLayerOptions(weathering_velocity_m_s=500.0),
        initial_bedrock_velocity_m_s=1800.0,
        layer_assignment_policy=policy,
        layers=(
            RefractionStaticLayerOptions(
                kind='v2_t1',
                min_offset_m=0.0,
                max_offset_m=600.0,
            ),
            RefractionStaticLayerOptions(
                kind='v3_t2',
                min_offset_m=500.0,
                max_offset_m=None,
                initial_velocity_m_s=2600.0,
            ),
        ),
    )

    config = normalize_refraction_layer_config(model)

    assert config.assignment_policy == policy
