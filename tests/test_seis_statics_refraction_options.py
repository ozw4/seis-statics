from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from seis_statics.refraction import (
    RefractionStaticConversionOptions,
    RefractionStaticDatumOptions,
    RefractionStaticFirstLayerOptions,
    RefractionStaticLayerOptions,
    RefractionStaticModelOptions,
    RefractionStaticMoveoutOptions,
    RefractionStaticRefractorCellOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
)


def test_refraction_options_construct_shared_defaults_without_mutable_defaults() -> None:
    first_layer = RefractionStaticFirstLayerOptions(weathering_velocity_m_s=500.0)
    model = RefractionStaticModelOptions(
        first_layer=first_layer,
        initial_bedrock_velocity_m_s=1700.0,
    )
    solver_a = RefractionStaticSolverOptions()
    solver_b = RefractionStaticSolverOptions()

    assert model.first_layer_mode == 'constant'
    assert model.resolved_weathering_velocity_m_s == 500.0
    assert solver_a.robust == RefractionStaticRobustOptions()
    assert solver_a.robust is not solver_b.robust

    with pytest.raises(FrozenInstanceError):
        solver_a.damping = 1.0  # type: ignore[misc]


def test_refraction_first_layer_estimate_requires_direct_offset_gate() -> None:
    options = RefractionStaticFirstLayerOptions(
        mode='estimate_direct_arrival',
        min_direct_offset_m=100.0,
        max_direct_offset_m=500.0,
    )

    assert options.weathering_velocity_m_s is None

    with pytest.raises(ValueError, match='min_direct_offset_m'):
        RefractionStaticFirstLayerOptions(mode='estimate_direct_arrival')

    with pytest.raises(ValueError, match='weathering_velocity_m_s'):
        RefractionStaticFirstLayerOptions(
            mode='estimate_direct_arrival',
            weathering_velocity_m_s=500.0,
            min_direct_offset_m=100.0,
            max_direct_offset_m=500.0,
        )


def test_refraction_options_validate_cell_layer_model_and_datum_constraints() -> None:
    cell = RefractionStaticRefractorCellOptions(
        number_of_cell_x=3,
        size_of_cell_x_m=250.0,
        x_coordinate_origin_m=1000.0,
    )
    layer = RefractionStaticLayerOptions(
        kind='v2_t1',
        min_offset_m=100.0,
        max_offset_m=None,
        velocity_mode='solve_cell',
        initial_velocity_m_s=1800.0,
    )
    model = RefractionStaticModelOptions(
        method='multilayer_time_term',
        first_layer=RefractionStaticFirstLayerOptions(weathering_velocity_m_s=500.0),
        refractor_cell=cell,
        layers=(layer,),
    )

    assert model.enabled_refraction_layer_count == 1

    with pytest.raises(ValueError, match='size_of_cell_y_m'):
        RefractionStaticRefractorCellOptions(
            number_of_cell_x=3,
            size_of_cell_x_m=250.0,
            x_coordinate_origin_m=1000.0,
            number_of_cell_y=2,
        )

    with pytest.raises(ValueError, match='floating_datum_elevation_m'):
        RefractionStaticDatumOptions(floating_datum_mode='constant')


def test_refraction_moveout_robust_solver_and_conversion_options_validate() -> None:
    assert RefractionStaticMoveoutOptions(distance_source='offset_header').offset_byte == 37
    assert RefractionStaticSolverOptions(
        robust=RefractionStaticRobustOptions(method='sigma', min_used_fraction=1.0)
    ).robust.method == 'sigma'
    assert RefractionStaticConversionOptions(
        mode='t1lsst_multilayer',
        layer_count=3,
    ).layer_count == 3

    with pytest.raises(ValueError, match='offset_byte'):
        RefractionStaticMoveoutOptions(distance_source='offset_header', offset_byte=None)

    with pytest.raises(ValueError, match='min_used_fraction'):
        RefractionStaticRobustOptions(min_used_fraction=1.5)

    with pytest.raises(ValueError, match='layer_count'):
        RefractionStaticConversionOptions(mode='t1lsst_multilayer')
