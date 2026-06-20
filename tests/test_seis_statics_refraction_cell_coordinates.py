from __future__ import annotations

import numpy as np

from seis_statics.refraction import RefractionStaticRefractorCellOptions
from seis_statics.refraction.cell_coordinates import (
    effective_refraction_cell_grid_config,
    project_refraction_cell_coordinates,
    project_refraction_cell_points,
    refraction_cell_coordinate_metadata_from_config,
)


def test_grid_3d_coordinate_mode_passes_coordinates_through() -> None:
    projected = project_refraction_cell_points(
        x_m=np.array([10.0, 20.0]),
        y_m=np.array([30.0, 40.0]),
        mode='grid_3d',
        line_origin_x_m=1.0,
        line_origin_y_m=2.0,
        line_azimuth_deg=3.0,
    )

    np.testing.assert_allclose(projected.x_m, [10.0, 20.0])
    np.testing.assert_allclose(projected.y_m, [30.0, 40.0])
    assert projected.projected_inline_m is None
    assert projected.projected_crossline_m is None
    assert projected.qc == {
        'coordinate_mode': 'grid_3d',
        'line_origin_x_m': None,
        'line_origin_y_m': None,
        'line_azimuth_deg': None,
    }


def test_line_2d_projected_coordinates_match_reference_axis_convention() -> None:
    projected = project_refraction_cell_points(
        x_m=np.array([100.0, 110.0, 100.0]),
        y_m=np.array([200.0, 200.0, 210.0]),
        mode='line_2d_projected',
        line_origin_x_m=100.0,
        line_origin_y_m=200.0,
        line_azimuth_deg=90.0,
    )

    np.testing.assert_allclose(projected.x_m, [0.0, 10.0, 0.0])
    np.testing.assert_allclose(projected.y_m, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(projected.projected_inline_m, [0.0, 10.0, 0.0])
    np.testing.assert_allclose(projected.projected_crossline_m, [0.0, 0.0, -10.0])
    assert projected.qc == {
        'coordinate_mode': 'line_2d_projected',
        'line_origin_x_m': 100.0,
        'line_origin_y_m': 200.0,
        'line_azimuth_deg': 90.0,
        'projected_inline_m_min': 0.0,
        'projected_inline_m_max': 10.0,
        'projected_crossline_m_min': -10.0,
        'projected_crossline_m_max': 0.0,
    }


def test_line_2d_source_receiver_projection_qc_and_effective_grid_config() -> None:
    config = RefractionStaticRefractorCellOptions(
        number_of_cell_x=4,
        size_of_cell_x_m=25.0,
        x_coordinate_origin_m=0.0,
        coordinate_mode='line_2d_projected',
        line_origin_x_m=10.0,
        line_origin_y_m=20.0,
        line_azimuth_deg=0.0,
    )

    projected = project_refraction_cell_coordinates(
        source_x_m=np.array([10.0, 15.0]),
        source_y_m=np.array([20.0, 30.0]),
        receiver_x_m=np.array([20.0, 5.0]),
        receiver_y_m=np.array([40.0, 10.0]),
        mode=config.coordinate_mode,
        line_origin_x_m=config.line_origin_x_m,
        line_origin_y_m=config.line_origin_y_m,
        line_azimuth_deg=config.line_azimuth_deg,
    )
    effective = effective_refraction_cell_grid_config(config)

    np.testing.assert_allclose(projected.source_x_m, [0.0, 10.0])
    np.testing.assert_allclose(projected.source_y_m, [0.0, 0.0])
    np.testing.assert_allclose(projected.receiver_x_m, [20.0, -10.0])
    np.testing.assert_allclose(projected.receiver_y_m, [0.0, 0.0])
    np.testing.assert_allclose(projected.source_projected_crossline_m, [0.0, 5.0])
    np.testing.assert_allclose(projected.receiver_projected_crossline_m, [10.0, -5.0])
    assert projected.qc == {
        'coordinate_mode': 'line_2d_projected',
        'line_origin_x_m': 10.0,
        'line_origin_y_m': 20.0,
        'line_azimuth_deg': 0.0,
        'projected_inline_m_min': 0.0,
        'projected_inline_m_max': 10.0,
        'projected_crossline_m_min': 0.0,
        'projected_crossline_m_max': 5.0,
        'source_projected_inline_m_min': 0.0,
        'source_projected_inline_m_max': 10.0,
        'source_projected_crossline_m_min': 0.0,
        'source_projected_crossline_m_max': 5.0,
        'receiver_projected_inline_m_min': -10.0,
        'receiver_projected_inline_m_max': 20.0,
        'receiver_projected_crossline_m_min': -5.0,
        'receiver_projected_crossline_m_max': 10.0,
    }
    assert effective.number_of_cell_y == 1
    assert effective.size_of_cell_y_m is None
    assert effective.y_coordinate_origin_m == 0.0
    assert refraction_cell_coordinate_metadata_from_config(config) == {
        'coordinate_mode': 'line_2d_projected',
        'line_origin_x_m': 10.0,
        'line_origin_y_m': 20.0,
        'line_azimuth_deg': 0.0,
    }
