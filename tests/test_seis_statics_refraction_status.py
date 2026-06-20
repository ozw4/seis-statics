from __future__ import annotations

import math

from seis_statics.refraction import (
    LOW_FOLD_CELL_REJECTION_REASON,
    LOW_FOLD_CELL_VELOCITY_STATUS,
    REFRACTION_STATIC_STATUSES,
    classify_refraction_endpoint_static_status,
)


def _status(**overrides: object) -> str:
    values: dict[str, object] = {
        'node_missing': False,
        'x_m': 1.0,
        'y_m': 2.0,
        'surface_elevation_m': 3.0,
        't1_s': 0.1,
        'weathering_thickness_m': 10.0,
        'total_shift_s': 0.02,
        'solution_status': 'solved',
        'weathering_status': 'ok',
        'datum_status': 'ok',
    }
    values.update(overrides)
    return classify_refraction_endpoint_static_status(**values)


def test_refraction_status_classification_matches_reference_priority() -> None:
    assert _status() == 'ok'
    assert _status(node_missing=True) == 'missing_linkage'
    assert _status(x_m=math.nan) == 'missing_geometry'
    assert _status(datum_status='outside_refractor_cell_grid') == 'outside_refractor_cell_grid'
    assert _status(solution_status='inactive') == 'inactive_endpoint'
    assert _status(weathering_status='low_fold') == 'insufficient_pick_fold'
    assert _status(solution_status='missing_solution') == 'invalid_t1'
    assert _status(weathering_status='negative_weathering_thickness') == 'invalid_weathering_thickness'
    assert _status(datum_status='invalid_datum_shift') == 'invalid_datum'
    assert _status(total_shift_s=math.nan) == 'not_applied'


def test_refraction_status_vocabulary_and_cell_velocity_constants() -> None:
    assert 'v2_not_greater_than_v1' in REFRACTION_STATIC_STATUSES
    assert LOW_FOLD_CELL_REJECTION_REASON == 'below_min_observations_per_cell'
    assert LOW_FOLD_CELL_VELOCITY_STATUS == 'low_fold'
