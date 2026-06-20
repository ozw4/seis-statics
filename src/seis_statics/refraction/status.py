"""Shared status vocabulary for refraction statics outputs."""

from __future__ import annotations

import numpy as np


REFRACTION_STATIC_STATUSES: frozenset[str] = frozenset(
    {
        'ok',
        'inactive_endpoint',
        'missing_geometry',
        'missing_linkage',
        'insufficient_pick_fold',
        'invalid_t1',
        'invalid_datum',
        'not_applied',
        'solved',
        'not_observed',
        'inactive',
        'low_fold',
        'clipped_lower',
        'clipped_upper',
        'clipped_half_intercept_lower',
        'clipped_half_intercept_upper',
        'zero_thickness',
        'exceeds_max_abs_shift',
        'exceeds_max_thickness',
        'invalid_shift',
        'invalid_solution',
        'missing_solution',
        'invalid_velocity',
        'invalid_bedrock_velocity',
        'invalid_surface_elevation',
        'invalid_floating_datum_elevation',
        'invalid_flat_datum_elevation',
        'invalid_weathering_replacement',
        'invalid_weathering_thickness',
        'invalid_nonfinite_input',
        'invalid_velocity_order',
        'invalid_negative_thickness',
        'negative_weathering_thickness',
        'negative_thickness',
        'invalid_half_intercept',
        'invalid_refractor_elevation',
        'invalid_datum_shift',
        'floating_datum_below_refractor',
        'flat_datum_below_refractor',
        'missing_endpoint',
        'missing_node',
        'outside_refractor_cell_grid',
        'inactive_v2_cell',
        'low_fold_v2_cell',
        'invalid_local_v2',
        'v2_not_greater_than_v1',
    }
)

LOCAL_V2_STATUS_VALUES: frozenset[str] = frozenset(
    {
        'outside_refractor_cell_grid',
        'inactive_v2_cell',
        'low_fold_v2_cell',
        'invalid_local_v2',
        'v2_not_greater_than_v1',
    }
)


def classify_refraction_endpoint_static_status(
    *,
    node_missing: bool,
    x_m: object,
    y_m: object,
    surface_elevation_m: object,
    t1_s: object,
    weathering_thickness_m: object,
    total_shift_s: object,
    solution_status: object,
    weathering_status: object,
    datum_status: object,
) -> str:
    """Classify final endpoint static status from solution/weathering/datum inputs."""
    solution = str(solution_status)
    weathering = str(weathering_status)
    datum = str(datum_status)
    if node_missing or 'missing_node' in {solution, weathering, datum}:
        return 'missing_linkage'
    if not all(np.isfinite(_float_or_nan(value)) for value in (x_m, y_m, surface_elevation_m)):
        return 'missing_geometry'
    for status in (datum, weathering, solution):
        if status in LOCAL_V2_STATUS_VALUES:
            return status
        if status in {'invalid_nonfinite_input', 'invalid_velocity_order'}:
            return status
    if 'inactive' in {solution, weathering, datum}:
        return 'inactive_endpoint'
    if 'low_fold' in {solution, weathering, datum}:
        return 'insufficient_pick_fold'
    if (
        not np.isfinite(_float_or_nan(t1_s))
        or solution in {'invalid_solution', 'missing_solution'}
        or weathering == 'invalid_half_intercept'
    ):
        return 'invalid_t1'
    if (
        not np.isfinite(_float_or_nan(weathering_thickness_m))
        or weathering
        in {
            'invalid_weathering_thickness',
            'invalid_negative_thickness',
            'negative_weathering_thickness',
            'negative_thickness',
            'exceeds_max_thickness',
            'invalid_weathering_replacement',
        }
        or datum == 'invalid_weathering_replacement'
    ):
        return 'invalid_weathering_thickness'
    if datum in {
        'invalid_datum_shift',
        'invalid_floating_datum_elevation',
        'invalid_flat_datum_elevation',
        'floating_datum_below_refractor',
        'flat_datum_below_refractor',
    }:
        return 'invalid_datum'
    for status in (datum, weathering, solution):
        if status not in {'ok', 'solved', 'zero_thickness'}:
            return status
    if not np.isfinite(_float_or_nan(total_shift_s)):
        return 'not_applied'
    return 'ok'


def _float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out if np.isfinite(out) else float('nan')


__all__ = [
    'LOCAL_V2_STATUS_VALUES',
    'REFRACTION_STATIC_STATUSES',
    'classify_refraction_endpoint_static_status',
]
