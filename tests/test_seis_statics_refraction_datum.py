from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from seis_statics.refraction.datum import (
    RefractionDatumError,
    ResolvedFloatingDatum,
    build_refraction_datum_statics,
    compute_refraction_datum_elevation_shift_s,
    compute_refraction_datum_elevation_shift_scalar_s,
    smooth_refraction_floating_datum_elevation,
)
from seis_statics.refraction.field_composition import (
    compose_refraction_endpoint_field_corrections,
    compose_refraction_trace_field_corrections,
)


def _keys(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def _ids(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _values(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _statuses(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype='<U64')


def _base_kwargs() -> dict[str, object]:
    return {
        'source_endpoint_key': _keys(['s0', 's1']),
        'source_endpoint_id': _ids([0, 1]),
        'source_node_id': _ids([10, 11]),
        'source_surface_elevation_m': _values([100.0, 90.0]),
        'source_refractor_elevation_m': _values([70.0, 60.0]),
        'source_weathering_replacement_shift_s': _values([-0.010, -0.008]),
        'source_weathering_replacement_status': _statuses(['ok', 'ok']),
        'receiver_endpoint_key': _keys(['r0', 'r1']),
        'receiver_endpoint_id': _ids([0, 1]),
        'receiver_node_id': _ids([20, 21]),
        'receiver_surface_elevation_m': _values([95.0, 85.0]),
        'receiver_refractor_elevation_m': _values([68.0, 55.0]),
        'receiver_weathering_replacement_shift_s': _values([-0.006, -0.004]),
        'receiver_weathering_replacement_status': _statuses(['ok', 'ok']),
        'source_endpoint_key_sorted': _keys(['s0', 's1']),
        'receiver_endpoint_key_sorted': _keys(['r0', 'r1']),
        'replacement_velocity_m_s': 1000.0,
    }


@pytest.mark.parametrize(
    ('mode', 'expected'),
    [
        ('none', [-0.016, -0.012]),
        ('floating_only', [-0.031, -0.027]),
        ('flat_only', [-0.051, -0.027]),
        ('floating_and_flat', [-0.051, -0.027]),
    ],
)
def test_refraction_datum_modes_compose_expected_trace_shift(
    mode: str,
    expected: list[float],
) -> None:
    result = build_refraction_datum_statics(
        **_base_kwargs(),
        mode=mode,
        floating_datum=ResolvedFloatingDatum(
            source_elevation_m=_values([92.0, 82.0]),
            receiver_elevation_m=_values([88.0, 78.0]),
        ),
        flat_datum_elevation_m=80.0,
    )

    np.testing.assert_allclose(result.refraction_trace_shift_s_sorted, expected)
    np.testing.assert_allclose(result.final_trace_shift_s_sorted, expected)
    np.testing.assert_array_equal(result.trace_static_status_sorted, ['ok', 'ok'])
    np.testing.assert_array_equal(result.trace_static_valid_mask_sorted, [True, True])
    assert result.qc['sign_convention'] == 'corrected(t) = raw(t - shift_s)'


def test_refraction_datum_accepts_constant_and_array_resolved_datums() -> None:
    result = build_refraction_datum_statics(
        **_base_kwargs(),
        mode='floating_only',
        source_floating_datum_elevation_m=82.0,
        receiver_floating_datum_elevation_m=_values([88.0, 78.0]),
    )

    np.testing.assert_allclose(
        result.source_endpoint_datum.floating_datum_elevation_m,
        [82.0, 82.0],
    )
    np.testing.assert_allclose(
        result.receiver_endpoint_datum.floating_datum_elevation_m,
        [88.0, 78.0],
    )
    np.testing.assert_allclose(
        result.source_endpoint_datum.floating_datum_shift_s,
        [-0.018, -0.008],
    )


def test_refraction_datum_shift_scalar_and_array_api() -> None:
    assert compute_refraction_datum_elevation_shift_scalar_s(
        elevation_m=100.0,
        datum_elevation_m=90.0,
        replacement_velocity_m_s=1000.0,
    ) == pytest.approx(-0.010)

    np.testing.assert_allclose(
        compute_refraction_datum_elevation_shift_s(
            elevation_m=_values([100.0, 80.0]),
            datum_elevation_m=_values([90.0, 85.0]),
            replacement_velocity_m_s=1000.0,
        ),
        [-0.010, 0.005],
    )


def test_refraction_datum_status_nan_checks_and_priority() -> None:
    result = build_refraction_datum_statics(
        **{
            **_base_kwargs(),
            'source_weathering_replacement_status': _statuses(['low_fold', 'ok']),
            'receiver_refractor_elevation_m': _values([68.0, 81.0]),
        },
        mode='floating_and_flat',
        floating_datum=ResolvedFloatingDatum(
            source_elevation_m=_values([92.0, np.nan]),
            receiver_elevation_m=_values([88.0, 78.0]),
        ),
        flat_datum_elevation_m=80.0,
    )

    np.testing.assert_array_equal(
        result.source_endpoint_datum.datum_static_status,
        ['low_fold', 'invalid_floating_datum_elevation'],
    )
    np.testing.assert_array_equal(
        result.receiver_endpoint_datum.datum_static_status,
        ['ok', 'floating_datum_below_refractor'],
    )
    np.testing.assert_array_equal(
        result.trace_static_status_sorted,
        ['low_fold', 'floating_datum_below_refractor'],
    )
    assert np.isnan(result.refraction_trace_shift_s_sorted[0])
    assert np.isnan(result.refraction_trace_shift_s_sorted[1])


def test_refraction_datum_flat_constraints_and_max_shift() -> None:
    below = build_refraction_datum_statics(
        **_base_kwargs(),
        mode='flat_only',
        flat_datum_elevation_m=50.0,
        allow_flat_datum_below_refractor=False,
    )

    np.testing.assert_array_equal(
        below.source_endpoint_datum.datum_static_status,
        ['flat_datum_below_refractor', 'flat_datum_below_refractor'],
    )

    above = build_refraction_datum_statics(
        **_base_kwargs(),
        mode='flat_only',
        flat_datum_elevation_m=120.0,
        allow_flat_datum_above_topography=False,
    )

    np.testing.assert_array_equal(
        above.receiver_endpoint_datum.datum_static_status,
        ['invalid_flat_datum_elevation', 'invalid_flat_datum_elevation'],
    )

    max_shift = build_refraction_datum_statics(
        **_base_kwargs(),
        mode='flat_only',
        flat_datum_elevation_m=80.0,
        max_abs_datum_shift_ms=20.0,
    )

    np.testing.assert_array_equal(
        max_shift.trace_static_status_sorted,
        ['invalid_datum_shift', 'invalid_datum_shift'],
    )
    np.testing.assert_array_equal(
        max_shift.trace_static_valid_mask_sorted,
        [False, False],
    )


def test_refraction_datum_composes_field_shift_into_final_trace_shift() -> None:
    source_field = compose_refraction_endpoint_field_corrections(
        endpoint_kind='source',
        endpoint_key=_keys(['s0', 's1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([10, 11]),
        manual_static_shift_s=_values([0.001, 0.0]),
        manual_static_status=_statuses(['ok', 'missing_manual_static']),
    )
    receiver_field = compose_refraction_endpoint_field_corrections(
        endpoint_kind='receiver',
        endpoint_key=_keys(['r0', 'r1']),
        endpoint_id=_ids([0, 1]),
        node_id=_ids([20, 21]),
        manual_static_shift_s=_values([0.004, -0.001]),
        manual_static_status=_statuses(['ok', 'ok']),
    )
    trace_field = compose_refraction_trace_field_corrections(
        source_endpoint_field=source_field,
        receiver_endpoint_field=receiver_field,
        source_endpoint_key_sorted=_keys(['s0', 's1']),
        receiver_endpoint_key_sorted=_keys(['r0', 'r1']),
    )

    result = build_refraction_datum_statics(
        **_base_kwargs(),
        mode='flat_only',
        flat_datum_elevation_m=80.0,
        trace_field_correction=trace_field,
        apply_field_correction_to_trace_shift=True,
    )

    np.testing.assert_allclose(result.refraction_trace_shift_s_sorted, [-0.051, -0.027])
    np.testing.assert_allclose(result.applied_field_shift_s_sorted, [0.005, -0.001])
    np.testing.assert_allclose(result.final_trace_shift_s_sorted, [-0.046, -0.028])


def test_refraction_floating_datum_smoothing_is_array_only() -> None:
    np.testing.assert_allclose(
        smooth_refraction_floating_datum_elevation(
            _values([100.0, np.nan, 80.0, 70.0, 60.0]),
            window_nodes=3,
            method='moving_average',
        ),
        [100.0, 90.0, 75.0, 70.0, 65.0],
    )
    np.testing.assert_allclose(
        smooth_refraction_floating_datum_elevation(
            _values([100.0, np.nan, 80.0, 70.0, 60.0]),
            window_nodes=3,
            method='median',
        ),
        [100.0, 90.0, 75.0, 70.0, 65.0],
    )

    with pytest.raises(RefractionDatumError, match='window_nodes'):
        smooth_refraction_floating_datum_elevation(_values([1.0, 2.0]), window_nodes=2)


def test_refraction_datum_module_has_no_artifact_or_file_io_imports() -> None:
    path = Path(__file__).resolve().parents[1] / 'src/seis_statics/refraction/datum.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    imported: list[str] = []
    calls: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or '')
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

    assert 'pathlib' not in imported
    assert 'Path' not in imported
    assert 'ArtifactWriter' not in imported
    assert 'ArtifactRegistry' not in imported
    assert 'open' not in calls
