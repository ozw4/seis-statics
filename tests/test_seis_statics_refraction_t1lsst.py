from __future__ import annotations

import numpy as np
import pytest

from seis_statics.refraction.t1lsst import (
    T1LSST_SIGN_CONVENTION,
    RefractionT1LSST1LayerEndpointComponents,
    RefractionT1LSSTError,
    compose_t1lsst_1layer_endpoint_component_rows,
    compute_t1lsst_1layer_thickness,
    compute_t1lsst_1layer_weathering_correction,
)


def test_t1lsst_1layer_scalar_formula_matches_canonical_design() -> None:
    t1_s = 0.010
    v1_m_s = 800.0
    v2_m_s = 2400.0

    sh1 = compute_t1lsst_1layer_thickness(
        np.asarray([t1_s]),
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
    )[0]
    wcor = compute_t1lsst_1layer_weathering_correction(
        np.asarray([sh1]),
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
    )[0]

    expected_sh1 = t1_s * v2_m_s * v1_m_s / np.sqrt(v2_m_s**2 - v1_m_s**2)
    assert sh1 == pytest.approx(expected_sh1)
    assert wcor == pytest.approx(expected_sh1 * (1.0 / v2_m_s - 1.0 / v1_m_s))
    assert wcor < 0.0


def test_t1lsst_1layer_vector_formula_supports_local_v2() -> None:
    t1_s = np.asarray([0.010, 0.012, 0.014], dtype=np.float64)
    v1_m_s = 800.0
    v2_m_s = np.asarray([2200.0, 2500.0, 3000.0], dtype=np.float64)

    sh1 = compute_t1lsst_1layer_thickness(
        t1_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
    )
    wcor = compute_t1lsst_1layer_weathering_correction(
        sh1,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
    )

    np.testing.assert_allclose(
        sh1,
        t1_s * v2_m_s * v1_m_s / np.sqrt(v2_m_s**2 - v1_m_s**2),
    )
    np.testing.assert_allclose(wcor, sh1 * (1.0 / v2_m_s - 1.0 / v1_m_s))
    assert sh1.dtype == np.float64
    assert wcor.flags.c_contiguous


def test_t1lsst_1layer_nan_t1_propagates_to_sh1_and_wcor() -> None:
    sh1 = compute_t1lsst_1layer_thickness(
        np.asarray([0.010, np.nan], dtype=np.float64),
        v1_m_s=800.0,
        v2_m_s=2400.0,
    )
    wcor = compute_t1lsst_1layer_weathering_correction(
        sh1,
        v1_m_s=800.0,
        v2_m_s=2400.0,
    )

    assert np.isfinite(sh1[0])
    assert np.isnan(sh1[1])
    assert np.isfinite(wcor[0])
    assert np.isnan(wcor[1])


def test_t1lsst_1layer_rejects_v2_less_than_or_equal_v1() -> None:
    with pytest.raises(RefractionT1LSSTError, match='v2_m_s must be greater'):
        compute_t1lsst_1layer_thickness(
            np.asarray([0.010]),
            v1_m_s=800.0,
            v2_m_s=800.0,
        )

    with pytest.raises(RefractionT1LSSTError, match='v2_m_s must be greater'):
        compute_t1lsst_1layer_weathering_correction(
            np.asarray([10.0]),
            v1_m_s=800.0,
            v2_m_s=799.0,
        )


def test_t1lsst_1layer_component_row_composition_is_pure_and_sign_conventioned() -> None:
    components = RefractionT1LSST1LayerEndpointComponents(
        endpoint_kind='source',
        endpoint_key=np.asarray(['s0', 's1'], dtype='<U2'),
        node_id=np.asarray([10, 11], dtype=np.int64),
        x_m=np.asarray([100.0, 200.0], dtype=np.float64),
        y_m=np.asarray([300.0, 400.0], dtype=np.float64),
        surface_elevation_m=np.asarray([80.0, 81.0], dtype=np.float64),
        floating_datum_elevation_m=np.asarray([75.0, np.nan], dtype=np.float64),
        flat_datum_elevation_m=np.asarray([70.0, 70.0], dtype=np.float64),
        t1_s=np.asarray([0.010, 0.012], dtype=np.float64),
        v1_m_s=800.0,
        v2_m_s=np.asarray([2400.0, 2600.0], dtype=np.float64),
        sh1_m=np.asarray([8.485281374, 9.9], dtype=np.float64),
        refractor_elevation_m=np.asarray([71.514718626, 71.1], dtype=np.float64),
        weathering_correction_s=np.asarray([-0.007071, -0.00825], dtype=np.float64),
        floating_datum_correction_s=np.asarray([0.001, np.nan], dtype=np.float64),
        flat_datum_correction_s=np.asarray([0.002, 0.003], dtype=np.float64),
        total_static_s=np.asarray([-0.004071, -0.00525], dtype=np.float64),
        solution_status=np.asarray(['solved', 'solved'], dtype='<U16'),
        weathering_status=np.asarray(['ok', 'zero_thickness'], dtype='<U16'),
        datum_status=np.asarray(['ok', 'ok'], dtype='<U16'),
        static_status=np.asarray(['ok', 'ok'], dtype='<U16'),
    )

    rows = compose_t1lsst_1layer_endpoint_component_rows(components)

    assert len(rows) == 2
    assert rows[0]['endpoint_kind'] == 'source'
    assert rows[0]['endpoint_key'] == 's0'
    assert rows[0]['node_id'] == 10
    assert rows[0]['t1_ms'] == pytest.approx(10.0)
    assert rows[0]['weathering_correction_ms'] == pytest.approx(-7.071)
    assert rows[0]['elevation_correction_ms'] == pytest.approx(3.0)
    assert rows[0]['total_applied_shift_ms'] == rows[0]['total_static_ms']
    assert rows[0]['sign_convention'] == T1LSST_SIGN_CONVENTION
    assert rows[1]['floating_datum_elevation_m'] == ''
    assert rows[1]['elevation_correction_ms'] == ''
