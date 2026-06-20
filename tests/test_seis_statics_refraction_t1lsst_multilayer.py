from __future__ import annotations

import numpy as np
import pytest

from seis_statics.refraction.t1lsst import (
    RefractionT1LSSTError,
    compute_t1lsst_2layer_thicknesses,
    compute_t1lsst_2layer_thicknesses_with_status,
    compute_t1lsst_2layer_weathering_correction,
    compute_t1lsst_3layer_thicknesses,
    compute_t1lsst_3layer_thicknesses_with_status,
    compute_t1lsst_3layer_weathering_correction,
)


def test_t1lsst_2layer_scalar_formula_matches_known_truth() -> None:
    v1_m_s = 800.0
    v2_m_s = 2400.0
    v3_m_s = 3600.0
    expected_sh1_m = np.asarray([10.0], dtype=np.float64)
    expected_sh2_m = np.asarray([20.0], dtype=np.float64)
    t1_s, t2_s = _forward_t1_t2_s(
        sh1_m=expected_sh1_m,
        sh2_m=expected_sh2_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    sh1_m, sh2_m = compute_t1lsst_2layer_thicknesses(
        t1_s=t1_s,
        t2_s=t2_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    np.testing.assert_allclose(sh1_m, expected_sh1_m)
    np.testing.assert_allclose(sh2_m, expected_sh2_m)


def test_t1lsst_2layer_supports_vectorized_endpoint_local_velocities() -> None:
    v1_m_s = np.asarray([700.0, 800.0, 900.0], dtype=np.float64)
    v2_m_s = np.asarray([2200.0, 2500.0, 2800.0], dtype=np.float64)
    v3_m_s = np.asarray([3300.0, 3600.0, 4100.0], dtype=np.float64)
    expected_sh1_m = np.asarray([8.0, 10.0, 12.0], dtype=np.float64)
    expected_sh2_m = np.asarray([14.0, 18.0, 22.0], dtype=np.float64)
    t1_s, t2_s = _forward_t1_t2_s(
        sh1_m=expected_sh1_m,
        sh2_m=expected_sh2_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=t1_s,
        t2_s=t2_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
    )

    assert result.status.tolist() == ['ok', 'ok', 'ok']
    np.testing.assert_allclose(result.sh1_m, expected_sh1_m)
    np.testing.assert_allclose(result.sh2_m, expected_sh2_m)
    np.testing.assert_allclose(
        result.weathering_correction_s,
        expected_sh1_m * (1.0 / v3_m_s - 1.0 / v1_m_s)
        + expected_sh2_m * (1.0 / v3_m_s - 1.0 / v2_m_s),
    )


def test_t1lsst_2layer_statuses_invalid_velocity_order_and_nonfinite() -> None:
    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=np.asarray([0.01, 0.01, np.nan], dtype=np.float64),
        t2_s=np.asarray([0.02, 0.02, 0.02], dtype=np.float64),
        v1_m_s=800.0,
        v2_m_s=np.asarray([2400.0, 3800.0, 2400.0], dtype=np.float64),
        v3_m_s=3600.0,
    )

    assert result.status.tolist() == [
        'ok',
        'invalid_velocity_order',
        'invalid_nonfinite_input',
    ]
    assert np.isfinite(result.sh1_m[0])
    assert np.isfinite(result.sh2_m[0])
    assert np.isfinite(result.weathering_correction_s[0])
    assert np.isnan(result.sh1_m[1])
    assert np.isnan(result.sh2_m[1])
    assert np.isnan(result.weathering_correction_s[1])
    assert np.isnan(result.sh1_m[2])
    assert np.isnan(result.sh2_m[2])


def test_t1lsst_2layer_negative_sh1_blanks_dependent_outputs() -> None:
    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=np.asarray([-0.001], dtype=np.float64),
        t2_s=np.asarray([0.1], dtype=np.float64),
        v1_m_s=800.0,
        v2_m_s=2000.0,
        v3_m_s=3000.0,
    )

    assert result.status.tolist() == ['invalid_negative_thickness']
    assert np.isnan(result.sh1_m[0])
    assert np.isnan(result.sh2_m[0])
    assert np.isnan(result.weathering_correction_s[0])


def test_t1lsst_2layer_weathering_correction_rejects_invalid_order() -> None:
    with pytest.raises(RefractionT1LSSTError, match='v3_m_s must be greater'):
        compute_t1lsst_2layer_weathering_correction(
            sh1_m=np.asarray([10.0]),
            sh2_m=np.asarray([20.0]),
            v1_m_s=800.0,
            v2_m_s=2400.0,
            v3_m_s=2400.0,
        )


def test_t1lsst_3layer_scalar_formula_matches_known_truth() -> None:
    v1_m_s = 800.0
    v2_m_s = 2400.0
    v3_m_s = 3600.0
    vsub_m_s = 5000.0
    expected_sh1_m = 10.0
    expected_sh2_m = 20.0
    expected_sh3_m = 30.0
    t1_s, t2_s, t3_s = _forward_t1_t2_t3_s(
        sh1_m=expected_sh1_m,
        sh2_m=expected_sh2_m,
        sh3_m=expected_sh3_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        vsub_m_s=vsub_m_s,
    )

    sh1_m, sh2_m, sh3_m = compute_t1lsst_3layer_thicknesses(
        t1_s=t1_s,
        t2_s=t2_s,
        t3_s=t3_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        vsub_m_s=vsub_m_s,
    )

    np.testing.assert_allclose(sh1_m, expected_sh1_m)
    np.testing.assert_allclose(sh2_m, expected_sh2_m)
    np.testing.assert_allclose(sh3_m, expected_sh3_m)


def test_t1lsst_3layer_vector_formula_and_wcor() -> None:
    v1_m_s = np.asarray([700.0, 800.0, 900.0], dtype=np.float64)
    v2_m_s = np.asarray([2200.0, 2500.0, 2800.0], dtype=np.float64)
    v3_m_s = np.asarray([3300.0, 3600.0, 4100.0], dtype=np.float64)
    vsub_m_s = 5200.0
    expected_sh1_m = np.asarray([8.0, 10.0, 12.0], dtype=np.float64)
    expected_sh2_m = np.asarray([14.0, 18.0, 22.0], dtype=np.float64)
    expected_sh3_m = np.asarray([24.0, 28.0, 32.0], dtype=np.float64)
    t1_s, t2_s, t3_s = _forward_t1_t2_t3_s(
        sh1_m=expected_sh1_m,
        sh2_m=expected_sh2_m,
        sh3_m=expected_sh3_m,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        vsub_m_s=vsub_m_s,
    )

    result = compute_t1lsst_3layer_thicknesses_with_status(
        t1_s=t1_s,
        t2_s=t2_s,
        t3_s=t3_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        vsub_m_s=vsub_m_s,
    )

    assert result.status.tolist() == ['ok', 'ok', 'ok']
    np.testing.assert_allclose(result.sh1_m, expected_sh1_m)
    np.testing.assert_allclose(result.sh2_m, expected_sh2_m)
    np.testing.assert_allclose(result.sh3_m, expected_sh3_m)
    np.testing.assert_allclose(
        result.weathering_correction_s,
        expected_sh1_m * (1.0 / vsub_m_s - 1.0 / v1_m_s)
        + expected_sh2_m * (1.0 / vsub_m_s - 1.0 / v2_m_s)
        + expected_sh3_m * (1.0 / vsub_m_s - 1.0 / v3_m_s),
    )


def test_t1lsst_3layer_status_invalid_velocity_order() -> None:
    result = compute_t1lsst_3layer_thicknesses_with_status(
        t1_s=np.asarray([0.01, 0.01], dtype=np.float64),
        t2_s=np.asarray([0.02, 0.02], dtype=np.float64),
        t3_s=np.asarray([0.03, 0.03], dtype=np.float64),
        v1_m_s=800.0,
        v2_m_s=np.asarray([2400.0, 3800.0], dtype=np.float64),
        v3_m_s=3600.0,
        vsub_m_s=5000.0,
    )

    assert result.status.tolist() == ['ok', 'invalid_velocity_order']
    assert np.isfinite(result.sh1_m[0])
    assert np.isfinite(result.sh2_m[0])
    assert np.isfinite(result.sh3_m[0])
    assert np.isnan(result.sh1_m[1])
    assert np.isnan(result.sh2_m[1])
    assert np.isnan(result.sh3_m[1])
    assert np.isnan(result.weathering_correction_s[1])


def test_t1lsst_3layer_negative_sh3_blanks_dependent_outputs() -> None:
    v1_m_s = 800.0
    v2_m_s = 2400.0
    v3_m_s = 3600.0
    vsub_m_s = 5000.0
    t1_s, t2_s, t3_s = _forward_t1_t2_t3_s(
        sh1_m=np.asarray([10.0], dtype=np.float64),
        sh2_m=np.asarray([20.0], dtype=np.float64),
        sh3_m=np.asarray([0.0], dtype=np.float64),
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        vsub_m_s=vsub_m_s,
    )

    result = compute_t1lsst_3layer_thicknesses_with_status(
        t1_s=t1_s,
        t2_s=t2_s,
        t3_s=t3_s - 0.001,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        vsub_m_s=vsub_m_s,
    )

    assert result.status.tolist() == ['invalid_negative_thickness']
    assert np.isnan(result.sh1_m[0])
    assert np.isnan(result.sh2_m[0])
    assert np.isnan(result.sh3_m[0])
    assert np.isnan(result.weathering_correction_s[0])


def test_t1lsst_3layer_weathering_correction_rejects_invalid_order() -> None:
    with pytest.raises(RefractionT1LSSTError, match='vsub_m_s must be greater'):
        compute_t1lsst_3layer_weathering_correction(
            sh1_m=np.asarray([10.0]),
            sh2_m=np.asarray([20.0]),
            sh3_m=np.asarray([30.0]),
            v1_m_s=800.0,
            v2_m_s=2400.0,
            v3_m_s=3600.0,
            vsub_m_s=3600.0,
        )


def _forward_t1_t2_s(
    *,
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    t1_s = sh1_m * np.sqrt(v2_m_s**2 - v1_m_s**2) / (v1_m_s * v2_m_s)
    t2_s = (
        sh1_m * np.sqrt(v3_m_s**2 - v1_m_s**2) / (v1_m_s * v3_m_s)
        + sh2_m * np.sqrt(v3_m_s**2 - v2_m_s**2) / (v2_m_s * v3_m_s)
    )
    return np.asarray(t1_s, dtype=np.float64), np.asarray(t2_s, dtype=np.float64)


def _forward_t1_t2_t3_s(
    *,
    sh1_m: np.ndarray | float,
    sh2_m: np.ndarray | float,
    sh3_m: np.ndarray | float,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
    vsub_m_s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c12 = np.sqrt(1.0 - (np.asarray(v1_m_s) / np.asarray(v2_m_s)) ** 2)
    c13 = np.sqrt(1.0 - (np.asarray(v1_m_s) / np.asarray(v3_m_s)) ** 2)
    c23 = np.sqrt(1.0 - (np.asarray(v2_m_s) / np.asarray(v3_m_s)) ** 2)
    c1sub = np.sqrt(1.0 - (np.asarray(v1_m_s) / np.asarray(vsub_m_s)) ** 2)
    c2sub = np.sqrt(1.0 - (np.asarray(v2_m_s) / np.asarray(vsub_m_s)) ** 2)
    c3sub = np.sqrt(1.0 - (np.asarray(v3_m_s) / np.asarray(vsub_m_s)) ** 2)
    t1_s = sh1_m * c12 / v1_m_s
    t2_s = sh1_m * c13 / v1_m_s + sh2_m * c23 / v2_m_s
    t3_s = (
        sh1_m * c1sub / v1_m_s
        + sh2_m * c2sub / v2_m_s
        + sh3_m * c3sub / v3_m_s
    )
    return (
        np.asarray(t1_s, dtype=np.float64),
        np.asarray(t2_s, dtype=np.float64),
        np.asarray(t3_s, dtype=np.float64),
    )
