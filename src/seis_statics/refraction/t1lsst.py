"""Pure T1LSST formula helpers for refraction statics.

Canonical formula and sign-convention references:
``docs/refraction_static.md#t1lsst-1-layer-components`` and
``docs/statics/refraction_multilayer_time_term.md#5-t1lsst-formulas``.

The repo static-shift convention is ``corrected(t) = raw(t - shift_s)``.
Under normal velocity order, WCOR is negative because slower near-surface
intervals are replaced by faster deeper velocities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from seis_statics._validation import coerce_finite_float as _coerce_finite_float


T1LSST_SIGN_CONVENTION = 'corrected(t) = raw(t - shift_s)'
_STATUS_DTYPE = '<U32'


class RefractionT1LSSTError(ValueError):
    """Raised when T1LSST values cannot be computed."""


@dataclass(frozen=True)
class RefractionT1LSST2LayerThicknessResult:
    """Two-layer T1LSST thicknesses with endpoint conversion status."""

    sh1_m: np.ndarray
    sh2_m: np.ndarray
    status: np.ndarray
    weathering_correction_s: np.ndarray | None = None


@dataclass(frozen=True)
class RefractionT1LSST3LayerThicknessResult:
    """Three-layer T1LSST thicknesses with endpoint conversion status."""

    sh1_m: np.ndarray
    sh2_m: np.ndarray
    sh3_m: np.ndarray
    status: np.ndarray
    weathering_correction_s: np.ndarray | None = None


@dataclass(frozen=True)
class RefractionT1LSST1LayerEndpointComponents:
    """Pure endpoint values used to compose IRAS-style T1LSST component rows."""

    endpoint_kind: Literal['source', 'receiver']
    endpoint_key: np.ndarray
    node_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    surface_elevation_m: np.ndarray
    floating_datum_elevation_m: np.ndarray
    flat_datum_elevation_m: np.ndarray
    t1_s: np.ndarray
    v1_m_s: float | np.ndarray
    v2_m_s: float | np.ndarray
    sh1_m: np.ndarray
    refractor_elevation_m: np.ndarray
    weathering_correction_s: np.ndarray
    floating_datum_correction_s: np.ndarray
    flat_datum_correction_s: np.ndarray
    total_static_s: np.ndarray
    solution_status: np.ndarray
    weathering_status: np.ndarray
    datum_status: np.ndarray
    static_status: np.ndarray


def compute_t1lsst_1layer_thickness(
    t1_s: np.ndarray,
    v1_m_s: float,
    v2_m_s: float | np.ndarray,
) -> np.ndarray:
    """Compute one-layer ``SH1`` weathering thickness from T1, V1, and V2."""
    t1 = _coerce_float_array(t1_s, name='t1_s', allow_nonfinite=True)
    v1 = _positive_finite(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    t1, v2 = _broadcast_t1lsst_arrays((t1, v2), names=('t1_s', 'v2_m_s'))
    if np.any(v2 <= v1):
        raise RefractionT1LSSTError('v2_m_s must be greater than v1_m_s')
    denom = np.sqrt(v2 * v2 - v1 * v1)
    return np.ascontiguousarray(t1 * v2 * v1 / denom, dtype=np.float64)


def compute_t1lsst_1layer_weathering_correction(
    sh1_m: np.ndarray,
    v1_m_s: float,
    v2_m_s: float | np.ndarray,
) -> np.ndarray:
    """Compute one-layer ``WCOR`` from SH1, V1, and V2 in seconds."""
    sh1 = _coerce_float_array(sh1_m, name='sh1_m', allow_nonfinite=True)
    v1 = _positive_finite(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    sh1, v2 = _broadcast_t1lsst_arrays((sh1, v2), names=('sh1_m', 'v2_m_s'))
    if np.any(v2 <= v1):
        raise RefractionT1LSSTError('v2_m_s must be greater than v1_m_s')
    return np.ascontiguousarray(sh1 * (1.0 / v2 - 1.0 / v1), dtype=np.float64)


def compute_t1lsst_2layer_thicknesses(
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two-layer ``SH1`` and ``SH2`` thicknesses from T1/T2 terms."""
    result = compute_t1lsst_2layer_thicknesses_with_status(
        t1_s=t1_s,
        t2_s=t2_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        strict_velocity_order=True,
    )
    return result.sh1_m, result.sh2_m


def compute_t1lsst_2layer_thicknesses_with_status(
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
    *,
    strict_velocity_order: bool = False,
) -> RefractionT1LSST2LayerThicknessResult:
    """Compute two-layer ``SH1``/``SH2`` and status-code invalid endpoints."""
    if strict_velocity_order:
        t1, t2, v1, v2, v3 = _coerce_2layer_inputs(
            t1_s=t1_s,
            t2_s=t2_s,
            v1_m_s=v1_m_s,
            v2_m_s=v2_m_s,
            v3_m_s=v3_m_s,
        )
        invalid_nonfinite = np.zeros(t1.shape, dtype=bool)
        invalid_velocity_order = np.zeros(t1.shape, dtype=bool)
    else:
        t1, t2, v1, v2, v3 = _coerce_2layer_status_inputs(
            t1_s=t1_s,
            t2_s=t2_s,
            v1_m_s=v1_m_s,
            v2_m_s=v2_m_s,
            v3_m_s=v3_m_s,
        )
        invalid_nonfinite = ~(
            np.isfinite(t1)
            & np.isfinite(t2)
            & np.isfinite(v1)
            & np.isfinite(v2)
            & np.isfinite(v3)
        )
        invalid_velocity_order = (
            ~invalid_nonfinite
            & ((v1 <= 0.0) | (v2 <= v1) | (v3 <= v2))
        )

    status = np.full(t1.shape, 'ok', dtype=_STATUS_DTYPE)
    status[invalid_nonfinite] = 'invalid_nonfinite_input'
    status[invalid_velocity_order] = 'invalid_velocity_order'
    sh1 = np.full(t1.shape, np.nan, dtype=np.float64)
    sh2 = np.full(t1.shape, np.nan, dtype=np.float64)
    valid = status == 'ok'
    if np.any(valid):
        valid_t1 = t1[valid]
        valid_t2 = t2[valid]
        valid_v1 = v1[valid]
        valid_v2 = v2[valid]
        valid_v3 = v3[valid]
        c12 = np.sqrt(1.0 - (valid_v1 / valid_v2) ** 2)
        c13 = np.sqrt(1.0 - (valid_v1 / valid_v3) ** 2)
        c23 = np.sqrt(1.0 - (valid_v2 / valid_v3) ** 2)
        valid_sh1 = valid_t1 * valid_v1 / c12
        valid_sh2 = (
            (valid_t2 - valid_sh1 * c13 / valid_v1)
            * valid_v2
            / c23
        )
        sh1[valid] = valid_sh1
        sh2[valid] = valid_sh2

    negative_sh1 = np.isfinite(sh1) & (sh1 < 0.0)
    negative_sh2 = np.isfinite(sh2) & (sh2 < 0.0)
    status[negative_sh1 | negative_sh2] = 'invalid_negative_thickness'
    sh1[negative_sh1] = np.nan
    sh2[negative_sh1 | negative_sh2] = np.nan
    wcor = np.full(t1.shape, np.nan, dtype=np.float64)
    wcor_valid = status == 'ok'
    if np.any(wcor_valid):
        wcor[wcor_valid] = (
            sh1[wcor_valid] * (1.0 / v3[wcor_valid] - 1.0 / v1[wcor_valid])
            + sh2[wcor_valid] * (1.0 / v3[wcor_valid] - 1.0 / v2[wcor_valid])
        )
    return RefractionT1LSST2LayerThicknessResult(
        sh1_m=np.array(sh1, dtype=np.float64, copy=True, order='C'),
        sh2_m=np.array(sh2, dtype=np.float64, copy=True, order='C'),
        status=np.array(status, dtype=_STATUS_DTYPE, copy=True, order='C'),
        weathering_correction_s=np.array(wcor, dtype=np.float64, copy=True, order='C'),
    )


def compute_t1lsst_2layer_weathering_correction(
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
) -> np.ndarray:
    """Compute two-layer replacement ``WCOR`` to V3 in seconds."""
    sh1 = _coerce_float_array(sh1_m, name='sh1_m', allow_nonfinite=True)
    sh2 = _coerce_float_array(sh2_m, name='sh2_m', allow_nonfinite=True)
    v1 = _positive_finite_float_array(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    v3 = _positive_finite_float_array(v3_m_s, name='v3_m_s')
    sh1, sh2, v1, v2, v3 = _broadcast_t1lsst_arrays(
        (sh1, sh2, v1, v2, v3),
        names=('sh1_m', 'sh2_m', 'v1_m_s', 'v2_m_s', 'v3_m_s'),
    )
    _validate_2layer_velocity_order(v1=v1, v2=v2, v3=v3)
    return np.array(
        sh1 * (1.0 / v3 - 1.0 / v1)
        + sh2 * (1.0 / v3 - 1.0 / v2),
        dtype=np.float64,
        copy=True,
        order='C',
    )


def compute_t1lsst_3layer_thicknesses(
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    t3_s: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
    vsub_m_s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute three-layer ``SH1``/``SH2``/``SH3`` thicknesses."""
    result = compute_t1lsst_3layer_thicknesses_with_status(
        t1_s=t1_s,
        t2_s=t2_s,
        t3_s=t3_s,
        v1_m_s=v1_m_s,
        v2_m_s=v2_m_s,
        v3_m_s=v3_m_s,
        vsub_m_s=vsub_m_s,
        strict_velocity_order=True,
    )
    return result.sh1_m, result.sh2_m, result.sh3_m


def compute_t1lsst_3layer_thicknesses_with_status(
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    t3_s: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
    vsub_m_s: np.ndarray | float,
    *,
    strict_velocity_order: bool = False,
) -> RefractionT1LSST3LayerThicknessResult:
    """Compute three-layer thicknesses and status-code invalid endpoints."""
    if strict_velocity_order:
        t1, t2, t3, v1, v2, v3, vsub = _coerce_3layer_inputs(
            t1_s=t1_s,
            t2_s=t2_s,
            t3_s=t3_s,
            v1_m_s=v1_m_s,
            v2_m_s=v2_m_s,
            v3_m_s=v3_m_s,
            vsub_m_s=vsub_m_s,
        )
        invalid_nonfinite = np.zeros(t1.shape, dtype=bool)
        invalid_velocity_order = np.zeros(t1.shape, dtype=bool)
    else:
        t1, t2, t3, v1, v2, v3, vsub = _coerce_3layer_status_inputs(
            t1_s=t1_s,
            t2_s=t2_s,
            t3_s=t3_s,
            v1_m_s=v1_m_s,
            v2_m_s=v2_m_s,
            v3_m_s=v3_m_s,
            vsub_m_s=vsub_m_s,
        )
        invalid_nonfinite = ~(
            np.isfinite(t1)
            & np.isfinite(t2)
            & np.isfinite(t3)
            & np.isfinite(v1)
            & np.isfinite(v2)
            & np.isfinite(v3)
            & np.isfinite(vsub)
        )
        invalid_velocity_order = (
            ~invalid_nonfinite
            & ((v1 <= 0.0) | (v2 <= v1) | (v3 <= v2) | (vsub <= v3))
        )

    status = np.full(t1.shape, 'ok', dtype=_STATUS_DTYPE)
    status[invalid_nonfinite] = 'invalid_nonfinite_input'
    status[invalid_velocity_order] = 'invalid_velocity_order'
    sh1 = np.full(t1.shape, np.nan, dtype=np.float64)
    sh2 = np.full(t1.shape, np.nan, dtype=np.float64)
    sh3 = np.full(t1.shape, np.nan, dtype=np.float64)
    valid = status == 'ok'
    if np.any(valid):
        valid_t1 = t1[valid]
        valid_t2 = t2[valid]
        valid_t3 = t3[valid]
        valid_v1 = v1[valid]
        valid_v2 = v2[valid]
        valid_v3 = v3[valid]
        valid_vsub = vsub[valid]
        c12 = np.sqrt(1.0 - (valid_v1 / valid_v2) ** 2)
        c13 = np.sqrt(1.0 - (valid_v1 / valid_v3) ** 2)
        c23 = np.sqrt(1.0 - (valid_v2 / valid_v3) ** 2)
        c1sub = np.sqrt(1.0 - (valid_v1 / valid_vsub) ** 2)
        c2sub = np.sqrt(1.0 - (valid_v2 / valid_vsub) ** 2)
        c3sub = np.sqrt(1.0 - (valid_v3 / valid_vsub) ** 2)
        valid_sh1 = valid_t1 * valid_v1 / c12
        valid_sh2 = (
            (valid_t2 - valid_sh1 * c13 / valid_v1)
            * valid_v2
            / c23
        )
        valid_sh3 = (
            valid_t3
            - valid_sh1 * c1sub / valid_v1
            - valid_sh2 * c2sub / valid_v2
        ) * valid_v3 / c3sub
        sh1[valid] = valid_sh1
        sh2[valid] = valid_sh2
        sh3[valid] = valid_sh3

    invalid_negative = (
        (np.isfinite(sh1) & (sh1 < 0.0))
        | (np.isfinite(sh2) & (sh2 < 0.0))
        | (np.isfinite(sh3) & (sh3 < 0.0))
    )
    status[invalid_negative] = 'invalid_negative_thickness'
    sh1[invalid_negative] = np.nan
    sh2[invalid_negative] = np.nan
    sh3[invalid_negative] = np.nan
    wcor = np.full(t1.shape, np.nan, dtype=np.float64)
    wcor_valid = status == 'ok'
    if np.any(wcor_valid):
        wcor[wcor_valid] = (
            sh1[wcor_valid] * (1.0 / vsub[wcor_valid] - 1.0 / v1[wcor_valid])
            + sh2[wcor_valid] * (1.0 / vsub[wcor_valid] - 1.0 / v2[wcor_valid])
            + sh3[wcor_valid] * (1.0 / vsub[wcor_valid] - 1.0 / v3[wcor_valid])
        )
    return RefractionT1LSST3LayerThicknessResult(
        sh1_m=np.array(sh1, dtype=np.float64, copy=True, order='C'),
        sh2_m=np.array(sh2, dtype=np.float64, copy=True, order='C'),
        sh3_m=np.array(sh3, dtype=np.float64, copy=True, order='C'),
        status=np.array(status, dtype=_STATUS_DTYPE, copy=True, order='C'),
        weathering_correction_s=np.array(wcor, dtype=np.float64, copy=True, order='C'),
    )


def compute_t1lsst_3layer_weathering_correction(
    sh1_m: np.ndarray,
    sh2_m: np.ndarray,
    sh3_m: np.ndarray,
    v1_m_s: np.ndarray | float,
    v2_m_s: np.ndarray | float,
    v3_m_s: np.ndarray | float,
    vsub_m_s: np.ndarray | float,
) -> np.ndarray:
    """Compute three-layer replacement ``WCOR`` to Vsub in seconds."""
    sh1 = _coerce_float_array(sh1_m, name='sh1_m', allow_nonfinite=True)
    sh2 = _coerce_float_array(sh2_m, name='sh2_m', allow_nonfinite=True)
    sh3 = _coerce_float_array(sh3_m, name='sh3_m', allow_nonfinite=True)
    v1 = _positive_finite_float_array(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    v3 = _positive_finite_float_array(v3_m_s, name='v3_m_s')
    vsub = _positive_finite_float_array(vsub_m_s, name='vsub_m_s')
    sh1, sh2, sh3, v1, v2, v3, vsub = _broadcast_t1lsst_arrays(
        (sh1, sh2, sh3, v1, v2, v3, vsub),
        names=(
            'sh1_m',
            'sh2_m',
            'sh3_m',
            'v1_m_s',
            'v2_m_s',
            'v3_m_s',
            'vsub_m_s',
        ),
    )
    _validate_3layer_velocity_order(v1=v1, v2=v2, v3=v3, vsub=vsub)
    return np.array(
        sh1 * (1.0 / vsub - 1.0 / v1)
        + sh2 * (1.0 / vsub - 1.0 / v2)
        + sh3 * (1.0 / vsub - 1.0 / v3),
        dtype=np.float64,
        copy=True,
        order='C',
    )


def compose_t1lsst_1layer_endpoint_component_rows(
    components: RefractionT1LSST1LayerEndpointComponents,
) -> list[dict[str, object]]:
    """Compose source or receiver T1LSST rows without application I/O."""
    if components.endpoint_kind not in {'source', 'receiver'}:
        raise RefractionT1LSSTError('endpoint_kind must be source or receiver')
    keys = np.asarray(components.endpoint_key)
    if keys.ndim != 1:
        raise RefractionT1LSSTError('endpoint_key must be a 1D array')
    count = int(keys.shape[0])
    node_id = _coerce_integer_array(
        components.node_id,
        name='node_id',
        expected_shape=(count,),
    )
    x_m = _component_float_array(components.x_m, name='x_m', expected_shape=(count,))
    y_m = _component_float_array(components.y_m, name='y_m', expected_shape=(count,))
    surface = _component_float_array(
        components.surface_elevation_m,
        name='surface_elevation_m',
        expected_shape=(count,),
    )
    floating_datum = _component_float_array(
        components.floating_datum_elevation_m,
        name='floating_datum_elevation_m',
        expected_shape=(count,),
    )
    flat_datum = _component_float_array(
        components.flat_datum_elevation_m,
        name='flat_datum_elevation_m',
        expected_shape=(count,),
    )
    t1 = _component_float_array(components.t1_s, name='t1_s', expected_shape=(count,))
    v1 = _component_broadcast_float_array(components.v1_m_s, name='v1_m_s', shape=count)
    v2 = _component_broadcast_float_array(components.v2_m_s, name='v2_m_s', shape=count)
    sh1 = _component_float_array(components.sh1_m, name='sh1_m', expected_shape=(count,))
    refractor = _component_float_array(
        components.refractor_elevation_m,
        name='refractor_elevation_m',
        expected_shape=(count,),
    )
    wcor = _component_float_array(
        components.weathering_correction_s,
        name='weathering_correction_s',
        expected_shape=(count,),
    )
    floating_shift = _component_float_array(
        components.floating_datum_correction_s,
        name='floating_datum_correction_s',
        expected_shape=(count,),
    )
    flat_shift = _component_float_array(
        components.flat_datum_correction_s,
        name='flat_datum_correction_s',
        expected_shape=(count,),
    )
    total = _component_float_array(
        components.total_static_s,
        name='total_static_s',
        expected_shape=(count,),
    )
    solution_status = _component_string_array(
        components.solution_status,
        name='solution_status',
        expected_shape=(count,),
    )
    weathering_status = _component_string_array(
        components.weathering_status,
        name='weathering_status',
        expected_shape=(count,),
    )
    datum_status = _component_string_array(
        components.datum_status,
        name='datum_status',
        expected_shape=(count,),
    )
    static_status = _component_string_array(
        components.static_status,
        name='static_status',
        expected_shape=(count,),
    )

    rows: list[dict[str, object]] = []
    for index in range(count):
        elevation_correction_s = _sum_correction_s(floating_shift[index], flat_shift[index])
        rows.append(
            {
                'endpoint_kind': components.endpoint_kind,
                'endpoint_key': str(keys[index]),
                'node_id': int(node_id[index]),
                'x_m': _csv_float(x_m[index]),
                'y_m': _csv_float(y_m[index]),
                'surface_elevation_m': _csv_float(surface[index]),
                'floating_datum_elevation_m': _csv_float(floating_datum[index]),
                'flat_datum_elevation_m': _csv_float(flat_datum[index]),
                't1_ms': _csv_ms(t1[index]),
                'v1_m_s': _csv_float(v1[index]),
                'v2_m_s': _csv_float(v2[index]),
                'sh1_weathering_thickness_m': _csv_float(sh1[index]),
                'refractor_elevation_m': _csv_float(refractor[index]),
                'weathering_correction_ms': _csv_ms(wcor[index]),
                'floating_datum_correction_ms': _csv_ms(floating_shift[index]),
                'flat_datum_correction_ms': _csv_ms(flat_shift[index]),
                'elevation_correction_ms': _csv_ms(elevation_correction_s),
                'total_static_ms': _csv_ms(total[index]),
                'total_applied_shift_ms': _csv_ms(total[index]),
                'solution_status': str(solution_status[index]),
                'weathering_status': str(weathering_status[index]),
                'datum_status': str(datum_status[index]),
                'static_status': str(static_status[index]),
                'sign_convention': T1LSST_SIGN_CONVENTION,
            }
        )
    return rows


def _coerce_float_array(
    value: object,
    *,
    name: str,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RefractionT1LSSTError(f'{name} must be numeric') from exc
    if array.dtype == object:
        raise RefractionT1LSSTError(f'{name} must not have object dtype')
    if not allow_nonfinite and not np.all(np.isfinite(array)):
        raise RefractionT1LSSTError(f'{name} must contain finite values')
    return np.ascontiguousarray(array, dtype=np.float64)


def _coerce_2layer_inputs(
    *,
    t1_s: object,
    t2_s: object,
    v1_m_s: object,
    v2_m_s: object,
    v3_m_s: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t1 = _coerce_float_array(t1_s, name='t1_s')
    t2 = _coerce_float_array(t2_s, name='t2_s')
    v1 = _positive_finite_float_array(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    v3 = _positive_finite_float_array(v3_m_s, name='v3_m_s')
    t1, t2, v1, v2, v3 = _broadcast_t1lsst_arrays(
        (t1, t2, v1, v2, v3),
        names=('t1_s', 't2_s', 'v1_m_s', 'v2_m_s', 'v3_m_s'),
    )
    _validate_2layer_velocity_order(v1=v1, v2=v2, v3=v3)
    return t1, t2, v1, v2, v3


def _coerce_2layer_status_inputs(
    *,
    t1_s: object,
    t2_s: object,
    v1_m_s: object,
    v2_m_s: object,
    v3_m_s: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t1 = _coerce_float_array(t1_s, name='t1_s', allow_nonfinite=True)
    t2 = _coerce_float_array(t2_s, name='t2_s', allow_nonfinite=True)
    v1 = _coerce_float_array(v1_m_s, name='v1_m_s', allow_nonfinite=True)
    v2 = _coerce_float_array(v2_m_s, name='v2_m_s', allow_nonfinite=True)
    v3 = _coerce_float_array(v3_m_s, name='v3_m_s', allow_nonfinite=True)
    return _broadcast_t1lsst_arrays(
        (t1, t2, v1, v2, v3),
        names=('t1_s', 't2_s', 'v1_m_s', 'v2_m_s', 'v3_m_s'),
    )


def _coerce_3layer_inputs(
    *,
    t1_s: object,
    t2_s: object,
    t3_s: object,
    v1_m_s: object,
    v2_m_s: object,
    v3_m_s: object,
    vsub_m_s: object,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    t1 = _coerce_float_array(t1_s, name='t1_s')
    t2 = _coerce_float_array(t2_s, name='t2_s')
    t3 = _coerce_float_array(t3_s, name='t3_s')
    v1 = _positive_finite_float_array(v1_m_s, name='v1_m_s')
    v2 = _positive_finite_float_array(v2_m_s, name='v2_m_s')
    v3 = _positive_finite_float_array(v3_m_s, name='v3_m_s')
    vsub = _positive_finite_float_array(vsub_m_s, name='vsub_m_s')
    t1, t2, t3, v1, v2, v3, vsub = _broadcast_t1lsst_arrays(
        (t1, t2, t3, v1, v2, v3, vsub),
        names=(
            't1_s',
            't2_s',
            't3_s',
            'v1_m_s',
            'v2_m_s',
            'v3_m_s',
            'vsub_m_s',
        ),
    )
    _validate_3layer_velocity_order(v1=v1, v2=v2, v3=v3, vsub=vsub)
    return t1, t2, t3, v1, v2, v3, vsub


def _coerce_3layer_status_inputs(
    *,
    t1_s: object,
    t2_s: object,
    t3_s: object,
    v1_m_s: object,
    v2_m_s: object,
    v3_m_s: object,
    vsub_m_s: object,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    t1 = _coerce_float_array(t1_s, name='t1_s', allow_nonfinite=True)
    t2 = _coerce_float_array(t2_s, name='t2_s', allow_nonfinite=True)
    t3 = _coerce_float_array(t3_s, name='t3_s', allow_nonfinite=True)
    v1 = _coerce_float_array(v1_m_s, name='v1_m_s', allow_nonfinite=True)
    v2 = _coerce_float_array(v2_m_s, name='v2_m_s', allow_nonfinite=True)
    v3 = _coerce_float_array(v3_m_s, name='v3_m_s', allow_nonfinite=True)
    vsub = _coerce_float_array(vsub_m_s, name='vsub_m_s', allow_nonfinite=True)
    return _broadcast_t1lsst_arrays(
        (t1, t2, t3, v1, v2, v3, vsub),
        names=(
            't1_s',
            't2_s',
            't3_s',
            'v1_m_s',
            'v2_m_s',
            'v3_m_s',
            'vsub_m_s',
        ),
    )


def _broadcast_t1lsst_arrays(
    arrays: tuple[np.ndarray, ...],
    *,
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    try:
        broadcasted = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        joined = ', '.join(names)
        raise RefractionT1LSSTError(
            f'{joined} must be broadcastable to a common shape'
        ) from exc
    return tuple(
        np.array(array, dtype=np.float64, copy=True, order='C')
        for array in broadcasted
    )


def _validate_2layer_velocity_order(
    *,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
) -> None:
    if np.any(v2 <= v1):
        raise RefractionT1LSSTError('v2_m_s must be greater than v1_m_s')
    if np.any(v3 <= v2):
        raise RefractionT1LSSTError('v3_m_s must be greater than v2_m_s')


def _validate_3layer_velocity_order(
    *,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    vsub: np.ndarray,
) -> None:
    _validate_2layer_velocity_order(v1=v1, v2=v2, v3=v3)
    if np.any(vsub <= v3):
        raise RefractionT1LSSTError('vsub_m_s must be greater than v3_m_s')


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise RefractionT1LSSTError(f'{name} must be finite and positive')
    try:
        out = _coerce_finite_float(
            value,
            name=name,
            error_type=RefractionT1LSSTError,
        )
    except RefractionT1LSSTError as exc:
        raise RefractionT1LSSTError(f'{name} must be finite and positive') from exc
    if out <= 0.0:
        raise RefractionT1LSSTError(f'{name} must be finite and positive')
    return out


def _positive_finite_float_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    out = _coerce_float_array(value, name=name)
    if raw.ndim == 0:
        return np.asarray(_positive_finite(value, name=name), dtype=np.float64)
    if np.any(out <= 0.0):
        raise RefractionT1LSSTError(f'{name} must be finite and positive')
    return out


def _coerce_integer_array(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise RefractionT1LSSTError(f'{name} must be numeric') from exc
    if array.shape != expected_shape:
        raise RefractionT1LSSTError(
            f'{name} shape mismatch: expected {expected_shape}, got {array.shape}'
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise RefractionT1LSSTError(f'{name} must contain integer values')
    return np.ascontiguousarray(array, dtype=np.int64)


def _component_float_array(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = _coerce_float_array(value, name=name, allow_nonfinite=True)
    if array.shape != expected_shape:
        raise RefractionT1LSSTError(
            f'{name} shape mismatch: expected {expected_shape}, got {array.shape}'
        )
    return array


def _component_broadcast_float_array(
    value: object,
    *,
    name: str,
    shape: int,
) -> np.ndarray:
    array = _coerce_float_array(value, name=name, allow_nonfinite=True)
    try:
        out = np.broadcast_to(array, (shape,))
    except ValueError as exc:
        raise RefractionT1LSSTError(
            f'{name} must be broadcastable to shape {(shape,)}'
        ) from exc
    return np.ascontiguousarray(out, dtype=np.float64)


def _component_string_array(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise RefractionT1LSSTError(
            f'{name} shape mismatch: expected {expected_shape}, got {array.shape}'
        )
    return np.ascontiguousarray(array.astype(str), dtype=str)


def _sum_correction_s(left: object, right: object) -> float:
    left_value = _as_float_or_nan(left)
    right_value = _as_float_or_nan(right)
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return float('nan')
    return float(left_value + right_value)


def _as_float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    if not np.isfinite(out):
        return float('nan')
    return out


def _csv_float(value: object) -> str | float:
    out = _as_float_or_nan(value)
    if not np.isfinite(out):
        return ''
    return float(out)


def _csv_ms(value_s: object) -> str | float:
    out = _as_float_or_nan(value_s)
    if not np.isfinite(out):
        return ''
    return float(out * 1000.0)


__all__ = [
    'T1LSST_SIGN_CONVENTION',
    'RefractionT1LSST1LayerEndpointComponents',
    'RefractionT1LSST2LayerThicknessResult',
    'RefractionT1LSST3LayerThicknessResult',
    'RefractionT1LSSTError',
    'compose_t1lsst_1layer_endpoint_component_rows',
    'compute_t1lsst_1layer_thickness',
    'compute_t1lsst_1layer_weathering_correction',
    'compute_t1lsst_2layer_thicknesses',
    'compute_t1lsst_2layer_thicknesses_with_status',
    'compute_t1lsst_2layer_weathering_correction',
    'compute_t1lsst_3layer_thicknesses',
    'compute_t1lsst_3layer_thicknesses_with_status',
    'compute_t1lsst_3layer_weathering_correction',
]
