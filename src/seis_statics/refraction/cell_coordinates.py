"""Coordinate projection helpers for refraction-static cell assignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from seis_statics.refraction.options import (
    RefractionStaticRefractorCellCoordinateMode,
    RefractionStaticRefractorCellOptions,
)

RefractionCellCoordinateMode = RefractionStaticRefractorCellCoordinateMode


@dataclass(frozen=True)
class RefractionCellProjectedPoints:
    """Coordinates used by the refractor-cell grid plus diagnostic projection."""

    x_m: np.ndarray
    y_m: np.ndarray
    projected_inline_m: np.ndarray | None
    projected_crossline_m: np.ndarray | None
    qc: dict[str, Any]


@dataclass(frozen=True)
class RefractionCellProjectedSourceReceiver:
    """Projected source and receiver coordinates for cell assignment."""

    source_x_m: np.ndarray
    source_y_m: np.ndarray
    receiver_x_m: np.ndarray
    receiver_y_m: np.ndarray
    source_projected_inline_m: np.ndarray | None
    source_projected_crossline_m: np.ndarray | None
    receiver_projected_inline_m: np.ndarray | None
    receiver_projected_crossline_m: np.ndarray | None
    qc: dict[str, Any]


def project_refraction_cell_points(
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    mode: RefractionCellCoordinateMode,
    line_origin_x_m: float | None = None,
    line_origin_y_m: float | None = None,
    line_azimuth_deg: float | None = None,
) -> RefractionCellProjectedPoints:
    """Project point coordinates into the coordinate system used by cells."""
    x = _as_1d_float_array(x_m, name='x_m')
    y = _as_1d_float_array(y_m, name='y_m')
    if x.shape != y.shape:
        raise ValueError('x_m and y_m must have the same shape')

    coordinate_mode = _validate_mode(mode)
    if coordinate_mode == 'grid_3d':
        return RefractionCellProjectedPoints(
            x_m=x,
            y_m=y,
            projected_inline_m=None,
            projected_crossline_m=None,
            qc=refraction_cell_coordinate_metadata(
                coordinate_mode=coordinate_mode,
                line_origin_x_m=None,
                line_origin_y_m=None,
                line_azimuth_deg=None,
            ),
        )

    origin_x = _finite_float(
        line_origin_x_m,
        name='model.refractor_cell.line_origin_x_m',
    )
    origin_y = _finite_float(
        line_origin_y_m,
        name='model.refractor_cell.line_origin_y_m',
    )
    azimuth = _finite_float(
        line_azimuth_deg,
        name='model.refractor_cell.line_azimuth_deg',
    )
    inline, crossline = _project_line_2d(
        x_m=x,
        y_m=y,
        line_origin_x_m=origin_x,
        line_origin_y_m=origin_y,
        line_azimuth_deg=azimuth,
    )
    return RefractionCellProjectedPoints(
        x_m=inline,
        y_m=np.zeros(inline.shape, dtype=np.float64),
        projected_inline_m=inline,
        projected_crossline_m=crossline,
        qc={
            **refraction_cell_coordinate_metadata(
                coordinate_mode=coordinate_mode,
                line_origin_x_m=origin_x,
                line_origin_y_m=origin_y,
                line_azimuth_deg=azimuth,
            ),
            **_projected_range_qc(prefix='projected', inline=inline, crossline=crossline),
        },
    )


def project_refraction_cell_coordinates(
    *,
    source_x_m: np.ndarray,
    source_y_m: np.ndarray,
    receiver_x_m: np.ndarray,
    receiver_y_m: np.ndarray,
    mode: RefractionCellCoordinateMode,
    line_origin_x_m: float | None = None,
    line_origin_y_m: float | None = None,
    line_azimuth_deg: float | None = None,
) -> RefractionCellProjectedSourceReceiver:
    """Project source/receiver coordinates into the cell-assignment system."""
    source = project_refraction_cell_points(
        x_m=source_x_m,
        y_m=source_y_m,
        mode=mode,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )
    receiver = project_refraction_cell_points(
        x_m=receiver_x_m,
        y_m=receiver_y_m,
        mode=mode,
        line_origin_x_m=line_origin_x_m,
        line_origin_y_m=line_origin_y_m,
        line_azimuth_deg=line_azimuth_deg,
    )
    if source.x_m.shape != receiver.x_m.shape:
        raise ValueError('source and receiver coordinate arrays must have the same shape')
    qc = dict(source.qc)
    if source.projected_inline_m is not None and receiver.projected_inline_m is not None:
        qc.update(
            _projected_range_qc(
                prefix='source_projected',
                inline=source.projected_inline_m,
                crossline=source.projected_crossline_m,
            )
        )
        qc.update(
            _projected_range_qc(
                prefix='receiver_projected',
                inline=receiver.projected_inline_m,
                crossline=receiver.projected_crossline_m,
            )
        )

    return RefractionCellProjectedSourceReceiver(
        source_x_m=source.x_m,
        source_y_m=source.y_m,
        receiver_x_m=receiver.x_m,
        receiver_y_m=receiver.y_m,
        source_projected_inline_m=source.projected_inline_m,
        source_projected_crossline_m=source.projected_crossline_m,
        receiver_projected_inline_m=receiver.projected_inline_m,
        receiver_projected_crossline_m=receiver.projected_crossline_m,
        qc=qc,
    )


def effective_refraction_cell_grid_config(
    config: RefractionStaticRefractorCellOptions,
) -> RefractionStaticRefractorCellOptions:
    """Return the grid definition actually used by the selected coordinate mode."""
    if getattr(config, 'coordinate_mode', 'grid_3d') != 'line_2d_projected':
        return config
    return replace(
        config,
        number_of_cell_y=1,
        size_of_cell_y_m=None,
        y_coordinate_origin_m=0.0,
    )


def refraction_cell_coordinate_metadata_from_config(
    config: RefractionStaticRefractorCellOptions,
) -> dict[str, Any]:
    """Build strict-JSON coordinate mode metadata from cell options."""
    return refraction_cell_coordinate_metadata(
        coordinate_mode=getattr(config, 'coordinate_mode', 'grid_3d'),
        line_origin_x_m=getattr(config, 'line_origin_x_m', None),
        line_origin_y_m=getattr(config, 'line_origin_y_m', None),
        line_azimuth_deg=getattr(config, 'line_azimuth_deg', None),
    )


def refraction_cell_coordinate_metadata(
    *,
    coordinate_mode: RefractionCellCoordinateMode,
    line_origin_x_m: float | None,
    line_origin_y_m: float | None,
    line_azimuth_deg: float | None,
) -> dict[str, Any]:
    """Return the JSON-safe coordinate metadata common to QC artifacts."""
    mode = _validate_mode(coordinate_mode)
    if mode == 'grid_3d':
        line_origin_x_m = None
        line_origin_y_m = None
        line_azimuth_deg = None
    return {
        'coordinate_mode': mode,
        'line_origin_x_m': _optional_json_float(line_origin_x_m),
        'line_origin_y_m': _optional_json_float(line_origin_y_m),
        'line_azimuth_deg': _optional_json_float(line_azimuth_deg),
    }


def _project_line_2d(
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    line_origin_x_m: float,
    line_origin_y_m: float,
    line_azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth_rad = np.deg2rad(float(line_azimuth_deg))
    inline_unit_x = float(np.sin(azimuth_rad))
    inline_unit_y = float(np.cos(azimuth_rad))
    dx = x_m - float(line_origin_x_m)
    dy = y_m - float(line_origin_y_m)
    inline = _round_projected_coordinate(dx * inline_unit_x + dy * inline_unit_y)
    crossline = _round_projected_coordinate(dx * inline_unit_y - dy * inline_unit_x)
    return (
        np.ascontiguousarray(inline, dtype=np.float64),
        np.ascontiguousarray(crossline, dtype=np.float64),
    )


def _projected_range_qc(
    *,
    prefix: str,
    inline: np.ndarray | None,
    crossline: np.ndarray | None,
) -> dict[str, float | None]:
    return {
        f'{prefix}_inline_m_min': _finite_stat(inline, 'min'),
        f'{prefix}_inline_m_max': _finite_stat(inline, 'max'),
        f'{prefix}_crossline_m_min': _finite_stat(crossline, 'min'),
        f'{prefix}_crossline_m_max': _finite_stat(crossline, 'max'),
    }


def _round_projected_coordinate(values: np.ndarray) -> np.ndarray:
    # Keep exact line-grid boundaries stable after sin/cos rotation roundoff.
    rounded = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(rounded)
    rounded[finite] = np.round(rounded[finite], decimals=9)
    return rounded


def _finite_stat(values: np.ndarray | None, stat: Literal['min', 'max']) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if stat == 'min':
        return float(np.min(arr))
    return float(np.max(arr))


def _validate_mode(value: object) -> RefractionCellCoordinateMode:
    if value == 'grid_3d':
        return 'grid_3d'
    if value == 'line_2d_projected':
        return 'line_2d_projected'
    raise ValueError('coordinate_mode must be grid_3d or line_2d_projected')


def _as_1d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f'{name} must contain numeric values')
    return np.ascontiguousarray(arr, dtype=np.float64)


def _finite_float(value: float | None, *, name: str) -> float:
    if value is None:
        raise ValueError(f'{name} is required when coordinate_mode is line_2d_projected')
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not np.isfinite(out):
        raise ValueError(f'{name} must be finite')
    return out


def _optional_json_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    'RefractionCellCoordinateMode',
    'RefractionCellProjectedPoints',
    'RefractionCellProjectedSourceReceiver',
    'effective_refraction_cell_grid_config',
    'project_refraction_cell_coordinates',
    'project_refraction_cell_points',
    'refraction_cell_coordinate_metadata',
    'refraction_cell_coordinate_metadata_from_config',
]
