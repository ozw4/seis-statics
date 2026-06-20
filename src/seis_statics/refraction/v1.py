"""Global V1 estimation from near-offset direct-arrival first breaks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seis_statics.refraction.options import RefractionStaticFirstLayerOptions
from seis_statics.refraction.types import (
    RefractionStaticInputModel,
    RefractionV1EstimateResult,
)

_GROUP_KIND = 'source_endpoint'
_STATUS_DTYPE = '<U40'


class RefractionV1EstimationError(ValueError):
    """Raised when direct-arrival V1 cannot be estimated."""

    def __init__(
        self,
        message: str,
        *,
        n_valid_groups: int | None = None,
        min_groups: int | None = None,
        group_status_counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.n_valid_groups = n_valid_groups
        self.min_groups = min_groups
        self.group_status_counts = (
            dict(group_status_counts) if group_status_counts is not None else None
        )


@dataclass(frozen=True)
class _FitResult:
    slope_s_per_m: float
    intercept_s: float
    n_used: int
    residual_rms_s: float
    residual_mad_s: float
    status: str


def estimate_global_v1_from_direct_arrivals(
    *,
    input_model: RefractionStaticInputModel,
    first_layer: RefractionStaticFirstLayerOptions,
) -> RefractionV1EstimateResult:
    """Estimate one global weathering velocity from source-gather direct arrivals."""
    _validate_first_layer_options(first_layer)
    min_offset, max_offset = _direct_offset_gate(first_layer)
    min_velocity = _positive_float(
        first_layer.min_weathering_velocity_m_s,
        name='model.first_layer.min_weathering_velocity_m_s',
    )
    max_velocity = _positive_float(
        first_layer.max_weathering_velocity_m_s,
        name='model.first_layer.max_weathering_velocity_m_s',
    )
    if min_velocity >= max_velocity:
        raise RefractionV1EstimationError(
            'model.first_layer.min_weathering_velocity_m_s must be less than '
            'model.first_layer.max_weathering_velocity_m_s'
        )
    min_picks = _positive_int(
        first_layer.min_picks_per_fit,
        name='model.first_layer.min_picks_per_fit',
    )
    min_groups = _positive_int(
        first_layer.min_groups,
        name='model.first_layer.min_groups',
    )
    robust_enabled = bool(first_layer.robust_enabled)
    robust_threshold = _positive_float(
        first_layer.robust_threshold,
        name='model.first_layer.robust_threshold',
    )

    data = _validate_input_model(input_model)
    candidate_mask = (
        data['valid_observation_mask_sorted']
        & data['valid_pick_mask_sorted']
        & np.isfinite(data['pick_time_s_sorted'])
        & np.isfinite(data['distance_m_sorted'])
        & (data['distance_m_sorted'] >= min_offset)
        & (data['distance_m_sorted'] <= max_offset)
    )
    n_candidate = int(np.count_nonzero(candidate_mask))
    if n_candidate < min_picks:
        raise RefractionV1EstimationError(
            'Insufficient near-offset direct-arrival picks for V1 estimation: '
            f'{n_candidate} candidate picks, require at least {min_picks}.'
        )

    group_keys = np.unique(data['source_endpoint_key_sorted'][candidate_mask])
    if group_keys.size == 0:
        raise RefractionV1EstimationError(
            'Insufficient near-offset direct-arrival source groups for V1 estimation.'
        )

    group_v1: list[float] = []
    group_slope: list[float] = []
    group_intercept: list[float] = []
    group_n_candidates: list[int] = []
    group_n_used: list[int] = []
    group_offset_min: list[float] = []
    group_offset_max: list[float] = []
    group_rms: list[float] = []
    group_mad: list[float] = []
    group_status: list[str] = []

    for key in group_keys:
        mask = candidate_mask & (data['source_endpoint_key_sorted'] == key)
        x = np.ascontiguousarray(data['distance_m_sorted'][mask], dtype=np.float64)
        t = np.ascontiguousarray(data['pick_time_s_sorted'][mask], dtype=np.float64)
        n_group_candidates = int(x.shape[0])
        fit = _fit_group(
            distance_m=x,
            pick_time_s=t,
            min_picks=min_picks,
            robust_enabled=robust_enabled,
            robust_threshold=robust_threshold,
        )
        v1 = np.nan
        status = fit.status
        if status == 'ok':
            if fit.slope_s_per_m <= 0.0 or not np.isfinite(fit.slope_s_per_m):
                status = 'nonpositive_slope'
            else:
                v1 = float(1.0 / fit.slope_s_per_m)
                if not min_velocity <= v1 <= max_velocity:
                    status = 'velocity_out_of_bounds'
        group_v1.append(v1)
        group_slope.append(fit.slope_s_per_m)
        group_intercept.append(fit.intercept_s)
        group_n_candidates.append(n_group_candidates)
        group_n_used.append(fit.n_used)
        group_offset_min.append(float(np.min(x)) if n_group_candidates else np.nan)
        group_offset_max.append(float(np.max(x)) if n_group_candidates else np.nan)
        group_rms.append(fit.residual_rms_s)
        group_mad.append(fit.residual_mad_s)
        group_status.append(status)

    group_status_arr = np.asarray(group_status, dtype=_STATUS_DTYPE)
    group_v1_arr = np.ascontiguousarray(group_v1, dtype=np.float64)
    valid_group_mask = group_status_arr == 'ok'
    n_valid_groups = int(np.count_nonzero(valid_group_mask))
    if n_valid_groups < min_groups:
        status_counts = _status_counts(group_status_arr)
        if _all_groups_velocity_out_of_bounds(
            n_valid_groups=n_valid_groups,
            status_counts=status_counts,
        ):
            raise RefractionV1EstimationError(
                _v1_group_failure_message(
                    'No valid direct-arrival V1 groups remain within '
                    'model.first_layer velocity bounds',
                    n_valid_groups=n_valid_groups,
                    min_groups=min_groups,
                    status_counts=status_counts,
                ),
                n_valid_groups=n_valid_groups,
                min_groups=min_groups,
                group_status_counts=status_counts,
            )
        raise RefractionV1EstimationError(
            _v1_group_failure_message(
                'Insufficient valid direct-arrival V1 groups',
                n_valid_groups=n_valid_groups,
                min_groups=min_groups,
                status_counts=status_counts,
            ),
            n_valid_groups=n_valid_groups,
            min_groups=min_groups,
            group_status_counts=status_counts,
        )

    valid_v1 = group_v1_arr[valid_group_mask]
    resolved_v1 = float(np.median(valid_v1))
    if not np.isfinite(resolved_v1):
        raise RefractionV1EstimationError('Estimated global V1 must be finite.')

    qc = {
        'v1_mode': 'estimate_direct_arrival',
        'resolved_weathering_velocity_m_s': resolved_v1,
        'n_candidate_picks': n_candidate,
        'n_candidate_groups': int(group_keys.shape[0]),
        'n_used_groups': n_valid_groups,
        'min_direct_offset_m': float(min_offset),
        'max_direct_offset_m': float(max_offset),
        'min_weathering_velocity_m_s': float(min_velocity),
        'max_weathering_velocity_m_s': float(max_velocity),
        'min_picks_per_fit': int(min_picks),
        'min_groups': int(min_groups),
        'robust_enabled': bool(robust_enabled),
        'robust_threshold': float(robust_threshold),
        'v1_min_m_s': float(np.min(valid_v1)),
        'v1_median_m_s': resolved_v1,
        'v1_max_m_s': float(np.max(valid_v1)),
        'v1_status': 'estimated',
        'group_status_counts': _status_counts(group_status_arr),
        'warnings': [],
    }

    return RefractionV1EstimateResult(
        mode='estimate_direct_arrival',
        resolved_weathering_velocity_m_s=resolved_v1,
        group_kind=_GROUP_KIND,
        group_key=np.ascontiguousarray(group_keys.astype(str), dtype='<U192'),
        group_v1_m_s=group_v1_arr,
        group_slope_s_per_m=np.ascontiguousarray(group_slope, dtype=np.float64),
        group_intercept_s=np.ascontiguousarray(group_intercept, dtype=np.float64),
        group_n_candidates=np.ascontiguousarray(group_n_candidates, dtype=np.int64),
        group_n_used=np.ascontiguousarray(group_n_used, dtype=np.int64),
        group_offset_min_m=np.ascontiguousarray(group_offset_min, dtype=np.float64),
        group_offset_max_m=np.ascontiguousarray(group_offset_max, dtype=np.float64),
        group_residual_rms_s=np.ascontiguousarray(group_rms, dtype=np.float64),
        group_residual_mad_s=np.ascontiguousarray(group_mad, dtype=np.float64),
        group_status=group_status_arr,
        qc=qc,
    )


def _fit_group(
    *,
    distance_m: np.ndarray,
    pick_time_s: np.ndarray,
    min_picks: int,
    robust_enabled: bool,
    robust_threshold: float,
) -> _FitResult:
    n_candidates = int(distance_m.shape[0])
    if n_candidates < min_picks:
        return _empty_fit('insufficient_picks', n_used=n_candidates)
    if not np.all(np.isfinite(distance_m)) or not np.all(np.isfinite(pick_time_s)):
        return _empty_fit('nonfinite_candidate', n_used=0)
    if float(np.max(distance_m) - np.min(distance_m)) <= 0.0:
        return _empty_fit('insufficient_offset_aperture', n_used=n_candidates)

    initial = _ordinary_least_squares(distance_m, pick_time_s)
    if initial is None:
        return _empty_fit('fit_failed', n_used=n_candidates)
    intercept, slope = initial
    used_mask = np.ones(n_candidates, dtype=bool)

    if robust_enabled:
        residual = pick_time_s - (intercept + slope * distance_m)
        scale = _robust_scale_s(residual)
        centered = residual - float(np.median(residual))
        if scale <= 0.0:
            used_mask = np.isclose(centered, 0.0, atol=1.0e-12, rtol=0.0)
        else:
            used_mask = np.abs(centered) <= robust_threshold * scale
        if int(np.count_nonzero(used_mask)) < min_picks:
            return _empty_fit(
                'insufficient_picks_after_robust',
                n_used=int(np.count_nonzero(used_mask)),
            )
        if float(np.max(distance_m[used_mask]) - np.min(distance_m[used_mask])) <= 0.0:
            return _empty_fit(
                'insufficient_offset_aperture',
                n_used=int(np.count_nonzero(used_mask)),
            )
        refit = _ordinary_least_squares(distance_m[used_mask], pick_time_s[used_mask])
        if refit is None:
            return _empty_fit(
                'fit_failed',
                n_used=int(np.count_nonzero(used_mask)),
            )
        intercept, slope = refit

    residual_used = pick_time_s[used_mask] - (intercept + slope * distance_m[used_mask])
    rms = _residual_rms(residual_used)
    mad = _residual_mad(residual_used)
    if not np.isfinite(rms) or not np.isfinite(mad):
        status = 'invalid_residual_statistics'
    else:
        status = 'ok'
    return _FitResult(
        slope_s_per_m=float(slope),
        intercept_s=float(intercept),
        n_used=int(np.count_nonzero(used_mask)),
        residual_rms_s=float(rms),
        residual_mad_s=float(mad),
        status=status,
    )


def _ordinary_least_squares(
    distance_m: np.ndarray,
    pick_time_s: np.ndarray,
) -> tuple[float, float] | None:
    matrix = np.column_stack(
        (np.ones(distance_m.shape[0], dtype=np.float64), distance_m)
    )
    try:
        coeff, _residuals, rank, _singular = np.linalg.lstsq(
            matrix,
            pick_time_s,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None
    if rank < 2 or coeff.shape != (2,) or not np.all(np.isfinite(coeff)):
        return None
    return float(coeff[0]), float(coeff[1])


def _empty_fit(status: str, *, n_used: int) -> _FitResult:
    return _FitResult(
        slope_s_per_m=np.nan,
        intercept_s=np.nan,
        n_used=int(n_used),
        residual_rms_s=np.nan,
        residual_mad_s=np.nan,
        status=status,
    )


def _validate_first_layer_options(first_layer: RefractionStaticFirstLayerOptions) -> None:
    if not isinstance(first_layer, RefractionStaticFirstLayerOptions):
        raise RefractionV1EstimationError(
            'first_layer must be a RefractionStaticFirstLayerOptions instance'
        )
    if first_layer.mode != 'estimate_direct_arrival':
        raise RefractionV1EstimationError(
            'model.first_layer.mode must be estimate_direct_arrival'
        )


def _validate_input_model(input_model: RefractionStaticInputModel) -> dict[str, np.ndarray]:
    if not isinstance(input_model, RefractionStaticInputModel):
        raise RefractionV1EstimationError(
            'input_model must be a RefractionStaticInputModel instance'
        )
    n_traces = _array_1d(input_model.pick_time_s_sorted, name='pick_time_s_sorted').shape[0]
    if n_traces <= 0:
        raise RefractionV1EstimationError('input_model must contain at least one trace')
    values = {
        'pick_time_s_sorted': _array_1d(
            input_model.pick_time_s_sorted,
            name='pick_time_s_sorted',
            dtype=np.float64,
        ),
        'valid_pick_mask_sorted': _array_1d(
            input_model.valid_pick_mask_sorted,
            name='valid_pick_mask_sorted',
            dtype=bool,
        ),
        'distance_m_sorted': _array_1d(
            input_model.distance_m_sorted,
            name='distance_m_sorted',
            dtype=np.float64,
        ),
        'valid_observation_mask_sorted': _array_1d(
            input_model.valid_observation_mask_sorted,
            name='valid_observation_mask_sorted',
            dtype=bool,
        ),
        'source_endpoint_key_sorted': _array_1d(
            input_model.source_endpoint_key_sorted,
            name='source_endpoint_key_sorted',
        ).astype(str, copy=False),
    }
    for name, arr in values.items():
        if arr.shape != (n_traces,):
            raise RefractionV1EstimationError(f'{name} length mismatch')
    return values


def _array_1d(
    value: object,
    *,
    name: str,
    dtype: object | None = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        raise RefractionV1EstimationError(f'{name} must be one-dimensional')
    return arr


def _direct_offset_gate(first_layer: RefractionStaticFirstLayerOptions) -> tuple[float, float]:
    min_offset = first_layer.min_direct_offset_m
    max_offset = first_layer.max_direct_offset_m
    if min_offset is None or max_offset is None:
        raise RefractionV1EstimationError(
            'model.first_layer.min_direct_offset_m and '
            'model.first_layer.max_direct_offset_m are required when '
            'model.first_layer.mode is estimate_direct_arrival'
        )
    min_value = _nonnegative_float(
        min_offset,
        name='model.first_layer.min_direct_offset_m',
    )
    max_value = _nonnegative_float(
        max_offset,
        name='model.first_layer.max_direct_offset_m',
    )
    if min_value >= max_value:
        raise RefractionV1EstimationError(
            'model.first_layer.min_direct_offset_m must be less than '
            'model.first_layer.max_direct_offset_m'
        )
    return min_value, max_value


def _positive_float(value: object, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionV1EstimationError(f'{name} must be a finite number') from exc
    if not np.isfinite(number) or number <= 0.0:
        raise RefractionV1EstimationError(f'{name} must be positive and finite')
    return number


def _nonnegative_float(value: object, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RefractionV1EstimationError(f'{name} must be a finite number') from exc
    if not np.isfinite(number) or number < 0.0:
        raise RefractionV1EstimationError(f'{name} must be nonnegative and finite')
    return number


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise RefractionV1EstimationError(f'{name} must be a positive integer')
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RefractionV1EstimationError(f'{name} must be a positive integer') from exc
    if number <= 0:
        raise RefractionV1EstimationError(f'{name} must be a positive integer')
    return number


def _robust_scale_s(residual_s: np.ndarray) -> float:
    centered = residual_s - float(np.median(residual_s))
    mad = float(np.median(np.abs(centered)))
    if not np.isfinite(mad):
        return np.nan
    return 1.4826 * mad


def _residual_rms(residual_s: np.ndarray) -> float:
    if residual_s.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(residual_s))))


def _residual_mad(residual_s: np.ndarray) -> float:
    if residual_s.size == 0:
        return np.nan
    return float(np.median(np.abs(residual_s - float(np.median(residual_s)))))


def _status_counts(status: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(status).astype(str), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts, strict=True)}


def _all_groups_velocity_out_of_bounds(
    *,
    n_valid_groups: int,
    status_counts: dict[str, int],
) -> bool:
    n_groups = sum(status_counts.values())
    return (
        n_valid_groups == 0
        and n_groups > 0
        and status_counts.get('velocity_out_of_bounds', 0) == n_groups
    )


def _v1_group_failure_message(
    prefix: str,
    *,
    n_valid_groups: int,
    min_groups: int,
    status_counts: dict[str, int],
) -> str:
    return (
        f'{prefix}: {n_valid_groups} valid groups, require at least {min_groups}. '
        f'Status counts: {_format_status_counts(status_counts)}.'
    )


def _format_status_counts(status_counts: dict[str, int]) -> str:
    preferred_order = (
        'ok',
        'velocity_out_of_bounds',
        'insufficient_picks',
        'insufficient_picks_after_robust',
        'nonpositive_slope',
        'insufficient_offset_aperture',
        'invalid_residual_statistics',
        'nonfinite_candidate',
        'fit_failed',
    )
    ordered = [status for status in preferred_order if status in status_counts]
    ordered.extend(sorted(set(status_counts) - set(ordered)))
    if not ordered:
        return 'none'
    return ', '.join(f'{status}={status_counts[status]}' for status in ordered)


__all__ = [
    'RefractionV1EstimateResult',
    'RefractionV1EstimationError',
    'estimate_global_v1_from_direct_arrivals',
]
