"""Reusable NumPy array validation helpers for statics core modules."""

from __future__ import annotations

from typing import NoReturn

import numpy as np


def is_real_numeric_dtype(dtype: np.dtype) -> bool:
    """Return true for non-bool, non-complex numeric dtypes."""
    np_dtype = np.dtype(dtype)
    return (
        not np.issubdtype(np_dtype, np.bool_)
        and np.issubdtype(np_dtype, np.number)
        and not np.issubdtype(np_dtype, np.complexfloating)
    )


def coerce_1d_real_numeric_float64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    require_finite: bool = False,
    allow_nonfinite: bool | None = None,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    if allow_nonfinite is not None:
        if require_finite and allow_nonfinite:
            _raise(
                'require_finite and allow_nonfinite cannot both be true',
                error_type,
            )
        require_finite = not allow_nonfinite

    arr = _as_1d_array(values, name=name, expected_shape=expected_shape, error_type=error_type)
    if not is_real_numeric_dtype(arr.dtype):
        _raise(f'{name} must have a real numeric dtype', error_type)
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if require_finite and np.any(~np.isfinite(out)):
        _raise(f'{name} must contain only finite values', error_type)
    return out


def coerce_1d_finite_float64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    return coerce_1d_real_numeric_float64(
        values,
        name=name,
        expected_shape=expected_shape,
        require_finite=True,
        error_type=error_type,
    )


def coerce_1d_castable_finite_float64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    reject_bool_dtype: bool = False,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    arr = _as_1d_array(values, name=name, expected_shape=expected_shape, error_type=error_type)
    if reject_bool_dtype and np.issubdtype(arr.dtype, np.bool_):
        _raise(f'{name} must be numeric', error_type)
    try:
        out = np.ascontiguousarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        _raise_from(f'{name} must be numeric', error_type, exc)
    if np.any(~np.isfinite(out)):
        _raise(f'{name} must contain only finite values', error_type)
    return out


def coerce_1d_integer_int64(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    allow_integer_like_float: bool = True,
    nonfinite_message: str | None = None,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    arr = _as_1d_array(values, name=name, expected_shape=expected_shape, error_type=error_type)
    if not is_real_numeric_dtype(arr.dtype):
        _raise(f'{name} must have a real numeric dtype', error_type)
    if np.issubdtype(arr.dtype, np.integer):
        out = np.ascontiguousarray(arr, dtype=np.int64)
        if not np.array_equal(arr, out):
            _raise(f'{name} values must fit in int64', error_type)
        return out
    if not allow_integer_like_float or np.issubdtype(arr.dtype, np.complexfloating):
        _raise(f'{name} must contain integer values', error_type)

    try:
        float_values = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        _raise_from(f'{name} must contain integer values', error_type, exc)
    int64_info = np.iinfo(np.int64)
    int64_upper_exclusive = np.float64(2**63)
    if np.any(~np.isfinite(float_values)):
        message = nonfinite_message or 'must contain integer values'
        _raise(f'{name} {message}', error_type)
    if (
        np.any(float_values != np.trunc(float_values))
        or np.any(float_values < int64_info.min)
        or np.any(float_values >= int64_upper_exclusive)
    ):
        _raise(f'{name} must contain integer values', error_type)
    return np.ascontiguousarray(float_values.astype(np.int64), dtype=np.int64)


def coerce_1d_bool_array(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    arr = _as_1d_array(values, name=name, expected_shape=expected_shape, error_type=error_type)
    if not np.issubdtype(arr.dtype, np.bool_):
        _raise(f'{name} must have bool dtype', error_type)
    return np.ascontiguousarray(arr, dtype=bool)


def coerce_1d_string_array(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    reject_object_dtype: bool = True,
    allow_non_string_dtype: bool = False,
    output_dtype: object = str,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    arr = _as_1d_array(values, name=name, expected_shape=expected_shape, error_type=error_type)
    if allow_non_string_dtype:
        return np.ascontiguousarray(arr.astype(output_dtype, copy=False))
    if arr.dtype == object:
        if reject_object_dtype:
            _raise(f'{name} must not have object dtype', error_type)
        return np.ascontiguousarray(arr.astype(output_dtype))
    if arr.dtype.kind not in {'U', 'S'}:
        _raise(f'{name} must have string dtype', error_type)
    return np.ascontiguousarray(arr.astype(output_dtype, copy=False))


def coerce_positive_int(
    value: object,
    *,
    name: str,
    error_type: type[Exception] = ValueError,
) -> int:
    out = _coerce_int(value, name=name, error_type=error_type)
    if out <= 0:
        _raise(f'{name} must be greater than 0', error_type)
    return out


def coerce_nonnegative_int(
    value: object,
    *,
    name: str,
    error_type: type[Exception] = ValueError,
) -> int:
    out = _coerce_int(value, name=name, error_type=error_type)
    if out < 0:
        _raise(f'{name} must be non-negative', error_type)
    return out


def coerce_finite_float(
    value: object,
    *,
    name: str,
    error_type: type[Exception] = ValueError,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        _raise(f'{name} must be finite', error_type)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        _raise_from(f'{name} must be finite', error_type, exc)
    if not np.isfinite(out):
        _raise(f'{name} must be finite', error_type)
    return out


def coerce_optional_finite_float(
    value: object,
    *,
    name: str,
    error_type: type[Exception] = ValueError,
) -> float | None:
    if value is None:
        return None
    return coerce_finite_float(value, name=name, error_type=error_type)


def coerce_positive_finite_float(
    value: object,
    *,
    name: str,
    error_type: type[Exception] = ValueError,
) -> float:
    out = coerce_finite_float(value, name=name, error_type=error_type)
    if out <= 0.0:
        _raise(f'{name} must be greater than 0', error_type)
    return out


def coerce_nonnegative_finite_float(
    value: object,
    *,
    name: str,
    error_type: type[Exception] = ValueError,
) -> float:
    out = coerce_finite_float(value, name=name, error_type=error_type)
    if out < 0.0:
        _raise(f'{name} must be non-negative', error_type)
    return out


def coerce_header_byte(
    value: object,
    *,
    name: str,
    error_type: type[Exception] = ValueError,
) -> int:
    out = coerce_positive_int(value, name=name, error_type=error_type)
    if out > 240:
        _raise(f'{name} must be between 1 and 240', error_type)
    return out


def _as_1d_array(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None,
    error_type: type[Exception],
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        _raise(f'{name} must be a 1D array', error_type)
    if expected_shape is not None and arr.shape != expected_shape:
        _raise(
            f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}',
            error_type,
        )
    return arr


def _coerce_int(
    value: object,
    *,
    name: str,
    error_type: type[Exception],
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _raise(f'{name} must be an integer', error_type)
    return int(value)


def _raise(message: str, error_type: type[Exception]) -> NoReturn:
    raise error_type(message)


def _raise_from(
    message: str,
    error_type: type[Exception],
    exc: BaseException,
) -> NoReturn:
    raise error_type(message) from exc


__all__ = [
    'coerce_1d_bool_array',
    'coerce_1d_castable_finite_float64',
    'coerce_1d_finite_float64',
    'coerce_1d_integer_int64',
    'coerce_1d_real_numeric_float64',
    'coerce_1d_string_array',
    'coerce_finite_float',
    'coerce_header_byte',
    'coerce_nonnegative_finite_float',
    'coerce_nonnegative_int',
    'coerce_optional_finite_float',
    'coerce_positive_finite_float',
    'coerce_positive_int',
    'is_real_numeric_dtype',
]
