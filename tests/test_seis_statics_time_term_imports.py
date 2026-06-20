from __future__ import annotations

import importlib


def test_time_term_public_api_imports_from_package_root() -> None:
    time_term = importlib.import_module('seis_statics.time_term')

    for name in time_term.__all__:
        assert getattr(time_term, name) is not None

    assert callable(time_term.solve_time_term_robust_least_squares)
    assert callable(time_term.compose_time_term_applied_shifts)
    assert callable(time_term.delay_to_applied_shift)
