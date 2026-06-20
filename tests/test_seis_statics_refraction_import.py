from __future__ import annotations

import importlib


def test_refraction_public_api_imports_from_package_root() -> None:
    refraction = importlib.import_module('seis_statics.refraction')

    for name in refraction.__all__:
        assert getattr(refraction, name) is not None

    prohibited_public_fragments = (
        'DesignMatrix',
        'SolverResult',
        'WeatheringResult',
        'DatumResult',
        'TraceStore',
        'Artifact',
    )
    assert not any(
        fragment in name
        for name in refraction.__all__
        for fragment in prohibited_public_fragments
    )
