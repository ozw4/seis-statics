from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from seis_statics.refraction import RefractionStaticDatumOptions
from seis_statics.time_term import TimeTermInversionInputs

_CONSUMER_OWNED_FIELD_PARTS = (
    'artifact',
    'description',
    'file_id',
    'job',
    'metadata',
    'path',
    'provenance',
)


def _annotation_contains_path(annotation: Any) -> bool:
    if annotation is Path:
        return True
    origin = get_origin(annotation)
    if origin is Path:
        return True
    return any(_annotation_contains_path(arg) for arg in get_args(annotation))


def test_time_term_public_contract_excludes_path_artifact_and_provenance_fields() -> None:
    field_names = {field.name for field in fields(TimeTermInversionInputs)}

    assert field_names == {
        'n_traces',
        'dt',
        'valid_pick_mask_sorted',
        'pick_time_after_static_s_sorted',
        'source_node_id_sorted',
        'receiver_node_id_sorted',
        'n_nodes',
        'offset_sorted',
        'source_x_m_sorted',
        'source_y_m_sorted',
        'receiver_x_m_sorted',
        'receiver_y_m_sorted',
        'order',
        'sign_convention',
    }
    assert not any(
        part in field_name
        for field_name in field_names
        for part in _CONSUMER_OWNED_FIELD_PARTS
    )
    assert not any(
        _annotation_contains_path(annotation)
        for annotation in get_type_hints(TimeTermInversionInputs).values()
    )


def test_refraction_datum_contract_excludes_path_artifact_and_provenance_fields() -> None:
    field_names = {field.name for field in fields(RefractionStaticDatumOptions)}

    assert 'floating_datum_job_id' not in field_names
    assert 'floating_datum_artifact_name' not in field_names
    assert not any(
        part in field_name
        for field_name in field_names
        for part in _CONSUMER_OWNED_FIELD_PARTS
    )
    assert not any(
        _annotation_contains_path(annotation)
        for annotation in get_type_hints(RefractionStaticDatumOptions).values()
    )
