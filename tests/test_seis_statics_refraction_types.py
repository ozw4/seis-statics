from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import numpy as np

from seis_statics.refraction import (
    RefractionEndpointTable,
    RefractionLayerObservationMasks,
    RefractionStaticInputModel,
)
from seis_statics.refraction.source_depth import (
    resolve_refraction_source_depth_for_input_model,
)
from seis_statics.refraction import types as refraction_types
from seis_statics.refraction.uphole import resolve_refraction_uphole_for_input_model


def test_refraction_types_construct_shared_input_model_without_path_containers() -> None:
    n_traces = 2
    endpoint_table = RefractionEndpointTable(
        node_id=np.array([0, 1]),
        endpoint_id=np.array([10, 11]),
        x_m=np.array([100.0, 200.0]),
        y_m=np.array([0.0, 0.0]),
        elevation_m=np.array([50.0, 60.0]),
        kind=np.array(['source', 'receiver']),
        pick_count=np.array([1, 1]),
    )
    layer_masks = RefractionLayerObservationMasks(
        layer_kind=np.array(['v2_t1']),
        layer_enabled=np.array([True]),
        layer_min_offset_m=np.array([0.0]),
        layer_max_offset_m=np.array([1000.0]),
        layer_used_mask_sorted={'v2_t1': np.array([True, True])},
        layer_rejection_reason_sorted={'v2_t1': np.array(['', ''])},
        layer_candidate_count={'v2_t1': n_traces},
        layer_observation_count={'v2_t1': n_traces},
    )

    model = RefractionStaticInputModel(
        file_id='file-1',
        n_traces=n_traces,
        sorted_trace_index=np.array([0, 1]),
        pick_time_s_sorted=np.array([0.1, 0.2]),
        valid_pick_mask_sorted=np.array([True, True]),
        valid_observation_mask_sorted=np.array([True, True]),
        source_id_sorted=np.array([1, 1]),
        receiver_id_sorted=np.array([2, 3]),
        source_x_m_sorted=np.array([0.0, 0.0]),
        source_y_m_sorted=np.array([0.0, 0.0]),
        receiver_x_m_sorted=np.array([100.0, 200.0]),
        receiver_y_m_sorted=np.array([0.0, 0.0]),
        source_elevation_m_sorted=np.array([50.0, 50.0]),
        receiver_elevation_m_sorted=np.array([60.0, 65.0]),
        source_depth_m_sorted=None,
        geometry_distance_m_sorted=np.array([100.0, 200.0]),
        offset_m_sorted=None,
        distance_m_sorted=np.array([100.0, 200.0]),
        source_endpoint_key_sorted=np.array(['s1', 's1']),
        receiver_endpoint_key_sorted=np.array(['r2', 'r3']),
        source_node_id_sorted=np.array([0, 0]),
        receiver_node_id_sorted=np.array([1, 2]),
        node_x_m=np.array([0.0, 100.0, 200.0]),
        node_y_m=np.array([0.0, 0.0, 0.0]),
        node_elevation_m=np.array([50.0, 60.0, 65.0]),
        node_kind=np.array(['source', 'receiver', 'receiver']),
        rejection_reason_sorted=np.array(['', '']),
        qc={},
        endpoint_table=endpoint_table,
        metadata={},
        layer_observation_masks=layer_masks,
    )

    assert model.endpoint_table is endpoint_table
    assert model.layer_observation_masks is layer_masks
    assert model.source_depth_m_sorted is None

    source_depth = resolve_refraction_source_depth_for_input_model(
        input_model=model,
        mode='none',
        source_depth_byte=None,
    )
    uphole = resolve_refraction_uphole_for_input_model(
        input_model=model,
        uphole_time_sorted=np.array([np.nan, np.inf]),
        mode='none',
        uphole_time_byte=None,
    )
    np.testing.assert_array_equal(source_depth.source_endpoint_id, [0])
    np.testing.assert_array_equal(uphole.source_endpoint_id, [0])
    assert source_depth.source_depth_status.tolist() == ['ok']
    assert uphole.uphole_status.tolist() == ['ok']


def test_refraction_public_types_do_not_expose_path_fields() -> None:
    for public_name in refraction_types.__all__:
        obj = getattr(refraction_types, public_name)
        if not hasattr(obj, '__dataclass_fields__'):
            continue
        assert all(type_hint is not Path for type_hint in get_type_hints(obj).values())


def test_refraction_package_import_does_not_eager_import_scipy_modules() -> None:
    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'scipy' or name.startswith('scipy.'):
        raise ImportError('scipy import blocked')
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import seis_statics.refraction
"""

    result = subprocess.run(
        [sys.executable, '-c', code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
