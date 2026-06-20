from __future__ import annotations

import numpy as np

from seis_statics.refraction import (
    ALREADY_ASSIGNED_REJECTION_REASON,
    INVALID_OBSERVATION_REJECTION_REASON,
    INVALID_OFFSET_REJECTION_REASON,
    OUTSIDE_LAYER_GATE_REJECTION_REASON,
    RefractionLayerConfig,
    RefractionLayerConfigLayer,
    build_refraction_layer_observation_masks,
    refraction_layer_observation_qc,
)


def _three_layer_config() -> RefractionLayerConfig:
    return RefractionLayerConfig(
        layers=(
            RefractionLayerConfigLayer(
                kind='v2_t1',
                min_offset_m=0.0,
                max_offset_m=100.0,
                velocity_mode='solve_global',
            ),
            RefractionLayerConfigLayer(
                kind='v3_t2',
                min_offset_m=100.0,
                max_offset_m=200.0,
                velocity_mode='solve_global',
            ),
            RefractionLayerConfigLayer(
                kind='vsub_t3',
                min_offset_m=200.0,
                max_offset_m=None,
                velocity_mode='solve_global',
            ),
        ),
    )


def test_refraction_layer_observations_use_half_open_contact_gates() -> None:
    masks = build_refraction_layer_observation_masks(
        layer_config=_three_layer_config(),
        offset_m_sorted=np.asarray([-50.0, 100.0, 199.0, 200.0, np.nan, 99.0, 100.0, 250.0]),
        valid_observation_mask_sorted=np.asarray(
            [True, True, False, True, True, True, True, False]
        ),
        rejection_reason_sorted=np.asarray(['', '', 'bad_pick', '', '', '', '', '']),
    )

    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['v2_t1'],
        np.asarray([True, False, False, False, False, True, False, False]),
    )
    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['v3_t2'],
        np.asarray([False, True, False, False, False, False, True, False]),
    )
    np.testing.assert_array_equal(
        masks.layer_used_mask_sorted['vsub_t3'],
        np.asarray([False, False, False, True, False, False, False, False]),
    )

    used_stack = np.vstack(
        [masks.layer_used_mask_sorted[str(kind)] for kind in masks.layer_kind]
    )
    assert np.all(np.count_nonzero(used_stack, axis=0) <= 1)
    assert masks.layer_candidate_count == {
        'v2_t1': 2,
        'v3_t2': 3,
        'vsub_t3': 2,
    }
    assert masks.layer_observation_count == {
        'v2_t1': 2,
        'v3_t2': 2,
        'vsub_t3': 1,
    }
    assert masks.layer_rejection_reason_sorted['v2_t1'][1] == (
        OUTSIDE_LAYER_GATE_REJECTION_REASON
    )
    assert masks.layer_rejection_reason_sorted['v2_t1'][4] == (
        INVALID_OFFSET_REJECTION_REASON
    )
    assert masks.layer_rejection_reason_sorted['v3_t2'][2] == 'bad_pick'
    assert masks.layer_rejection_reason_sorted['vsub_t3'][7] == (
        INVALID_OBSERVATION_REJECTION_REASON
    )
    np.testing.assert_array_equal(
        masks.layer_min_offset_m,
        np.asarray([0.0, 100.0, 200.0]),
    )
    np.testing.assert_array_equal(
        masks.layer_max_offset_m,
        np.asarray([100.0, 200.0, np.inf]),
    )

    assert refraction_layer_observation_qc(masks) == {
        'layer_count': 3,
        'layer_candidate_count': {
            'v2_t1': 2,
            'v3_t2': 3,
            'vsub_t3': 2,
        },
        'layer_observation_count': {
            'v2_t1': 2,
            'v3_t2': 2,
            'vsub_t3': 1,
        },
        'total_layer_candidate_count': 7,
        'total_layer_observation_count': 5,
    }


def test_refraction_layer_observations_keep_used_masks_exclusive_with_overlap() -> None:
    config = RefractionLayerConfig(
        layers=(
            RefractionLayerConfigLayer(
                kind='v2_t1',
                min_offset_m=0.0,
                max_offset_m=150.0,
                velocity_mode='solve_global',
            ),
            RefractionLayerConfigLayer(
                kind='v3_t2',
                min_offset_m=100.0,
                max_offset_m=None,
                velocity_mode='solve_global',
            ),
        ),
        allow_overlapping_layer_gates=True,
    )

    masks = build_refraction_layer_observation_masks(
        layer_config=config,
        offset_m_sorted=np.asarray([125.0]),
    )

    assert masks.layer_candidate_count == {'v2_t1': 1, 'v3_t2': 1}
    assert masks.layer_observation_count == {'v2_t1': 1, 'v3_t2': 0}
    np.testing.assert_array_equal(masks.layer_used_mask_sorted['v2_t1'], np.asarray([True]))
    np.testing.assert_array_equal(masks.layer_used_mask_sorted['v3_t2'], np.asarray([False]))
    assert masks.layer_rejection_reason_sorted['v3_t2'][0] == (
        ALREADY_ASSIGNED_REJECTION_REASON
    )


def test_refraction_layer_observations_default_invalid_reason_is_not_truncated() -> None:
    masks = build_refraction_layer_observation_masks(
        layer_config=_three_layer_config(),
        offset_m_sorted=np.asarray([250.0]),
        valid_observation_mask_sorted=np.asarray([False]),
    )

    assert masks.layer_rejection_reason_sorted['vsub_t3'][0] == (
        INVALID_OBSERVATION_REJECTION_REASON
    )
