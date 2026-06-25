from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARITY_MANIFEST = REPO_ROOT / 'tests' / 'parity_manifest.md'

EXPECTED_TIME_TERM_PUBLIC_API = (
    'MoveoutDistanceSource',
    'ORDER',
    'SIGN_CONVENTION',
    'ROBUST_SCALE_FLOOR_S',
    'TimeTermDesignMatrix',
    'TimeTermDesignMatrixOptions',
    'TimeTermGaugeMode',
    'TimeTermGaugeResolution',
    'TimeTermAppliedShiftResult',
    'TimeTermInversionInputs',
    'TimeTermMoveoutConfig',
    'TimeTermMoveoutModel',
    'TimeTermMoveoutResult',
    'TimeTermRobustIterationSummary',
    'TimeTermRobustMethod',
    'TimeTermRobustOptions',
    'TimeTermRobustSolveResult',
    'TimeTermRobustStopReason',
    'TimeTermSolverSystem',
    'TimeTermSparseSolverName',
    'TimeTermSparseSolverOptions',
    'TimeTermSparseSolverResult',
    'TimeTermTracePredictionPolicy',
    'build_node_components',
    'build_reciprocal_pair_index',
    'build_time_term_outlier_mask',
    'build_time_term_design_matrix',
    'build_time_term_solver_system',
    'compose_time_term_applied_shifts',
    'compute_geometry_distance_m',
    'compute_time_term_moveout',
    'compute_time_term_robust_center_scale',
    'delay_to_applied_shift',
    'solve_time_term_sparse_least_squares',
    'solve_time_term_robust_least_squares',
    'summarize_time_term_design_matrix',
    'summarize_time_term_moveout',
    'summarize_time_term_robust_solver_result',
    'summarize_time_term_sparse_solver_result',
    'validate_time_term_robust_options',
)

EXPECTED_REFRACTION_PUBLIC_API = (
    'BedrockVelocityMode',
    'ALREADY_ASSIGNED_REJECTION_REASON',
    'GlobalBedrockSlownessEstimateResult',
    'INVALID_OBSERVATION_REJECTION_REASON',
    'INVALID_OFFSET_REJECTION_REASON',
    'LOCAL_V2_STATUS_VALUES',
    'LOW_FOLD_CELL_REJECTION_REASON',
    'LOW_FOLD_CELL_VELOCITY_STATUS',
    'LOW_FOLD_NODE_REJECTION_REASON',
    'LOW_FOLD_NODE_STATUS',
    'ManualStaticSignConvention',
    'OK_REJECTION_REASON',
    'OUTSIDE_LAYER_GATE_REJECTION_REASON',
    'OUTSIDE_REFRACTOR_CELL_GRID_REASON',
    'REFRACTION_FIELD_CORRECTION_COMPONENT_NAMES',
    'REFRACTION_STATIC_STATUSES',
    'ROBUST_REJECTION_REASON',
    'RefractionCellAssignment',
    'RefractionCellCoordinateMode',
    'RefractionCellGrid',
    'RefractionCellProjectedPoints',
    'RefractionCellProjectedSourceReceiver',
    'RefractionBedrockEstimationError',
    'RefractionDesignMatrixNodeDiagnostics',
    'RefractionDatumEndpointResult',
    'RefractionDatumError',
    'RefractionDatumStaticsResult',
    'RefractionSmoothedFloatingDatum',
    'RefractionEndpointTable',
    'RefractionEndpointFieldCorrectionResult',
    'RefractionFieldCorrectionComponentName',
    'RefractionFieldComposedTraceShiftResult',
    'RefractionFieldCompositionError',
    'RefractionFieldInvalidComponentPolicy',
    'RefractionFirstLayerMode',
    'RefractionHalfInterceptEndpointResult',
    'RefractionHalfInterceptError',
    'RefractionHalfInterceptResult',
    'RefractionLayerAssignmentPolicy',
    'RefractionLayerConfig',
    'RefractionLayerConfigLayer',
    'RefractionLayerKind',
    'RefractionLayerObservationMasks',
    'RefractionLayerVelocityMode',
    'RefractionManualStaticResult',
    'RefractionMultilayerConversionError',
    'RefractionMultilayerConversionResult',
    'RefractionMultilayerDatumStaticsResult',
    'RefractionMultilayerEndpointConversion',
    'RefractionMultilayerTimeTermLayerResult',
    'RefractionMultilayerTimeTermSolveResult',
    'RefractionMultilayerTimeTermSolverError',
    'RefractionManualStaticTableRow',
    'RefractionSourceDepthMode',
    'RefractionSourceDepthResult',
    'RefractionSourceDepthStatus',
    'RefractionStaticConversionMode',
    'RefractionStaticConversionOptions',
    'RefractionStaticDatumMode',
    'RefractionStaticDatumOptions',
    'RefractionStaticDatumSmoothingMethod',
    'RefractionStaticDistanceSource',
    'RefractionStaticFirstLayerOptions',
    'RefractionStaticFloatingDatumMode',
    'RefractionStaticInputModel',
    'RefractionStaticDesignMatrix',
    'RefractionStaticDesignMatrixError',
    'RefractionTraceFieldCorrectionResult',
    'RefractionV1EstimateResult',
    'RefractionV1EstimationError',
    'RefractionWeatheringEndpointComponents',
    'RefractionWeatheringError',
    'RefractionWeatheringModel',
    'RefractionWeatheringReplacementError',
    'RefractionWeatheringReplacementResult',
    'RefractionWeatheringThicknessComputation',
    'RefractionStaticLayerKind',
    'RefractionStaticLayerAssignmentPolicy',
    'RefractionStaticLayerOptions',
    'RefractionStaticLayerVelocityMode',
    'RefractionStaticMethod',
    'RefractionStaticModelOptions',
    'RefractionStaticMoveoutModel',
    'RefractionStaticMoveoutOptions',
    'RefractionStaticReducedTimeQcOptions',
    'RefractionStaticReducedTimeQcVelocityMode',
    'RefractionStaticRefractorCellAssignmentMode',
    'RefractionStaticRefractorCellCoordinateMode',
    'RefractionStaticRefractorCellOptions',
    'RefractionStaticRefractorCellOutsideGridPolicy',
    'RefractionStaticRobustIterationSummary',
    'RefractionStaticRobustMethod',
    'RefractionStaticRobustOptions',
    'RefractionStaticRobustStopReason',
    'RefractionStaticSolveResult',
    'RefractionStaticSolveSystem',
    'RefractionStaticSolverOptions',
    'RefractionStaticSolverError',
    'RefractionT1LSST1LayerEndpointComponents',
    'RefractionT1LSST2LayerThicknessResult',
    'RefractionT1LSST3LayerThicknessResult',
    'RefractionT1LSSTError',
    'RefractionUpholeResult',
    'RefractionUpholeStatus',
    'ResolvedRefractionFirstLayer',
    'ResolvedFloatingDatum',
    'T1LSST_SIGN_CONVENTION',
    'VELOCITY_ORDER_REJECTION_REASON',
    'assign_observation_midpoint_cells',
    'assign_points_to_refraction_cells',
    'build_refraction_layer_observation_masks',
    'build_refraction_multilayer_conversion',
    'build_refraction_cell_grid',
    'build_refraction_datum_statics',
    'build_refraction_design_matrix_node_diagnostics',
    'build_refraction_endpoint_datum_statics',
    'build_refraction_half_intercept_design',
    'build_refraction_half_intercept_result',
    'build_refraction_half_intercept_result_from_bedrock_result',
    'build_refraction_static_cell_design_matrix',
    'build_refraction_static_design_matrix',
    'build_refraction_static_design_matrix_from_arrays',
    'build_refraction_static_solver_system',
    'build_refraction_weathering_model_from_half_intercept_result',
    'build_refraction_weathering_replacement_statics',
    'classify_refraction_endpoint_static_status',
    'compose_t1lsst_1layer_endpoint_component_rows',
    'compose_refraction_endpoint_field_corrections',
    'compose_refraction_final_trace_shift',
    'compose_refraction_trace_field_corrections',
    'compute_t1lsst_1layer_thickness',
    'compute_t1lsst_1layer_weathering_correction',
    'compute_t1lsst_2layer_thicknesses',
    'compute_t1lsst_2layer_thicknesses_with_status',
    'compute_t1lsst_2layer_weathering_correction',
    'compute_t1lsst_3layer_thicknesses',
    'compute_t1lsst_3layer_thicknesses_with_status',
    'compute_t1lsst_3layer_weathering_correction',
    'compute_refraction_datum_elevation_shift_s',
    'compute_refraction_datum_elevation_shift_scalar_s',
    'compute_refraction_multilayer_datum_statics_from_input_model',
    'compute_weathering_thickness_from_half_intercept_time',
    'compute_weathering_thickness_from_half_intercept_time_with_status',
    'compute_weathering_thickness_scalar_from_half_intercept_time',
    'compute_weathering_replacement_shift_s',
    'compute_weathering_replacement_shift_scalar_s',
    'compute_source_depth_weathering_time_correction',
    'compute_source_depth_weathering_time_correction_from_result',
    'compute_source_receiver_midpoints',
    'compute_uphole_time_correction',
    'compute_uphole_time_correction_from_result',
    'estimate_global_v1_from_direct_arrivals',
    'estimate_global_bedrock_slowness_from_input_model',
    'estimate_refraction_half_intercept_from_design',
    'estimate_refraction_half_intercept_from_input_model',
    'effective_refraction_cell_grid_config',
    'layer_offset_gate_contains',
    'manual_static_inline_rows',
    'normalize_refraction_first_layer_request',
    'normalize_refraction_layer_config',
    'normalize_refraction_manual_static_rows',
    'project_refraction_cell_coordinates',
    'project_refraction_cell_points',
    'refraction_cell_coordinate_metadata',
    'refraction_cell_coordinate_metadata_from_config',
    'refraction_layer_observation_qc',
    'resolve_weathering_velocity_m_s',
    'resolve_refraction_source_depth',
    'resolve_refraction_source_depth_for_input_model',
    'resolve_refraction_manual_static',
    'resolve_refraction_uphole',
    'resolve_refraction_uphole_for_input_model',
    'resolve_smoothed_refraction_floating_datum',
    'resolved_first_layer_weathering_velocity_m_s',
    'solve_refraction_static_design_least_squares',
    'solve_refraction_static_least_squares',
    'solve_refraction_multilayer_time_terms',
    'smooth_refraction_floating_datum_elevation',
    'summarize_refraction_static_design_matrix',
    'summarize_refraction_static_solve_result',
    'validate_refraction_static_solver_options',
    'validate_resolved_first_layer_velocity_match',
)


def _assert_public_api(module_name: str, expected: tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)

    assert tuple(module.__all__) == expected
    assert len(module.__all__) == len(set(module.__all__))
    for name in expected:
        assert getattr(module, name) is not None


def test_package_root_stays_light_and_exports_no_eager_public_api() -> None:
    package = importlib.import_module('seis_statics')

    assert package.__all__ == []


def test_lightweight_imports_do_not_load_scipy_heavy_modules() -> None:
    script = """
import importlib
import sys

package = importlib.import_module('seis_statics')
assert package.__all__ == []
assert 'scipy' not in sys.modules
assert 'seis_statics.refraction.solver' not in sys.modules
assert 'seis_statics.refraction.design_matrix' not in sys.modules

refraction = importlib.import_module('seis_statics.refraction')
assert 'RefractionStaticModelOptions' in refraction.__all__
assert 'scipy' not in sys.modules
assert 'seis_statics.refraction.solver' not in sys.modules
assert 'seis_statics.refraction.design_matrix' not in sys.modules

status = importlib.import_module('seis_statics.refraction.status')
assert callable(status.classify_refraction_endpoint_static_status)
assert 'scipy' not in sys.modules
assert 'seis_statics.refraction.solver' not in sys.modules
assert 'seis_statics.refraction.design_matrix' not in sys.modules
"""
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        str(REPO_ROOT / 'src')
        if not env.get('PYTHONPATH')
        else f"{REPO_ROOT / 'src'}{os.pathsep}{env['PYTHONPATH']}"
    )
    subprocess.run(
        [sys.executable, '-c', script],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_time_term_public_api_is_pinned() -> None:
    _assert_public_api('seis_statics.time_term', EXPECTED_TIME_TERM_PUBLIC_API)


def test_refraction_public_api_is_pinned() -> None:
    _assert_public_api('seis_statics.refraction', EXPECTED_REFRACTION_PUBLIC_API)


def test_distribution_version_is_release_handoff_minor() -> None:
    text = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')

    assert re.search(r'^version = "0\.4\.0"$', text, flags=re.MULTILINE)


def test_parity_manifest_records_migrated_test_mapping() -> None:
    text = PARITY_MANIFEST.read_text(encoding='utf-8')
    migrated_sources = (
        'test_refraction_static_datum.py',
        'test_refraction_static_multilayer_orchestration.py',
        'test_refraction_static_multilayer_v3_t2_solver.py',
        'test_refraction_static_multilayer_vsub_t3_solver.py',
        'test_refraction_static_multilayer_cell_layers.py',
        'test_refraction_static_multilayer_2layer_e2e.py',
        'test_refraction_static_multilayer_3layer_composition.py',
        'test_refraction_static_multilayer_3layer_e2e.py',
        'test_refraction_static_multilayer_types.py',
    )

    assert 'Canonical source' in text
    assert 'time-term' in text
    assert 'refraction' in text
    assert 'tests/test_seis_statics_time_term_sparse_solver.py' in text
    assert 'tests/test_seis_statics_refraction_cell_solver.py' in text
    for source_test in migrated_sources:
        assert f'`{source_test}`' in text


def test_wheel_install_smoke_without_application_dependencies(tmp_path: Path) -> None:
    wheel_dir = tmp_path / 'wheelhouse'
    wheel_dir.mkdir()
    subprocess.run(
        [sys.executable, '-m', 'pip', 'wheel', '.', '--no-deps', '-w', str(wheel_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob('seis_statics-0.4.0-*.whl'))

    install_dir = tmp_path / 'install'
    subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'install',
            '--no-deps',
            '--target',
            str(install_dir),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    script = r"""
import importlib
import importlib.abc

import numpy as np

BLOCKED_ROOTS = {'app', 'fastapi', 'pydantic', 'segyio'}


class BlockApplicationDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in BLOCKED_ROOTS:
            raise ImportError(f'blocked application dependency: {fullname}')
        return None


import sys

sys.meta_path.insert(0, BlockApplicationDependencies())

from seis_statics.refraction import (
    RefractionStaticModelOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
    build_refraction_static_design_matrix_from_arrays,
    solve_refraction_static_design_least_squares,
)

source_node_id = np.asarray([10, 10, 20, 20, 10, 20], dtype=np.int64)
receiver_node_id = np.asarray([30, 40, 30, 40, 30, 40], dtype=np.int64)
distance_m = np.asarray([500.0, 700.0, 600.0, 850.0, 900.0, 950.0])
t1_by_node = {10: 0.03, 20: 0.05, 30: 0.035, 40: 0.045}
slowness = 1.0 / 2500.0
pick_time = np.asarray(
    [
        t1_by_node[int(src)] + t1_by_node[int(rec)] + dist * slowness
        for src, rec, dist in zip(source_node_id, receiver_node_id, distance_m)
    ],
    dtype=np.float64,
)
design = build_refraction_static_design_matrix_from_arrays(
    pick_time_s_sorted=pick_time,
    valid_observation_mask_sorted=np.ones(6, dtype=bool),
    source_node_id_sorted=source_node_id,
    receiver_node_id_sorted=receiver_node_id,
    distance_m_sorted=distance_m,
    node_id=np.asarray([10, 20, 30, 40]),
    bedrock_velocity_mode='solve_global',
    n_traces=6,
)
result = solve_refraction_static_design_least_squares(
    design,
    model=RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode='solve_global',
        initial_bedrock_velocity_m_s=3000.0,
    ),
    solver_options=RefractionStaticSolverOptions(
        half_intercept_damping_lambda=0.0,
        robust=RefractionStaticRobustOptions(enabled=False),
    ),
)
assert abs(result.bedrock_velocity_m_s - 2500.0) < 1.0e-6
assert result.qc['solver_name'] == 'lsq_linear'
"""
    env = os.environ.copy()
    env['PYTHONPATH'] = str(install_dir)
    subprocess.run(
        [sys.executable, '-c', script],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
