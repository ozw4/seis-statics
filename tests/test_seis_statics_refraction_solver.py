from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy import optimize, sparse

from seis_statics.refraction import (
    RefractionStaticModelOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverError,
    RefractionStaticSolverOptions,
    build_refraction_static_solver_system,
    build_refraction_static_design_matrix_from_arrays,
    solve_refraction_static_design_least_squares,
)
import seis_statics.refraction.solver as solver_module
from seis_statics.refraction.solver import _column_scaled_numerical_rank


def _solver_options(**overrides: object) -> RefractionStaticSolverOptions:
    values: dict[str, object] = {
        'half_intercept_damping_lambda': 0.0,
        'max_abs_half_intercept_time_ms': 100.0,
        'robust': RefractionStaticRobustOptions(enabled=False),
    }
    values.update(overrides)
    return RefractionStaticSolverOptions(**values)


def _model(
    *,
    mode: str,
    fixed_velocity: float | None = None,
    min_velocity: float = 1200.0,
    max_velocity: float = 6000.0,
    initial_velocity: float = 3000.0,
) -> RefractionStaticModelOptions:
    return RefractionStaticModelOptions(
        weathering_velocity_m_s=500.0,
        bedrock_velocity_mode=mode,  # type: ignore[arg-type]
        bedrock_velocity_m_s=fixed_velocity,
        initial_bedrock_velocity_m_s=(
            None if mode == 'fixed_global' else initial_velocity
        ),
        min_bedrock_velocity_m_s=min_velocity,
        max_bedrock_velocity_m_s=max_velocity,
    )


def _simple_lsq_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> solver_module.RefractionStaticSolveSystem:
    csr = sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))
    rhs_array = np.ascontiguousarray(rhs, dtype=np.float64)
    n_parameters = int(csr.shape[1])
    lower_bounds = (
        np.zeros(n_parameters, dtype=np.float64)
        if lower is None
        else np.ascontiguousarray(lower, dtype=np.float64)
    )
    upper_bounds = (
        np.full(n_parameters, np.inf, dtype=np.float64)
        if upper is None
        else np.ascontiguousarray(upper, dtype=np.float64)
    )
    return solver_module.RefractionStaticSolveSystem(
        augmented_matrix=csr,
        augmented_rhs_s=rhs_array,
        observation_matrix=csr,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        initial_parameter_vector=np.zeros(n_parameters, dtype=np.float64),
        n_observation_rows=int(csr.shape[0]),
        n_smoothing_rows=0,
        n_damping_rows=0,
        n_gauge_rows=0,
        n_augmented_rows=int(csr.shape[0]),
        n_parameters=n_parameters,
        component_id_by_node=np.arange(n_parameters, dtype=np.int64),
        n_node_components=n_parameters,
        is_bipartite_by_component=np.zeros(n_parameters, dtype=bool),
        signed_partition_by_node=np.ones(n_parameters, dtype=np.int64),
        gauge_required_by_component=np.zeros(n_parameters, dtype=bool),
        n_bipartite_node_components=0,
        gauge_resolution='not_required',
        half_intercept_damping_lambda=0.0,
        regularized_parameter_group='node_half_intercept_time_s',
        regularization_row_count=0,
        node_lower_bound_s=0.0,
        node_upper_bound_s=float(np.max(upper_bounds[np.isfinite(upper_bounds)]))
        if np.any(np.isfinite(upper_bounds))
        else np.inf,
        slowness_lower_bound_s_per_m=None,
        slowness_upper_bound_s_per_m=None,
        initial_bedrock_slowness_s_per_m=None,
        smoothing_rows=None,
        identifiability=solver_module._NumericalRankDiagnostic(
            method='test',
            n_rows=int(csr.shape[0]),
            n_columns=n_parameters,
            expected_rank=n_parameters,
            estimated_rank=n_parameters,
            expected_nullity=0,
            gauge_nullity=0,
            threshold=0.0,
            critical_singular_value=1.0,
            largest_singular_value=1.0,
            rtol=0.0,
        ),
    )


def _known_global_arrays() -> tuple[np.ndarray, ...]:
    source_node_id = np.asarray([10, 10, 20, 20, 10, 20], dtype=np.int64)
    receiver_node_id = np.asarray([30, 40, 30, 40, 30, 40], dtype=np.int64)
    distance_m = np.asarray([500.0, 700.0, 600.0, 850.0, 900.0, 950.0])
    true_t1_by_node = {
        10: 0.03,
        20: 0.05,
        30: 0.035,
        40: 0.045,
    }
    true_slowness = 1.0 / 2500.0
    pick_time = np.asarray(
        [
            true_t1_by_node[int(src)]
            + true_t1_by_node[int(rec)]
            + dist * true_slowness
            for src, rec, dist in zip(
                source_node_id,
                receiver_node_id,
                distance_m,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    valid_mask = np.asarray([True, True, True, True, True, False])
    return source_node_id, receiver_node_id, distance_m, pick_time, valid_mask


def test_refraction_solver_solve_global_matches_known_parameters_and_residual() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_global',
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='solve_global'),
        solver_options=_solver_options(),
    )

    assert result.bedrock_velocity_mode == 'solve_global'
    assert result.bedrock_velocity_m_s == pytest.approx(2500.0, abs=1.0e-6)
    assert result.bedrock_slowness_s_per_m == pytest.approx(1.0 / 2500.0)
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.03, 0.05, 0.035, 0.045],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(result.row_modeled_pick_time_s, pick_time[:5])
    np.testing.assert_allclose(result.row_residual_s, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(result.modeled_pick_time_s_sorted[:5], pick_time[:5])
    assert np.isnan(result.modeled_pick_time_s_sorted[5])
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, True, True, True, True, False],
    )
    assert set(result.node_solution_status.tolist()) == {'solved'}
    assert result.system.n_gauge_rows == 0
    assert result.system.n_node_components == 1
    assert result.system.n_bipartite_node_components == 1
    assert result.system.gauge_resolution == 'postsolve_minimum_norm'
    assert result.qc['n_node_components'] == 1
    assert result.qc['n_bipartite_node_components'] == 1
    assert result.qc['n_gauge_required_node_components'] == 1
    assert result.qc['gauge_resolution'] == 'postsolve_minimum_norm'
    assert result.qc['solver_name'] == 'lsq_linear'
    assert result.qc['physical_identifiability']['expected_rank'] == 4
    assert result.qc['physical_identifiability']['estimated_numerical_rank'] == 4


@pytest.mark.parametrize('excess_s', [1.0e-6, 0.1])
def test_refraction_solver_fixed_global_exact_fit_is_scale_stable(
    excess_s: float,
) -> None:
    fixed_velocity = 2500.0
    distance_m = np.asarray([500.0, 500.0], dtype=np.float64)
    pick_time = distance_m / fixed_velocity + np.asarray(
        [excess_s, 0.0],
        dtype=np.float64,
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 20], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([30, 30], dtype=np.int64),
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=2,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(min_picks_per_node=1),
    )

    np.testing.assert_allclose(result.row_residual_s, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [excess_s, 0.0, 0.0],
        atol=1.0e-10,
    )
    assert result.solver_success
    assert result.qc['solver_quality']['verified'] is True
    assert result.qc['solver_quality']['solve_scale'] > 0.0


def test_refraction_solver_fixed_global_sparse_consumer_equivalent_quality_regression() -> None:
    fixed_velocity = 2600.0
    n_sources = 48
    n_receivers = 48
    source_ids = np.arange(1000, 1000 + n_sources, dtype=np.int64)
    receiver_ids = np.arange(2000, 2000 + n_receivers, dtype=np.int64)
    source_x = np.linspace(0.0, 4700.0, n_sources, dtype=np.float64)
    receiver_x = np.linspace(120.0, 4820.0, n_receivers, dtype=np.float64)
    source_grid, receiver_grid = np.meshgrid(
        np.arange(n_sources, dtype=np.int64),
        np.arange(n_receivers, dtype=np.int64),
        indexing='ij',
    )
    source_index = source_grid.ravel()
    receiver_index = receiver_grid.ravel()
    source_node_id = source_ids[source_index]
    receiver_node_id = receiver_ids[receiver_index]
    distance_m = (
        np.abs(receiver_x[receiver_index] - source_x[source_index]) + 350.0
    )
    source_t1 = 0.018 + 0.0003 * (np.arange(n_sources, dtype=np.float64) % 7.0)
    receiver_t1 = 0.022 + 0.00025 * (
        np.arange(n_receivers, dtype=np.float64) % 11.0
    )
    pick_time = (
        source_t1[source_index]
        + receiver_t1[receiver_index]
        + distance_m / fixed_velocity
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(pick_time.shape, dtype=bool),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.concatenate([source_ids, receiver_ids]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=int(pick_time.size),
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(min_picks_per_node=1),
    )
    repeat = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(min_picks_per_node=1),
    )

    quality = result.qc['solver_quality']
    repeat_quality = repeat.qc['solver_quality']
    matrix_shape = tuple(int(value) for value in quality['matrix_shape'])
    scipy_default_lsmr_maxiter = min(matrix_shape)
    assert result.solver_success
    assert quality['verified'] is True
    assert quality['stage'] in {'first_attempt', 'retry_strict_lsmr'}
    expected_lsmr_tol = (
        solver_module._LSQ_FIRST_LSMR_TOL
        if quality['stage'] == 'first_attempt'
        else solver_module._LSQ_RETRY_LSMR_TOL
    )
    assert quality['lsmr_tol'] == pytest.approx(expected_lsmr_tol)
    assert quality['lsmr_maxiter'] == solver_module._lsq_lsmr_maxiter(
        result.system,
        stage=str(quality['stage']),
    )
    assert int(quality['lsmr_maxiter']) > scipy_default_lsmr_maxiter
    assert matrix_shape == result.system.augmented_matrix.shape
    assert matrix_shape[0] * matrix_shape[1] > solver_module._LSQ_DENSE_RETRY_MAX_ELEMENTS
    assert quality['projected_gradient_ratio'] < 1.0
    assert quality['matrix_nnz'] == result.system.augmented_matrix.nnz
    assert np.all(np.diff(result.system.augmented_matrix.indptr) > 0)
    np.testing.assert_allclose(result.row_residual_s, 0.0, atol=1.0e-10)
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        np.ones(pick_time.shape, dtype=bool),
    )
    assert repeat.solver_success
    assert repeat_quality['verified'] is True
    assert repeat_quality['stage'] == quality['stage']
    assert repeat.solver_status == result.solver_status
    np.testing.assert_allclose(
        repeat.node_half_intercept_time_s,
        result.node_half_intercept_time_s,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(repeat.row_residual_s, result.row_residual_s, atol=1.0e-12)


def test_refraction_solver_polished_solver_cost_is_unscaled_in_qc() -> None:
    fixed_velocity = 2500.0
    distance_m = np.asarray([500.0, 500.0], dtype=np.float64)
    pick_time = distance_m / fixed_velocity + np.asarray([0.02, 0.04])
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 10], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([10, 10], dtype=np.int64),
        distance_m_sorted=distance_m,
        node_id=np.asarray([10], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        n_traces=2,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(),
    )

    quality = result.qc['solver_quality']
    assert quality['stage'] == 'active_set_polish'
    assert quality['solve_scale'] != pytest.approx(1.0)
    assert result.solver_cost == pytest.approx(quality['unscaled_objective'])
    assert result.qc['solver_cost'] == pytest.approx(quality['unscaled_objective'])
    np.testing.assert_allclose(result.row_residual_s, [-0.01, 0.01], atol=1.0e-12)


def test_refraction_lsq_solution_is_invariant_to_common_system_scale() -> None:
    base = _simple_lsq_system(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64),
        np.asarray([0.2, 0.4, 0.6], dtype=np.float64),
        lower=np.zeros(2, dtype=np.float64),
        upper=np.ones(2, dtype=np.float64),
    )
    base_result = solver_module._run_lsq_linear(base)

    for common_scale in (1.0e-9, 1.0e9):
        scaled = replace(
            base,
            augmented_matrix=base.augmented_matrix * common_scale,
            augmented_rhs_s=base.augmented_rhs_s * common_scale,
            observation_matrix=base.observation_matrix * common_scale,
        )
        result = solver_module._run_lsq_linear(scaled)
        np.testing.assert_allclose(result.x, base_result.x, atol=1.0e-10)
        np.testing.assert_array_equal(
            result.active_mask,
            base_result.active_mask,
        )
        assert result.refraction_solver_quality['verified'] is True


def test_refraction_lsq_sparse_attempts_forward_distinct_lsmr_maxiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _simple_lsq_system(
        np.eye(1, dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        lower=np.zeros(1, dtype=np.float64),
        upper=np.ones(1, dtype=np.float64),
    )
    calls: list[dict[str, object]] = []

    def fake_lsq_linear(*args: object, **kwargs: object) -> optimize.OptimizeResult:
        del args
        calls.append(dict(kwargs))
        x = np.asarray([0.0 if len(calls) == 1 else 1.0], dtype=np.float64)
        return optimize.OptimizeResult(
            x=x,
            success=True,
            status=1,
            message='forced',
            cost=0.0,
            optimality=0.0,
            nit=len(calls),
            unbounded_sol=(x.copy(), 1, len(calls) * 10, 0.1, 0.01, 1.0, 2.0, 3.0),
        )

    monkeypatch.setattr(solver_module.optimize, 'lsq_linear', fake_lsq_linear)

    result = solver_module._run_lsq_linear(system)

    assert len(calls) == 2
    assert calls[0]['lsq_solver'] == 'lsmr'
    assert calls[1]['lsq_solver'] == 'lsmr'
    assert calls[0]['lsmr_tol'] == solver_module._LSQ_FIRST_LSMR_TOL
    assert calls[1]['lsmr_tol'] == solver_module._LSQ_RETRY_LSMR_TOL
    assert calls[0]['lsmr_maxiter'] == solver_module._lsq_lsmr_maxiter(
        system,
        stage='first_attempt',
    )
    assert calls[1]['lsmr_maxiter'] == solver_module._lsq_lsmr_maxiter(
        system,
        stage='retry_strict_lsmr',
    )
    assert int(calls[0]['lsmr_maxiter']) < int(calls[1]['lsmr_maxiter'])
    assert result.refraction_solver_quality['stage'] == 'retry_strict_lsmr'
    assert result.refraction_solver_quality['verified'] is True
    assert result.refraction_solver_quality['lsmr_maxiter'] == calls[1]['lsmr_maxiter']
    assert result.refraction_solver_quality['lsmr_tol'] == calls[1]['lsmr_tol']
    assert result.refraction_solver_quality['lsmr_iterations'] == 20


def test_refraction_lsq_strict_retry_recovers_real_sparse_lsmr_budget_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(solver_module, '_LSQ_FIRST_LSMR_MAXITER_MIN', 1)
    monkeypatch.setattr(solver_module, '_LSQ_FIRST_LSMR_MAXITER_FACTOR', 0)
    system = _simple_lsq_system(
        np.asarray([[1.0, 0.1], [0.0, 1.0]], dtype=np.float64),
        np.asarray([0.19, 0.9], dtype=np.float64),
        lower=np.zeros(2, dtype=np.float64),
        upper=np.ones(2, dtype=np.float64),
    )
    first_lsmr_maxiter = solver_module._lsq_lsmr_maxiter(
        system,
        stage='first_attempt',
    )
    retry_lsmr_maxiter = solver_module._lsq_lsmr_maxiter(
        system,
        stage='retry_strict_lsmr',
    )
    scale = solver_module._common_lsq_solve_scale(
        system.augmented_matrix,
        system.augmented_rhs_s,
    )
    first = solver_module._run_scaled_lsq_linear_attempt(
        system,
        solve_scale=scale,
        stage='first_attempt',
        tol=solver_module._LSQ_FIRST_TOL,
        lsmr_tol=solver_module._LSQ_FIRST_LSMR_TOL,
        max_iter=max(100, 20 * int(system.n_parameters)),
        lsmr_maxiter=first_lsmr_maxiter,
        dense=False,
    )
    _, first_quality = solver_module._verify_lsq_linear_solution(
        system,
        first.x,
        solve_scale=scale,
        stage='first_attempt',
        scipy_result=first,
    )

    result = solver_module._run_lsq_linear(system)

    assert first_lsmr_maxiter == 1
    assert retry_lsmr_maxiter > first_lsmr_maxiter
    assert first_quality.verified is False
    assert first_quality.projected_gradient_ratio > 1.0
    assert result.refraction_solver_quality['stage'] == 'retry_strict_lsmr'
    assert result.refraction_solver_quality['verified'] is True
    assert result.refraction_solver_quality['lsmr_maxiter'] == retry_lsmr_maxiter
    np.testing.assert_allclose(result.x, [0.1, 0.9], atol=1.0e-12)


def test_refraction_lsq_dense_attempt_does_not_forward_lsmr_maxiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _simple_lsq_system(
        np.eye(1, dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        lower=np.zeros(1, dtype=np.float64),
        upper=np.ones(1, dtype=np.float64),
    )
    captured: dict[str, object] = {}

    def fake_lsq_linear(*args: object, **kwargs: object) -> optimize.OptimizeResult:
        del args
        captured.update(kwargs)
        return optimize.OptimizeResult(
            x=np.ones(1, dtype=np.float64),
            success=True,
            status=1,
            message='forced',
            cost=0.0,
            optimality=0.0,
            nit=1,
        )

    monkeypatch.setattr(solver_module.optimize, 'lsq_linear', fake_lsq_linear)

    result = solver_module._run_scaled_lsq_linear_attempt(
        system,
        solve_scale=1.0,
        stage='dense_bvls_retry',
        tol=solver_module._LSQ_RETRY_TOL,
        lsmr_tol=solver_module._LSQ_RETRY_LSMR_TOL,
        max_iter=10,
        lsmr_maxiter=2000,
        dense=True,
    )

    assert captured['method'] == 'bvls'
    assert captured['lsq_solver'] is None
    assert captured['lsmr_tol'] is None
    assert captured['lsmr_maxiter'] is None
    assert result.refraction_solver_lsmr_tol is None
    assert result.refraction_solver_lsmr_maxiter is None


def test_refraction_lsq_quality_diagnostics_include_lsmr_and_kkt_ratio() -> None:
    system = _simple_lsq_system(
        np.eye(1, dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        lower=np.zeros(1, dtype=np.float64),
        upper=np.ones(1, dtype=np.float64),
    )
    fake = optimize.OptimizeResult(
        x=np.zeros(1, dtype=np.float64),
        success=True,
        status=1,
        optimality=0.0,
        nit=3,
        unbounded_sol=(
            np.zeros(1, dtype=np.float64),
            7,
            123,
            1.0e-3,
            1.0e-4,
            2.0,
            30.0,
            4.0,
        ),
    )
    fake.refraction_solver_lsmr_tol = 1.0e-14
    fake.refraction_solver_lsmr_maxiter = 2000

    _, quality = solver_module._verify_lsq_linear_solution(
        system,
        np.zeros(1, dtype=np.float64),
        solve_scale=1.0,
        stage='retry_strict_lsmr',
        scipy_result=fake,
    )
    payload = solver_module._lsq_quality_json(quality)

    assert payload['verified'] is False
    assert payload['stage'] == 'retry_strict_lsmr'
    assert payload['matrix_shape'] == [1, 1]
    assert payload['matrix_nnz'] == 1
    assert payload['outer_iterations'] == 3
    assert payload['outer_status'] == 1
    assert payload['outer_success'] is True
    assert payload['projected_gradient_inf_norm'] == pytest.approx(1.0)
    assert payload['kkt_tolerance'] is not None
    assert payload['projected_gradient_ratio'] is not None
    assert float(payload['projected_gradient_ratio']) > 1.0
    assert payload['lsmr_tol'] == pytest.approx(1.0e-14)
    assert payload['lsmr_maxiter'] == 2000
    assert payload['lsmr_stop_code'] == 7
    assert payload['lsmr_iterations'] == 123
    assert payload['lsmr_condition_estimate'] == pytest.approx(30.0)


def test_refraction_solver_public_result_is_invariant_to_common_system_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_velocity = 2500.0
    distance_m = np.asarray([500.0, 500.0], dtype=np.float64)
    pick_time = distance_m / fixed_velocity + np.asarray(
        [1.0e-6, 0.0],
        dtype=np.float64,
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 20], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([30, 30], dtype=np.int64),
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=2,
    )
    model = _model(mode='fixed_global', fixed_velocity=fixed_velocity)
    options = _solver_options(min_picks_per_node=1)
    base = solve_refraction_static_design_least_squares(
        design,
        model=model,
        solver_options=options,
    )
    original_builder = solver_module.build_refraction_static_solver_system

    def scaled_builder(*args: object, **kwargs: object):
        system = original_builder(*args, **kwargs)
        common_scale = 1.0e-9
        return replace(
            system,
            augmented_matrix=system.augmented_matrix * common_scale,
            augmented_rhs_s=system.augmented_rhs_s * common_scale,
            observation_matrix=system.observation_matrix * common_scale,
        )

    monkeypatch.setattr(
        solver_module,
        'build_refraction_static_solver_system',
        scaled_builder,
    )

    scaled = solve_refraction_static_design_least_squares(
        design,
        model=model,
        solver_options=options,
    )

    assert scaled.solver_success
    assert scaled.qc['solver_quality']['verified'] is True
    np.testing.assert_allclose(
        scaled.node_half_intercept_time_s,
        base.node_half_intercept_time_s,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        scaled.row_modeled_pick_time_s,
        base.row_modeled_pick_time_s,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        scaled.row_residual_s,
        base.row_residual_s,
        atol=1.0e-10,
    )


def test_refraction_lsq_kkt_verifier_rejects_suboptimal_feasible_candidate() -> None:
    system = _simple_lsq_system(
        np.eye(2, dtype=np.float64),
        np.asarray([1.0e-6, 0.0], dtype=np.float64),
        lower=np.zeros(2, dtype=np.float64),
        upper=np.ones(2, dtype=np.float64),
    )
    candidate = np.zeros(2, dtype=np.float64)
    fake = optimize.OptimizeResult(
        success=True,
        status=1,
        optimality=0.0,
        nit=1,
    )

    _, quality = solver_module._verify_lsq_linear_solution(
        system,
        candidate,
        solve_scale=1.0,
        stage='test',
        scipy_result=fake,
    )

    assert not quality.verified
    assert quality.projected_gradient_inf_norm > quality.kkt_tolerance


def test_refraction_lsq_kkt_verifier_rejects_large_cancellation_bound_case() -> None:
    system = _simple_lsq_system(
        np.ones((1, 1), dtype=np.float64),
        np.asarray([1.0e12 + 1.0], dtype=np.float64),
        lower=np.asarray([1.0e12], dtype=np.float64),
        upper=np.full(1, np.inf, dtype=np.float64),
    )
    fake = optimize.OptimizeResult(
        success=True,
        status=1,
        optimality=0.0,
        nit=1,
    )

    _, quality = solver_module._verify_lsq_linear_solution(
        system,
        np.asarray([1.0e12], dtype=np.float64),
        solve_scale=1.0,
        stage='test',
        scipy_result=fake,
    )

    assert not quality.verified
    assert quality.projected_gradient_inf_norm == pytest.approx(1.0)
    assert quality.kkt_tolerance < quality.projected_gradient_inf_norm


def test_refraction_lsq_kkt_verifier_handles_bound_gradient_signs() -> None:
    lower_system = _simple_lsq_system(
        np.ones((1, 1), dtype=np.float64),
        np.asarray([-1.0], dtype=np.float64),
        lower=np.zeros(1, dtype=np.float64),
        upper=np.full(1, np.inf, dtype=np.float64),
    )
    upper_system = _simple_lsq_system(
        np.ones((1, 1), dtype=np.float64),
        np.asarray([2.0], dtype=np.float64),
        lower=np.zeros(1, dtype=np.float64),
        upper=np.ones(1, dtype=np.float64),
    )
    fake = optimize.OptimizeResult(success=True, status=1, optimality=0.0, nit=1)

    _, lower_quality = solver_module._verify_lsq_linear_solution(
        lower_system,
        np.asarray([0.0], dtype=np.float64),
        solve_scale=1.0,
        stage='lower',
        scipy_result=fake,
    )
    _, lower_bad = solver_module._verify_lsq_linear_solution(
        _simple_lsq_system(
            np.ones((1, 1), dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            lower=np.zeros(1, dtype=np.float64),
            upper=np.full(1, np.inf, dtype=np.float64),
        ),
        np.asarray([0.0], dtype=np.float64),
        solve_scale=1.0,
        stage='lower_bad',
        scipy_result=fake,
    )
    _, upper_quality = solver_module._verify_lsq_linear_solution(
        upper_system,
        np.asarray([1.0], dtype=np.float64),
        solve_scale=1.0,
        stage='upper',
        scipy_result=fake,
    )
    _, upper_bad = solver_module._verify_lsq_linear_solution(
        _simple_lsq_system(
            np.ones((1, 1), dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            lower=np.zeros(1, dtype=np.float64),
            upper=np.ones(1, dtype=np.float64),
        ),
        np.asarray([1.0], dtype=np.float64),
        solve_scale=1.0,
        stage='upper_bad',
        scipy_result=fake,
    )

    assert lower_quality.verified
    assert upper_quality.verified
    assert not lower_bad.verified
    assert not upper_bad.verified


def test_refraction_lsq_accepts_noisy_inconsistent_kkt_solution() -> None:
    system = _simple_lsq_system(
        np.ones((2, 1), dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        lower=np.zeros(1, dtype=np.float64),
        upper=np.ones(1, dtype=np.float64),
    )

    result = solver_module._run_lsq_linear(system)

    np.testing.assert_allclose(result.x, [0.5], atol=1.0e-10)
    assert result.success
    assert result.refraction_solver_quality['unscaled_augmented_residual_norm'] > 0.0
    assert result.refraction_solver_quality['verified'] is True


def test_refraction_lsq_active_set_polish_improves_failed_attempt() -> None:
    system = _simple_lsq_system(
        np.eye(2, dtype=np.float64),
        np.asarray([0.25, 0.75], dtype=np.float64),
        lower=np.zeros(2, dtype=np.float64),
        upper=np.ones(2, dtype=np.float64),
    )
    fake = optimize.OptimizeResult(success=True, status=1, optimality=1.0, nit=1)
    _, _, before_objective = solver_module._lsq_residual_gradient_objective(
        system,
        np.asarray([0.4, 0.4], dtype=np.float64),
    )

    polished = solver_module._active_set_polish_lsq_solution(
        system,
        np.asarray([0.4, 0.4], dtype=np.float64),
        solve_scale=1.0,
        scipy_result=fake,
    )

    assert polished is not None
    assert polished.refraction_solver_quality['stage'] == 'active_set_polish'
    assert polished.refraction_solver_quality['verified'] is True
    assert polished.refraction_solver_quality['unscaled_objective'] <= before_objective
    np.testing.assert_allclose(polished.x, [0.25, 0.75], atol=1.0e-12)


def test_refraction_lsq_rejects_polished_result_with_worse_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _simple_lsq_system(
        np.eye(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        lower=np.zeros(1, dtype=np.float64),
        upper=np.ones(1, dtype=np.float64),
    )

    def worse_polish(
        system_arg: solver_module.RefractionStaticSolveSystem,
        parameter_vector: np.ndarray,
        *,
        solve_scale: float,
        scipy_result: optimize.OptimizeResult,
    ) -> optimize.OptimizeResult:
        del system_arg, parameter_vector, scipy_result
        quality = solver_module._LsqLinearQualityDiagnostic(
            verified=True,
            stage='active_set_polish',
            failure_reason='',
            solve_scale=float(solve_scale),
            matrix_shape=(1, 1),
            matrix_nnz=1,
            scipy_success=True,
            scipy_status=1,
            scipy_optimality=0.0,
            scipy_iterations=1,
            lsmr_tol=None,
            lsmr_maxiter=None,
            lsmr_stop_code=None,
            lsmr_iterations=None,
            lsmr_norm_r=None,
            lsmr_norm_ar=None,
            lsmr_norm_a=None,
            lsmr_condition_estimate=None,
            lsmr_norm_x=None,
            unscaled_augmented_residual_norm=1.0e-7,
            unscaled_objective=1.0e-14,
            projected_gradient_inf_norm=0.0,
            kkt_tolerance=1.0,
            projected_gradient_ratio=0.0,
            max_bound_violation=0.0,
            bound_tolerance=0.0,
        )
        result = optimize.OptimizeResult(
            x=np.zeros(1, dtype=np.float64),
            success=True,
            status=1,
            message='worse polished result',
            cost=quality.unscaled_objective,
            optimality=quality.projected_gradient_inf_norm,
            nit=1,
        )
        result.refraction_solver_quality_diagnostic = quality
        result.refraction_solver_quality = solver_module._lsq_quality_json(quality)
        return result

    monkeypatch.setattr(
        solver_module,
        '_active_set_polish_lsq_solution',
        worse_polish,
    )

    result = solver_module._run_lsq_linear(system)

    assert result.refraction_solver_quality['stage'] == 'first_attempt'
    assert result.refraction_solver_quality['unscaled_objective'] == pytest.approx(0.0)


def test_refraction_lsq_failure_remains_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _simple_lsq_system(
        np.eye(1, dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        lower=np.zeros(1, dtype=np.float64),
        upper=np.ones(1, dtype=np.float64),
    )

    def suboptimal_attempt(*args: object, **kwargs: object) -> optimize.OptimizeResult:
        return optimize.OptimizeResult(
            x=np.asarray([0.0], dtype=np.float64),
            success=True,
            status=1,
            message='forced',
            cost=0.5,
            optimality=0.0,
            nit=1,
        )

    monkeypatch.setattr(
        solver_module,
        '_run_scaled_lsq_linear_attempt',
        suboptimal_attempt,
    )
    monkeypatch.setattr(solver_module, '_active_set_polish_lsq_solution', lambda *a, **k: None)
    monkeypatch.setattr(solver_module, '_can_use_dense_lsq_retry', lambda matrix: False)

    with pytest.raises(
        RefractionStaticSolverError,
        match='failed quality verification',
    ):
        solver_module._run_lsq_linear(system)


def test_refraction_solver_rejects_global_slowness_underdetermined() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.25], dtype=np.float64),
        valid_observation_mask_sorted=np.asarray([True]),
        source_node_id_sorted=np.asarray([10], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20], dtype=np.int64),
        distance_m_sorted=np.asarray([500.0], dtype=np.float64),
        node_id=np.asarray([10, 20], dtype=np.int64),
        bedrock_velocity_mode='solve_global',
        n_traces=1,
    )

    with pytest.raises(
        RefractionStaticSolverError,
        match='solve_global.*expected_rank=2.*actual_rank=1.*gauge_nullity=1',
    ):
        solve_refraction_static_design_least_squares(
            design,
            model=_model(mode='solve_global'),
            solver_options=_solver_options(),
        )


def test_refraction_solver_global_slowness_identified_by_distance_variation() -> None:
    true_sum_t1 = 0.07
    true_velocity = 2500.0
    distance_m = np.asarray([500.0, 700.0], dtype=np.float64)
    pick_time = true_sum_t1 + distance_m / true_velocity
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 10], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20, 20], dtype=np.int64),
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20], dtype=np.int64),
        bedrock_velocity_mode='solve_global',
        n_traces=2,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='solve_global'),
        solver_options=_solver_options(),
    )

    assert result.bedrock_velocity_m_s == pytest.approx(true_velocity, abs=1.0e-5)
    assert result.system.identifiability.expected_rank == 2
    assert result.system.identifiability.estimated_rank == 2


def test_refraction_solver_rejects_duplicate_rows_missed_by_pattern_rank() -> None:
    distance_m = np.asarray([500.0, 500.0], dtype=np.float64)
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.27, 0.27], dtype=np.float64),
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 10], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20, 20], dtype=np.int64),
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20], dtype=np.int64),
        bedrock_velocity_mode='solve_global',
        n_traces=2,
    )

    with pytest.raises(RefractionStaticSolverError, match='actual_rank=1'):
        solve_refraction_static_design_least_squares(
            design,
            model=_model(mode='solve_global'),
            solver_options=_solver_options(),
        )


def test_refraction_solver_damping_does_not_identify_global_slowness() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.25], dtype=np.float64),
        valid_observation_mask_sorted=np.asarray([True]),
        source_node_id_sorted=np.asarray([10], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20], dtype=np.int64),
        distance_m_sorted=np.asarray([500.0], dtype=np.float64),
        node_id=np.asarray([10, 20], dtype=np.int64),
        bedrock_velocity_mode='solve_global',
        n_traces=1,
    )

    with pytest.raises(RefractionStaticSolverError, match='physical system is not identifiable'):
        solve_refraction_static_design_least_squares(
            design,
            model=_model(mode='solve_global'),
            solver_options=_solver_options(half_intercept_damping_lambda=100.0),
        )


def test_refraction_solver_robust_rejection_refuses_slowness_rank_loss() -> None:
    true_sum_t1 = 0.07
    distance_m = np.asarray([500.0, 700.0], dtype=np.float64)
    pick_time = true_sum_t1 + distance_m / 2500.0
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 10], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20, 20], dtype=np.int64),
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20], dtype=np.int64),
        bedrock_velocity_mode='solve_global',
        n_traces=2,
    )
    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='solve_global'),
        solver_options=_solver_options(),
    )
    from seis_statics.refraction.solver import _robust_row_mask_is_safe

    assert not _robust_row_mask_is_safe(
        design=design,
        system=system,
        initial_row_mask=np.ones(2, dtype=bool),
        row_used_mask=np.asarray([True, False], dtype=bool),
        min_used_fraction=0.5,
        min_used_observations=1,
    )


def test_refraction_solver_identifiability_rank_is_column_scale_stable() -> None:
    base_array = np.asarray(
        [
            [1.0, 0.0, 1.0e-6],
            [0.0, 1.0, 2.0e-6],
            [1.0, 1.0, 4.0e-6],
        ],
        dtype=np.float64,
    )
    rescaled_array = base_array.copy()
    rescaled_array[:, 2] *= 1.0e12

    base_rank = _column_scaled_numerical_rank(
        sparse.csr_matrix(base_array),
        expected_rank=3,
        expected_nullity=0,
        rtol=1.0e-10,
    )
    rescaled_rank = _column_scaled_numerical_rank(
        sparse.csr_matrix(rescaled_array),
        expected_rank=3,
        expected_nullity=0,
        rtol=1.0e-10,
    )

    assert base_rank.estimated_rank == 3
    assert rescaled_rank.estimated_rank == 3


def test_refraction_solver_large_sparse_skinny_rank_uses_sparse_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = sparse.eye(512, 2000, format='csr', dtype=np.float64)
    seen_shape: list[tuple[int, int]] = []

    def fail_dense(*args: object, **kwargs: object) -> object:
        raise AssertionError('large sparse identifiability matrix was densified')

    def fake_sparse(
        scaled_matrix: sparse.csr_matrix,
        *,
        expected_rank: int,
        expected_nullity: int,
        rtol: float,
    ) -> solver_module._NumericalRankDiagnostic:
        seen_shape.append(tuple(map(int, scaled_matrix.shape)))
        return solver_module._NumericalRankDiagnostic(
            method='sparse_svds',
            n_rows=int(scaled_matrix.shape[0]),
            n_columns=int(scaled_matrix.shape[1]),
            expected_rank=int(expected_rank),
            estimated_rank=int(expected_rank),
            expected_nullity=int(expected_nullity),
            gauge_nullity=int(expected_nullity),
            threshold=float(rtol),
            critical_singular_value=1.0,
            largest_singular_value=1.0,
            rtol=float(rtol),
        )

    monkeypatch.setattr(solver_module, '_dense_column_scaled_numerical_rank', fail_dense)
    monkeypatch.setattr(solver_module, '_sparse_column_scaled_numerical_rank', fake_sparse)

    diagnostic = _column_scaled_numerical_rank(
        matrix,
        expected_rank=512,
        expected_nullity=1488,
        rtol=1.0e-10,
    )

    assert diagnostic.method == 'sparse_svds'
    assert seen_shape == [(512, 2000)]


def test_refraction_solver_sparse_rank_one_uses_largest_singular_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_svds(*args: object, **kwargs: object) -> object:
        raise AssertionError('rank-one sparse certification should not request SM svds')

    monkeypatch.setattr(solver_module.sparse_linalg, 'svds', fail_svds)

    diagnostic = solver_module._sparse_column_scaled_numerical_rank(
        sparse.csr_matrix(([1.0], ([0], [0])), shape=(3, 3), dtype=np.float64),
        expected_rank=1,
        expected_nullity=2,
        rtol=1.0e-10,
    )

    assert diagnostic.estimated_rank == 1
    assert diagnostic.critical_singular_value == pytest.approx(1.0)
    assert diagnostic.certification_status == 'certified'
    assert diagnostic.requested_smallest_count == 0
    assert diagnostic.method == 'sparse_normal_eigsh'
    assert diagnostic.sparse_solver_name == 'eigsh_normal'


def test_refraction_solver_sparse_rank_zero_reports_largest_solver_only() -> None:
    diagnostic = solver_module._sparse_column_scaled_numerical_rank(
        sparse.eye(3, format='csr', dtype=np.float64),
        expected_rank=0,
        expected_nullity=3,
        rtol=1.0e-10,
    )

    assert diagnostic.certification_status == 'rank_deficient'
    assert diagnostic.method == 'sparse_normal_eigsh'
    assert diagnostic.sparse_solver_name == 'eigsh_normal'
    assert diagnostic.requested_smallest_count == 0


def _sparse_banded_rank_control_matrix(n: int) -> sparse.csr_matrix:
    diagonal = np.ones(int(n), dtype=np.float64)
    subdiagonal = np.linspace(0.05, 0.15, int(n) - 1, dtype=np.float64)
    return sparse.diags(
        diagonals=(diagonal, subdiagonal),
        offsets=(0, -1),
        format='csr',
    )


def _sparse_banded_rank_deficient_matrix(
    n: int,
    *,
    nullity: int,
) -> sparse.csr_matrix:
    diagonal = np.ones(int(n), dtype=np.float64)
    diagonal[-int(nullity) :] = 0.0
    subdiagonal = np.linspace(0.05, 0.15, int(n) - 1, dtype=np.float64)
    if nullity > 1:
        subdiagonal[-(int(nullity) - 1) :] = 0.0
    return sparse.diags(
        diagonals=(diagonal, subdiagonal),
        offsets=(0, -1),
        format='csr',
    )


def test_refraction_solver_large_sparse_exact_zero_is_not_full_rank() -> None:
    n = 1100
    diagnostic = solver_module._sparse_column_scaled_numerical_rank(
        sparse.diags(np.r_[np.ones(n - 1), 0.0], format='csr'),
        expected_rank=n,
        expected_nullity=0,
        rtol=1.0e-10,
    )

    assert diagnostic.method == 'sparse_structural_rank'
    assert diagnostic.structural_rank == n - 1
    assert diagnostic.estimated_rank == n - 1
    assert diagnostic.certification_status == 'rank_deficient'
    assert diagnostic.failure_reason == 'structural rank is below expected rank'
    assert diagnostic.requested_smallest_count == 0


def test_refraction_solver_large_sparse_full_rank_control_is_certified() -> None:
    n = 1100
    small_control = _sparse_banded_rank_control_matrix(32)
    assert np.linalg.matrix_rank(small_control.toarray()) == 32

    diagnostic = solver_module._sparse_column_scaled_numerical_rank(
        _sparse_banded_rank_control_matrix(n),
        expected_rank=n,
        expected_nullity=0,
        rtol=1.0e-10,
    )

    assert diagnostic.estimated_rank == n
    assert diagnostic.certification_status == 'certified'
    assert diagnostic.structural_rank == n
    assert diagnostic.critical_singular_value > diagnostic.threshold
    assert diagnostic.max_singular_triplet_residual < 1.0e-8
    assert diagnostic.selected_candidate in {'propack_svds', 'eigsh_normal'}


def test_refraction_solver_large_sparse_allowed_nullity_boundary() -> None:
    n = 1100
    one_null_fixture = _sparse_banded_rank_deficient_matrix(n, nullity=1)
    two_null_fixture = _sparse_banded_rank_deficient_matrix(n, nullity=2)
    assert np.linalg.matrix_rank(
        _sparse_banded_rank_deficient_matrix(32, nullity=1).toarray()
    ) == 31
    assert np.linalg.matrix_rank(
        _sparse_banded_rank_deficient_matrix(32, nullity=2).toarray()
    ) == 30

    one_null = solver_module._sparse_column_scaled_numerical_rank(
        one_null_fixture,
        expected_rank=n - 1,
        expected_nullity=1,
        rtol=1.0e-10,
    )
    two_null = solver_module._sparse_column_scaled_numerical_rank(
        two_null_fixture,
        expected_rank=n - 1,
        expected_nullity=1,
        rtol=1.0e-10,
    )

    assert one_null.estimated_rank == n - 1
    assert one_null.certification_status == 'certified'
    assert one_null.requested_smallest_count == 2
    assert two_null.estimated_rank == n - 2
    assert two_null.certification_status == 'rank_deficient'
    assert two_null.critical_singular_value <= two_null.threshold


def test_refraction_solver_sparse_near_threshold_policy() -> None:
    n = 1100
    above = solver_module._sparse_column_scaled_numerical_rank(
        sparse.diags(np.r_[np.ones(n - 1), 1.0e-6], format='csr'),
        expected_rank=n,
        expected_nullity=0,
        rtol=1.0e-10,
    )
    below = solver_module._sparse_column_scaled_numerical_rank(
        sparse.diags(np.r_[np.ones(n - 1), 1.0e-12], format='csr'),
        expected_rank=n,
        expected_nullity=0,
        rtol=1.0e-10,
    )

    assert above.certification_status == 'certified'
    assert above.estimated_rank == n
    assert below.certification_status == 'rank_deficient'
    assert below.estimated_rank == n - 1
    with pytest.raises(RefractionStaticSolverError, match='too close'):
        solver_module._sparse_column_scaled_numerical_rank(
            sparse.diags(np.r_[np.ones(n - 1), 1.0e-10], format='csr'),
            expected_rank=n,
            expected_nullity=0,
            rtol=1.0e-10,
        )


def test_refraction_solver_sparse_single_candidate_nonconvergence_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_svds(*args: object, **kwargs: object) -> object:
        raise RuntimeError('no convergence')

    monkeypatch.setattr(solver_module.sparse_linalg, 'svds', fail_svds)

    diagnostic = solver_module._sparse_column_scaled_numerical_rank(
        _sparse_banded_rank_control_matrix(1100),
        expected_rank=1100,
        expected_nullity=0,
        rtol=1.0e-10,
    )

    assert diagnostic.certification_status == 'certified'
    assert diagnostic.selected_candidate == 'eigsh_normal'
    assert diagnostic.rejected_candidate_names == ('propack_svds',)
    assert 'did not converge' in diagnostic.rejected_candidate_reasons[0]


def test_refraction_solver_sparse_all_candidates_invalid_does_not_certify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_normal_triplets = solver_module._sparse_normal_singular_triplets

    def fail_svds(
        scaled_matrix: sparse.csr_matrix,
        *,
        k: int,
        name: str,
    ) -> solver_module._SparseSingularTripletDiagnostic:
        del scaled_matrix, k, name
        raise RefractionStaticSolverError('propack failed')

    def fail_normal(
        scaled_matrix: sparse.csr_matrix,
        *,
        k: int,
        which: object,
        name: str,
    ) -> solver_module._SparseSingularTripletDiagnostic:
        if which == 'LA':
            return original_normal_triplets(
                scaled_matrix,
                k=k,
                which='LA',
                name=name,
            )
        del scaled_matrix, k, name
        raise RefractionStaticSolverError('normal failed')

    monkeypatch.setattr(solver_module, '_sparse_svds_singular_triplets', fail_svds)
    monkeypatch.setattr(solver_module, '_sparse_normal_singular_triplets', fail_normal)

    with pytest.raises(RefractionStaticSolverError, match='certification unavailable'):
        solver_module._sparse_column_scaled_numerical_rank(
            _sparse_banded_rank_control_matrix(1100),
            expected_rank=1100,
            expected_nullity=0,
            rtol=1.0e-10,
        )


def test_refraction_solver_sparse_bad_candidate_residual_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_triplets(
        scaled_matrix: sparse.csr_matrix,
        *,
        k: int,
        name: str,
    ) -> solver_module._SparseSingularTripletDiagnostic:
        return solver_module._SparseSingularTripletDiagnostic(
            singular_values=np.ones(int(k), dtype=np.float64),
            max_residual=1.0,
        )

    monkeypatch.setattr(solver_module, '_sparse_svds_singular_triplets', bad_triplets)

    diagnostic = solver_module._sparse_column_scaled_numerical_rank(
        _sparse_banded_rank_control_matrix(1100),
        expected_rank=1100,
        expected_nullity=0,
        rtol=1.0e-10,
    )

    assert diagnostic.certification_status == 'certified'
    assert diagnostic.selected_candidate == 'eigsh_normal'
    assert diagnostic.rejected_candidate_names == ('propack_svds',)
    assert diagnostic.rejected_candidate_reasons == (
        'singular triplet residual is too large',
    )


def test_refraction_solver_sparse_bad_allowed_null_triplet_residual_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boundary_triplets(
        scaled_matrix: sparse.csr_matrix,
        *,
        k: int,
        name: str,
    ) -> solver_module._SparseSingularTripletDiagnostic:
        del scaled_matrix, name
        residuals = np.zeros(int(k), dtype=np.float64)
        residuals[0] = 1.0
        return solver_module._SparseSingularTripletDiagnostic(
            singular_values=np.ones(int(k), dtype=np.float64),
            max_residual=float(np.max(residuals)),
            residuals=residuals,
        )

    def normal_triplets(
        scaled_matrix: sparse.csr_matrix,
        *,
        k: int,
        which: object,
        name: str,
    ) -> solver_module._SparseSingularTripletDiagnostic:
        del scaled_matrix, which, name
        return solver_module._SparseSingularTripletDiagnostic(
            singular_values=np.ones(int(k), dtype=np.float64),
            max_residual=0.0,
            residuals=np.zeros(int(k), dtype=np.float64),
        )

    monkeypatch.setattr(
        solver_module,
        '_sparse_svds_singular_triplets',
        boundary_triplets,
    )
    monkeypatch.setattr(
        solver_module,
        '_sparse_normal_singular_triplets',
        normal_triplets,
    )

    diagnostic = solver_module._sparse_column_scaled_numerical_rank(
        _sparse_banded_rank_control_matrix(4),
        expected_rank=3,
        expected_nullity=1,
        rtol=1.0e-10,
    )

    assert diagnostic.certification_status == 'certified'
    assert diagnostic.selected_candidate == 'eigsh_normal'
    assert diagnostic.rejected_candidate_names == ('propack_svds',)


def test_refraction_solver_sparse_bad_corroborating_boundary_residual_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def good_boundary_triplets(
        scaled_matrix: sparse.csr_matrix,
        *,
        k: int,
        name: str,
    ) -> solver_module._SparseSingularTripletDiagnostic:
        return solver_module._SparseSingularTripletDiagnostic(
            singular_values=np.ones(int(k), dtype=np.float64),
            max_residual=0.0,
            residuals=np.zeros(int(k), dtype=np.float64),
        )

    def normal_triplets(
        scaled_matrix: sparse.csr_matrix,
        *,
        k: int,
        which: object,
        name: str,
    ) -> solver_module._SparseSingularTripletDiagnostic:
        bad_residual = 1.0 if which == 'SA' else 0.0
        singular_value = 1.0
        return solver_module._SparseSingularTripletDiagnostic(
            singular_values=np.full(int(k), singular_value, dtype=np.float64),
            max_residual=bad_residual,
            residuals=np.full(int(k), bad_residual, dtype=np.float64),
        )

    monkeypatch.setattr(
        solver_module,
        '_sparse_svds_singular_triplets',
        good_boundary_triplets,
    )
    monkeypatch.setattr(
        solver_module,
        '_sparse_normal_singular_triplets',
        normal_triplets,
    )

    diagnostic = solver_module._sparse_column_scaled_numerical_rank(
        _sparse_banded_rank_control_matrix(4),
        expected_rank=4,
        expected_nullity=0,
        rtol=1.0e-10,
    )

    assert diagnostic.certification_status == 'certified'
    assert diagnostic.selected_candidate == 'propack_svds'
    assert diagnostic.rejected_candidate_names == ('eigsh_normal',)


def test_refraction_lsq_scale_accounts_for_large_matrix_with_nonzero_rhs() -> None:
    matrix = sparse.csr_matrix(np.asarray([[1.0e100]], dtype=np.float64))
    rhs = np.asarray([1.0], dtype=np.float64)

    scale = solver_module._common_lsq_solve_scale(matrix, rhs)

    assert scale < 1.0
    assert abs(scale * matrix.data[0]) <= solver_module._LSQ_SOLVE_SCALED_ABS_MAX


def test_refraction_lsq_kkt_tolerance_does_not_overflow_to_success() -> None:
    system = _simple_lsq_system(
        np.asarray([[1.0e155]], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        lower=-np.ones(1, dtype=np.float64),
        upper=np.ones(1, dtype=np.float64),
    )
    fake = optimize.OptimizeResult(
        success=True,
        status=1,
        optimality=0.0,
        nit=1,
    )
    candidate = np.asarray([1.0e-155 + 1.0e-163], dtype=np.float64)

    _, quality = solver_module._verify_lsq_linear_solution(
        system,
        candidate,
        solve_scale=1.0,
        stage='test',
        scipy_result=fake,
    )

    assert not quality.verified
    assert np.isfinite(quality.kkt_tolerance)
    assert quality.projected_gradient_inf_norm > quality.kkt_tolerance


def test_refraction_solver_large_sparse_global_slowness_duplicate_path_regression() -> None:
    n_nodes = 1100
    source_node_id = np.r_[np.arange(n_nodes - 1), 0].astype(np.int64)
    receiver_node_id = np.r_[np.arange(1, n_nodes), 1].astype(np.int64)
    distance_m = np.full(n_nodes, 500.0, dtype=np.float64)
    pick_time = np.full(n_nodes, 0.25, dtype=np.float64)

    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(n_nodes, dtype=bool),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.arange(n_nodes, dtype=np.int64),
        bedrock_velocity_mode='solve_global',
        n_traces=n_nodes,
    )

    with pytest.raises(
        RefractionStaticSolverError,
        match='certification unavailable',
    ):
        solve_refraction_static_design_least_squares(
            design,
            model=_model(mode='solve_global'),
            solver_options=_solver_options(),
        )


def test_refraction_solver_large_sparse_global_slowness_distance_variation_is_certified_without_densifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_nodes = 1100
    source_node_id = np.r_[np.zeros(n_nodes - 1, dtype=np.int64), 0]
    receiver_node_id = np.r_[np.arange(1, n_nodes), 1].astype(np.int64)
    distance_m = np.full(n_nodes, 500.0, dtype=np.float64)
    distance_m[-1] = 700.0
    true_node_half_intercept_s = np.full(n_nodes, 0.02, dtype=np.float64)
    true_velocity = 2500.0
    pick_time = (
        true_node_half_intercept_s[source_node_id]
        + true_node_half_intercept_s[receiver_node_id]
        + distance_m / true_velocity
    )

    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(n_nodes, dtype=bool),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.arange(n_nodes, dtype=np.int64),
        bedrock_velocity_mode='solve_global',
        n_traces=n_nodes,
    )

    def fail_toarray(self: sparse.csr_matrix, *args: object, **kwargs: object) -> object:
        raise AssertionError('large sparse solve_global system was densified')

    monkeypatch.setattr(sparse.csr_matrix, 'toarray', fail_toarray)

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='solve_global'),
        solver_options=_solver_options(),
    )

    assert result.bedrock_velocity_m_s == pytest.approx(true_velocity, abs=1.0e-5)
    assert result.system.identifiability.method == 'sparse_normal_eigsh'
    assert result.system.identifiability.expected_rank == n_nodes
    assert result.system.identifiability.estimated_rank == n_nodes
    assert result.system.identifiability.certification_status == 'certified'
    assert result.system.identifiability.critical_singular_value > (
        result.system.identifiability.threshold
    )


def test_refraction_solver_system_gauge_rows_are_conceptual_not_matrix_rows() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.30, 0.40], dtype=np.float64),
        valid_observation_mask_sorted=np.asarray([True, True]),
        source_node_id_sorted=np.asarray([10, 20], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([11, 21], dtype=np.int64),
        distance_m_sorted=np.asarray([500.0, 500.0], dtype=np.float64),
        node_id=np.asarray([10, 11, 20, 21], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        n_traces=2,
    )

    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='fixed_global', fixed_velocity=2500.0),
        solver_options=_solver_options(),
    )

    assert system.n_node_components == 2
    assert system.n_bipartite_node_components == 2
    assert np.count_nonzero(system.gauge_required_by_component) == 2
    assert system.n_gauge_rows == 0
    assert system.gauge_resolution == 'postsolve_minimum_norm'
    assert system.n_augmented_rows == system.n_observation_rows


def test_refraction_solver_zero_damping_canonicalizes_bound_clipped_exact_fit() -> None:
    fixed_velocity = 2500.0
    distance_m = np.asarray([500.0, 500.0], dtype=np.float64)
    pick_time = np.asarray([0.02, 0.10], dtype=np.float64) + distance_m / fixed_velocity
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 30], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20, 20], dtype=np.int64),
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=2,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(min_picks_per_node=1),
    )

    np.testing.assert_allclose(result.row_residual_s, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.0, 0.02, 0.08],
        atol=1.0e-9,
    )
    assert result.system.n_gauge_rows == 0
    assert result.system.gauge_resolution == 'postsolve_minimum_norm'


def test_refraction_solver_postsolve_canonicalization_preserves_predictions() -> None:
    fixed_velocity = 2500.0
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.22, 0.30], dtype=np.float64),
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 30], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20, 20], dtype=np.int64),
        distance_m_sorted=np.asarray([500.0, 500.0], dtype=np.float64),
        node_id=np.asarray([10, 20, 30], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=2,
    )
    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(min_picks_per_node=1),
    )
    before = np.asarray([0.02, 0.0, 0.10], dtype=np.float64)

    after = solver_module._canonicalize_refraction_parameter_vector(
        before,
        system=system,
        design=design,
    )

    np.testing.assert_allclose(design.matrix @ after, design.matrix @ before)
    np.testing.assert_allclose(after, [0.0, 0.02, 0.08], atol=1.0e-12)
    assert np.all(after >= system.lower_bounds)
    assert np.all(after <= system.upper_bounds)


def test_refraction_solver_positive_damping_uses_no_postsolve_shift() -> None:
    fixed_velocity = 2500.0
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.22, 0.30], dtype=np.float64),
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 30], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20, 20], dtype=np.int64),
        distance_m_sorted=np.asarray([500.0, 500.0], dtype=np.float64),
        node_id=np.asarray([10, 20, 30], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=2,
    )
    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(
            half_intercept_damping_lambda=4.0,
            min_picks_per_node=1,
        ),
    )
    before = np.asarray([0.02, 0.0, 0.10], dtype=np.float64)

    after = solver_module._canonicalize_refraction_parameter_vector(
        before,
        system=system,
        design=design,
    )

    assert system.n_gauge_rows == 0
    assert system.gauge_resolution == 'node_damping'
    assert system.n_augmented_rows == system.n_observation_rows + system.n_damping_rows
    np.testing.assert_allclose(after, before)


def test_refraction_solver_canonicalizes_disconnected_components_independently() -> None:
    fixed_velocity = 2500.0
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.30, 0.28], dtype=np.float64),
        valid_observation_mask_sorted=np.ones(2, dtype=bool),
        source_node_id_sorted=np.asarray([10, 30], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([20, 40], dtype=np.int64),
        distance_m_sorted=np.asarray([500.0, 500.0], dtype=np.float64),
        node_id=np.asarray([10, 20, 30, 40], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=2,
    )
    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(min_picks_per_node=1),
    )
    before = np.asarray([0.10, 0.0, 0.08, 0.0], dtype=np.float64)

    after = solver_module._canonicalize_refraction_parameter_vector(
        before,
        system=system,
        design=design,
    )

    np.testing.assert_allclose(design.matrix @ after, design.matrix @ before)
    np.testing.assert_allclose(after, [0.05, 0.05, 0.04, 0.04], atol=1.0e-12)
    assert np.count_nonzero(system.gauge_required_by_component) == 2


def test_refraction_solver_canonicalization_checks_final_observation_rows() -> None:
    fixed_velocity = 2500.0
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.30, 0.28, 0.32], dtype=np.float64),
        valid_observation_mask_sorted=np.ones(3, dtype=bool),
        source_node_id_sorted=np.asarray([10, 20, 20], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([30, 40, 30], dtype=np.int64),
        distance_m_sorted=np.asarray([500.0, 500.0, 500.0], dtype=np.float64),
        node_id=np.asarray([10, 20, 30, 40], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=3,
    )
    final_row_mask = np.asarray([True, True, False], dtype=bool)
    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(min_picks_per_node=1),
        row_used_mask=final_row_mask,
    )
    before = np.asarray([0.10, 0.0, 0.0, 0.08], dtype=np.float64)

    after = solver_module._canonicalize_refraction_parameter_vector(
        before,
        system=system,
        design=design,
    )

    np.testing.assert_allclose(system.observation_matrix @ after, system.observation_matrix @ before)
    np.testing.assert_allclose(after, [0.05, 0.04, 0.05, 0.04], atol=1.0e-12)
    assert not np.allclose(design.matrix @ after, design.matrix @ before)
    assert np.count_nonzero(system.gauge_required_by_component) == 2


def test_refraction_solver_non_bipartite_component_is_not_shifted() -> None:
    fixed_velocity = 2500.0
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.24], dtype=np.float64),
        valid_observation_mask_sorted=np.ones(1, dtype=bool),
        source_node_id_sorted=np.asarray([10], dtype=np.int64),
        receiver_node_id_sorted=np.asarray([10], dtype=np.int64),
        distance_m_sorted=np.asarray([500.0], dtype=np.float64),
        node_id=np.asarray([10], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=1,
    )
    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(min_picks_per_node=1),
    )
    before = np.asarray([0.02], dtype=np.float64)

    after = solver_module._canonicalize_refraction_parameter_vector(
        before,
        system=system,
        design=design,
    )

    assert not np.any(system.gauge_required_by_component)
    assert system.gauge_resolution == 'not_required'
    np.testing.assert_allclose(after, before)


def test_refraction_solver_global_damping_regularizes_only_half_intercepts() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_global',
        n_traces=6,
    )
    damping_lambda = 4.0
    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='solve_global'),
        solver_options=_solver_options(half_intercept_damping_lambda=damping_lambda),
    )

    damping_slice = slice(
        system.n_observation_rows,
        system.n_observation_rows + system.n_damping_rows,
    )
    damping_block = system.augmented_matrix[damping_slice].toarray()
    parameter_vector = np.asarray(
        [0.02, -0.01, 0.04, 0.03, 1.0 / 2200.0],
        dtype=np.float64,
    )
    damping_residual = (
        system.augmented_matrix[damping_slice] @ parameter_vector
        - system.augmented_rhs_s[damping_slice]
    )

    assert system.n_damping_rows == design.n_active_nodes
    assert system.regularization_row_count == design.n_active_nodes
    assert system.regularized_parameter_group == 'node_half_intercept_time_s'
    assert system.half_intercept_damping_lambda == damping_lambda
    np.testing.assert_allclose(
        damping_block[:, : design.n_active_nodes],
        np.eye(design.n_active_nodes, dtype=np.float64) * 2.0,
    )
    np.testing.assert_allclose(damping_block[:, design.n_active_nodes :], 0.0)
    np.testing.assert_allclose(system.augmented_rhs_s[damping_slice], 0.0)
    np.testing.assert_allclose(
        float(damping_residual @ damping_residual),
        damping_lambda * float(np.sum(parameter_vector[: design.n_active_nodes] ** 2)),
    )


def test_refraction_solver_zero_lambda_adds_no_damping_rows() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_global',
        n_traces=6,
    )

    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='solve_global'),
        solver_options=_solver_options(half_intercept_damping_lambda=0.0),
    )

    assert system.n_damping_rows == 0
    assert system.regularization_row_count == 0


def test_refraction_solver_fixed_global_damping_scope_is_half_intercepts() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        n_traces=6,
    )

    system = build_refraction_static_solver_system(
        design,
        model=_model(mode='fixed_global', fixed_velocity=2500.0),
        solver_options=_solver_options(half_intercept_damping_lambda=9.0),
    )

    damping_slice = slice(
        system.n_observation_rows,
        system.n_observation_rows + system.n_damping_rows,
    )
    np.testing.assert_allclose(
        system.augmented_matrix[damping_slice].toarray(),
        np.eye(design.n_active_nodes, dtype=np.float64) * 3.0,
    )


def test_refraction_solver_fixed_global_uses_fixed_velocity_distance_term() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=2500.0),
        solver_options=_solver_options(),
    )

    assert result.bedrock_velocity_mode == 'fixed_global'
    assert result.bedrock_velocity_status == 'fixed'
    assert result.bedrock_velocity_m_s == 2500.0
    assert result.bedrock_slowness_s_per_m == 1.0 / 2500.0
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.03, 0.05, 0.035, 0.045],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(result.row_modeled_pick_time_s, pick_time[:5])
    np.testing.assert_allclose(result.row_residual_s, 0.0, atol=1.0e-10)


def test_refraction_solver_sparse_trace_index_infers_full_output_length() -> None:
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=np.asarray([0.26, 0.30]),
        valid_observation_mask_sorted=np.asarray([True, True]),
        source_node_id_sorted=np.asarray([10, 10]),
        receiver_node_id_sorted=np.asarray([10, 10]),
        distance_m_sorted=np.asarray([500.0, 600.0]),
        node_id=np.asarray([10]),
        sorted_trace_index=np.asarray([41, 99]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=2500.0),
        solver_options=_solver_options(),
    )

    assert result.modeled_pick_time_s_sorted.shape == (100,)
    assert result.used_observation_mask_sorted.shape == (100,)
    np.testing.assert_array_equal(np.flatnonzero(result.used_observation_mask_sorted), [41, 99])
    np.testing.assert_allclose(result.modeled_pick_time_s_sorted[[41, 99]], [0.26, 0.30])
    np.testing.assert_allclose(result.node_half_intercept_time_s, [0.03], atol=1.0e-10)


def test_refraction_solver_marks_global_velocity_bound_status() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_global',
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(
            mode='solve_global',
            min_velocity=1200.0,
            max_velocity=2400.0,
            initial_velocity=2000.0,
        ),
        solver_options=_solver_options(),
    )

    assert result.bedrock_velocity_m_s == pytest.approx(2400.0)
    assert result.bedrock_velocity_status == 'clipped_upper'


def test_refraction_solver_robust_global_rejects_outlier_and_recovers_solution() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    valid_mask = np.ones(valid_mask.shape, dtype=bool)
    pick_time = pick_time.copy()
    pick_time[5] += 0.12
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_global',
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='solve_global'),
        solver_options=_solver_options(
            robust=RefractionStaticRobustOptions(
                enabled=True,
                method='mad',
                threshold=2.0,
                scale_floor_ms=0.0,
                max_iterations=5,
                min_used_fraction=0.5,
                min_used_observations=1,
            )
        ),
    )

    assert result.robust_enabled is True
    assert result.robust_stop_reason == 'converged'
    assert result.n_rejected_observations == 1
    np.testing.assert_array_equal(
        result.rejected_observation_mask_sorted,
        [False, False, False, False, False, True],
    )
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, True, True, True, True, False],
    )
    np.testing.assert_array_equal(
        result.rejected_iteration_sorted,
        [-1, -1, -1, -1, -1, 0],
    )
    assert result.system.n_observation_rows == 5
    assert result.system.n_augmented_rows == (
        result.system.n_observation_rows
        + result.system.n_smoothing_rows
        + result.system.n_damping_rows
        + result.system.n_gauge_rows
    )
    observation_block = result.system.augmented_matrix[
        : result.system.n_observation_rows
    ].tocsr()
    np.testing.assert_allclose(observation_block.toarray(), design.matrix[:5].toarray())
    np.testing.assert_allclose(
        result.system.augmented_rhs_s[: result.system.n_observation_rows],
        design.rhs_s[:5],
    )
    assert np.all(np.diff(observation_block.indptr) > 0)
    assert len(result.robust_iteration_summaries) == 2
    assert result.robust_iteration_summaries[0].n_rejected_this_iteration == 1
    assert result.qc['robust_iteration_count'] == 2
    assert result.qc['n_final_used_observations'] == 5
    assert result.qc['n_observation_rows'] == 5
    assert result.qc['n_final_observation_rows'] == 5
    np.testing.assert_array_equal(result.node_observation_count, [3, 2, 3, 2])
    assert result.qc['design_matrix']['node_observation_count'] == [3, 2, 3, 2]
    assert result.bedrock_velocity_m_s == pytest.approx(2500.0, abs=1.0e-6)
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.03, 0.05, 0.035, 0.045],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(result.row_residual_s[:5], 0.0, atol=1.0e-10)


def test_refraction_solver_robust_fixed_global_recovers_solution() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    valid_mask = np.ones(valid_mask.shape, dtype=bool)
    pick_time = pick_time.copy()
    pick_time[5] += 0.12
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=2500.0,
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=2500.0),
        solver_options=_solver_options(
            robust=RefractionStaticRobustOptions(
                enabled=True,
                method='mad',
                threshold=2.0,
                scale_floor_ms=0.0,
                max_iterations=5,
                min_used_fraction=0.5,
                min_used_observations=1,
            )
        ),
    )

    assert result.robust_stop_reason == 'converged'
    assert result.n_rejected_observations >= 1
    assert result.rejected_observation_mask_sorted[5]
    np.testing.assert_allclose(
        result.node_half_intercept_time_s,
        [0.03, 0.05, 0.035, 0.045],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        result.row_residual_s[result.used_observation_mask_sorted],
        0.0,
        atol=1.0e-10,
    )


def test_refraction_solver_robust_safe_rejection_preserves_used_floor() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )
    valid_mask = np.ones(valid_mask.shape, dtype=bool)
    pick_time = pick_time.copy()
    pick_time[5] += 0.12
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_global',
        n_traces=6,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='solve_global'),
        solver_options=_solver_options(
            robust=RefractionStaticRobustOptions(
                enabled=True,
                method='mad',
                threshold=2.0,
                scale_floor_ms=0.0,
                max_iterations=5,
                min_used_fraction=1.0,
                min_used_observations=1,
            )
        ),
    )

    assert result.robust_stop_reason == 'safe_rejection'
    assert result.n_rejected_observations == 0
    np.testing.assert_array_equal(result.used_observation_mask_sorted, valid_mask)


def test_refraction_solver_robust_safe_rejection_deduplicates_same_node_rows() -> None:
    fixed_velocity = 2500.0
    source_node_id = np.asarray([10, 10], dtype=np.int64)
    receiver_node_id = np.asarray([10, 10], dtype=np.int64)
    distance_m = np.asarray([500.0, 600.0], dtype=np.float64)
    valid_mask = np.ones(2, dtype=bool)
    pick_time = 2.0 * 0.03 + distance_m / fixed_velocity
    pick_time[1] += 0.1
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=2,
        n_traces=2,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(
            min_picks_per_node=2,
            robust=RefractionStaticRobustOptions(
                enabled=True,
                method='mad',
                threshold=0.5,
                scale_floor_ms=0.0,
                max_iterations=5,
                min_used_fraction=0.5,
                min_used_observations=1,
            ),
        ),
    )

    assert result.robust_stop_reason == 'safe_rejection'
    assert result.n_rejected_observations == 0
    np.testing.assert_array_equal(result.used_observation_mask_sorted, valid_mask)
    np.testing.assert_array_equal(result.node_observation_count, [2])


def test_refraction_solver_robust_rejection_allows_identifiable_graph_split() -> None:
    fixed_velocity = 2500.0
    source_node_id = np.asarray([10, 10, 20, 20, 20], dtype=np.int64)
    receiver_node_id = np.asarray([30, 30, 40, 40, 30], dtype=np.int64)
    distance_m = np.asarray([500.0, 600.0, 500.0, 600.0, 700.0])
    valid_mask = np.ones(5, dtype=bool)
    true_t1_by_node = {
        10: 0.02,
        20: 0.04,
        30: 0.03,
        40: 0.05,
    }
    pick_time = np.asarray(
        [
            true_t1_by_node[int(src)]
            + true_t1_by_node[int(rec)]
            + dist / fixed_velocity
            for src, rec, dist in zip(
                source_node_id,
                receiver_node_id,
                distance_m,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    pick_time[4] += 0.12
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=5,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(
            min_picks_per_node=1,
            robust=RefractionStaticRobustOptions(
                enabled=True,
                method='mad',
                threshold=1.0,
                scale_floor_ms=0.0,
                max_iterations=5,
                min_used_fraction=0.5,
                min_used_observations=1,
            ),
        ),
    )

    assert result.robust_stop_reason == 'zero_scale'
    assert result.n_rejected_observations == 2
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, True, False, True, False],
    )
    assert result.robust_iteration_summaries[0].n_rejected_this_iteration == 1
    assert result.robust_iteration_summaries[1].n_rejected_this_iteration == 1
    assert result.system.n_observation_rows == 3
    assert result.system.n_node_components == 2
    assert result.system.n_gauge_rows == 0
    assert np.count_nonzero(result.system.gauge_required_by_component) == 2
    assert result.system.gauge_resolution == 'postsolve_minimum_norm'
    assert result.qc['n_initial_node_components'] == 1
    assert result.qc['n_final_node_components'] == 2
    assert result.qc['n_initial_gauge_rows'] == 0
    assert result.qc['n_final_gauge_rows'] == 0
    assert result.qc['n_initial_gauge_required_node_components'] == 1
    assert result.qc['n_final_gauge_required_node_components'] == 2


def test_refraction_solver_robust_rejection_refuses_bridge_losing_node_coverage() -> None:
    fixed_velocity = 2500.0
    source_node_id = np.asarray([10, 10, 20], dtype=np.int64)
    receiver_node_id = np.asarray([30, 30, 30], dtype=np.int64)
    distance_m = np.asarray([500.0, 600.0, 700.0], dtype=np.float64)
    true_t1_by_node = {
        10: 0.02,
        20: 0.04,
        30: 0.03,
    }
    pick_time = np.asarray(
        [
            true_t1_by_node[int(src)]
            + true_t1_by_node[int(rec)]
            + dist / fixed_velocity
            for src, rec, dist in zip(
                source_node_id,
                receiver_node_id,
                distance_m,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    pick_time[2] += 1.0
    design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=np.ones(3, dtype=bool),
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30], dtype=np.int64),
        bedrock_velocity_mode='fixed_global',
        fixed_bedrock_velocity_m_s=fixed_velocity,
        min_observations_per_node=1,
        n_traces=3,
    )

    result = solve_refraction_static_design_least_squares(
        design,
        model=_model(mode='fixed_global', fixed_velocity=fixed_velocity),
        solver_options=_solver_options(
            min_picks_per_node=1,
            robust=RefractionStaticRobustOptions(
                enabled=True,
                method='mad',
                threshold=1.0,
                scale_floor_ms=0.0,
                max_iterations=5,
                min_used_fraction=0.5,
                min_used_observations=1,
            ),
        ),
    )

    assert result.robust_stop_reason == 'safe_rejection'
    assert result.n_rejected_observations == 0
    np.testing.assert_array_equal(result.used_observation_mask_sorted, [True] * 3)
    np.testing.assert_array_equal(result.node_observation_count, [2, 1, 3])
    assert result.system.n_node_components == 1


def test_refraction_solver_rejects_mismatched_design_model_modes() -> None:
    source_node_id, receiver_node_id, distance_m, pick_time, valid_mask = (
        _known_global_arrays()
    )

    cell_design = build_refraction_static_design_matrix_from_arrays(
        pick_time_s_sorted=pick_time,
        valid_observation_mask_sorted=valid_mask,
        source_node_id_sorted=source_node_id,
        receiver_node_id_sorted=receiver_node_id,
        distance_m_sorted=distance_m,
        node_id=np.asarray([10, 20, 30, 40]),
        bedrock_velocity_mode='solve_cell',
        midpoint_cell_id_sorted=np.asarray([0, 0, 0, 0, 0, 0]),
        n_total_cells=1,
        number_of_cell_x=1,
        number_of_cell_y=1,
        cell_assignment_mode='midpoint',
        n_traces=6,
    )
    with pytest.raises(RefractionStaticSolverError, match='must match'):
        solve_refraction_static_design_least_squares(
            cell_design,
            model=_model(mode='solve_global'),
            solver_options=_solver_options(),
        )
