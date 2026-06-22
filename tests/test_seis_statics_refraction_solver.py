from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

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
    assert result.system.n_gauge_rows == 1
    assert result.system.n_node_components == 1
    assert result.system.n_bipartite_node_components == 1
    assert result.qc['n_node_components'] == 1
    assert result.qc['n_bipartite_node_components'] == 1
    assert result.qc['n_gauge_required_node_components'] == 1
    assert result.qc['solver_name'] == 'lsq_linear'
    assert result.qc['physical_identifiability']['expected_rank'] == 4
    assert result.qc['physical_identifiability']['estimated_numerical_rank'] == 4


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

    assert result.bedrock_velocity_m_s == pytest.approx(true_velocity, abs=1.0e-6)
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


def test_refraction_solver_system_gauge_adds_row_per_bipartite_component() -> None:
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
    node_block = system.augmented_matrix[
        : system.n_observation_rows + system.n_gauge_rows,
        : design.n_active_nodes,
    ].toarray()

    assert system.n_node_components == 2
    assert system.n_bipartite_node_components == 2
    assert system.n_gauge_rows == 2
    np.testing.assert_allclose(
        system.augmented_matrix[-2:, : design.n_active_nodes].toarray(),
        [
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
        ],
    )
    assert np.linalg.matrix_rank(node_block) == design.n_active_nodes


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
    assert result.n_rejected_observations == 1
    np.testing.assert_array_equal(
        result.used_observation_mask_sorted,
        [True, True, True, True, False],
    )
    assert result.robust_iteration_summaries[0].n_rejected_this_iteration == 1
    assert result.system.n_observation_rows == 4
    assert result.system.n_node_components == 2
    assert result.system.n_gauge_rows == 2
    assert result.qc['n_initial_node_components'] == 1
    assert result.qc['n_final_node_components'] == 2
    assert result.qc['n_initial_gauge_rows'] == 1
    assert result.qc['n_final_gauge_rows'] == 2


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
