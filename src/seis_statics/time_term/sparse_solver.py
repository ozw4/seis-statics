"""Sparse least-squares solver for source/receiver node time-term delays."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_1d_real_numeric_float64 as _coerce_1d_real_numeric_float64,
    coerce_finite_float as _coerce_finite_float,
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
    coerce_nonnegative_int as _coerce_nonnegative_int,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
    is_real_numeric_dtype as _is_real_numeric_dtype,
)
from seis_statics._endpoint_sum_graph import (
    EndpointSumGraphSummary,
    analyze_endpoint_sum_graph,
    build_endpoint_sum_gauge_matrix,
)
from seis_statics.time_term.design_matrix import TimeTermDesignMatrix

_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    allow_integer_like_float=False,
)

TimeTermGaugeMode = Literal[
    'auto_component',
    'none',
]
TimeTermSparseSolverName = Literal['lsmr', 'lsqr']
TimeTermTracePredictionPolicy = Literal['all_supported', 'fit_used_only']


@dataclass(frozen=True)
class TimeTermSparseSolverOptions:
    """Options for fitting node terms and selecting trace prediction coverage.

    ``damping_lambda`` is the objective-level Tikhonov coefficient in
    ``||A x - b||^2 + lambda ||x - damping_prior_s||^2``. Augmented damping
    rows therefore use ``sqrt(damping_lambda)`` as their coefficient.
    """

    damping_lambda: float = 0.01
    damping_prior_s: float | np.ndarray = 0.0

    gauge: TimeTermGaugeMode = 'auto_component'
    gauge_weight: float = 1.0

    solver: TimeTermSparseSolverName = 'lsmr'
    atol: float = 1.0e-8
    btol: float = 1.0e-8
    conlim: float = 1.0e8
    maxiter: int | None = None

    min_observations: int = 1
    min_total_observations_per_node: int = 1
    require_all_nodes_observed: bool = True
    trace_prediction_policy: TimeTermTracePredictionPolicy = 'all_supported'

    max_abs_node_time_term_ms: float | None = 500.0
    max_abs_estimated_trace_delay_ms: float | None = 500.0


@dataclass(frozen=True)
class TimeTermSolverSystem:
    augmented_matrix: sparse.csr_matrix
    augmented_data_s: np.ndarray

    n_observation_rows: int
    n_damping_rows: int
    n_gauge_rows: int
    n_augmented_rows: int
    n_nodes: int

    damping_prior_s: np.ndarray
    gauge_mode: TimeTermGaugeMode
    component_id_by_node: np.ndarray
    n_components: int
    is_bipartite_by_component: np.ndarray
    signed_partition_by_node: np.ndarray
    gauge_required_by_component: np.ndarray
    n_bipartite_components: int

    damping_lambda: float
    gauge_weight: float
    min_total_observations_per_node: int
    total_observation_count_by_node: np.ndarray
    endpoint_supported_trace_mask_sorted: np.ndarray


@dataclass(frozen=True)
class TimeTermSparseSolverResult:
    """Sparse solve result.

    ``used_trace_mask_sorted`` is fit eligibility for this solve. It does not
    define where the final node model may be applied.
    ``prediction_valid_trace_mask_sorted`` marks traces eligible for final
    model application under the selected trace prediction policy. The
    ``all_supported`` policy uses endpoint support from final used
    observations; ``fit_used_only`` uses the final fit mask exactly.
    """

    node_time_term_s: np.ndarray
    estimated_trace_time_term_delay_s_sorted: np.ndarray
    row_estimated_time_term_delay_s: np.ndarray

    row_residual_before_s: np.ndarray
    row_residual_after_s: np.ndarray
    row_residual_after_ms: np.ndarray
    rms_residual_before_s: float
    rms_residual_after_s: float

    used_trace_mask_sorted: np.ndarray
    prediction_valid_trace_mask_sorted: np.ndarray
    row_trace_index_sorted: np.ndarray

    solver_name: TimeTermSparseSolverName
    solver_istop: int
    solver_iterations: int
    solver_normr: float
    solver_normar: float | None
    solver_conda: float | None
    solver_message: str

    system: TimeTermSolverSystem


@dataclass(frozen=True)
class _ValidatedTimeTermDesign:
    matrix: sparse.csr_matrix
    data_s: np.ndarray

    n_traces: int
    n_observations: int
    n_nodes: int

    used_trace_mask_sorted: np.ndarray
    row_trace_index_sorted: np.ndarray
    row_source_node_id: np.ndarray
    row_receiver_node_id: np.ndarray
    source_node_id_sorted: np.ndarray
    receiver_node_id_sorted: np.ndarray
    total_observation_count_by_node: np.ndarray


@dataclass(frozen=True)
class _SparseSolverRawResult:
    x: np.ndarray
    istop: int
    itn: int
    normr: float
    normar: float | None
    conda: float | None
    message: str


def build_time_term_solver_system(
    design: TimeTermDesignMatrix,
    *,
    options: TimeTermSparseSolverOptions | None = None,
) -> TimeTermSolverSystem:
    """Build the objective ``||A x - b||^2 + lambda ||x - prior||^2``."""
    system, _, _ = _build_time_term_solver_system(design, options=options)
    return system


def solve_time_term_sparse_least_squares(
    design: TimeTermDesignMatrix,
    *,
    options: TimeTermSparseSolverOptions | None = None,
) -> TimeTermSparseSolverResult:
    """Estimate endpoint node time-term delays from a time-term design matrix."""
    system, validated_design, validated_options = _build_time_term_solver_system(
        design,
        options=options,
    )
    raw_result = _run_sparse_solver(
        system.augmented_matrix,
        system.augmented_data_s,
        options=validated_options,
    )

    node_time_term = np.ascontiguousarray(raw_result.x, dtype=np.float64)
    _validate_all_finite(node_time_term, name='node_time_term_s')
    _validate_max_abs_ms(
        node_time_term,
        max_abs_ms=validated_options.max_abs_node_time_term_ms,
        name='node_time_term_s',
        limit_name='max_abs_node_time_term_ms',
    )

    row_estimated_delay = np.ascontiguousarray(
        validated_design.matrix @ node_time_term,
        dtype=np.float64,
    )
    if row_estimated_delay.shape != (validated_design.n_observations,):
        raise ValueError('row_estimated_time_term_delay_s shape mismatch')
    _validate_all_finite(
        row_estimated_delay,
        name='row_estimated_time_term_delay_s',
    )

    prediction_valid_mask = _build_trace_prediction_valid_mask(
        validated_design,
        options=validated_options,
    )
    estimated_trace_delay = np.full(
        validated_design.n_traces,
        np.nan,
        dtype=np.float64,
    )
    estimated_trace_delay[prediction_valid_mask] = (
        node_time_term[validated_design.source_node_id_sorted[prediction_valid_mask]]
        + node_time_term[validated_design.receiver_node_id_sorted[prediction_valid_mask]]
    )
    if estimated_trace_delay.shape != (validated_design.n_traces,):
        raise ValueError('estimated_trace_time_term_delay_s_sorted shape mismatch')
    _validate_all_finite(
        estimated_trace_delay[prediction_valid_mask],
        name='estimated_trace_time_term_delay_s_sorted prediction-valid traces',
    )
    _validate_max_abs_ms(
        estimated_trace_delay[prediction_valid_mask],
        max_abs_ms=validated_options.max_abs_estimated_trace_delay_ms,
        name='estimated_trace_time_term_delay_s_sorted prediction-valid traces',
        limit_name='max_abs_estimated_trace_delay_ms',
    )

    residual_before = np.ascontiguousarray(validated_design.data_s, dtype=np.float64)
    residual_after = np.ascontiguousarray(
        validated_design.data_s - row_estimated_delay,
        dtype=np.float64,
    )
    _validate_all_finite(residual_after, name='row_residual_after_s')

    return TimeTermSparseSolverResult(
        node_time_term_s=node_time_term,
        estimated_trace_time_term_delay_s_sorted=estimated_trace_delay,
        row_estimated_time_term_delay_s=row_estimated_delay,
        row_residual_before_s=residual_before,
        row_residual_after_s=residual_after,
        row_residual_after_ms=np.ascontiguousarray(
            residual_after * 1000.0,
            dtype=np.float64,
        ),
        rms_residual_before_s=_rms(residual_before),
        rms_residual_after_s=_rms(residual_after),
        used_trace_mask_sorted=np.ascontiguousarray(
            validated_design.used_trace_mask_sorted,
            dtype=bool,
        ),
        prediction_valid_trace_mask_sorted=np.ascontiguousarray(
            prediction_valid_mask,
            dtype=bool,
        ),
        row_trace_index_sorted=np.ascontiguousarray(
            validated_design.row_trace_index_sorted,
            dtype=np.int64,
        ),
        solver_name=validated_options.solver,
        solver_istop=raw_result.istop,
        solver_iterations=raw_result.itn,
        solver_normr=raw_result.normr,
        solver_normar=raw_result.normar,
        solver_conda=raw_result.conda,
        solver_message=raw_result.message,
        system=system,
    )


def summarize_time_term_sparse_solver_result(
    result: TimeTermSparseSolverResult,
) -> dict[str, object]:
    """Return a JSON-safe summary for future artifacts and job logs."""
    system = result.system
    total_count = _coerce_1d_integer_int64(
        system.total_observation_count_by_node,
        name='total_observation_count_by_node',
        expected_shape=(system.n_nodes,),
    )
    min_count = _coerce_nonnegative_int(
        system.min_total_observations_per_node,
        name='min_total_observations_per_node',
    )
    used_mask = _coerce_1d_bool_array(
        result.used_trace_mask_sorted,
        name='used_trace_mask_sorted',
    )
    prediction_mask = _coerce_1d_bool_array(
        result.prediction_valid_trace_mask_sorted,
        name='prediction_valid_trace_mask_sorted',
        expected_shape=used_mask.shape,
    )
    endpoint_supported_mask = _coerce_1d_bool_array(
        system.endpoint_supported_trace_mask_sorted,
        name='endpoint_supported_trace_mask_sorted',
        expected_shape=used_mask.shape,
    )

    return {
        'n_nodes': int(system.n_nodes),
        'n_observations': int(system.n_observation_rows),
        'n_augmented_rows': int(system.n_augmented_rows),
        'n_damping_rows': int(system.n_damping_rows),
        'n_gauge_rows': int(system.n_gauge_rows),
        'gauge_mode': system.gauge_mode,
        'damping_lambda': _json_float(system.damping_lambda),
        'solver_name': result.solver_name,
        'solver_istop': int(result.solver_istop),
        'solver_iterations': int(result.solver_iterations),
        'solver_normr': _optional_json_float(result.solver_normr),
        'solver_normar': _optional_json_float(result.solver_normar),
        'solver_conda': _optional_json_float(result.solver_conda),
        'solver_message': str(result.solver_message),
        'rms_residual_before_ms': _json_float(
            _coerce_finite_float(result.rms_residual_before_s, name='rms') * 1000.0
        ),
        'rms_residual_after_ms': _json_float(
            _coerce_finite_float(result.rms_residual_after_s, name='rms') * 1000.0
        ),
        'node_time_term_ms': _stats_payload(
            _coerce_1d_real_numeric_float64(
                result.node_time_term_s,
                name='node_time_term_s',
                expected_shape=(system.n_nodes,),
            )
            * 1000.0
        ),
        'estimated_trace_time_term_delay_ms': _stats_payload(
            _coerce_1d_real_numeric_float64(
                result.estimated_trace_time_term_delay_s_sorted,
                name='estimated_trace_time_term_delay_s_sorted',
            )
            * 1000.0
        ),
        'row_residual_after_ms': _stats_payload(
            _coerce_1d_real_numeric_float64(
                result.row_residual_after_ms,
                name='row_residual_after_ms',
                expected_shape=(system.n_observation_rows,),
            )
        ),
        'n_components': int(system.n_components),
        'n_bipartite_components': int(system.n_bipartite_components),
        'n_gauge_required_components': int(
            np.count_nonzero(system.gauge_required_by_component)
        ),
        'n_fit_used_traces': int(np.count_nonzero(used_mask)),
        'n_robust_rejected_traces': 0,
        'n_prediction_valid_traces': int(np.count_nonzero(prediction_mask)),
        'n_fit_unused_prediction_valid_traces': int(
            np.count_nonzero(~used_mask & prediction_mask)
        ),
        'n_unsupported_endpoint_traces': int(
            np.count_nonzero(~endpoint_supported_mask)
        ),
        'n_unobserved_nodes': int(np.count_nonzero(total_count == 0)),
        'n_nodes_below_min_observations': int(np.count_nonzero(total_count < min_count)),
    }


def build_node_components(
    n_nodes: int,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
) -> np.ndarray:
    """Build deterministic connected-component ids from used source/receiver edges."""
    graph = analyze_endpoint_sum_graph(
        n_nodes=n_nodes,
        row_source_node_id=row_source_node_id,
        row_receiver_node_id=row_receiver_node_id,
    )
    return graph.component_id_by_node


def _build_gauge_matrix(
    *,
    graph: EndpointSumGraphSummary,
    gauge: TimeTermGaugeMode,
    gauge_weight: float,
) -> sparse.csr_matrix:
    if gauge == 'none':
        return sparse.csr_matrix(
            (0, int(graph.component_id_by_node.shape[0])),
            dtype=np.float64,
        )
    if gauge != 'auto_component':
        raise ValueError(f'unsupported gauge: {gauge!r}')
    return build_endpoint_sum_gauge_matrix(
        graph=graph,
        n_columns=int(graph.component_id_by_node.shape[0]),
        gauge_weight=gauge_weight,
    )


def _build_time_term_solver_system(
    design: TimeTermDesignMatrix,
    *,
    options: TimeTermSparseSolverOptions | None,
) -> tuple[
    TimeTermSolverSystem,
    _ValidatedTimeTermDesign,
    TimeTermSparseSolverOptions,
]:
    validated_options = _validate_options(options)
    validated_design = _validate_design(design)
    damping_prior = _coerce_damping_prior(
        validated_options.damping_prior_s,
        n_nodes=validated_design.n_nodes,
    )
    _validate_minimum_observations(
        validated_design,
        options=validated_options,
    )

    graph = analyze_endpoint_sum_graph(
        n_nodes=validated_design.n_nodes,
        row_source_node_id=validated_design.row_source_node_id,
        row_receiver_node_id=validated_design.row_receiver_node_id,
    )
    _validate_zero_regularization_gauge(
        graph=graph,
        options=validated_options,
    )

    damping_matrix, damping_data = _build_damping_system(
        n_nodes=validated_design.n_nodes,
        damping_lambda=validated_options.damping_lambda,
        damping_prior_s=damping_prior,
    )
    gauge_matrix = _build_gauge_matrix(
        graph=graph,
        gauge=validated_options.gauge,
        gauge_weight=validated_options.gauge_weight,
    )
    gauge_data = np.zeros(gauge_matrix.shape[0], dtype=np.float64)

    augmented_matrix = sparse.vstack(
        [validated_design.matrix, damping_matrix, gauge_matrix],
        format='csr',
        dtype=np.float64,
    )
    augmented_matrix.sort_indices()
    augmented_data = np.ascontiguousarray(
        np.concatenate([validated_design.data_s, damping_data, gauge_data]),
        dtype=np.float64,
    )
    n_augmented_rows = int(augmented_matrix.shape[0])
    if augmented_matrix.shape != (n_augmented_rows, validated_design.n_nodes):
        raise ValueError('augmented_matrix shape mismatch')
    if augmented_data.shape != (n_augmented_rows,):
        raise ValueError('augmented_data_s shape mismatch')
    _validate_all_finite(augmented_matrix.data, name='augmented_matrix.data')
    _validate_all_finite(augmented_data, name='augmented_data_s')

    system = TimeTermSolverSystem(
        augmented_matrix=augmented_matrix,
        augmented_data_s=augmented_data,
        n_observation_rows=validated_design.n_observations,
        n_damping_rows=int(damping_matrix.shape[0]),
        n_gauge_rows=int(gauge_matrix.shape[0]),
        n_augmented_rows=n_augmented_rows,
        n_nodes=validated_design.n_nodes,
        damping_prior_s=damping_prior,
        gauge_mode=validated_options.gauge,
        component_id_by_node=graph.component_id_by_node,
        n_components=graph.n_components,
        is_bipartite_by_component=graph.is_bipartite_by_component,
        signed_partition_by_node=graph.signed_partition_by_node,
        gauge_required_by_component=graph.gauge_required_by_component,
        n_bipartite_components=int(np.count_nonzero(graph.is_bipartite_by_component)),
        damping_lambda=validated_options.damping_lambda,
        gauge_weight=validated_options.gauge_weight,
        min_total_observations_per_node=(
            validated_options.min_total_observations_per_node
        ),
        total_observation_count_by_node=(
            validated_design.total_observation_count_by_node
        ),
        endpoint_supported_trace_mask_sorted=_build_endpoint_supported_trace_mask(
            validated_design,
            options=validated_options,
        ),
    )
    return system, validated_design, validated_options


def _build_damping_system(
    *,
    n_nodes: int,
    damping_lambda: float,
    damping_prior_s: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    if damping_lambda == 0.0:
        return (
            sparse.csr_matrix((0, n_nodes), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    damping_weight = float(np.sqrt(damping_lambda))
    damping_matrix = sparse.eye(
        n_nodes,
        n_nodes,
        format='csr',
        dtype=np.float64,
    ) * damping_weight
    damping_data = np.ascontiguousarray(
        damping_weight * damping_prior_s,
        dtype=np.float64,
    )
    return damping_matrix, damping_data


def _run_sparse_solver(
    matrix: sparse.csr_matrix,
    rhs_s: np.ndarray,
    *,
    options: TimeTermSparseSolverOptions,
) -> _SparseSolverRawResult:
    matrix_csr = _coerce_sparse_matrix_float64_csr(matrix)
    n_rows, n_cols = matrix_csr.shape
    rhs = _coerce_1d_real_numeric_float64(
        rhs_s,
        name='rhs_s',
        expected_shape=(n_rows,),
    )
    _validate_all_finite(rhs, name='rhs_s')

    try:
        if options.solver == 'lsmr':
            x, istop, itn, normr, normar, _, conda, _ = sparse_linalg.lsmr(
                matrix_csr,
                rhs,
                damp=0.0,
                atol=options.atol,
                btol=options.btol,
                conlim=options.conlim,
                maxiter=options.maxiter,
                show=False,
            )
        elif options.solver == 'lsqr':
            x, istop, itn, normr, _, _, conda, normar, _, _ = sparse_linalg.lsqr(
                matrix_csr,
                rhs,
                damp=0.0,
                atol=options.atol,
                btol=options.btol,
                conlim=options.conlim,
                iter_lim=options.maxiter,
                show=False,
                calc_var=False,
            )
        else:
            raise ValueError(f'unsupported solver: {options.solver!r}')
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f'{options.solver} solve failed') from exc

    solution = np.ascontiguousarray(x, dtype=np.float64)
    if solution.shape != (n_cols,):
        raise ValueError(
            f'solver x shape mismatch: expected {(n_cols,)}, got {solution.shape}'
        )
    _validate_all_finite(solution, name='solver x')
    return _SparseSolverRawResult(
        x=solution,
        istop=int(istop),
        itn=int(itn),
        normr=float(normr),
        normar=float(normar),
        conda=float(conda),
        message=_solver_message(options.solver, int(istop)),
    )


def _validate_options(
    options: TimeTermSparseSolverOptions | None,
) -> TimeTermSparseSolverOptions:
    opts = TimeTermSparseSolverOptions() if options is None else options
    if not isinstance(opts, TimeTermSparseSolverOptions):
        raise ValueError('options must be a TimeTermSparseSolverOptions instance')

    damping_lambda = _coerce_nonnegative_finite_float(
        opts.damping_lambda,
        name='damping_lambda',
    )
    gauge = _validate_gauge_mode(opts.gauge)
    gauge_weight = _coerce_finite_float(opts.gauge_weight, name='gauge_weight')
    if gauge == 'auto_component' and gauge_weight <= 0.0:
        raise ValueError(
            'gauge_weight must be greater than 0 when gauge is auto_component'
        )
    solver = _validate_solver_name(opts.solver)

    return TimeTermSparseSolverOptions(
        damping_lambda=damping_lambda,
        damping_prior_s=opts.damping_prior_s,
        gauge=gauge,
        gauge_weight=gauge_weight,
        solver=solver,
        atol=_coerce_positive_finite_float(opts.atol, name='atol'),
        btol=_coerce_positive_finite_float(opts.btol, name='btol'),
        conlim=_coerce_positive_finite_float(opts.conlim, name='conlim'),
        maxiter=_coerce_optional_positive_int(opts.maxiter, name='maxiter'),
        min_observations=_coerce_positive_int(
            opts.min_observations,
            name='min_observations',
        ),
        min_total_observations_per_node=_coerce_nonnegative_int(
            opts.min_total_observations_per_node,
            name='min_total_observations_per_node',
        ),
        require_all_nodes_observed=_coerce_bool(
            opts.require_all_nodes_observed,
            name='require_all_nodes_observed',
        ),
        trace_prediction_policy=_validate_trace_prediction_policy(
            opts.trace_prediction_policy
        ),
        max_abs_node_time_term_ms=_coerce_optional_nonnegative_finite_float(
            opts.max_abs_node_time_term_ms,
            name='max_abs_node_time_term_ms',
        ),
        max_abs_estimated_trace_delay_ms=_coerce_optional_nonnegative_finite_float(
            opts.max_abs_estimated_trace_delay_ms,
            name='max_abs_estimated_trace_delay_ms',
        ),
    )


def _build_trace_prediction_valid_mask(
    design: _ValidatedTimeTermDesign,
    *,
    options: TimeTermSparseSolverOptions,
) -> np.ndarray:
    if options.trace_prediction_policy == 'fit_used_only':
        return np.ascontiguousarray(
            design.used_trace_mask_sorted,
            dtype=bool,
        )

    return _build_endpoint_supported_trace_mask(design, options=options)


def _build_endpoint_supported_trace_mask(
    design: _ValidatedTimeTermDesign,
    *,
    options: TimeTermSparseSolverOptions,
) -> np.ndarray:
    support_threshold = max(1, int(options.min_total_observations_per_node))
    node_supported = np.ascontiguousarray(
        design.total_observation_count_by_node >= support_threshold,
        dtype=bool,
    )
    endpoint_supported = (
        node_supported[design.source_node_id_sorted]
        & node_supported[design.receiver_node_id_sorted]
    )
    return np.ascontiguousarray(endpoint_supported, dtype=bool)


def _validate_design(design: TimeTermDesignMatrix) -> _ValidatedTimeTermDesign:
    n_nodes = _coerce_positive_int(design.n_nodes, name='design.n_nodes')
    n_observations = _coerce_positive_int(
        design.n_observations,
        name='design.n_observations',
    )
    n_traces = _coerce_positive_int(design.n_traces, name='design.n_traces')

    matrix = _coerce_sparse_matrix_float64_csr(design.matrix)
    if matrix.shape != (n_observations, n_nodes):
        raise ValueError('design.matrix shape does not match design dimensions')

    data_s = _coerce_1d_real_numeric_float64(
        design.data_s,
        name='design.data_s',
        expected_shape=(n_observations,),
    )
    _validate_all_finite(data_s, name='design.data_s')
    used_mask = _coerce_1d_bool_array(
        design.used_trace_mask_sorted,
        name='design.used_trace_mask_sorted',
        expected_shape=(n_traces,),
    )
    row_trace_index = _coerce_1d_integer_int64(
        design.row_trace_index_sorted,
        name='design.row_trace_index_sorted',
        expected_shape=(n_observations,),
    )
    _validate_index_range(
        row_trace_index,
        n_unique=n_traces,
        name='design.row_trace_index_sorted',
    )
    if not np.array_equal(row_trace_index, np.flatnonzero(used_mask)):
        raise ValueError(
            'design.row_trace_index_sorted must match design.used_trace_mask_sorted'
        )
    row_source = _coerce_1d_integer_int64(
        design.row_source_node_id,
        name='design.row_source_node_id',
        expected_shape=(n_observations,),
    )
    row_receiver = _coerce_1d_integer_int64(
        design.row_receiver_node_id,
        name='design.row_receiver_node_id',
        expected_shape=(n_observations,),
    )
    _validate_index_range(row_source, n_unique=n_nodes, name='design.row_source_node_id')
    _validate_index_range(
        row_receiver,
        n_unique=n_nodes,
        name='design.row_receiver_node_id',
    )

    source_sorted = _coerce_1d_integer_int64(
        design.source_node_id_sorted,
        name='design.source_node_id_sorted',
        expected_shape=(n_traces,),
    )
    receiver_sorted = _coerce_1d_integer_int64(
        design.receiver_node_id_sorted,
        name='design.receiver_node_id_sorted',
        expected_shape=(n_traces,),
    )
    _validate_index_range(
        source_sorted,
        n_unique=n_nodes,
        name='design.source_node_id_sorted',
    )
    _validate_index_range(
        receiver_sorted,
        n_unique=n_nodes,
        name='design.receiver_node_id_sorted',
    )

    total_count = _node_total_observation_count(
        row_source,
        row_receiver,
        n_nodes=n_nodes,
    )
    design_total_count = _coerce_1d_integer_int64(
        design.total_observation_count_by_node,
        name='design.total_observation_count_by_node',
        expected_shape=(n_nodes,),
    )
    if np.any(total_count != design_total_count):
        raise ValueError('design.total_observation_count_by_node does not match rows')

    return _ValidatedTimeTermDesign(
        matrix=matrix,
        data_s=data_s,
        n_traces=n_traces,
        n_observations=n_observations,
        n_nodes=n_nodes,
        used_trace_mask_sorted=used_mask,
        row_trace_index_sorted=row_trace_index,
        row_source_node_id=row_source,
        row_receiver_node_id=row_receiver,
        source_node_id_sorted=source_sorted,
        receiver_node_id_sorted=receiver_sorted,
        total_observation_count_by_node=total_count,
    )


def _validate_minimum_observations(
    design: _ValidatedTimeTermDesign,
    *,
    options: TimeTermSparseSolverOptions,
) -> None:
    if design.n_observations < options.min_observations:
        raise ValueError(
            'not enough time-term observations: '
            f'{design.n_observations} < {options.min_observations}'
        )
    total_count = design.total_observation_count_by_node
    if options.require_all_nodes_observed and np.any(total_count == 0):
        raise ValueError('all nodes must have at least one observation')
    if np.any(total_count < options.min_total_observations_per_node):
        raise ValueError(
            'node observation count is below min_total_observations_per_node'
        )


def _validate_zero_regularization_gauge(
    *,
    graph: EndpointSumGraphSummary,
    options: TimeTermSparseSolverOptions,
) -> None:
    if options.damping_lambda != 0.0 or options.gauge != 'none':
        return
    if np.any(graph.gauge_required_by_component):
        raise ValueError(
            "gauge='none' with zero damping is only allowed when every "
            'endpoint-sum component is non-bipartite and already full rank'
        )


def _coerce_damping_prior(values: float | np.ndarray, *, n_nodes: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        prior = np.full(n_nodes, float(arr), dtype=np.float64)
    elif arr.ndim == 1 and arr.shape == (n_nodes,):
        if not _is_real_numeric_dtype(arr.dtype):
            raise ValueError('damping_prior_s must have a numeric dtype')
        prior = np.ascontiguousarray(arr, dtype=np.float64)
    else:
        raise ValueError(
            f'damping_prior_s must be scalar or shape {(n_nodes,)}, got {arr.shape}'
        )
    _validate_all_finite(prior, name='damping_prior_s')
    return prior


def _node_total_observation_count(
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    source_count = np.bincount(row_source_node_id, minlength=n_nodes).astype(
        np.int64,
        copy=False,
    )
    receiver_count = np.bincount(row_receiver_node_id, minlength=n_nodes).astype(
        np.int64,
        copy=False,
    )
    return np.ascontiguousarray(source_count + receiver_count, dtype=np.int64)


def _coerce_sparse_matrix_float64_csr(matrix: object) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        raise ValueError('matrix must be a SciPy sparse matrix')
    if len(matrix.shape) != 2:
        raise ValueError('matrix must be 2D')
    n_rows = _coerce_positive_int(matrix.shape[0], name='matrix.shape[0]')
    n_cols = _coerce_positive_int(matrix.shape[1], name='matrix.shape[1]')
    dtype = np.dtype(matrix.dtype)
    if not _is_real_numeric_dtype(dtype):
        raise ValueError('matrix dtype must be real numeric')
    matrix_csr = matrix.tocsr().astype(np.float64, copy=False)
    if matrix_csr.shape != (n_rows, n_cols):
        raise ValueError('matrix shape changed during CSR conversion')
    _validate_all_finite(matrix_csr.data, name='matrix.data')
    return matrix_csr


def _coerce_optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _coerce_positive_int(value, name=name)


def _coerce_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be bool')
    return bool(value)


def _coerce_optional_nonnegative_finite_float(
    value: object,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    return _coerce_nonnegative_finite_float(value, name=name)


def _validate_index_range(
    indices: np.ndarray,
    *,
    n_unique: int,
    name: str,
) -> None:
    if indices.size == 0:
        return
    if np.any(indices < 0):
        raise ValueError(f'{name} must be greater than or equal to 0')
    if np.any(indices >= n_unique):
        raise ValueError(f'{name} contains values outside 0..{n_unique - 1}')


def _validate_all_finite(values: np.ndarray, *, name: str) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f'{name} must contain only finite values')


def _validate_max_abs_ms(
    values: np.ndarray,
    *,
    max_abs_ms: float | None,
    name: str,
    limit_name: str,
) -> float:
    arr = _coerce_1d_real_numeric_float64(values, name=name)
    _validate_all_finite(arr, name=name)
    max_abs_s = float(np.max(np.abs(arr)))
    if max_abs_ms is not None and max_abs_s > max_abs_ms / 1000.0:
        raise ValueError(f'{name} exceeds {limit_name}')
    return max_abs_s


def _validate_gauge_mode(value: object) -> TimeTermGaugeMode:
    if value in {'auto_component', 'none'}:
        return value  # type: ignore[return-value]
    raise ValueError(f'unsupported gauge: {value!r}')


def _validate_solver_name(value: object) -> TimeTermSparseSolverName:
    if value in {'lsmr', 'lsqr'}:
        return value  # type: ignore[return-value]
    raise ValueError(f'unsupported solver: {value!r}')


def _validate_trace_prediction_policy(value: object) -> TimeTermTracePredictionPolicy:
    if value in {'all_supported', 'fit_used_only'}:
        return value  # type: ignore[return-value]
    raise ValueError(f'unsupported trace_prediction_policy: {value!r}')


def _rms(values: np.ndarray) -> float:
    arr = _coerce_1d_real_numeric_float64(values, name='rms values')
    _validate_all_finite(arr, name='rms values')
    return float(np.sqrt(np.mean(np.square(arr))))


def _json_float(value: float) -> float:
    out = _coerce_finite_float(value, name='summary value')
    return float(out)


def _optional_json_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _stats_payload(values: np.ndarray) -> dict[str, float | int | None]:
    arr = _coerce_1d_real_numeric_float64(values, name='summary values')
    finite = np.ascontiguousarray(arr[np.isfinite(arr)], dtype=np.float64)
    count = int(finite.shape[0])
    if count == 0:
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'max_abs': None,
        }
    return {
        'count': count,
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
        'median': float(np.median(finite)),
        'std': float(np.std(finite, ddof=0)),
        'max_abs': float(np.max(np.abs(finite))),
    }


def _solver_message(solver: TimeTermSparseSolverName, istop: int) -> str:
    messages = {
        0: 'zero solution is exact',
        1: 'Ax-b is small enough',
        2: 'least-squares solution is good enough',
        3: 'condition estimate exceeded conlim',
        4: 'Ax-b is small enough for machine precision',
        5: 'least-squares solution is good enough for machine precision',
        6: 'condition estimate exceeded conlim for machine precision',
        7: 'iteration limit reached',
    }
    return f'{solver}: {messages.get(istop, f"istop={istop}")}'


__all__ = [
    'TimeTermGaugeMode',
    'TimeTermSolverSystem',
    'TimeTermSparseSolverName',
    'TimeTermSparseSolverOptions',
    'TimeTermSparseSolverResult',
    'TimeTermTracePredictionPolicy',
    'build_node_components',
    'build_time_term_solver_system',
    'solve_time_term_sparse_least_squares',
    'summarize_time_term_sparse_solver_result',
]
