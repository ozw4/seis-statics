"""Sparse least-squares solver for residual static estimation."""

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
    coerce_nonnegative_finite_float as _coerce_nonnegative_finite_float,
    coerce_positive_finite_float as _coerce_positive_finite_float,
    coerce_positive_int as _coerce_positive_int,
    is_real_numeric_dtype as _is_real_numeric_dtype,
)
from seis_statics.residual.design_matrix import (
    ResidualStaticColumnLayout,
    ResidualStaticModelEvaluation,
    ResidualStaticObservationMatrixTriplets,
    ResidualStaticParameterParts,
    build_residual_static_column_layout,
    build_residual_static_observation_matrix_triplets,
    evaluate_residual_static_model,
    unpack_residual_static_parameters,
)
from seis_statics.residual.types import ResidualStaticSolverInputs

ResidualStaticGauge = Literal['zero_mean_source_receiver']

_coerce_1d_integer_int64 = partial(
    _common_coerce_1d_integer_int64,
    allow_integer_like_float=False,
)


@dataclass(frozen=True)
class ResidualStaticStabilizationOptions:
    gauge: ResidualStaticGauge = 'zero_mean_source_receiver'
    damping_lambda: float = 0.0
    min_valid_picks: int = 10
    min_picks_per_source: int = 1
    min_picks_per_receiver: int = 1
    max_abs_estimated_delay_ms: float = 250.0


@dataclass(frozen=True)
class ResidualStaticObservationGraphSummary:
    n_components: int
    source_component_index: np.ndarray
    receiver_component_index: np.ndarray
    component_observation_counts: np.ndarray
    component_source_counts: np.ndarray
    component_receiver_counts: np.ndarray


@dataclass(frozen=True)
class ResidualStaticMinimumDataSummary:
    n_used_picks: int
    n_sources: int
    n_receivers: int
    n_effective_parameters: int

    source_used_pick_counts: np.ndarray
    receiver_used_pick_counts: np.ndarray
    underconstrained_source_ids: np.ndarray
    underconstrained_receiver_ids: np.ndarray

    graph: ResidualStaticObservationGraphSummary

    abs_offset_min: float | None
    abs_offset_max: float | None
    abs_offset_span: float | None


@dataclass(frozen=True)
class ResidualStaticAugmentedSystem:
    matrix: object
    rhs_s: np.ndarray

    n_observation_rows: int
    n_gauge_rows: int
    n_damping_rows: int
    n_rows: int
    n_cols: int

    observation_row_to_sorted_trace_index: np.ndarray
    used_mask_sorted: np.ndarray
    minimum_data: ResidualStaticMinimumDataSummary


@dataclass(frozen=True)
class ResidualStaticStabilizedLeastSquaresResult:
    parameter_vector: np.ndarray
    parameter_parts: ResidualStaticParameterParts
    model_evaluation: ResidualStaticModelEvaluation
    diagnostics: ResidualStaticLsmrDiagnostics

    layout: ResidualStaticColumnLayout
    used_mask_sorted: np.ndarray
    observation_row_to_sorted_trace_index: np.ndarray
    minimum_data: ResidualStaticMinimumDataSummary
    stabilization_options: ResidualStaticStabilizationOptions

    n_observations: int
    n_model_parameters: int
    n_gauge_rows: int
    n_damping_rows: int

    max_abs_estimated_delay_s: float
    rank_deficient_possible: bool


@dataclass(frozen=True)
class ResidualStaticLsmrOptions:
    atol: float = 1.0e-10
    btol: float = 1.0e-10
    conlim: float = 1.0e8
    maxiter: int | None = None


@dataclass(frozen=True)
class ResidualStaticLsmrDiagnostics:
    istop: int
    itn: int
    normr: float
    normar: float
    norma: float
    conda: float
    normx: float


@dataclass(frozen=True)
class ResidualStaticRawLsmrResult:
    parameter_vector: np.ndarray
    diagnostics: ResidualStaticLsmrDiagnostics


@dataclass(frozen=True)
class ResidualStaticLeastSquaresResult:
    parameter_vector: np.ndarray
    parameter_parts: ResidualStaticParameterParts
    model_evaluation: ResidualStaticModelEvaluation
    diagnostics: ResidualStaticLsmrDiagnostics

    layout: ResidualStaticColumnLayout
    used_mask_sorted: np.ndarray
    row_to_sorted_trace_index: np.ndarray

    n_observations: int
    n_model_parameters: int
    rank_deficient_possible: bool


def validate_lsmr_options(
    options: ResidualStaticLsmrOptions,
) -> ResidualStaticLsmrOptions:
    """Validate and normalize LSMR numeric options."""
    if not isinstance(options, ResidualStaticLsmrOptions):
        raise ValueError('options must be a ResidualStaticLsmrOptions instance')
    return ResidualStaticLsmrOptions(
        atol=_coerce_nonnegative_finite_float(options.atol, name='atol'),
        btol=_coerce_nonnegative_finite_float(options.btol, name='btol'),
        conlim=_coerce_positive_finite_float(options.conlim, name='conlim'),
        maxiter=_coerce_optional_positive_int(options.maxiter, name='maxiter'),
    )


def validate_residual_static_stabilization_options(
    options: ResidualStaticStabilizationOptions,
) -> ResidualStaticStabilizationOptions:
    """Validate and normalize residual-static stabilization options."""
    if not isinstance(options, ResidualStaticStabilizationOptions):
        raise ValueError(
            'options must be a ResidualStaticStabilizationOptions instance'
        )
    if options.gauge != 'zero_mean_source_receiver':
        raise ValueError('gauge must be zero_mean_source_receiver')
    return ResidualStaticStabilizationOptions(
        gauge='zero_mean_source_receiver',
        damping_lambda=_coerce_nonnegative_finite_float(
            options.damping_lambda,
            name='damping_lambda',
        ),
        min_valid_picks=_coerce_positive_int(
            options.min_valid_picks,
            name='min_valid_picks',
        ),
        min_picks_per_source=_coerce_positive_int(
            options.min_picks_per_source,
            name='min_picks_per_source',
        ),
        min_picks_per_receiver=_coerce_positive_int(
            options.min_picks_per_receiver,
            name='min_picks_per_receiver',
        ),
        max_abs_estimated_delay_ms=_coerce_positive_finite_float(
            options.max_abs_estimated_delay_ms,
            name='max_abs_estimated_delay_ms',
        ),
    )


def validate_residual_static_used_mask(
    inputs: ResidualStaticSolverInputs,
    used_mask_sorted: np.ndarray | None,
) -> np.ndarray:
    """Validate the robust-solver used-pick mask against valid picks."""
    n_traces = _input_n_traces(inputs)
    valid_mask = _coerce_1d_bool_array(
        inputs.valid_pick_mask_sorted,
        name='valid_pick_mask_sorted',
        expected_shape=(n_traces,),
    )
    if used_mask_sorted is None:
        used_mask = np.ascontiguousarray(valid_mask, dtype=bool)
    else:
        used_mask = _coerce_1d_bool_array(
            used_mask_sorted,
            name='used_mask_sorted',
            expected_shape=(n_traces,),
        )
    _validate_mask_subset(
        used_mask,
        valid_mask,
        mask_name='used_mask_sorted',
        base_name='valid_pick_mask_sorted',
    )

    pick_time_after_datum = _coerce_1d_real_numeric_float64(
        inputs.pick_time_after_datum_s_sorted,
        name='pick_time_after_datum_s_sorted',
        expected_shape=(n_traces,),
    )
    if np.any(~np.isfinite(pick_time_after_datum[used_mask])):
        raise ValueError('pick_time_after_datum_s_sorted must be finite for used picks')
    return np.ascontiguousarray(used_mask, dtype=bool)


def build_residual_static_observation_graph_summary(
    inputs: ResidualStaticSolverInputs,
    *,
    used_mask_sorted: np.ndarray,
) -> ResidualStaticObservationGraphSummary:
    """Summarize source/receiver connectivity for used residual-static picks."""
    used_mask = validate_residual_static_used_mask(inputs, used_mask_sorted)
    layout = build_residual_static_column_layout(inputs)
    n_traces = _input_n_traces(inputs)
    source_index = _coerce_1d_integer_int64(
        inputs.source_index_sorted,
        name='source_index_sorted',
        expected_shape=(n_traces,),
    )
    receiver_index = _coerce_1d_integer_int64(
        inputs.receiver_index_sorted,
        name='receiver_index_sorted',
        expected_shape=(n_traces,),
    )
    _validate_index_range(
        source_index,
        n_unique=layout.n_sources,
        name='source_index_sorted',
    )
    _validate_index_range(
        receiver_index,
        n_unique=layout.n_receivers,
        name='receiver_index_sorted',
    )

    n_nodes = layout.n_sources + layout.n_receivers
    parent = np.arange(n_nodes, dtype=np.int64)
    rank = np.zeros(n_nodes, dtype=np.int8)

    def find(node: int) -> int:
        root = node
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[node]) != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    used_trace_indices = np.flatnonzero(used_mask)
    for trace_index in used_trace_indices:
        union(
            int(source_index[trace_index]),
            layout.n_sources + int(receiver_index[trace_index]),
        )

    component_by_root: dict[int, int] = {}
    node_component = np.empty(n_nodes, dtype=np.int64)
    for node in range(n_nodes):
        root = find(node)
        component = component_by_root.setdefault(root, len(component_by_root))
        node_component[node] = component

    n_components = int(len(component_by_root))
    component_observation_counts = np.zeros(n_components, dtype=np.int64)
    for trace_index in used_trace_indices:
        component = node_component[int(source_index[trace_index])]
        component_observation_counts[component] += 1

    source_component_index = np.ascontiguousarray(
        node_component[: layout.n_sources],
        dtype=np.int64,
    )
    receiver_component_index = np.ascontiguousarray(
        node_component[layout.n_sources :],
        dtype=np.int64,
    )
    component_source_counts = np.bincount(
        source_component_index,
        minlength=n_components,
    ).astype(np.int64, copy=False)
    component_receiver_counts = np.bincount(
        receiver_component_index,
        minlength=n_components,
    ).astype(np.int64, copy=False)

    return ResidualStaticObservationGraphSummary(
        n_components=n_components,
        source_component_index=source_component_index,
        receiver_component_index=receiver_component_index,
        component_observation_counts=np.ascontiguousarray(
            component_observation_counts,
            dtype=np.int64,
        ),
        component_source_counts=np.ascontiguousarray(
            component_source_counts,
            dtype=np.int64,
        ),
        component_receiver_counts=np.ascontiguousarray(
            component_receiver_counts,
            dtype=np.int64,
        ),
    )


def validate_minimum_residual_static_data(
    inputs: ResidualStaticSolverInputs,
    *,
    used_mask_sorted: np.ndarray,
    options: ResidualStaticStabilizationOptions,
) -> ResidualStaticMinimumDataSummary:
    """Validate that used picks can support a stabilized residual-static solve."""
    validated_options = validate_residual_static_stabilization_options(options)
    used_mask = validate_residual_static_used_mask(inputs, used_mask_sorted)
    layout = build_residual_static_column_layout(inputs)
    n_traces = _input_n_traces(inputs)
    source_unique_ids = _coerce_1d_integer_preserve_dtype(
        inputs.source_unique_ids,
        name='source_unique_ids',
        expected_shape=(layout.n_sources,),
    )
    receiver_unique_ids = _coerce_1d_integer_preserve_dtype(
        inputs.receiver_unique_ids,
        name='receiver_unique_ids',
        expected_shape=(layout.n_receivers,),
    )
    source_index = _coerce_1d_integer_int64(
        inputs.source_index_sorted,
        name='source_index_sorted',
        expected_shape=(n_traces,),
    )
    receiver_index = _coerce_1d_integer_int64(
        inputs.receiver_index_sorted,
        name='receiver_index_sorted',
        expected_shape=(n_traces,),
    )
    _validate_index_range(
        source_index,
        n_unique=layout.n_sources,
        name='source_index_sorted',
    )
    _validate_index_range(
        receiver_index,
        n_unique=layout.n_receivers,
        name='receiver_index_sorted',
    )

    used_trace_indices = np.flatnonzero(used_mask)
    n_used_picks = int(used_trace_indices.shape[0])
    n_effective_parameters = _effective_parameter_count(layout)
    if n_used_picks < validated_options.min_valid_picks:
        raise ValueError(
            'n_used_picks must be greater than or equal to min_valid_picks'
        )
    if n_used_picks < n_effective_parameters:
        raise ValueError(
            'n_used_picks must be greater than or equal to n_effective_parameters'
        )

    source_used_pick_counts = np.bincount(
        source_index[used_mask],
        minlength=layout.n_sources,
    ).astype(np.int64, copy=False)
    receiver_used_pick_counts = np.bincount(
        receiver_index[used_mask],
        minlength=layout.n_receivers,
    ).astype(np.int64, copy=False)
    underconstrained_source_mask = (
        source_used_pick_counts < validated_options.min_picks_per_source
    )
    underconstrained_receiver_mask = (
        receiver_used_pick_counts < validated_options.min_picks_per_receiver
    )
    underconstrained_source_ids = np.ascontiguousarray(
        source_unique_ids[underconstrained_source_mask],
        dtype=source_unique_ids.dtype,
    )
    underconstrained_receiver_ids = np.ascontiguousarray(
        receiver_unique_ids[underconstrained_receiver_mask],
        dtype=receiver_unique_ids.dtype,
    )
    if underconstrained_source_ids.size > 0:
        raise ValueError('used pick count is below min_picks_per_source')
    if underconstrained_receiver_ids.size > 0:
        raise ValueError('used pick count is below min_picks_per_receiver')

    graph = build_residual_static_observation_graph_summary(
        inputs,
        used_mask_sorted=used_mask,
    )
    if graph.n_components != 1:
        raise ValueError('source/receiver observation graph must be connected')

    abs_offset_min: float | None
    abs_offset_max: float | None
    abs_offset_span: float | None
    if layout.moveout_model == 'linear_abs_offset':
        abs_offset = _required_abs_offset(inputs, expected_shape=(n_traces,))
        used_abs_offset = abs_offset[used_mask]
        if np.any(~np.isfinite(used_abs_offset)):
            raise ValueError('abs_offset_sorted must be finite for used traces')
        if np.any(used_abs_offset < 0.0):
            raise ValueError('abs_offset_sorted must be non-negative for used traces')
        abs_offset_min = float(np.min(used_abs_offset))
        abs_offset_max = float(np.max(used_abs_offset))
        abs_offset_span = float(abs_offset_max - abs_offset_min)
        if abs_offset_span <= 0.0:
            raise ValueError('abs_offset span must be greater than 0')
    else:
        abs_offset_min = None
        abs_offset_max = None
        abs_offset_span = None

    return ResidualStaticMinimumDataSummary(
        n_used_picks=n_used_picks,
        n_sources=layout.n_sources,
        n_receivers=layout.n_receivers,
        n_effective_parameters=n_effective_parameters,
        source_used_pick_counts=np.ascontiguousarray(
            source_used_pick_counts,
            dtype=np.int64,
        ),
        receiver_used_pick_counts=np.ascontiguousarray(
            receiver_used_pick_counts,
            dtype=np.int64,
        ),
        underconstrained_source_ids=underconstrained_source_ids,
        underconstrained_receiver_ids=underconstrained_receiver_ids,
        graph=graph,
        abs_offset_min=abs_offset_min,
        abs_offset_max=abs_offset_max,
        abs_offset_span=abs_offset_span,
    )


def build_zero_mean_gauge_rows(
    layout: ResidualStaticColumnLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build COO rows enforcing zero-mean source and receiver delays."""
    _validate_layout_for_delay_regularization(layout)
    source_cols = _coerce_1d_integer_int64(
        layout.source_delay_cols,
        name='layout.source_delay_cols',
        expected_shape=(layout.n_sources,),
    )
    receiver_cols = _coerce_1d_integer_int64(
        layout.receiver_delay_cols,
        name='layout.receiver_delay_cols',
        expected_shape=(layout.n_receivers,),
    )
    row_indices = np.concatenate(
        [
            np.zeros(layout.n_sources, dtype=np.int64),
            np.ones(layout.n_receivers, dtype=np.int64),
        ]
    )
    col_indices = np.concatenate([source_cols, receiver_cols]).astype(
        np.int64,
        copy=False,
    )
    data = np.concatenate(
        [
            np.full(layout.n_sources, 1.0 / layout.n_sources, dtype=np.float64),
            np.full(
                layout.n_receivers,
                1.0 / layout.n_receivers,
                dtype=np.float64,
            ),
        ]
    )
    rhs_s = np.zeros(2, dtype=np.float64)
    return (
        np.ascontiguousarray(row_indices, dtype=np.int64),
        np.ascontiguousarray(col_indices, dtype=np.int64),
        np.ascontiguousarray(data, dtype=np.float64),
        np.ascontiguousarray(rhs_s, dtype=np.float64),
    )


def build_delay_damping_rows(
    layout: ResidualStaticColumnLayout,
    *,
    damping_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build COO rows for Tikhonov damping on source/receiver delays."""
    _validate_layout_for_delay_regularization(layout)
    damping = _coerce_nonnegative_finite_float(
        damping_lambda,
        name='damping_lambda',
    )
    if damping == 0.0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    delay_cols = np.concatenate(
        [
            _coerce_1d_integer_int64(
                layout.source_delay_cols,
                name='layout.source_delay_cols',
                expected_shape=(layout.n_sources,),
            ),
            _coerce_1d_integer_int64(
                layout.receiver_delay_cols,
                name='layout.receiver_delay_cols',
                expected_shape=(layout.n_receivers,),
            ),
        ]
    )
    n_rows = int(delay_cols.shape[0])
    row_indices = np.arange(n_rows, dtype=np.int64)
    data = np.full(n_rows, damping, dtype=np.float64)
    rhs_s = np.zeros(n_rows, dtype=np.float64)
    return (
        np.ascontiguousarray(row_indices, dtype=np.int64),
        np.ascontiguousarray(delay_cols, dtype=np.int64),
        np.ascontiguousarray(data, dtype=np.float64),
        np.ascontiguousarray(rhs_s, dtype=np.float64),
    )


def build_stabilized_residual_static_augmented_system(
    inputs: ResidualStaticSolverInputs,
    *,
    used_mask_sorted: np.ndarray | None = None,
    options: ResidualStaticStabilizationOptions | None = None,
) -> ResidualStaticAugmentedSystem:
    """Build observation, gauge, and damping rows as one CSR system."""
    validated_options = validate_residual_static_stabilization_options(
        options or ResidualStaticStabilizationOptions()
    )
    used_mask = validate_residual_static_used_mask(inputs, used_mask_sorted)
    minimum_data = validate_minimum_residual_static_data(
        inputs,
        used_mask_sorted=used_mask,
        options=validated_options,
    )
    triplets = build_residual_static_observation_matrix_triplets(
        inputs,
        used_mask_sorted=used_mask,
    )
    observation_matrix = build_csr_matrix_from_residual_static_triplets(triplets)
    n_cols = int(observation_matrix.shape[1])

    gauge_rows, gauge_cols, gauge_data, gauge_rhs = build_zero_mean_gauge_rows(
        triplets.layout
    )
    gauge_matrix = sparse.coo_matrix(
        (gauge_data, (gauge_rows, gauge_cols)),
        shape=(2, n_cols),
        dtype=np.float64,
    ).tocsr()

    damping_rows, damping_cols, damping_data, damping_rhs = build_delay_damping_rows(
        triplets.layout,
        damping_lambda=validated_options.damping_lambda,
    )
    n_damping_rows = int(damping_rhs.shape[0])
    damping_matrix = sparse.coo_matrix(
        (damping_data, (damping_rows, damping_cols)),
        shape=(n_damping_rows, n_cols),
        dtype=np.float64,
    ).tocsr()

    matrix = sparse.vstack(
        [observation_matrix, gauge_matrix, damping_matrix],
        format='csr',
        dtype=np.float64,
    )
    rhs_s = np.ascontiguousarray(
        np.concatenate([triplets.rhs_s, gauge_rhs, damping_rhs]),
        dtype=np.float64,
    )
    n_rows = int(matrix.shape[0])
    if rhs_s.shape != (n_rows,):
        raise ValueError('augmented rhs_s shape must match augmented matrix rows')

    return ResidualStaticAugmentedSystem(
        matrix=matrix,
        rhs_s=rhs_s,
        n_observation_rows=int(observation_matrix.shape[0]),
        n_gauge_rows=2,
        n_damping_rows=n_damping_rows,
        n_rows=n_rows,
        n_cols=n_cols,
        observation_row_to_sorted_trace_index=np.ascontiguousarray(
            triplets.row_to_sorted_trace_index,
            dtype=np.int64,
        ),
        used_mask_sorted=np.ascontiguousarray(triplets.used_mask_sorted, dtype=bool),
        minimum_data=minimum_data,
    )


def build_csr_matrix_from_residual_static_triplets(
    triplets: ResidualStaticObservationMatrixTriplets,
):
    """Build a SciPy CSR matrix from residual-static COO triplets."""
    n_rows = _coerce_positive_int(triplets.n_rows, name='triplets.n_rows')
    n_cols = _coerce_positive_int(triplets.n_cols, name='triplets.n_cols')
    layout_n_cols = _coerce_positive_int(
        triplets.layout.n_model_parameters,
        name='triplets.layout.n_model_parameters',
    )
    if n_cols != layout_n_cols:
        raise ValueError('triplets.n_cols must match layout.n_model_parameters')

    row_indices = _coerce_1d_integer_int64(
        triplets.row_indices,
        name='triplets.row_indices',
    )
    col_indices = _coerce_1d_integer_int64(
        triplets.col_indices,
        name='triplets.col_indices',
    )
    data = _coerce_1d_real_numeric_float64(triplets.data, name='triplets.data')
    if row_indices.shape != col_indices.shape or row_indices.shape != data.shape:
        raise ValueError('triplet row_indices, col_indices, and data must match')
    _validate_index_range(row_indices, n_unique=n_rows, name='triplets.row_indices')
    _validate_index_range(col_indices, n_unique=n_cols, name='triplets.col_indices')
    _validate_all_finite(data, name='triplets.data')

    rhs_s = _coerce_1d_real_numeric_float64(
        triplets.rhs_s,
        name='triplets.rhs_s',
        expected_shape=(n_rows,),
    )
    _validate_all_finite(rhs_s, name='triplets.rhs_s')
    _coerce_1d_integer_int64(
        triplets.row_to_sorted_trace_index,
        name='triplets.row_to_sorted_trace_index',
        expected_shape=(n_rows,),
    )
    _coerce_1d_bool_array(
        triplets.used_mask_sorted,
        name='triplets.used_mask_sorted',
    )

    return sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_rows, n_cols),
        dtype=np.float64,
    ).tocsr()


def run_sparse_lsmr(
    matrix,
    rhs_s: np.ndarray,
    *,
    options: ResidualStaticLsmrOptions | None = None,
) -> ResidualStaticRawLsmrResult:
    """Run SciPy LSMR against a validated sparse linear system."""
    matrix_csr = _coerce_sparse_matrix_float64_csr(matrix)
    n_rows, n_cols = matrix_csr.shape
    rhs = _coerce_1d_real_numeric_float64(
        rhs_s,
        name='rhs_s',
        expected_shape=(n_rows,),
    )
    _validate_all_finite(rhs, name='rhs_s')
    validated_options = validate_lsmr_options(options or ResidualStaticLsmrOptions())

    try:
        x, istop, itn, normr, normar, norma, conda, normx = sparse_linalg.lsmr(
            matrix_csr,
            rhs,
            damp=0.0,
            atol=validated_options.atol,
            btol=validated_options.btol,
            conlim=validated_options.conlim,
            maxiter=validated_options.maxiter,
            show=False,
        )
    except Exception as exc:
        raise RuntimeError('LSMR solve failed') from exc

    parameter_vector = np.ascontiguousarray(x, dtype=np.float64)
    if parameter_vector.shape != (n_cols,):
        msg = (
            'LSMR parameter_vector shape mismatch: '
            f'expected {(n_cols,)}, got {parameter_vector.shape}'
        )
        raise ValueError(msg)
    _validate_all_finite(parameter_vector, name='parameter_vector')
    return ResidualStaticRawLsmrResult(
        parameter_vector=parameter_vector,
        diagnostics=ResidualStaticLsmrDiagnostics(
            istop=int(istop),
            itn=int(itn),
            normr=float(normr),
            normar=float(normar),
            norma=float(norma),
            conda=float(conda),
            normx=float(normx),
        ),
    )


def solve_residual_static_least_squares(
    inputs: ResidualStaticSolverInputs,
    *,
    used_mask_sorted: np.ndarray | None = None,
    options: ResidualStaticLsmrOptions | None = None,
) -> ResidualStaticLeastSquaresResult:
    """Solve preliminary ungauged residual static delays with sparse LSMR."""
    triplets = build_residual_static_observation_matrix_triplets(
        inputs,
        used_mask_sorted=used_mask_sorted,
    )
    matrix = build_csr_matrix_from_residual_static_triplets(triplets)
    raw_result = run_sparse_lsmr(matrix, triplets.rhs_s, options=options)
    parameter_parts = unpack_residual_static_parameters(
        triplets.layout,
        raw_result.parameter_vector,
    )
    model_evaluation = evaluate_residual_static_model(
        inputs,
        triplets.layout,
        raw_result.parameter_vector,
        residual_mask_sorted=triplets.used_mask_sorted,
    )
    _validate_domain_result(
        parameter_vector=raw_result.parameter_vector,
        parameter_parts=parameter_parts,
        model_evaluation=model_evaluation,
        layout=triplets.layout,
    )

    return ResidualStaticLeastSquaresResult(
        parameter_vector=raw_result.parameter_vector,
        parameter_parts=parameter_parts,
        model_evaluation=model_evaluation,
        diagnostics=raw_result.diagnostics,
        layout=triplets.layout,
        used_mask_sorted=np.ascontiguousarray(triplets.used_mask_sorted, dtype=bool),
        row_to_sorted_trace_index=np.ascontiguousarray(
            triplets.row_to_sorted_trace_index,
            dtype=np.int64,
        ),
        n_observations=int(matrix.shape[0]),
        n_model_parameters=int(matrix.shape[1]),
        rank_deficient_possible=True,
    )


def solve_residual_static_stabilized_least_squares(
    inputs: ResidualStaticSolverInputs,
    *,
    used_mask_sorted: np.ndarray | None = None,
    stabilization_options: ResidualStaticStabilizationOptions | None = None,
    lsmr_options: ResidualStaticLsmrOptions | None = None,
) -> ResidualStaticStabilizedLeastSquaresResult:
    """Solve zero-mean-gauged residual static delays with sparse LSMR."""
    validated_options = validate_residual_static_stabilization_options(
        stabilization_options or ResidualStaticStabilizationOptions()
    )
    augmented = build_stabilized_residual_static_augmented_system(
        inputs,
        used_mask_sorted=used_mask_sorted,
        options=validated_options,
    )
    raw_result = run_sparse_lsmr(
        augmented.matrix,
        augmented.rhs_s,
        options=lsmr_options,
    )
    matrix_csr = _coerce_sparse_matrix_float64_csr(augmented.matrix)
    layout = build_residual_static_column_layout(inputs)
    if layout.n_model_parameters != augmented.n_cols:
        raise ValueError('layout.n_model_parameters must match augmented.n_cols')
    parameter_parts = unpack_residual_static_parameters(
        layout,
        raw_result.parameter_vector,
    )
    model_evaluation = evaluate_residual_static_model(
        inputs,
        layout,
        raw_result.parameter_vector,
        residual_mask_sorted=augmented.used_mask_sorted,
    )
    _validate_domain_result(
        parameter_vector=raw_result.parameter_vector,
        parameter_parts=parameter_parts,
        model_evaluation=model_evaluation,
        layout=layout,
    )
    _validate_zero_mean_delay_gauge(parameter_parts)
    max_abs_estimated_delay_s = validate_residual_static_estimated_delay_limit(
        model_evaluation,
        max_abs_estimated_delay_ms=validated_options.max_abs_estimated_delay_ms,
    )

    return ResidualStaticStabilizedLeastSquaresResult(
        parameter_vector=raw_result.parameter_vector,
        parameter_parts=parameter_parts,
        model_evaluation=model_evaluation,
        diagnostics=raw_result.diagnostics,
        layout=layout,
        used_mask_sorted=np.ascontiguousarray(augmented.used_mask_sorted, dtype=bool),
        observation_row_to_sorted_trace_index=np.ascontiguousarray(
            augmented.observation_row_to_sorted_trace_index,
            dtype=np.int64,
        ),
        minimum_data=augmented.minimum_data,
        stabilization_options=validated_options,
        n_observations=augmented.n_observation_rows,
        n_model_parameters=int(matrix_csr.shape[1]),
        n_gauge_rows=augmented.n_gauge_rows,
        n_damping_rows=augmented.n_damping_rows,
        max_abs_estimated_delay_s=max_abs_estimated_delay_s,
        rank_deficient_possible=False,
    )


def validate_residual_static_estimated_delay_limit(
    evaluation: ResidualStaticModelEvaluation,
    *,
    max_abs_estimated_delay_ms: float,
) -> float:
    """Validate the estimated trace delay limit and return the max abs seconds."""
    max_delay_ms = _coerce_positive_finite_float(
        max_abs_estimated_delay_ms,
        name='max_abs_estimated_delay_ms',
    )
    estimated_delay = _coerce_1d_real_numeric_float64(
        evaluation.estimated_trace_delay_s_sorted,
        name='estimated_trace_delay_s_sorted',
    )
    if estimated_delay.size == 0:
        raise ValueError('estimated_trace_delay_s_sorted must be non-empty')
    _validate_all_finite(
        estimated_delay,
        name='estimated_trace_delay_s_sorted',
    )
    max_abs_delay_s = float(np.max(np.abs(estimated_delay)))
    if max_abs_delay_s > max_delay_ms / 1000.0:
        raise ValueError(
            'estimated_trace_delay_s_sorted exceeds max_abs_estimated_delay_ms'
        )
    return max_abs_delay_s


def _validate_domain_result(
    *,
    parameter_vector: np.ndarray,
    parameter_parts: ResidualStaticParameterParts,
    model_evaluation: ResidualStaticModelEvaluation,
    layout: ResidualStaticColumnLayout,
) -> None:
    if parameter_vector.shape != (layout.n_model_parameters,):
        raise ValueError('parameter_vector shape must match layout.n_model_parameters')
    _validate_all_finite(parameter_vector, name='parameter_vector')
    if parameter_parts.source_delay_s.shape != (layout.n_sources,):
        raise ValueError('source_delay_s shape must match layout.n_sources')
    if parameter_parts.receiver_delay_s.shape != (layout.n_receivers,):
        raise ValueError('receiver_delay_s shape must match layout.n_receivers')
    _validate_all_finite(parameter_parts.source_delay_s, name='source_delay_s')
    _validate_all_finite(parameter_parts.receiver_delay_s, name='receiver_delay_s')
    residual_mask = _coerce_1d_bool_array(
        model_evaluation.residual_valid_mask_sorted,
        name='model_evaluation.residual_valid_mask_sorted',
    )
    residual = _coerce_1d_real_numeric_float64(
        model_evaluation.residual_s_sorted,
        name='model_evaluation.residual_s_sorted',
        expected_shape=residual_mask.shape,
    )
    if np.any(~np.isfinite(residual[residual_mask])):
        raise ValueError('residual_s_sorted must be finite for used traces')


def _validate_zero_mean_delay_gauge(
    parameter_parts: ResidualStaticParameterParts,
) -> None:
    source_mean = float(np.mean(parameter_parts.source_delay_s))
    receiver_mean = float(np.mean(parameter_parts.receiver_delay_s))
    if not np.isclose(source_mean, 0.0, atol=1.0e-8, rtol=0.0):
        raise ValueError('source_delay_s must have zero mean')
    if not np.isclose(receiver_mean, 0.0, atol=1.0e-8, rtol=0.0):
        raise ValueError('receiver_delay_s must have zero mean')


def _input_n_traces(inputs: ResidualStaticSolverInputs) -> int:
    return _coerce_positive_int(inputs.n_traces, name='n_traces')


def _effective_parameter_count(layout: ResidualStaticColumnLayout) -> int:
    _validate_layout_for_delay_regularization(layout)
    if layout.moveout_model == 'linear_abs_offset':
        return 2 + layout.n_sources + layout.n_receivers - 2
    if layout.moveout_model == 'none':
        return 1 + layout.n_sources + layout.n_receivers - 2
    raise ValueError(f'unsupported moveout_model: {layout.moveout_model!r}')


def _required_abs_offset(
    inputs: ResidualStaticSolverInputs,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if inputs.abs_offset_sorted is None:
        raise ValueError('abs_offset_sorted is required for linear_abs_offset moveout')
    return _coerce_1d_real_numeric_float64(
        inputs.abs_offset_sorted,
        name='abs_offset_sorted',
        expected_shape=expected_shape,
    )


def _validate_mask_subset(
    mask: np.ndarray,
    base_mask: np.ndarray,
    *,
    mask_name: str,
    base_name: str,
) -> None:
    if mask.shape != base_mask.shape:
        raise ValueError(f'{mask_name} shape must match {base_name}')
    if np.any(mask & ~base_mask):
        raise ValueError(f'{mask_name} must be a subset of {base_name}')


def _validate_layout_for_delay_regularization(
    layout: ResidualStaticColumnLayout,
) -> None:
    _coerce_positive_int(layout.n_model_parameters, name='layout.n_model_parameters')
    _coerce_positive_int(layout.n_sources, name='layout.n_sources')
    _coerce_positive_int(layout.n_receivers, name='layout.n_receivers')
    source_delay_cols = _coerce_1d_integer_int64(
        layout.source_delay_cols,
        name='layout.source_delay_cols',
        expected_shape=(layout.n_sources,),
    )
    receiver_delay_cols = _coerce_1d_integer_int64(
        layout.receiver_delay_cols,
        name='layout.receiver_delay_cols',
        expected_shape=(layout.n_receivers,),
    )
    _validate_index_range(
        source_delay_cols,
        n_unique=layout.n_model_parameters,
        name='layout.source_delay_cols',
    )
    _validate_index_range(
        receiver_delay_cols,
        n_unique=layout.n_model_parameters,
        name='layout.receiver_delay_cols',
    )


def _coerce_sparse_matrix_float64_csr(matrix):
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


def _coerce_1d_integer_preserve_dtype(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if expected_shape is not None and arr.shape != expected_shape:
        msg = f'{name} shape mismatch: expected {expected_shape}, got {arr.shape}'
        raise ValueError(msg)
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f'{name} must contain integer values')
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{name} must contain integer values')
    return np.ascontiguousarray(arr)


def _coerce_optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _coerce_positive_int(value, name=name)


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


__all__ = [
    'ResidualStaticAugmentedSystem',
    'ResidualStaticGauge',
    'ResidualStaticLeastSquaresResult',
    'ResidualStaticLsmrDiagnostics',
    'ResidualStaticLsmrOptions',
    'ResidualStaticMinimumDataSummary',
    'ResidualStaticObservationGraphSummary',
    'ResidualStaticRawLsmrResult',
    'ResidualStaticStabilizationOptions',
    'ResidualStaticStabilizedLeastSquaresResult',
    'build_delay_damping_rows',
    'build_csr_matrix_from_residual_static_triplets',
    'build_residual_static_observation_graph_summary',
    'build_stabilized_residual_static_augmented_system',
    'build_zero_mean_gauge_rows',
    'run_sparse_lsmr',
    'solve_residual_static_least_squares',
    'solve_residual_static_stabilized_least_squares',
    'validate_minimum_residual_static_data',
    'validate_lsmr_options',
    'validate_residual_static_estimated_delay_limit',
    'validate_residual_static_stabilization_options',
    'validate_residual_static_used_mask',
]
