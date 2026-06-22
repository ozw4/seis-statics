"""Endpoint-sum graph analysis for signless-incidence node inversions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from seis_statics._validation import (
    coerce_1d_bool_array as _coerce_1d_bool_array,
    coerce_1d_integer_int64 as _common_coerce_1d_integer_int64,
    coerce_finite_float as _coerce_finite_float,
    coerce_positive_int as _coerce_positive_int,
)


@dataclass(frozen=True)
class EndpointSumGraphSummary:
    """Connected components and signless-incidence gauge metadata.

    A connected component of endpoint-sum rows has one node gauge only when its
    graph is bipartite. The signed partition vector spans that null direction:
    adding ``+c`` on one partition and ``-c`` on the other leaves every
    endpoint-sum observation unchanged. Odd cycles and self-loops remove that
    null direction and therefore do not receive a gauge row.
    """

    component_id_by_node: np.ndarray
    n_components: int
    is_bipartite_by_component: np.ndarray
    signed_partition_by_node: np.ndarray
    gauge_required_by_component: np.ndarray


def analyze_endpoint_sum_graph(
    *,
    n_nodes: int,
    row_source_node_id: np.ndarray,
    row_receiver_node_id: np.ndarray,
) -> EndpointSumGraphSummary:
    """Analyze the used endpoint graph for component-aware node gauges."""
    node_count = _coerce_positive_int(n_nodes, name='n_nodes')
    source = _coerce_node_ids(
        row_source_node_id,
        n_nodes=node_count,
        name='row_source_node_id',
    )
    receiver = _coerce_node_ids(
        row_receiver_node_id,
        n_nodes=node_count,
        name='row_receiver_node_id',
    )
    if source.shape != receiver.shape:
        raise ValueError('row_source_node_id and row_receiver_node_id must match')

    adjacency: list[set[int]] = [set() for _ in range(node_count)]
    has_self_loop = np.zeros(node_count, dtype=bool)
    for left, right in zip(source.tolist(), receiver.tolist(), strict=True):
        left_node = int(left)
        right_node = int(right)
        if left_node == right_node:
            has_self_loop[left_node] = True
            continue
        adjacency[left_node].add(right_node)
        adjacency[right_node].add(left_node)

    raw_component = np.full(node_count, -1, dtype=np.int64)
    color = np.zeros(node_count, dtype=np.int8)
    component_nodes: list[list[int]] = []
    component_bipartite: list[bool] = []

    for start in range(node_count):
        if raw_component[start] >= 0:
            continue
        component_index = len(component_nodes)
        queue: deque[int] = deque([start])
        raw_component[start] = component_index
        color[start] = 1
        nodes: list[int] = []
        bipartite = not bool(has_self_loop[start])

        while queue:
            node = queue.popleft()
            nodes.append(node)
            if has_self_loop[node]:
                bipartite = False
            for neighbor in sorted(adjacency[node]):
                if raw_component[neighbor] < 0:
                    raw_component[neighbor] = component_index
                    color[neighbor] = np.int8(-int(color[node]))
                    queue.append(neighbor)
                elif int(color[neighbor]) == int(color[node]):
                    bipartite = False

        component_nodes.append(nodes)
        component_bipartite.append(bipartite)

    order = sorted(
        range(len(component_nodes)),
        key=lambda index: min(component_nodes[index]),
    )
    remap = {old: new for new, old in enumerate(order)}
    component_id = np.asarray(
        [remap[int(value)] for value in raw_component.tolist()],
        dtype=np.int64,
    )
    is_bipartite = np.asarray(
        [component_bipartite[old] for old in order],
        dtype=bool,
    )
    signed_partition = np.ascontiguousarray(color.astype(np.int8, copy=False))
    gauge_required = np.ascontiguousarray(is_bipartite.copy(), dtype=bool)

    return EndpointSumGraphSummary(
        component_id_by_node=np.ascontiguousarray(component_id, dtype=np.int64),
        n_components=int(len(component_nodes)),
        is_bipartite_by_component=np.ascontiguousarray(is_bipartite, dtype=bool),
        signed_partition_by_node=signed_partition,
        gauge_required_by_component=gauge_required,
    )


def build_endpoint_sum_gauge_matrix(
    *,
    graph: EndpointSumGraphSummary,
    n_columns: int,
    gauge_weight: float,
    node_column_offset: int = 0,
) -> sparse.csr_matrix:
    """Build normalized signed gauge rows for bipartite endpoint components."""
    column_count = _coerce_positive_int(n_columns, name='n_columns')
    offset = _coerce_nonnegative_int(node_column_offset, name='node_column_offset')
    component_id = _coerce_component_ids(graph.component_id_by_node, graph=graph)
    signed_partition = np.asarray(graph.signed_partition_by_node)
    if signed_partition.shape != component_id.shape:
        raise ValueError('graph.signed_partition_by_node shape mismatch')
    if offset + component_id.size > column_count:
        raise ValueError('node columns exceed n_columns')
    weight = _coerce_finite_float(gauge_weight, name='gauge_weight')
    if weight <= 0.0:
        raise ValueError('gauge_weight must be greater than 0')

    required_components = np.flatnonzero(graph.gauge_required_by_component).astype(
        np.int64,
        copy=False,
    )
    if required_components.size == 0:
        return sparse.csr_matrix((0, column_count), dtype=np.float64)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for row, component in enumerate(required_components.tolist()):
        nodes = np.flatnonzero(component_id == int(component)).astype(
            np.int64,
            copy=False,
        )
        if nodes.size == 0:
            raise ValueError('graph component ids must be contiguous')
        scale = weight / np.sqrt(float(nodes.size))
        row_parts.append(np.full(nodes.shape, row, dtype=np.int64))
        col_parts.append(np.ascontiguousarray(nodes + offset, dtype=np.int64))
        data_parts.append(
            np.ascontiguousarray(
                signed_partition[nodes].astype(np.float64, copy=False) * scale,
                dtype=np.float64,
            )
        )

    matrix = sparse.coo_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(int(required_components.size), column_count),
        dtype=np.float64,
    ).tocsr()
    matrix.sort_indices()
    return matrix


def _build_endpoint_sum_prediction_identifiable_mask(
    *,
    graph: EndpointSumGraphSummary,
    source_node_id: np.ndarray,
    receiver_node_id: np.ndarray,
    node_supported_mask: np.ndarray,
) -> np.ndarray:
    """Return traces whose endpoint sum is invariant over graph null vectors."""
    component_id = _coerce_component_ids(graph.component_id_by_node, graph=graph)
    n_nodes = int(component_id.shape[0])
    source = _coerce_node_ids(
        source_node_id,
        n_nodes=n_nodes,
        name='source_node_id',
    )
    receiver = _coerce_node_ids(
        receiver_node_id,
        n_nodes=n_nodes,
        name='receiver_node_id',
    )
    if source.shape != receiver.shape:
        raise ValueError('source_node_id and receiver_node_id must match')

    node_supported = _coerce_1d_bool_array(
        node_supported_mask,
        name='node_supported_mask',
        expected_shape=(n_nodes,),
    )
    signed_partition = _coerce_signed_partition(
        graph.signed_partition_by_node,
        expected_shape=(n_nodes,),
    )
    gauge_required = np.asarray(graph.gauge_required_by_component, dtype=bool)

    source_component = component_id[source]
    receiver_component = component_id[receiver]
    source_gauge_required = gauge_required[source_component]
    receiver_gauge_required = gauge_required[receiver_component]

    endpoint_supported = node_supported[source] & node_supported[receiver]
    same_component = source_component == receiver_component
    same_component_identifiable = (
        ~source_gauge_required
        | (signed_partition[source] + signed_partition[receiver] == 0)
    )
    different_component_identifiable = (
        ~source_gauge_required & ~receiver_gauge_required
    )
    identifiable = endpoint_supported & np.where(
        same_component,
        same_component_identifiable,
        different_component_identifiable,
    )
    return np.ascontiguousarray(identifiable, dtype=bool)


def _coerce_node_ids(values: np.ndarray, *, n_nodes: int, name: str) -> np.ndarray:
    arr = _common_coerce_1d_integer_int64(
        values,
        name=name,
        allow_integer_like_float=False,
    )
    if np.any(arr < 0):
        raise ValueError(f'{name} must be greater than or equal to 0')
    if np.any(arr >= n_nodes):
        raise ValueError(f'{name} contains values outside 0..{n_nodes - 1}')
    return arr


def _coerce_component_ids(
    values: np.ndarray,
    *,
    graph: EndpointSumGraphSummary,
) -> np.ndarray:
    component_id = _common_coerce_1d_integer_int64(
        values,
        name='graph.component_id_by_node',
        allow_integer_like_float=False,
    )
    n_components = _coerce_positive_int(
        graph.n_components,
        name='graph.n_components',
    )
    if component_id.size == 0:
        raise ValueError('graph.component_id_by_node must be non-empty')
    if np.any(component_id < 0) or np.any(component_id >= n_components):
        raise ValueError('graph.component_id_by_node contains invalid component ids')
    expected = np.arange(n_components, dtype=np.int64)
    if not np.array_equal(np.unique(component_id), expected):
        raise ValueError('graph.component_id_by_node must be 0-based and contiguous')
    for name, values in (
        ('graph.is_bipartite_by_component', graph.is_bipartite_by_component),
        ('graph.gauge_required_by_component', graph.gauge_required_by_component),
    ):
        arr = np.asarray(values)
        if arr.shape != (n_components,):
            raise ValueError(f'{name} shape mismatch')
        if not np.issubdtype(arr.dtype, np.bool_):
            raise ValueError(f'{name} must have bool dtype')
    return np.ascontiguousarray(component_id, dtype=np.int64)


def _coerce_signed_partition(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    arr = _common_coerce_1d_integer_int64(
        values,
        name='graph.signed_partition_by_node',
        allow_integer_like_float=False,
        expected_shape=expected_shape,
    )
    if np.any((arr != -1) & (arr != 1)):
        raise ValueError('graph.signed_partition_by_node must contain only -1 or 1')
    return np.ascontiguousarray(arr, dtype=np.int8)


def _coerce_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    out = int(value)
    if out < 0:
        raise ValueError(f'{name} must be greater than or equal to 0')
    return out


__all__ = [
    'EndpointSumGraphSummary',
    'analyze_endpoint_sum_graph',
    'build_endpoint_sum_gauge_matrix',
]
