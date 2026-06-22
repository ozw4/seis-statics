from __future__ import annotations

import numpy as np

from seis_statics._endpoint_sum_graph import (
    analyze_endpoint_sum_graph,
    build_endpoint_sum_gauge_matrix,
)


def test_endpoint_sum_graph_gauge_one_edge_signed_row_increases_rank() -> None:
    graph = analyze_endpoint_sum_graph(
        n_nodes=2,
        row_source_node_id=np.asarray([0], dtype=np.int64),
        row_receiver_node_id=np.asarray([1], dtype=np.int64),
    )

    gauge = build_endpoint_sum_gauge_matrix(
        graph=graph,
        n_columns=2,
        gauge_weight=1.0,
    ).toarray()

    unsigned_mean_row = np.asarray([[1.0, 1.0]], dtype=np.float64) / np.sqrt(2.0)
    observation = np.asarray([[1.0, 1.0]], dtype=np.float64)
    assert np.linalg.matrix_rank(np.vstack([observation, unsigned_mean_row])) == 1
    assert np.linalg.matrix_rank(np.vstack([observation, gauge])) == 2
    np.testing.assert_allclose(gauge, [[1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])


def test_endpoint_sum_graph_gauge_disconnected_bipartite_components_get_two_rows() -> None:
    graph = analyze_endpoint_sum_graph(
        n_nodes=4,
        row_source_node_id=np.asarray([0, 2], dtype=np.int64),
        row_receiver_node_id=np.asarray([1, 3], dtype=np.int64),
    )

    gauge = build_endpoint_sum_gauge_matrix(
        graph=graph,
        n_columns=4,
        gauge_weight=2.0,
    ).toarray()

    assert graph.n_components == 2
    np.testing.assert_array_equal(graph.gauge_required_by_component, [True, True])
    np.testing.assert_allclose(
        gauge,
        [
            [2.0 / np.sqrt(2.0), -2.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 0.0, 2.0 / np.sqrt(2.0), -2.0 / np.sqrt(2.0)],
        ],
    )


def test_endpoint_sum_graph_gauge_triangle_component_has_no_row_and_full_rank() -> None:
    graph = analyze_endpoint_sum_graph(
        n_nodes=3,
        row_source_node_id=np.asarray([0, 1, 2], dtype=np.int64),
        row_receiver_node_id=np.asarray([1, 2, 0], dtype=np.int64),
    )

    gauge = build_endpoint_sum_gauge_matrix(
        graph=graph,
        n_columns=3,
        gauge_weight=1.0,
    )
    observation = np.asarray(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    assert graph.n_components == 1
    np.testing.assert_array_equal(graph.is_bipartite_by_component, [False])
    assert gauge.shape == (0, 3)
    assert np.linalg.matrix_rank(observation) == 3


def test_endpoint_sum_graph_gauge_self_loop_component_has_no_row() -> None:
    graph = analyze_endpoint_sum_graph(
        n_nodes=1,
        row_source_node_id=np.asarray([0], dtype=np.int64),
        row_receiver_node_id=np.asarray([0], dtype=np.int64),
    )

    gauge = build_endpoint_sum_gauge_matrix(
        graph=graph,
        n_columns=1,
        gauge_weight=1.0,
    )

    np.testing.assert_array_equal(graph.is_bipartite_by_component, [False])
    assert gauge.shape == (0, 1)


def test_endpoint_sum_graph_gauge_is_deterministic_for_edge_order() -> None:
    first = analyze_endpoint_sum_graph(
        n_nodes=4,
        row_source_node_id=np.asarray([2, 0], dtype=np.int64),
        row_receiver_node_id=np.asarray([3, 1], dtype=np.int64),
    )
    second = analyze_endpoint_sum_graph(
        n_nodes=4,
        row_source_node_id=np.asarray([0, 2], dtype=np.int64),
        row_receiver_node_id=np.asarray([1, 3], dtype=np.int64),
    )

    np.testing.assert_array_equal(first.component_id_by_node, second.component_id_by_node)
    np.testing.assert_array_equal(
        first.signed_partition_by_node,
        second.signed_partition_by_node,
    )
