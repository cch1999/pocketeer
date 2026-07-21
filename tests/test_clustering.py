"""Tests for alpha-sphere graph clustering."""

from pocketeer.core.clustering import connected_components


def test_connected_components_handles_large_graph():
    """Large components must not be limited by Python's recursion depth."""
    node_count = 5_000
    graph = {
        node: {neighbor for neighbor in (node - 1, node + 1) if 0 <= neighbor < node_count}
        for node in range(node_count)
    }

    components = connected_components(graph)

    assert components == [set(range(node_count))]
