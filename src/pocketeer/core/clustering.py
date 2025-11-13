"""Graph operations and clustering of alpha-spheres into pockets."""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from .types import AlphaSphere


def build_sphere_graph(
    spheres: list[AlphaSphere],
    distance_threshold: float,
) -> dict[int, set[int]]:
    """Build adjacency graph of spheres within distance threshold.

    Args:
        spheres: list of alpha-spheres
        distance_threshold: maximum distance for edge

    Returns:
        Adjacency graph mapping list index -> set of neighbor list indices
        (uses list positions, not sphere_id values)
    """
    if not spheres:
        return {}

    # Extract centers
    centers = np.array([s.center for s in spheres])

    # Build KD-tree for efficient neighbor search
    tree = cKDTree(centers)

    # Find neighbors within threshold
    graph: dict[int, set[int]] = defaultdict(set)

    for idx, center in enumerate(centers):
        neighbor_indices = tree.query_ball_point(center, distance_threshold)
        for neighbor_idx in neighbor_indices:
            if neighbor_idx != idx:
                graph[idx].add(neighbor_idx)
                graph[neighbor_idx].add(idx)

    return dict(graph)


def connected_components(graph: dict[int, set[int]]) -> list[set[int]]:
    """Find connected components in undirected graph using DFS.

    Args:
        graph: adjacency list

    Returns:
        List of connected components (each is a set of node indices)
    """
    visited = set()
    components = []

    def dfs(node: int, component: set[int]) -> None:
        """Depth-first search to explore component."""
        visited.add(node)
        component.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, component)

    # Visit all nodes
    all_nodes = set(graph.keys()) | {n for neighbors in graph.values() for n in neighbors}

    for node in all_nodes:
        if node not in visited:
            component: set[int] = set()
            dfs(node, component)
            components.append(component)

    return components


def cluster_spheres(
    spheres: list[AlphaSphere],
    *,
    merge_distance: float = 1.2,
    min_spheres: int = 35,
) -> list[list[AlphaSphere]]:
    """Cluster alpha-spheres into pockets using graph connectivity.

    Args:
        spheres: list of alpha-spheres (typically apolar only)
        merge_distance: Distance threshold for merging nearby sphere clusters (Å)
        min_spheres: Minimum number of spheres per pocket cluster

    Returns:
        List of clusters, each a list of AlphaSphere objects
    """
    if not spheres:
        return []

    # Build proximity graph
    graph = build_sphere_graph(spheres, merge_distance)

    # Find connected components
    clusters = connected_components(graph)

    # Filter by minimum size
    valid_clusters = [c for c in clusters if len(c) >= min_spheres]

    # Convert index clusters to sphere clusters
    sphere_clusters = [[spheres[idx] for idx in cluster] for cluster in valid_clusters]

    return sphere_clusters
