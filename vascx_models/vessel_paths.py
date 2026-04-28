from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from skimage.morphology import skeletonize as skimage_skeletonize

_NEIGHBORS_8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class VesselPath:
    connection_index: int
    path_xy: np.ndarray
    path_yx: np.ndarray
    skeleton: np.ndarray


def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a one-pixel-wide skeleton."""
    if binary_mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask")
    return skimage_skeletonize(binary_mask.astype(bool))


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    ys, xs = np.nonzero(mask)
    for seed_y, seed_x in zip(ys, xs):
        if visited[seed_y, seed_x]:
            continue

        stack = [(int(seed_y), int(seed_x))]
        visited[seed_y, seed_x] = True
        coords: list[tuple[int, int]] = []

        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for dy, dx in _NEIGHBORS_8:
                ny = y + dy
                nx = x + dx
                if ny < 0 or nx < 0 or ny >= mask.shape[0] or nx >= mask.shape[1]:
                    continue
                if not mask[ny, nx] or visited[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        components.append(np.asarray(coords, dtype=np.int32))

    return components


def component_neighbor_map(
    component: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    points = {tuple(int(value) for value in coord) for coord in component.tolist()}
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in points:
        adjacent: list[tuple[int, int]] = []
        for dy, dx in _NEIGHBORS_8:
            candidate = (y + dy, x + dx)
            if candidate in points:
                if (
                    dy != 0
                    and dx != 0
                    and ((y + dy, x) in points or (y, x + dx) in points)
                ):
                    continue
                adjacent.append(candidate)
        neighbors[(y, x)] = adjacent
    return neighbors


def boundary_roles_for_component(
    component: np.ndarray,
    distances: np.ndarray,
    inner_radius_px: float,
    outer_radius_px: float,
    boundary_tolerance_px: float,
) -> dict[tuple[int, int], str]:
    candidates: dict[str, list[tuple[float, tuple[int, int]]]] = {
        "inner": [],
        "outer": [],
    }
    roles: dict[tuple[int, int], str] = {}
    for y_value, x_value in component:
        y = int(y_value)
        x = int(x_value)
        inner_delta = abs(float(distances[y, x]) - inner_radius_px)
        outer_delta = abs(float(distances[y, x]) - outer_radius_px)
        is_inner = inner_delta <= boundary_tolerance_px
        is_outer = outer_delta <= boundary_tolerance_px
        if is_inner == is_outer:
            continue
        if is_inner:
            candidates["inner"].append((inner_delta, (y, x)))
        else:
            candidates["outer"].append((outer_delta, (y, x)))

    for role, role_candidates in candidates.items():
        candidate_deltas = {node: delta for delta, node in role_candidates}
        remaining = set(candidate_deltas)
        while remaining:
            seed = remaining.pop()
            group = [seed]
            stack = [seed]
            while stack:
                node_y, node_x = stack.pop()
                for dy, dx in _NEIGHBORS_8:
                    neighbor = (node_y + dy, node_x + dx)
                    if neighbor not in remaining:
                        continue
                    remaining.remove(neighbor)
                    group.append(neighbor)
                    stack.append(neighbor)

            representative = min(group, key=lambda node: (candidate_deltas[node], node))
            roles[representative] = role
    return roles


def active_subgraph(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    active_nodes: set[tuple[int, int]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    return {
        node: [neighbor for neighbor in neighbors[node] if neighbor in active_nodes]
        for node in active_nodes
    }


def prune_to_inner_outer_nodes(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    boundary_roles: dict[tuple[int, int], str],
) -> set[tuple[int, int]]:
    """Remove dead-end skeleton pixels that are not needed for inner-to-outer traces."""
    active_nodes = set(neighbors)
    terminal_nodes = set(boundary_roles)
    queue = deque(
        node
        for node in active_nodes
        if node not in terminal_nodes
        and sum(1 for neighbor in neighbors[node] if neighbor in active_nodes) <= 1
    )

    while queue:
        node = queue.popleft()
        if node not in active_nodes or node in terminal_nodes:
            continue
        active_degree = sum(
            1 for neighbor in neighbors[node] if neighbor in active_nodes
        )
        if active_degree > 1:
            continue

        active_nodes.remove(node)
        for neighbor in neighbors[node]:
            if neighbor not in active_nodes or neighbor in terminal_nodes:
                continue
            neighbor_degree = sum(
                1
                for next_neighbor in neighbors[neighbor]
                if next_neighbor in active_nodes
            )
            if neighbor_degree <= 1:
                queue.append(neighbor)

    return active_nodes


def connected_node_groups(
    graph: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[set[tuple[int, int]]]:
    remaining = set(graph)
    groups: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        stack = [seed]
        while stack:
            node = stack.pop()
            for neighbor in graph[node]:
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                group.add(neighbor)
                stack.append(neighbor)
        groups.append(group)
    return groups


def trace_segments_between_key_nodes(
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    key_nodes: set[tuple[int, int]],
    boundary_roles: dict[tuple[int, int], str],
) -> list[np.ndarray]:
    key_graph = {
        node: [neighbor for neighbor in graph[node] if neighbor in key_nodes]
        for node in key_nodes
    }
    key_groups = connected_node_groups(key_graph)
    key_group_by_node = {
        node: group_index
        for group_index, group in enumerate(key_groups)
        for node in group
    }
    boundary_groups = {
        group_index
        for group_index, group in enumerate(key_groups)
        if any(node in boundary_roles for node in group)
    }

    segments: list[np.ndarray] = []
    visited_edges: set[frozenset[tuple[int, int]]] = set()

    for start in sorted(key_nodes):
        for first_neighbor in sorted(graph[start]):
            if (
                first_neighbor in key_nodes
                and key_group_by_node[first_neighbor] == key_group_by_node[start]
            ):
                continue
            first_edge = frozenset((start, first_neighbor))
            if first_edge in visited_edges:
                continue

            start_group = key_group_by_node[start]
            path: list[tuple[int, int]] = []
            if start_group in boundary_groups:
                path.append(start)
            if first_neighbor not in key_nodes:
                path.append(first_neighbor)
            visited_edges.add(first_edge)
            previous = start
            current = first_neighbor

            while current not in key_nodes:
                candidates = [
                    neighbor for neighbor in graph[current] if neighbor != previous
                ]
                if len(candidates) != 1:
                    break
                next_node = candidates[0]
                next_edge = frozenset((current, next_node))
                if next_edge in visited_edges:
                    break
                visited_edges.add(next_edge)
                path.append(next_node)
                previous, current = current, next_node

            if current in key_nodes and key_group_by_node[current] in boundary_groups:
                path.append(current)

            if len(path) >= 2:
                segments.append(np.asarray(path, dtype=float))

    return segments


def inner_outer_branch_segments(
    component: np.ndarray,
    distances: np.ndarray,
    inner_radius_px: float,
    outer_radius_px: float,
    boundary_tolerance_px: float,
) -> list[np.ndarray]:
    """Return inner-side trunk segments from components with inner-to-outer traces."""
    neighbors = component_neighbor_map(component)
    boundary_roles = boundary_roles_for_component(
        component=component,
        distances=distances,
        inner_radius_px=inner_radius_px,
        outer_radius_px=outer_radius_px,
        boundary_tolerance_px=boundary_tolerance_px,
    )
    if "inner" not in boundary_roles.values() or "outer" not in boundary_roles.values():
        return []

    active_nodes = prune_to_inner_outer_nodes(neighbors, boundary_roles)
    active_graph = active_subgraph(neighbors, active_nodes)
    segments: list[np.ndarray] = []

    for group in connected_node_groups(active_graph):
        group_roles = {role for node, role in boundary_roles.items() if node in group}
        if group_roles != {"inner", "outer"}:
            continue

        group_graph = active_subgraph(active_graph, group)
        key_nodes = {
            node
            for node, adjacent in group_graph.items()
            if node in boundary_roles or len(adjacent) != 2
        }
        if len(key_nodes) < 2:
            continue

        for segment in trace_segments_between_key_nodes(
            group_graph, key_nodes, boundary_roles
        ):
            endpoint_roles = {
                boundary_roles.get((int(segment[0, 0]), int(segment[0, 1]))),
                boundary_roles.get((int(segment[-1, 0]), int(segment[-1, 1]))),
            }
            if "inner" not in endpoint_roles:
                continue
            if (
                distances[int(segment[0, 0]), int(segment[0, 1])]
                > distances[int(segment[-1, 0]), int(segment[-1, 1])]
            ):
                segment = segment[::-1]
            segments.append(segment)

    return segments


def path_cumulative_lengths(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0:
        return np.empty(0, dtype=float)
    if len(path_xy) == 1:
        return np.zeros(1, dtype=float)

    deltas = np.diff(path_xy, axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def interpolate_path_point(
    path_xy: np.ndarray,
    cumulative_lengths: np.ndarray,
    target_length: float,
) -> np.ndarray:
    if len(path_xy) == 1:
        return path_xy[0].copy()

    clamped_length = min(max(float(target_length), 0.0), float(cumulative_lengths[-1]))
    upper_index = int(np.searchsorted(cumulative_lengths, clamped_length, side="right"))
    if upper_index <= 0:
        return path_xy[0].copy()
    if upper_index >= len(path_xy):
        return path_xy[-1].copy()

    lower_index = upper_index - 1
    lower_length = float(cumulative_lengths[lower_index])
    upper_length = float(cumulative_lengths[upper_index])
    if upper_length <= lower_length:
        return path_xy[lower_index].copy()

    fraction = (clamped_length - lower_length) / (upper_length - lower_length)
    return path_xy[lower_index] + fraction * (
        path_xy[upper_index] - path_xy[lower_index]
    )


def trace_vessel_paths_between_disc_circle_pair(
    vessel_mask: np.ndarray,
    disc_center_xy: np.ndarray,
    inner_radius_px: float,
    outer_radius_px: float,
    boundary_tolerance_px: float,
) -> list[VesselPath]:
    if not np.any(vessel_mask):
        return []
    if outer_radius_px <= inner_radius_px:
        raise ValueError("outer_radius_px must be larger than inner_radius_px")

    skeleton = skeletonize(vessel_mask)
    if not np.any(skeleton):
        return []

    yy, xx = np.indices(vessel_mask.shape, dtype=float)
    distances = np.hypot(xx - disc_center_xy[0], yy - disc_center_xy[1])
    annulus_mask = (
        skeleton & (distances >= inner_radius_px) & (distances <= outer_radius_px)
    )
    components = connected_components(annulus_mask)

    vessel_paths: list[VesselPath] = []
    connection_index = 0
    for component in components:
        branch_segments_yx = inner_outer_branch_segments(
            component=component,
            distances=distances,
            inner_radius_px=inner_radius_px,
            outer_radius_px=outer_radius_px,
            boundary_tolerance_px=boundary_tolerance_px,
        )
        for path_yx in branch_segments_yx:
            if len(path_yx) > 1:
                keep = np.concatenate(
                    ([True], np.any(np.diff(path_yx, axis=0) != 0, axis=1))
                )
                path_yx = path_yx[keep]
            path_xy = path_yx[:, ::-1]
            cumulative_lengths = path_cumulative_lengths(path_xy)
            if len(cumulative_lengths) == 0 or cumulative_lengths[-1] <= 0.0:
                continue

            path_skeleton = np.zeros_like(skeleton, dtype=bool)
            segment_rows = path_yx[:, 0].astype(int)
            segment_cols = path_yx[:, 1].astype(int)
            path_skeleton[segment_rows, segment_cols] = True

            connection_index += 1
            vessel_paths.append(
                VesselPath(
                    connection_index=connection_index,
                    path_xy=path_xy,
                    path_yx=path_yx,
                    skeleton=path_skeleton,
                )
            )

    return vessel_paths


def _key_nodes_for_graph(
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    boundary_roles: dict[tuple[int, int], str],
) -> set[tuple[int, int]]:
    return {
        node
        for node, adjacent in graph.items()
        if node in boundary_roles or len(adjacent) != 2
    }


def _rooted_key_node_segments(
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    root: tuple[int, int],
    boundary_roles: dict[tuple[int, int], str],
) -> list[np.ndarray]:
    """Split a single-inner vessel tree into 1-to-1 key-node segments.

    Returns an empty list when the group is ambiguous, such as when it contains
    cycles or merge patterns that prevent a unique outward traversal.
    """
    edge_count = sum(len(adjacent) for adjacent in graph.values()) // 2
    if edge_count != len(graph) - 1:
        return []

    key_nodes = _key_nodes_for_graph(graph, boundary_roles)
    parent_by_node: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
    traversal_stack = [root]

    while traversal_stack:
        node = traversal_stack.pop()
        parent = parent_by_node[node]
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            if neighbor in parent_by_node:
                return []
            parent_by_node[neighbor] = node
            traversal_stack.append(neighbor)

    child_by_node: dict[tuple[int, int], list[tuple[int, int]]] = {
        node: [] for node in graph
    }
    for node, parent in parent_by_node.items():
        if parent is not None:
            child_by_node[parent].append(node)
    for children in child_by_node.values():
        children.sort()

    segments: list[np.ndarray] = []
    key_stack = [root]
    while key_stack:
        start = key_stack.pop()
        for child in reversed(child_by_node[start]):
            path = [start, child]
            current = child
            while current not in key_nodes:
                children = child_by_node[current]
                if len(children) != 1:
                    return []
                current = children[0]
                path.append(current)

            segments.append(np.asarray(path, dtype=float))
            key_stack.append(current)

    return segments


def trace_vessel_tortuosity_paths_between_disc_circle_pair(
    vessel_mask: np.ndarray,
    disc_center_xy: np.ndarray,
    inner_radius_px: float,
    outer_radius_px: float,
    boundary_tolerance_px: float,
) -> list[VesselPath]:
    """Return 1-to-1 tortuosity segments within the annulus.

    Components are only measured when they form a single-inner rooted tree. This
    keeps direct 1-to-1 segments and splits valid 1-to-many bifurcations into
    base and child branches, while discarding ambiguous n-to-1 and n-to-n
    patterns.
    """
    if not np.any(vessel_mask):
        return []
    if outer_radius_px <= inner_radius_px:
        raise ValueError("outer_radius_px must be larger than inner_radius_px")

    skeleton = skeletonize(vessel_mask)
    if not np.any(skeleton):
        return []

    yy, xx = np.indices(vessel_mask.shape, dtype=float)
    distances = np.hypot(xx - disc_center_xy[0], yy - disc_center_xy[1])
    annulus_mask = (
        skeleton & (distances >= inner_radius_px) & (distances <= outer_radius_px)
    )
    components = connected_components(annulus_mask)

    vessel_paths: list[VesselPath] = []
    connection_index = 0
    for component in components:
        neighbors = component_neighbor_map(component)
        boundary_roles = boundary_roles_for_component(
            component=component,
            distances=distances,
            inner_radius_px=inner_radius_px,
            outer_radius_px=outer_radius_px,
            boundary_tolerance_px=boundary_tolerance_px,
        )
        inner_nodes = {node for node, role in boundary_roles.items() if role == "inner"}
        outer_nodes = {node for node, role in boundary_roles.items() if role == "outer"}
        if not inner_nodes or not outer_nodes:
            continue

        active_nodes = prune_to_inner_outer_nodes(neighbors, boundary_roles)
        active_graph = active_subgraph(neighbors, active_nodes)
        for group in connected_node_groups(active_graph):
            group_inner_nodes = sorted(inner_nodes & group)
            group_outer_nodes = outer_nodes & group
            if len(group_inner_nodes) != 1 or not group_outer_nodes:
                continue

            group_graph = active_subgraph(active_graph, group)
            for path_yx in _rooted_key_node_segments(
                group_graph,
                root=group_inner_nodes[0],
                boundary_roles=boundary_roles,
            ):
                if len(path_yx) > 1:
                    keep = np.concatenate(
                        ([True], np.any(np.diff(path_yx, axis=0) != 0, axis=1))
                    )
                    path_yx = path_yx[keep]
                path_xy = path_yx[:, ::-1]
                cumulative_lengths = path_cumulative_lengths(path_xy)
                if len(cumulative_lengths) == 0 or cumulative_lengths[-1] <= 0.0:
                    continue

                path_skeleton = np.zeros_like(skeleton, dtype=bool)
                segment_rows = path_yx[:, 0].astype(int)
                segment_cols = path_yx[:, 1].astype(int)
                path_skeleton[segment_rows, segment_cols] = True

                connection_index += 1
                vessel_paths.append(
                    VesselPath(
                        connection_index=connection_index,
                        path_xy=path_xy,
                        path_yx=path_yx,
                        skeleton=path_skeleton,
                    )
                )

    return vessel_paths
