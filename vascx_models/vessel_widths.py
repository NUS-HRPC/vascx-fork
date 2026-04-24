import logging
import math
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize as skimage_skeletonize

from .config import OverlayCircle

logger = logging.getLogger(__name__)

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


def _skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a one-pixel-wide skeleton."""
    if binary_mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask")
    return skimage_skeletonize(binary_mask.astype(bool))


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    visited = np.zeros_like(mask, dtype=bool)
    components: List[np.ndarray] = []

    ys, xs = np.nonzero(mask)
    for seed_y, seed_x in zip(ys, xs):
        if visited[seed_y, seed_x]:
            continue

        stack = [(int(seed_y), int(seed_x))]
        visited[seed_y, seed_x] = True
        coords: List[Tuple[int, int]] = []

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


def _local_skeleton_points(
    skeleton: np.ndarray,
    point_xy: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    x, y = point_xy
    height, width = skeleton.shape
    x_min = max(0, int(np.floor(x - radius_px)))
    x_max = min(width - 1, int(np.ceil(x + radius_px)))
    y_min = max(0, int(np.floor(y - radius_px)))
    y_max = min(height - 1, int(np.ceil(y + radius_px)))

    window = skeleton[y_min : y_max + 1, x_min : x_max + 1]
    ys, xs = np.nonzero(window)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=float)

    xs = xs.astype(float) + float(x_min)
    ys = ys.astype(float) + float(y_min)
    deltas = np.column_stack((xs - x, ys - y))
    keep = np.sum(deltas * deltas, axis=1) <= radius_px * radius_px
    return np.column_stack((xs[keep], ys[keep]))


def _estimate_tangent(points_xy: np.ndarray) -> Optional[np.ndarray]:
    if len(points_xy) < 2:
        return None

    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0:
        return None
    return tangent / norm


def _sample_mask(mask: np.ndarray, point_xy: np.ndarray) -> float:
    x, y = point_xy
    height, width = mask.shape
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    dx = x - x0
    dy = y - y0

    v00 = float(mask[y0, x0])
    v10 = float(mask[y0, x1])
    v01 = float(mask[y1, x0])
    v11 = float(mask[y1, x1])

    top = v00 * (1.0 - dx) + v10 * dx
    bottom = v01 * (1.0 - dx) + v11 * dx
    return top * (1.0 - dy) + bottom * dy


def _trace_boundary_distance(
    vessel_mask: np.ndarray,
    point_xy: np.ndarray,
    direction_xy: np.ndarray,
    step_px: float,
) -> float:
    max_distance = float(np.hypot(*vessel_mask.shape)) + 2.0
    current_value = _sample_mask(vessel_mask, point_xy)
    if current_value < 0.5:
        return float("nan")

    t_inside = 0.0
    steps = int(np.ceil(max_distance / step_px))
    for step in range(1, steps + 1):
        t_outside = step * step_px
        sample_point = point_xy + direction_xy * t_outside
        if _sample_mask(vessel_mask, sample_point) < 0.5:
            low = t_inside
            high = t_outside
            for _ in range(12):
                mid = (low + high) / 2.0
                mid_point = point_xy + direction_xy * mid
                if _sample_mask(vessel_mask, mid_point) >= 0.5:
                    low = mid
                else:
                    high = mid
            return low
        t_inside = t_outside

    return float("nan")


def _boundary_point(
    center_xy: np.ndarray,
    direction_xy: np.ndarray,
    distance_px: float,
) -> np.ndarray:
    return center_xy + direction_xy * distance_px


def _measure_width_along_normal(
    vessel_mask: np.ndarray,
    center_xy: np.ndarray,
    tangent_xy: np.ndarray,
    step_px: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    normal_xy = np.array([-tangent_xy[1], tangent_xy[0]], dtype=float)
    norm = float(np.linalg.norm(normal_xy))
    if norm == 0.0:
        nan_point = np.array([float("nan"), float("nan")], dtype=float)
        return float("nan"), nan_point, nan_point
    normal_xy /= norm

    positive = _trace_boundary_distance(vessel_mask, center_xy, normal_xy, step_px)
    negative = _trace_boundary_distance(vessel_mask, center_xy, -normal_xy, step_px)
    if np.isnan(positive) or np.isnan(negative):
        nan_point = np.array([float("nan"), float("nan")], dtype=float)
        return float("nan"), nan_point, nan_point

    start_xy = _boundary_point(center_xy, -normal_xy, negative)
    end_xy = _boundary_point(center_xy, normal_xy, positive)
    return positive + negative, start_xy, end_xy


def measure_vessel_width_at_coordinate(
    vessel_mask: np.ndarray,
    point_xy: np.ndarray,
    skeleton: Optional[np.ndarray] = None,
    tangent_window_px: float = 10.0,
    measurement_step_px: float = 0.25,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Measure vessel width at an arbitrary coordinate using the local skeleton tangent."""
    skeleton = _skeletonize(vessel_mask) if skeleton is None else skeleton
    local_points = _local_skeleton_points(skeleton, point_xy, tangent_window_px)
    tangent_xy = _estimate_tangent(local_points)
    if tangent_xy is None:
        nan_point = np.array([float("nan"), float("nan")], dtype=float)
        return float("nan"), nan_point, nan_point

    return _measure_width_along_normal(
        vessel_mask=vessel_mask,
        center_xy=point_xy,
        tangent_xy=tangent_xy,
        step_px=measurement_step_px,
    )


def _component_neighbor_map(component: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    points = {tuple(int(value) for value in coord) for coord in component.tolist()}
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in points:
        adjacent: list[tuple[int, int]] = []
        for dy, dx in _NEIGHBORS_8:
            candidate = (y + dy, x + dx)
            if candidate in points:
                if dy != 0 and dx != 0 and (
                    (y + dy, x) in points or (y, x + dx) in points
                ):
                    continue
                adjacent.append(candidate)
        neighbors[(y, x)] = adjacent
    return neighbors


def _boundary_roles_for_component(
    component: np.ndarray,
    distances: np.ndarray,
    inner_radius_px: float,
    outer_radius_px: float,
    boundary_tolerance_px: float,
) -> dict[tuple[int, int], str]:
    candidates: dict[str, list[tuple[float, tuple[int, int]]]] = {"inner": [], "outer": []}
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


def _active_subgraph(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    active_nodes: set[tuple[int, int]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    return {
        node: [neighbor for neighbor in neighbors[node] if neighbor in active_nodes]
        for node in active_nodes
    }


def _prune_to_inner_outer_nodes(
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
        active_degree = sum(1 for neighbor in neighbors[node] if neighbor in active_nodes)
        if active_degree > 1:
            continue

        active_nodes.remove(node)
        for neighbor in neighbors[node]:
            if neighbor not in active_nodes or neighbor in terminal_nodes:
                continue
            neighbor_degree = sum(
                1 for next_neighbor in neighbors[neighbor] if next_neighbor in active_nodes
            )
            if neighbor_degree <= 1:
                queue.append(neighbor)

    return active_nodes


def _connected_node_groups(
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


def _trace_segments_between_key_nodes(
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    key_nodes: set[tuple[int, int]],
    boundary_roles: dict[tuple[int, int], str],
) -> list[np.ndarray]:
    key_graph = {
        node: [neighbor for neighbor in graph[node] if neighbor in key_nodes]
        for node in key_nodes
    }
    key_groups = _connected_node_groups(key_graph)
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
                candidates = [neighbor for neighbor in graph[current] if neighbor != previous]
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


def _inner_outer_branch_segments(
    component: np.ndarray,
    distances: np.ndarray,
    inner_radius_px: float,
    outer_radius_px: float,
    boundary_tolerance_px: float,
) -> list[np.ndarray]:
    """Return inner-side trunk segments from components with inner-to-outer traces."""
    neighbors = _component_neighbor_map(component)
    boundary_roles = _boundary_roles_for_component(
        component=component,
        distances=distances,
        inner_radius_px=inner_radius_px,
        outer_radius_px=outer_radius_px,
        boundary_tolerance_px=boundary_tolerance_px,
    )
    if "inner" not in boundary_roles.values() or "outer" not in boundary_roles.values():
        return []

    active_nodes = _prune_to_inner_outer_nodes(neighbors, boundary_roles)
    active_graph = _active_subgraph(neighbors, active_nodes)
    segments: list[np.ndarray] = []

    for group in _connected_node_groups(active_graph):
        group_roles = {
            role for node, role in boundary_roles.items() if node in group
        }
        if group_roles != {"inner", "outer"}:
            continue

        group_graph = _active_subgraph(active_graph, group)
        key_nodes = {
            node
            for node, adjacent in group_graph.items()
            if node in boundary_roles
            or len(adjacent) != 2
        }
        if len(key_nodes) < 2:
            continue

        for segment in _trace_segments_between_key_nodes(group_graph, key_nodes, boundary_roles):
            endpoint_roles = {
                boundary_roles.get((int(segment[0, 0]), int(segment[0, 1]))),
                boundary_roles.get((int(segment[-1, 0]), int(segment[-1, 1]))),
            }
            if "inner" not in endpoint_roles:
                continue
            if distances[int(segment[0, 0]), int(segment[0, 1])] > distances[
                int(segment[-1, 0]), int(segment[-1, 1])
            ]:
                segment = segment[::-1]
            segments.append(segment)

    return segments


def _path_cumulative_lengths(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0:
        return np.empty(0, dtype=float)
    if len(path_xy) == 1:
        return np.zeros(1, dtype=float)

    deltas = np.diff(path_xy, axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _interpolate_path_point(
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
    return path_xy[lower_index] + fraction * (path_xy[upper_index] - path_xy[lower_index])


def _typed_vessel_masks(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the vessel mask into artery and vein masks using the AV segmentation."""
    artery_mask = vessel_mask & np.isin(av_mask, (1, 3))
    vein_mask = vessel_mask & np.isin(av_mask, (2, 3))
    return artery_mask, vein_mask


def _path_records_for_image(
    image_id: str,
    vessel_mask: np.ndarray,
    vessel_type: str,
    disc_center_xy: np.ndarray,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    disc_radius_px: float,
    samples_per_connection: int,
    boundary_tolerance_px: float,
    tangent_window_px: float,
    measurement_step_px: float,
) -> List[dict]:
    if not np.any(vessel_mask):
        return []

    if samples_per_connection <= 0:
        raise ValueError("samples_per_connection must be positive")

    inner_radius_px = float(disc_radius_px * inner_circle.diameter)
    outer_radius_px = float(disc_radius_px * outer_circle.diameter)
    if outer_radius_px <= inner_radius_px:
        raise ValueError("outer_circle must have a larger radius than inner_circle")

    skeleton = _skeletonize(vessel_mask)
    if not np.any(skeleton):
        return []

    yy, xx = np.indices(vessel_mask.shape, dtype=float)
    distances = np.hypot(xx - disc_center_xy[0], yy - disc_center_xy[1])
    annulus_mask = skeleton & (distances >= inner_radius_px) & (distances <= outer_radius_px)
    components = _connected_components(annulus_mask)

    records: List[dict] = []
    connection_index = 0
    for component in components:
        branch_segments_yx = _inner_outer_branch_segments(
            component=component,
            distances=distances,
            inner_radius_px=inner_radius_px,
            outer_radius_px=outer_radius_px,
            boundary_tolerance_px=boundary_tolerance_px,
        )
        if not branch_segments_yx:
            logger.debug(
                "Skipping %s annulus component because it has no inner-to-outer trace",
                image_id,
            )
            continue

        for segment_yx in branch_segments_yx:
            path_xy = segment_yx[:, ::-1]

            cumulative_lengths = _path_cumulative_lengths(path_xy)
            if len(cumulative_lengths) == 0 or cumulative_lengths[-1] <= 0.0:
                logger.debug(
                    "Skipping %s annulus segment because its path length is zero",
                    image_id,
                )
                continue

            segment_skeleton = np.zeros_like(skeleton, dtype=bool)
            segment_rows = segment_yx[:, 0].astype(int)
            segment_cols = segment_yx[:, 1].astype(int)
            segment_skeleton[segment_rows, segment_cols] = True

            component_records: List[dict] = []
            connection_index += 1
            for sample_index in range(1, samples_per_connection + 1):
                target_fraction = sample_index / (samples_per_connection + 1)
                center_xy = _interpolate_path_point(
                    path_xy=path_xy,
                    cumulative_lengths=cumulative_lengths,
                    target_length=float(cumulative_lengths[-1] * target_fraction),
                )
                width_value, start_xy, end_xy = measure_vessel_width_at_coordinate(
                    vessel_mask=vessel_mask,
                    point_xy=center_xy,
                    skeleton=segment_skeleton,
                    tangent_window_px=tangent_window_px,
                    measurement_step_px=measurement_step_px,
                )
                if np.isnan(width_value):
                    # TODO: Recover from local tangent/width failures by re-sampling nearby
                    # skeleton points.
                    component_records = []
                    logger.debug(
                        "Skipping %s connection %d because width measurement failed at sample %d",
                        image_id,
                        connection_index,
                        sample_index,
                    )
                    break

                component_records.append(
                    {
                        "image_id": image_id,
                        "inner_circle": inner_circle.name,
                        "outer_circle": outer_circle.name,
                        "inner_circle_radius_px": inner_radius_px,
                        "outer_circle_radius_px": outer_radius_px,
                        "connection_index": connection_index,
                        "sample_index": sample_index,
                        "x": float(center_xy[0]),
                        "y": float(center_xy[1]),
                        "width_px": float(width_value),
                        "x_start": float(start_xy[0]),
                        "y_start": float(start_xy[1]),
                        "x_end": float(end_xy[0]),
                        "y_end": float(end_xy[1]),
                        "vessel_type": vessel_type,
                    }
                )

            if len(component_records) != samples_per_connection:
                continue
            records.extend(component_records)

    return records


def resolve_vessel_width_circle_pair(
    circles: Sequence[OverlayCircle],
    inner_circle_name: str | None = None,
    outer_circle_name: str | None = None,
) -> tuple[OverlayCircle, OverlayCircle]:
    """Select the circle pair used for between-circle vessel width sampling."""
    if len(circles) < 2:
        raise ValueError("At least two overlay circles are required for vessel width sampling")

    circles_by_name = {circle.name: circle for circle in circles}
    if inner_circle_name is not None:
        if inner_circle_name not in circles_by_name:
            raise ValueError(f"Unknown inner circle '{inner_circle_name}'")
        inner_circle = circles_by_name[inner_circle_name]
    else:
        inner_circle = min(circles, key=lambda circle: (circle.diameter, circle.name))

    if outer_circle_name is not None:
        if outer_circle_name not in circles_by_name:
            raise ValueError(f"Unknown outer circle '{outer_circle_name}'")
        outer_circle = circles_by_name[outer_circle_name]
    else:
        remaining_circles = [circle for circle in circles if circle.name != inner_circle.name]
        larger_circles = [
            circle for circle in remaining_circles if circle.diameter > inner_circle.diameter
        ]
        candidates = larger_circles if larger_circles else remaining_circles
        outer_circle = min(candidates, key=lambda circle: (circle.diameter, circle.name))

    if outer_circle.diameter <= inner_circle.diameter:
        raise ValueError("outer_circle must have a larger diameter than inner_circle")
    return inner_circle, outer_circle


def measure_vessel_widths_between_disc_circle_pair(
    vessels_dir: Path,
    av_dir: Path,
    disc_geometry_path: Path,
    inner_circle: OverlayCircle,
    outer_circle: OverlayCircle,
    output_path: Optional[Path] = None,
    samples_per_connection: int = 5,
    boundary_tolerance_px: float = 1.5,
    tangent_window_px: float = 10.0,
    measurement_step_px: float = 0.25,
) -> pd.DataFrame:
    """Measure artery and vein widths separately at interior points between two circles."""
    if not disc_geometry_path.exists():
        raise FileNotFoundError(f"Disc geometry file not found: {disc_geometry_path}")
    if not vessels_dir.exists():
        raise FileNotFoundError(f"Vessels directory not found: {vessels_dir}")
    if not av_dir.exists():
        raise FileNotFoundError(f"AV directory not found: {av_dir}")

    df_geometry = pd.read_csv(disc_geometry_path, index_col=0)
    records: List[dict] = []

    for image_id, row in df_geometry.iterrows():
        if pd.isna(row["x_disc_center"]) or pd.isna(row["y_disc_center"]) or pd.isna(
            row["disc_radius_px"]
        ):
            logger.warning("Skipping %s because disc geometry is missing", image_id)
            continue

        vessel_path = vessels_dir / f"{image_id}.png"
        av_path = av_dir / f"{image_id}.png"
        if not vessel_path.exists() or not av_path.exists():
            logger.warning("Skipping %s because vessel or AV mask is missing", image_id)
            continue

        vessel_mask = np.array(Image.open(vessel_path)) > 0
        av_mask = np.array(Image.open(av_path))
        artery_mask, vein_mask = _typed_vessel_masks(vessel_mask, av_mask)
        image_records: List[dict] = []
        image_records.extend(
            _path_records_for_image(
                image_id=image_id,
                vessel_mask=artery_mask,
                vessel_type="artery",
                disc_center_xy=np.array([row["x_disc_center"], row["y_disc_center"]], dtype=float),
                inner_circle=inner_circle,
                outer_circle=outer_circle,
                disc_radius_px=float(row["disc_radius_px"]),
                samples_per_connection=samples_per_connection,
                boundary_tolerance_px=boundary_tolerance_px,
                tangent_window_px=tangent_window_px,
                measurement_step_px=measurement_step_px,
            )
        )
        image_records.extend(
            _path_records_for_image(
                image_id=image_id,
                vessel_mask=vein_mask,
                vessel_type="vein",
                disc_center_xy=np.array([row["x_disc_center"], row["y_disc_center"]], dtype=float),
                inner_circle=inner_circle,
                outer_circle=outer_circle,
                disc_radius_px=float(row["disc_radius_px"]),
                samples_per_connection=samples_per_connection,
                boundary_tolerance_px=boundary_tolerance_px,
                tangent_window_px=tangent_window_px,
                measurement_step_px=measurement_step_px,
            )
        )
        records.extend(image_records)

    df_widths = pd.DataFrame.from_records(
        records,
        columns=[
            "image_id",
            "inner_circle",
            "outer_circle",
            "inner_circle_radius_px",
            "outer_circle_radius_px",
            "connection_index",
            "sample_index",
            "x",
            "y",
            "width_px",
            "x_start",
            "y_start",
            "x_end",
            "y_end",
            "vessel_type",
        ],
    )
    if output_path is not None:
        df_widths.to_csv(output_path, index=False)
        logger.info("Vessel path width measurements saved to %s", output_path)
    return df_widths


def revised_vessel_equivalent(
    diameters: Iterable[float],
    vessel_type: str,
    n_largest: int = 6,
    return_rounds: bool = False,
):
    """
    Compute revised CRAE-6 or CRVE-6 using the Knudtson revised formula.

    Parameters
    ----------
    diameters:
        Vessel diameters in consistent units: pixels, microns, etc.
    vessel_type:
        Accepts 'artery'|'arteriole' or 'vein'|'venule'.
    n_largest:
        Number of largest vessels to use. Standard revised method uses 6.
    return_rounds:
        If True, also return intermediate rounds for inspection.

    Returns
    -------
    float
        Final equivalent vessel diameter.
    """

    coeffs = {
        "arteriole": 0.88,
        "venule": 0.95,
    }

    # Accept synonyms
    if vessel_type in ("artery", "arteriole"):
        vt = "arteriole"
    elif vessel_type in ("vein", "venule"):
        vt = "venule"
    else:
        raise ValueError("vessel_type must be 'arteriole'/'artery' or 'venule'/'vein'.")

    coeff = coeffs[vt]

    values = sorted([float(x) for x in diameters if float(x) > 0.0], reverse=True)[:n_largest]

    if len(values) < 2:
        raise ValueError("Need at least two valid vessel diameters.")

    rounds: list[list[float]] = [values.copy()]

    while len(values) > 1:
        values = sorted(values, reverse=True)

        next_values: list[float] = []

        # Pair largest with smallest.
        while len(values) >= 2:
            largest = values.pop(0)
            smallest = values.pop(-1)

            combined = coeff * math.sqrt(largest**2 + smallest**2)
            next_values.append(combined)

        # If odd count remains, carry the middle value forward unchanged.
        if values:
            next_values.append(values[0])

        values = next_values
        rounds.append(values.copy())

    result = float(values[0])

    if return_rounds:
        return result, rounds

    return result


def compute_revised_crx_from_widths(
    df_widths: pd.DataFrame,
    n_largest: int = 6,
    return_rounds: bool = False,
):
    """Aggregate per-connection mean widths and compute revised CRAE/CRVE.

    Parameters
    ----------
    df_widths:
        DataFrame produced by `measure_vessel_widths_between_disc_circle_pair` containing
        per-sample `width_px` values and `connection_index` identifiers.
    n_largest:
        Number of largest vessel averages to consider (default 6).
    return_rounds:
        If True, also return the intermediate rounds from the revised algorithm
        for each image/vessel_type as a mapping.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] or tuple[pd.DataFrame, pd.DataFrame, dict]
        Returns `(df_connections, df_equivalents)` or
        `(df_connections, df_equivalents, rounds_map)` when `return_rounds` is True.
        `df_connections` contains the per-connection mean widths and selection flags.
    """

    cols_conn = [
        "image_id",
        "vessel_type",
        "connection_index",
        "vessel_id",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
        "mean_width_px",
        "n_samples",
        "selected_for_equivalent",
    ]
    cols_equiv = [
        "image_id",
        "metric",
        "vessel_type",
        "requested_n_largest",
        "n_vessels_available",
        "n_vessels_used",
        "vessel_ids_used",
        "mean_widths_used_px",
        "equivalent_px",
    ]

    if df_widths.empty:
        df_conn_empty = pd.DataFrame(columns=cols_conn)
        df_equiv_empty = pd.DataFrame(columns=cols_equiv)
        if return_rounds:
            return df_conn_empty, df_equiv_empty, {}
        return df_conn_empty, df_equiv_empty

    group_cols = [
        "image_id",
        "vessel_type",
        "connection_index",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
    ]

    df_conn = (
        df_widths.groupby(group_cols, dropna=False)
        .agg(mean_width_px=("width_px", "mean"), n_samples=("width_px", "count"))
        .reset_index()
    )
    df_conn["vessel_id"] = df_conn.apply(
        lambda row: f"{row['vessel_type']}_{int(row['connection_index'])}",
        axis=1,
    )
    df_conn["selected_for_equivalent"] = False
    df_conn = df_conn[cols_conn]

    results: list[dict] = []
    rounds_map: dict[tuple[str, str], list[list[float]]] = {}

    for (image_id, vessel_type), group in df_conn.groupby(["image_id", "vessel_type"]):
        top = group.sort_values(
            ["mean_width_px", "connection_index"],
            ascending=[False, True],
        ).head(n_largest)
        df_conn.loc[top.index, "selected_for_equivalent"] = True
        diameters = top["mean_width_px"].tolist()
        metric = "CRAE" if vessel_type == "artery" else "CRVE"
        result = {
            "image_id": image_id,
            "metric": metric,
            "vessel_type": vessel_type,
            "requested_n_largest": int(n_largest),
            "n_vessels_available": int(len(group)),
            "n_vessels_used": int(len(top)),
            "vessel_ids_used": ";".join(str(value) for value in top["vessel_id"]),
            "mean_widths_used_px": ";".join(f"{float(value):.6g}" for value in diameters),
            "equivalent_px": float("nan"),
        }
        try:
            if return_rounds:
                eq, rounds = revised_vessel_equivalent(
                    diameters,
                    vessel_type,
                    n_largest=n_largest,
                    return_rounds=True,
                )
                rounds_map[(image_id, vessel_type)] = rounds
            else:
                eq = revised_vessel_equivalent(diameters, vessel_type, n_largest=n_largest)
            result["equivalent_px"] = float(eq)
        except ValueError:
            pass
        results.append(result)

    df_equiv = pd.DataFrame.from_records(results, columns=cols_equiv)

    if return_rounds:
        return df_conn, df_equiv, rounds_map

    return df_conn, df_equiv


def select_vessel_width_measurements_for_equivalents(
    df_widths: pd.DataFrame,
    df_connections: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-sample width rows whose vessels were selected for CRAE/CRVE."""
    if df_widths.empty or df_connections.empty:
        return df_widths.iloc[0:0].copy()

    selected = df_connections[df_connections["selected_for_equivalent"]]
    if selected.empty:
        return df_widths.iloc[0:0].copy()

    merge_cols = [
        "image_id",
        "vessel_type",
        "connection_index",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
    ]
    return df_widths.merge(selected[merge_cols], on=merge_cols, how="inner")
