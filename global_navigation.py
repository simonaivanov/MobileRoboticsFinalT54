"""Global navigation: shortest obstacle-free path for the Thymio (visibility graph + A*).

Frame: origin top-left, x right, y down, units mm
"""

import math
import cv2
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from shapely.geometry import box, Point, LineString
from shapely.ops import unary_union, nearest_points

EPS = 1e-6                        # mm, numerical tolerance for geometric tests
ROBOT_W = 110                     # mm, Thymio width (to be measured)
MARGIN = 2                        # mm, safety margin
INFLATION = ROBOT_W / 2 + MARGIN  # mm, obstacle inflation
ARENA_MARGIN = 100                # mm, the robot may leave the grid by this much
SAMPLE_STEP = 30                  # mm, spacing of the sampled points along the path


# 1. Free space
def build_free_space(obstacles, arena_w, arena_h, inflation=INFLATION):
    """Space reachable by the robot centre: inflated arena minus inflated obstacles."""
    inflated_obstacles = [obs.buffer(inflation, join_style="mitre") for obs in obstacles]
    inflated_arena = box(0, 0, arena_w, arena_h).buffer(ARENA_MARGIN, join_style="mitre")
    free_space = inflated_arena.difference(unary_union(inflated_obstacles))
    return free_space, inflated_obstacles, inflated_arena


# 2. Start point
def get_start(free_space, thymio_pos):
    """Thymio position, or the closest free point if it lies in a forbidden zone."""
    if free_space.buffer(EPS).covers(Point(thymio_pos)):
        return thymio_pos
    p = nearest_points(free_space, Point(thymio_pos))[0]
    return (p.x, p.y)


# 3. Graph nodes
def free_corners(free_space):
    """All corners of the free space (inflated obstacles + inflated arena)."""
    # Free space is one polygon, or several if obstacles split the arena
    if free_space.geom_type == "Polygon":
        free_polygons = [free_space]
    else:
        free_polygons = list(free_space.geoms)

    corners = []
    for free_polygon in free_polygons:
        # Outer boundary + boundary of each hole
        polygon_boundaries = [free_polygon.exterior] + list(free_polygon.interiors)
        for boundary in polygon_boundaries:
            corners.extend(boundary.coords[:-1])  # last point repeats the first
    return corners


def get_nodes(free_space, start, goal):
    """Nodes = start + goal + free-space corners."""
    corners = [c for c in free_corners(free_space) if c not in (start, goal)]
    return [start, goal] + corners


# 4. Visibility graph
def build_visibility_graph(nodes, free_space):
    """Link two nodes if the segment between them stays in free space.

    Returns graph[node] = [(neighbour, length), ...]
    """
    free_space_tol = free_space.buffer(EPS)  # accept segments grazing corners or edges
    graph = {n: [] for n in nodes}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:  # each pair once
            if free_space_tol.covers(LineString([a, b])):
                d = math.dist(a, b)
                graph[a].append((b, d))
                graph[b].append((a, d))
    return graph


# 5. Graph search (adapted from grid_search, exercise session 5)
def heuristic(a, b, algo="A*"):
    """Euclidean distance for A*, 0 for Dijkstra."""
    return math.dist(a, b) if algo == "A*" else 0


def a_star(graph, S, G, algo="A*"):
    """Shortest path from S to G. Returns (path, length, explored), path=None if unreachable."""
    came_from = {}
    g_costs = {S: 0}
    explored = set()
    open_set = [(heuristic(S, G, algo), 0, S)]  # (f, g, node)

    while open_set:
        _, current_g, current = heappop(open_set)
        if current in explored:  # outdated queue entry
            continue
        explored.add(current)

        if current == G:  # rebuild path
            path = [current]
            while current != S:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, current_g, explored

        for neighbour, length in graph[current]:
            tentative_g = current_g + length
            if neighbour not in g_costs or tentative_g < g_costs[neighbour]:
                g_costs[neighbour] = tentative_g
                came_from[neighbour] = current
                heappush(open_set, (tentative_g + heuristic(neighbour, G, algo),
                                    tentative_g, neighbour))

    return None, None, explored  # goal unreachable


# 6. Main function
def plan_path(obstacles, thymio_pos, goal, arena_w, arena_h, algo="A*"):
    """Shortest path from thymio_pos to goal avoiding obstacles.

    obstacles  : list of shapely polygons (mm)
    thymio_pos : (x, y) in mm
    goal       : (x, y) in mm
    Returns a dict: "waypoints", "length", "error" + intermediate steps for plotting.
    """
    thymio_pos = tuple(map(float, thymio_pos[:2]))   # keep (x, y), drop theta
    goal = tuple(map(float, goal))
    result = {"waypoints": None, "length": None, "error": None}

    # Inflation and free space
    free_space, inflated_obstacles, inflated_arena = build_free_space(obstacles, arena_w, arena_h)
    result.update(inflated_obstacles=inflated_obstacles, free_space=free_space,
                  inflated_arena=inflated_arena)

    # Goal check
    if not free_space.buffer(EPS).covers(Point(goal)):
        result["error"] = "Goal inside an inflated obstacle or outside the arena"
        return result

    # Start, nodes, graph
    start = get_start(free_space, thymio_pos)
    nodes = get_nodes(free_space, start, goal)
    graph = build_visibility_graph(nodes, free_space)
    result.update(start=start, nodes=nodes, graph=graph)

    # Search
    path, length, explored = a_star(graph, start, goal, algo)
    result["explored"] = explored
    if path is None:
        result["error"] = "Goal not reachable from the Thymio position"
        return result

    # If the start was moved, prepend the escape segment
    if start != thymio_pos:
        path = [thymio_pos] + path
        length += math.dist(thymio_pos, start)

    result["waypoints"] = path
    result["length"] = length
    return result


# 7. Path densification
def densify_path(waypoints, step=SAMPLE_STEP):
    """Split each segment of the path into pieces of about `step` mm.

    waypoints : list of (x, y) in mm
    Returns a list of (x, y): the first point and every original waypoint
    are kept, with intermediate points added in between (no duplicates).
    """
    if not waypoints:
        return []

    points = [tuple(waypoints[0])]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        d = math.dist(a, b)
        if d == 0:
            continue  # skip repeated waypoints
        n = math.ceil(d / step)  # number of pieces, each <= step
        for i in range(1, n):
            t = i / n
            points.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        points.append(tuple(b))  # keep the original waypoint exactly
    return points


# 8. Planning helper for the main loop
def compute_path(obstacles, thymio, goal, arena_w, arena_h, step=SAMPLE_STEP):
    """Plan the path from the Thymio pose and sample it.

    thymio : (x, y) or (x, y, theta) in mm (theta is ignored)
    goal   : (x, y) in mm, or None if there is no move to play
    Returns (waypoints, samples), or (None, None) if no path was found.
    """
    if goal is None:
        print("No move to play")
        return None, None
    result = plan_path(obstacles, thymio, goal, arena_w, arena_h)
    if result["error"]:
        print(result["error"])
        return None, None
    waypoints = result["waypoints"]
    samples = densify_path(waypoints, step)
    print(f"Path: {result['length']:.0f} mm, {len(waypoints)} waypoints, "
          f"{len(samples)} sampled points")
    return waypoints, samples


# 9. Plotting (report / debug)
def plot_steps(obstacles, thymio_pos, goal, result, arena_w, arena_h, robot_w, cell=None):
    """Plot the 4 steps of global navigation side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    titles = ["1. Real map", "2. Inflated obstacles", "3. Visibility graph", "4. Shortest path"]

    for step, ax in enumerate(axes, start=1):
        # Common background: arena, grid, real obstacles
        ax.plot([0, arena_w, arena_w, 0, 0], [0, 0, arena_h, arena_h, 0], "k-", lw=2)
        if cell:
            for i in range(1, round(arena_w / cell)):
                ax.plot([i * cell] * 2, [0, arena_h], color="lightgray", lw=1)
            for i in range(1, round(arena_h / cell)):
                ax.plot([0, arena_w], [i * cell] * 2, color="lightgray", lw=1)

        # Steps 2 to 4: forbidden zones
        if step >= 2:
            for obs in result["inflated_obstacles"]:
                x, y = obs.exterior.xy
                ax.fill(x, y, color="tab:orange", alpha=0.3)
            x, y = result["inflated_arena"].exterior.xy
            ax.plot(x, y, "--", color="tab:orange")

        for obs in obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, color="navy")

        # Step 3: nodes and edges
        if step == 3 and "graph" in result:
            for a, neighbours in result["graph"].items():
                for b, _ in neighbours:
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:blue", lw=0.6, alpha=0.6)
            xs, ys = zip(*result["nodes"])
            ax.plot(xs, ys, ".", color="tab:blue", ms=7)

        # Step 4: nodes explored by A* + path
        if step == 4:
            if "explored" in result:
                xs, ys = zip(*result["explored"])
                ax.plot(xs, ys, "o", color="tab:purple", ms=6, alpha=0.5, label="explored nodes")
            if result["waypoints"]:
                wx, wy = zip(*result["waypoints"])
                ax.plot(wx, wy, "-o", color="tab:red", lw=2.5, ms=6,
                        label=f"path ({result['length']:.0f} mm)")
            else:
                ax.text(arena_w / 2, arena_h / 2, result["error"], ha="center", color="red", wrap=True)

        # Thymio: real-size disc at step 1, dot afterwards
        if step == 1:
            ax.add_patch(plt.Circle(thymio_pos, robot_w / 2, color="tab:green", alpha=0.6))
        else:
            ax.plot(*thymio_pos, "o", color="tab:green", ms=10)
        ax.plot(*goal, "s", color="tab:cyan", ms=12)

        pad = ARENA_MARGIN + 20  # show the whole inflated arena
        ax.set_title(titles[step - 1])
        ax.set_aspect("equal")
        ax.set_xlim(-pad, arena_w + pad)
        ax.set_ylim(-pad, arena_h + pad)
        if step == 4:
            ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    return fig


# 10. Drawing on the camera image (OpenCV, BGR colours)
# vision imports this module, so project_to_camera is imported inside the
# functions to avoid a circular import.
def draw_obstacles(frame, obstacles, inflated_obstacles, to_camera):
    """Draw inflated obstacles (orange, transparent) and real obstacles (blue outline)."""
    from vision import project_to_camera
    overlay = frame.copy()
    for poly in inflated_obstacles:
        pts = project_to_camera(list(poly.exterior.coords), to_camera)
        cv2.fillPoly(overlay, [pts], (0, 165, 255))
        cv2.polylines(frame, [pts], True, (0, 165, 255), 2)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    for poly in obstacles:
        pts = project_to_camera(list(poly.exterior.coords), to_camera)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
    return frame


def draw_path(frame, waypoints, samples, to_camera):
    """Draw the path: waypoints (large, red) and sampled points (small, cyan)."""
    from vision import project_to_camera
    if not waypoints:
        return frame
    wp_px = project_to_camera(waypoints, to_camera)
    cv2.polylines(frame, [wp_px], False, (0, 0, 255), 2)
    for p in project_to_camera(samples, to_camera):
        cv2.circle(frame, tuple(map(int, p)), 3, (255, 255, 0), -1)
    for p in wp_px:
        cv2.circle(frame, tuple(map(int, p)), 8, (0, 0, 255), 2)
    return frame


def draw_legend(frame, top=60):
    """Small legend in the top left corner, starting at y = top (below the Thymio label)."""
    items = [("real obstacles", (255, 0, 0)), ("inflated obstacles", (0, 165, 255)),
             ("waypoints", (0, 0, 255)), ("sampled points", (255, 255, 0))]
    for i, (text, color) in enumerate(items):
        y = top + 25 * i
        cv2.circle(frame, (20, y - 5), 7, color, -1)
        cv2.putText(frame, text, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
