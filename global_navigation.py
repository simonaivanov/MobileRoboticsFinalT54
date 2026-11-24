"""Global navigation: shortest obstacle-free path for the Thymio (visibility graph + A*).

Frame: origin top-left, x right, y down, units mm
"""

import math
from heapq import heappush, heappop
from shapely.geometry import box, Point, LineString
from shapely.ops import unary_union, nearest_points

EPS = 1e-6  # mm, numerical tolerance for geometric tests
ROBOT_W = 110            # mm, a mesurer sur le Thymio
MARGIN = 20                    # mm, marge de securite
INFLATION = ROBOT_W /2 + MARGIN

# 1. Free space
def build_free_space(obstacles, arena_w, arena_h, inflation=INFLATION):
    """Space reachable by the robot centre: shrunk arena minus inflated obstacles."""
    inflated_obstacles = [obs.buffer(inflation, join_style="mitre") for obs in obstacles]
    shrunk_arena = box(0, 0, arena_w, arena_h).buffer(-inflation, join_style="mitre")
    free_space = shrunk_arena.difference(unary_union(inflated_obstacles))
    return free_space, inflated_obstacles, shrunk_arena


# 3. Start point
def get_start(free_space, thymio_pos):
    """Thymio position, or the closest free point if it lies in a forbidden zone."""
    if free_space.buffer(EPS).covers(Point(thymio_pos)):
        return thymio_pos
    p = nearest_points(free_space, Point(thymio_pos))[0]
    return (p.x, p.y)


# 4. Graph nodes
def free_corners(free_space):
    """All corners of the free space (inflated obstacles + shrunk arena)."""
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


# 5. Visibility graph
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


# 6. Graph search (adapted from grid_search, exercise session 5)
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


# 7. Main function
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
    free_space, inflated_obstacles, shrunk_arena = build_free_space(obstacles, arena_w, arena_h, INFLATION)
    result.update(inflated_obstacles=inflated_obstacles, free_space=free_space,
                  shrunk_arena=shrunk_arena)

    # Goal check
    if not free_space.buffer(EPS).covers(Point(goal)):
        result["error"] = "Goal too close to an obstacle or to the arena border"
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

import matplotlib.pyplot as plt


def afficher_etapes(obstacles, thymio_pos, goal, result, arena_w, arena_h, robot_w, cell=None):
    """Affiche les 4 etapes de la global navigation cote a cote."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    titres = ["1. Carte reelle", "2. Obstacles gonfles", "3. Graphe de visibilite", "4. Plus court chemin"]

    for etape, ax in enumerate(axes, start=1):
        # Fond commun : arene, grille, obstacles reels
        ax.plot([0, arena_w, arena_w, 0, 0], [0, 0, arena_h, arena_h, 0], "k-", lw=2)
        if cell:
            for i in range(1, round(arena_w / cell)):
                ax.plot([i * cell] * 2, [0, arena_h], color="lightgray", lw=1)
            for i in range(1, round(arena_h / cell)):
                ax.plot([0, arena_w], [i * cell] * 2, color="lightgray", lw=1)

        # Etapes 2 a 4 : zones interdites
        if etape >= 2:
            for obs in result["inflated_obstacles"]:
                x, y = obs.exterior.xy
                ax.fill(x, y, color="tab:orange", alpha=0.3)
            x, y = result["shrunk_arena"].exterior.xy
            ax.plot(x, y, "--", color="tab:orange")

        for obs in obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, color="black")

        # Etape 3 : noeuds et aretes
        if etape == 3 and "graph" in result:
            for a, voisins in result["graph"].items():
                for b, _ in voisins:
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:blue", lw=0.6, alpha=0.6)
            xs, ys = zip(*result["nodes"])
            ax.plot(xs, ys, ".", color="tab:blue", ms=7)

        # Etape 4 : noeuds explores par A* + chemin
        if etape == 4:
            if "explored" in result:
                xs, ys = zip(*result["explored"])
                ax.plot(xs, ys, "o", color="tab:purple", ms=6, alpha=0.5, label="noeuds explores")
            if result["waypoints"]:
                wx, wy = zip(*result["waypoints"])
                ax.plot(wx, wy, "-o", color="tab:red", lw=2.5, ms=6,
                        label=f"chemin ({result['length']:.0f} mm)")
            else:
                ax.text(arena_w / 2, arena_h / 2, result["error"], ha="center", color="red", wrap=True)

        # Thymio : disque a sa vraie taille a l'etape 1, point ensuite
        if etape == 1:
            ax.add_patch(plt.Circle(thymio_pos, robot_w / 2, color="tab:green", alpha=0.6))
        else:
            ax.plot(*thymio_pos, "o", color="tab:green", ms=10)
        ax.plot(*goal, "s", color="tab:blue", ms=12)

        ax.set_title(titres[etape - 1])
        ax.set_aspect("equal")
        ax.set_xlim(-30, arena_w + 30)
        ax.set_ylim(-30, arena_h + 30)
        if etape == 4:
            ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    return fig