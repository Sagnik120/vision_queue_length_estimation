import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def compute_hybrid_queue(points):

    if len(points) < 4:
        return None

    points = np.array(points)

    # ==============================
    # STEP 0: adaptive scale
    # ==============================
    dists = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dists.append(np.linalg.norm(points[i] - points[j]))

    scale = np.median(dists)  # robust scale

    PROJ_TOL = scale * 0.8
    EDGE_TOL = scale * 1.2
    CLEAN_TOL = scale * 0.5

    # ==============================
    # STEP 1: Projection filtering
    # ==============================
    pca = PCA(n_components=2)
    pca.fit(points)
    
    direction = pca.components_[0]
    direction = direction / np.linalg.norm(direction)

    origin = points.mean(axis=0)

    filtered = []

    for p in points:
        v = p - origin
        t = np.dot(v, direction)
        closest = origin + t * direction
        dist = np.linalg.norm(p - closest)

        if dist < PROJ_TOL:
            filtered.append(p)

    if len(filtered) < 4:
        return None

    points = np.array(filtered)

    # ==============================
    # STEP 2: kNN graph
    # ==============================
    k = min(6, max(3, len(points)//3))

    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    G = nx.Graph()

    for i in range(len(points)):
        G.add_node(i)

    for i in range(len(points)):
        for j_idx in range(1, k):
            j = indices[i][j_idx]
            dist = distances[i][j_idx]

            if dist < EDGE_TOL:
                G.add_edge(i, j, weight=dist)

    # keep largest component
    components = list(nx.connected_components(G))
    if not components:
        return None

    largest = max(components, key=len)
    G = G.subgraph(largest).copy()

    # ==============================
    # STEP 3: MST
    # ==============================
    mst = nx.minimum_spanning_tree(G, weight="weight")

    # ==============================
    # STEP 4: Longest path
    # ==============================
    longest_path = []

    for node in mst.nodes:
        lengths = nx.single_source_dijkstra_path_length(mst, node)
        farthest = max(lengths, key=lengths.get)
        path = nx.shortest_path(mst, node, farthest)

        if len(path) > len(longest_path):
            longest_path = path

    queue_points = points[list(longest_path)]

    # ==============================
    # STEP 5: Direction + sorting
    # ==============================
    direction = queue_points[-1] - queue_points[0]
    direction = direction / np.linalg.norm(direction)

    projections = queue_points @ direction
    order = np.argsort(projections)
    queue_points = queue_points[order]

    # ==============================
    # STEP 6: Final cleanup
    # ==============================
    clean = []

    for p in queue_points:
        v = p - queue_points[0]
        proj = np.dot(v, direction)
        closest = queue_points[0] + proj * direction
        dist = np.linalg.norm(p - closest)

        if dist < CLEAN_TOL:
            clean.append(p)

    queue_points = np.array(clean)

    if len(queue_points) < 3:
        return None

    # ==============================
    # STEP 7: Correct length
    # ==============================
    projections = queue_points @ direction
    queue_length = projections.max() - projections.min()

    queue_size = len(queue_points)

    return queue_points, queue_size, queue_length