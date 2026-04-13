import numpy as np
import networkx as nx
from sklearn.decomposition import PCA


def compute_graph_queue(points):

    if len(points) < 3:
        return None

    points = np.array(points)

    # -------- STEP 1: queue direction --------
    pca = PCA(n_components=2)
    pca.fit(points)

    direction = pca.components_[0]
    direction = direction / np.linalg.norm(direction)

    DIST_THRESHOLD = 200
    ANGLE_THRESHOLD = 50

    G = nx.Graph()

    for i in range(len(points)):
        G.add_node(i)

    # -------- STEP 2: build graph --------
    for i in range(len(points)):
        for j in range(i + 1, len(points)):

            p1 = points[i]
            p2 = points[j]

            vec = p2 - p1
            dist = np.linalg.norm(vec)

            if dist > DIST_THRESHOLD:
                continue

            vec_norm = vec / (np.linalg.norm(vec) + 1e-6)

            alignment = np.dot(vec_norm, direction)
            angle = np.degrees(np.arccos(np.clip(abs(alignment), -1, 1)))

            if angle < ANGLE_THRESHOLD:
                G.add_edge(i, j)

    # -------- STEP 3: largest component --------
    components = list(nx.connected_components(G))

    if not components:
        return None

    queue_nodes = max(components, key=len)

    queue_points = points[list(queue_nodes)]

    # -------- STEP 4: order queue along direction --------
    projections = queue_points @ direction
    order = np.argsort(projections)

    queue_points = queue_points[order]

    queue_size = len(queue_points)

    # -------- STEP 5: queue length --------
    queue_length = np.linalg.norm(queue_points[-1] - queue_points[0])

    return queue_points, queue_size, queue_length