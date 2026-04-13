import numpy as np
from sklearn.decomposition import PCA


def compute_projection_queue(points):

    if len(points) < 3:
        return None

    points = np.array(points)

    # Step 1: estimate queue direction
    pca = PCA(n_components=2)
    pca.fit(points)

    direction = pca.components_[0]
    direction = direction / np.linalg.norm(direction)

    origin = points.mean(axis=0)

    projections = []
    queue_points = []

    for p in points:

        v = p - origin

        # projection along queue direction
        t = np.dot(v, direction)

        # closest point on queue axis
        closest = origin + t * direction

        # perpendicular distance
        perp_dist = np.linalg.norm(p - closest)

        if perp_dist < 80:  # tolerance
            projections.append((t, p))

    if len(projections) < 3:
        return None

    # Step 2: sort people along queue axis
    projections.sort(key=lambda x: x[0])

    queue_points = np.array([p for _, p in projections])

    queue_size = len(queue_points)

    queue_length = np.linalg.norm(queue_points[-1] - queue_points[0])

    return queue_points, queue_size, queue_length