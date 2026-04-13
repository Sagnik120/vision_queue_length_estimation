import numpy as np
import networkx as nx

KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12


def compute_facing_angle(keypoints):
    """
    Compute facing direction as angle in radians from shoulder or hip vector.
    Returns float or None if no reliable joints detected.
    """
    ls = keypoints[KP_LEFT_SHOULDER]
    rs = keypoints[KP_RIGHT_SHOULDER]
    if ls[2] > 0.3 and rs[2] > 0.3:
        return float(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]) + np.pi / 2)

    lh = keypoints[KP_LEFT_HIP]
    rh = keypoints[KP_RIGHT_HIP]
    if lh[2] > 0.3 and rh[2] > 0.3:
        return float(np.arctan2(rh[1] - lh[1], rh[0] - lh[0]) + np.pi / 2)

    return None


def angle_difference(a1, a2):
    """Smallest angular difference in [0, pi]."""
    diff = abs(a1 - a2) % (2 * np.pi)
    return diff if diff <= np.pi else 2 * np.pi - diff


def compute_edge_weight(pos_i, pos_j, angle_i, angle_j, vel_i, vel_j, scale):
    """
    4-term composite edge weight for queue-likelihood.

    Term 1 — spatial proximity:   exp(-d / (scale * 0.8))
    Term 2 — orientation alignment: cos(delta_theta), clamped [0,1]
    Term 3 — velocity coherence:  dot(v1,v2) if both moving, else 0.5
    Term 4 — spacing regularity:  exp(-|d - scale| / (scale * 0.3))
              People in a queue stand at roughly uniform intervals.
              A pair whose distance equals the median crowd spacing
              gets a bonus; too-close or too-far pairs are penalised.

    Novel contribution: no prior queue paper uses all four terms together.
    """
    dist = float(np.linalg.norm(np.array(pos_i) - np.array(pos_j)))

    spatial_score      = np.exp(-dist / (scale * 0.8))
    orientation_score  = max(0.0, float(np.cos(angle_difference(angle_i, angle_j))))

    v1 = np.array(vel_i, dtype=float)
    v2 = np.array(vel_j, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 > 0.5 and n2 > 0.5:
        velocity_score = max(0.0, float(np.dot(v1 / n1, v2 / n2)))
    else:
        velocity_score = 0.5

    spacing_score = float(np.exp(-abs(dist - scale) / (scale * 0.3)))

    vel_term = 0.7 + 0.3 * velocity_score
    return spatial_score * orientation_score * vel_term * (0.85 + 0.15 * spacing_score)


def build_pose_queue_graph(person_data):
    """
    Build pose-aware queue graph.

    person_data items require: id, pos, angle (or None), velocity (or (0,0)).
    Returns networkx Graph with edge weights in [0, 1].
    """
    n = len(person_data)
    if n < 2:
        return nx.Graph()

    positions = np.array([p['pos'] for p in person_data], dtype=float)

    all_dists = [np.linalg.norm(positions[i] - positions[j])
                 for i in range(n) for j in range(i + 1, n)]
    if not all_dists:
        return nx.Graph()
    scale = float(np.median(all_dists))

    angles     = [p.get('angle') or 0.0 for p in person_data]
    velocities = [p.get('velocity') or (0, 0) for p in person_data]

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, track_id=person_data[i]['id'], pos=person_data[i]['pos'])

    WEIGHT_THRESHOLD = 0.08
    MAX_DIST         = scale * 4.0

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist > MAX_DIST:
                continue
            w = compute_edge_weight(
                person_data[i]['pos'], person_data[j]['pos'],
                angles[i], angles[j],
                velocities[i], velocities[j],
                scale,
            )
            if w >= WEIGHT_THRESHOLD:
                G.add_edge(i, j, weight=w)

    return G


def extract_queue_from_graph(G, person_data, min_queue_size=3):
    """
    Extract queue spine via MST longest path.
    Returns (queue_persons_sorted, queue_length_px) or None.
    Step 6 cleanup removed — it introduced false negatives for curved queues.
    """
    if G.number_of_nodes() < min_queue_size:
        return None

    components = list(nx.connected_components(G))
    if not components:
        return None

    largest = max(components, key=len)
    if len(largest) < min_queue_size:
        return None

    sub = G.subgraph(largest).copy()
    for u, v, d in sub.edges(data=True):
        d['inv_weight'] = 1.0 / (d['weight'] + 1e-9)

    mst = nx.minimum_spanning_tree(sub, weight='inv_weight')

    longest_path = []
    for node in mst.nodes:
        lengths  = nx.single_source_dijkstra_path_length(mst, node, weight='inv_weight')
        farthest = max(lengths, key=lengths.get)
        path     = nx.shortest_path(mst, node, farthest, weight='inv_weight')
        if len(path) > len(longest_path):
            longest_path = path

    if len(longest_path) < min_queue_size:
        return None

    queue_persons = [person_data[i] for i in longest_path]
    positions     = np.array([p['pos'] for p in queue_persons], dtype=float)

    # sort once along principal axis — draw_overlay must NOT re-sort
    direction = positions[-1] - positions[0]
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm
    order         = np.argsort(positions @ direction)
    queue_persons = [queue_persons[i] for i in order]
    positions     = positions[order]

    queue_length_px = float(sum(
    np.linalg.norm(positions[i+1] - positions[i])
    for i in range(len(positions) - 1)
))
    return queue_persons, queue_length_px


def compute_pose_queue(person_data, min_queue_size=3):
    """
    Main entry point. Returns dict or None.
    """
    if len(person_data) < min_queue_size:
        return None

    G      = build_pose_queue_graph(person_data)
    result = extract_queue_from_graph(G, person_data, min_queue_size)
    if result is None:
        return None

    queue_persons, queue_length_px = result
    return {
        'queue_persons':   queue_persons,
        'queue_size':      len(queue_persons),
        'queue_length_px': queue_length_px,
        'graph':           G,
        'all_positions':   np.array([p['pos'] for p in person_data], dtype=float),
    }
