import numpy as np


# classification labels
LABEL_QUEUE     = "queue"
LABEL_CROWD     = "crowd"
LABEL_UNCERTAIN = "uncertain"


def extract_graph_features(person_data, graph):
    """
    Compute 5 interpretable features from the pose-aware person graph.

    These features capture the geometric and social properties that
    distinguish a queue from a random crowd.

    Returns a dict of named features, or None if not enough data.

    Features:
        angle_variance   : variance of facing angles (low = everyone faces same way = queue)
        aspect_ratio     : bounding box width / height of the group (high = horizontal line)
        avg_edge_weight  : mean edge weight in the graph (high = strong queue structure)
        density          : persons per unit area (high = tightly packed)
        linearity_score  : how well the group fits a straight line (R^2 of PCA projection)
    """
    if len(person_data) < 3:
        return None

    positions = np.array([p['pos'] for p in person_data], dtype=float)
    angles    = [p.get('angle') for p in person_data]
    angles    = [a for a in angles if a is not None]

    # --- feature 1: angle variance ---
    if len(angles) >= 2:
        # convert angles to unit vectors to handle wrap-around correctly
        vecs = np.array([[np.cos(a), np.sin(a)] for a in angles])
        mean_vec = vecs.mean(axis=0)
        # circular variance: 0 = all same direction, 1 = random
        r = np.linalg.norm(mean_vec)
        angle_variance = float(1.0 - r)
    else:
        angle_variance = 0.5  # neutral if no pose data

    # --- feature 2: aspect ratio of bounding box ---
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    width  = max_x - min_x + 1e-6
    height = max_y - min_y + 1e-6
    aspect_ratio = float(max(width, height) / min(width, height))

    # --- feature 3: average edge weight ---
    if graph.number_of_edges() > 0:
        weights = [d['weight'] for _, _, d in graph.edges(data=True)]
        avg_edge_weight = float(np.mean(weights))
    else:
        avg_edge_weight = 0.0

    # --- feature 4: density (persons per 10000 sq px) ---
    area = width * height
    density = float(len(person_data) / area * 10000)

    # --- feature 5: linearity score (R^2 of fitting a line) ---
    if len(positions) >= 3:
        # PCA: first component explains variance along the main axis
        centered = positions - positions.mean(axis=0)
        cov = np.cov(centered.T)
        if cov.ndim == 2:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            if eigenvalues.sum() > 0:
                linearity_score = float(eigenvalues[0] / eigenvalues.sum())
            else:
                linearity_score = 0.5
        else:
            linearity_score = 0.5
    else:
        linearity_score = 0.5

    return {
        'angle_variance':   angle_variance,
        'aspect_ratio':     aspect_ratio,
        'avg_edge_weight':  avg_edge_weight,
        'density':          density,
        'linearity_score':  linearity_score,
    }


def classify_scene(features, n_persons):
    """
    Rule-based classifier that maps graph features to a scene label.

    Thresholds are derived from geometric properties of queues:
    - A queue has low angle variance (people face same way)
    - A queue has high aspect ratio (long thin shape)
    - A queue has high linearity (fits a line well)
    - A queue has high average edge weight (strong pose alignment)

    Returns: (label, confidence_score, feature_dict)
        label            : "queue", "crowd", or "uncertain"
        confidence_score : float in [0, 1] indicating classifier certainty
        features         : the same features dict passed in (for logging)
    """
    if features is None or n_persons < 3:
        return LABEL_UNCERTAIN, 0.0, features

    score = 0.0
    max_score = 5.0

    # low angle variance -> queue signal
    if features['angle_variance'] < 0.25:
        score += 1.5
    elif features['angle_variance'] < 0.45:
        score += 0.75

    # high aspect ratio -> queue signal (elongated group)
    if features['aspect_ratio'] > 3.0:
        score += 1.5
    elif features['aspect_ratio'] > 1.8:
        score += 0.75

    # high linearity -> queue signal
    if features['linearity_score'] > 0.85:
        score += 1.0
    elif features['linearity_score'] > 0.70:
        score += 0.5

    # strong edge weights -> queue signal
    if features['avg_edge_weight'] > 0.30:
        score += 0.75
    elif features['avg_edge_weight'] > 0.15:
        score += 0.35

    # more people -> stronger signal (a 2-person line is ambiguous)
    if n_persons >= 6:
        score += 0.25

    confidence = score / max_score

    if confidence >= 0.45:
        label = LABEL_QUEUE
    elif confidence <= 0.25:
        label = LABEL_CROWD
    else:
        label = LABEL_UNCERTAIN

    return label, float(confidence), features


class CrowdClassifier:
    """
    Stateful wrapper around extract_graph_features + classify_scene.

    Keeps a short history of labels to smooth out per-frame jitter.
    The final label is the majority vote over the last `history_len` frames.
    """

    def __init__(self, history_len=8):
        self.history_len = history_len
        self._label_history = []

    def update(self, person_data, graph):
        """
        Run classification for this frame and return smoothed result.

        Returns:
            label      : smoothed label string
            confidence : float confidence of current frame raw result
            features   : feature dict (for display / logging)
        """
        features = extract_graph_features(person_data, graph)
        label, confidence, features = classify_scene(features, len(person_data))

        self._label_history.append(label)
        if len(self._label_history) > self.history_len:
            self._label_history.pop(0)

        smoothed_label = self._majority_label()

        return smoothed_label, confidence, features

    def _majority_label(self):
        if not self._label_history:
            return LABEL_UNCERTAIN
        counts = {
            LABEL_QUEUE:     self._label_history.count(LABEL_QUEUE),
            LABEL_CROWD:     self._label_history.count(LABEL_CROWD),
            LABEL_UNCERTAIN: self._label_history.count(LABEL_UNCERTAIN),
        }
        return max(counts, key=counts.get)

    def reset(self):
        self._label_history.clear()
