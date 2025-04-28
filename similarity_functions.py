# --- Similarity Functions ---

import numpy as np


def cosine_similarity(a, b):
    norm1 = np.linalg.norm(a)
    norm2 = np.linalg.norm(b)
    return 0.0 if norm1 == 0 or norm2 == 0 else np.dot(a, b) / (norm1 * norm2)


def dot_product(a, b):
    return float(np.dot(a, b))


def l2_distance(a, b):
    return -float(np.linalg.norm(a - b))


def l1_distance(a, b):
    return -float(np.linalg.norm(a - b, ord=1))