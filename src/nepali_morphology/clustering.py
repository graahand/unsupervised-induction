from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def kl_divergence(p: dict, q: dict, epsilon: float = 1e-12) -> float:
    total = 0.0
    for key, p_val in p.items():
        q_val = q.get(key, epsilon)
        total += p_val * math.log((p_val + epsilon) / (q_val + epsilon))
    return total


def js_divergence(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    m = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in keys}
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compute_distance_matrix(words: list[str], distributions: dict[str, dict]) -> np.ndarray:
    size = len(words)
    matrix = np.zeros((size, size), dtype=float)
    for i in range(size):
        p = distributions[words[i]]
        for j in range(i + 1, size):
            q = distributions[words[j]]
            dist = js_divergence(p, q)
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix


def create_agglomerative(n_clusters: int) -> AgglomerativeClustering:
    try:
        return AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")


def cluster_words(
    words: list[str],
    distributions: dict[str, dict],
    n_clusters: int,
) -> list[int]:
    distance_matrix = compute_distance_matrix(words, distributions)
    model = create_agglomerative(n_clusters)
    labels = model.fit_predict(distance_matrix)
    return labels.tolist()


def build_cluster_map(words: list[str], labels: Iterable[int]) -> dict[str, int]:
    return {word: int(label) for word, label in zip(words, labels)}
