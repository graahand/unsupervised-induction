from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class ClusterMorphemeData:
    morpheme_counts: Counter
    morpheme_stems: dict[str, set[str]]


def iter_splits(word: str) -> list[tuple[str, str]]:
    if len(word) < 2:
        return []
    return [(word[:i], word[i:]) for i in range(1, len(word))]


def collect_morpheme_stats(cluster_words: dict[int, list[str]]) -> dict[int, ClusterMorphemeData]:
    stats: dict[int, ClusterMorphemeData] = {}
    for cluster_id, words in cluster_words.items():
        counts: Counter = Counter()
        stems: dict[str, set[str]] = defaultdict(set)
        for word in words:
            for stem, morpheme in iter_splits(word):
                counts[morpheme] += 1
                stems[morpheme].add(stem)
        stats[cluster_id] = ClusterMorphemeData(counts, stems)
    return stats


def compute_morpheme_probabilities(stats: dict[int, ClusterMorphemeData]) -> dict[int, dict[str, float]]:
    probabilities: dict[int, dict[str, float]] = {}
    for cluster_id, data in stats.items():
        total = sum(data.morpheme_counts.values())
        if total == 0:
            probabilities[cluster_id] = {}
            continue
        probabilities[cluster_id] = {
            morpheme: count / total for morpheme, count in data.morpheme_counts.items()
        }
    return probabilities


def filter_morphemes_by_prob(
    probabilities: dict[int, dict[str, float]], threshold: float
) -> dict[int, set[str]]:
    allowed: dict[int, set[str]] = {}
    for cluster_id, values in probabilities.items():
        allowed[cluster_id] = {morpheme for morpheme, prob in values.items() if prob >= threshold}
    return allowed
