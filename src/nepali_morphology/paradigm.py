from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .morpheme import ClusterMorphemeData


@dataclass
class Paradigm:
    morpheme_cluster_pairs: list[tuple[str, int]]
    stems: set[str]


def select_top_morphemes(
    stats: dict[int, ClusterMorphemeData],
    allowed: dict[int, set[str]],
    max_per_cluster: int,
) -> dict[int, dict[str, set[str]]]:
    selected: dict[int, dict[str, set[str]]] = {}
    for cluster_id, data in stats.items():
        candidates = [
            (morpheme, data.morpheme_counts[morpheme])
            for morpheme in allowed.get(cluster_id, set())
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        top_morphemes = {morpheme for morpheme, _ in candidates[:max_per_cluster]}
        selected[cluster_id] = {
            morpheme: data.morpheme_stems[morpheme] for morpheme in top_morphemes
        }
    return selected


def find_best_morpheme_pair(
    cluster_morpheme_stems: dict[int, dict[str, set[str]]],
    used_stems: dict[int, set[str]],
) -> tuple[int, str, int, str, set[str]] | None:
    cluster_ids = list(cluster_morpheme_stems.keys())
    best_pair = None
    best_count = 0

    for idx, c1 in enumerate(cluster_ids):
        for c2 in cluster_ids[idx + 1 :]:
            for m1, stems1 in cluster_morpheme_stems[c1].items():
                available1 = stems1 - used_stems[c1]
                if not available1:
                    continue
                for m2, stems2 in cluster_morpheme_stems[c2].items():
                    available2 = stems2 - used_stems[c2]
                    common = available1 & available2
                    if len(common) > best_count:
                        best_count = len(common)
                        best_pair = (c1, m1, c2, m2, common)
    return best_pair


def induce_initial_paradigms(
    cluster_morpheme_stems: dict[int, dict[str, set[str]]],
    max_iters: int,
) -> list[Paradigm]:
    paradigms: list[Paradigm] = []
    used_stems: dict[int, set[str]] = defaultdict(set)

    for _ in range(max_iters):
        best = find_best_morpheme_pair(cluster_morpheme_stems, used_stems)
        if not best:
            break
        c1, m1, c2, m2, common = best
        paradigms.append(Paradigm([(m1, c1), (m2, c2)], set(common)))
        used_stems[c1].update(common)
        used_stems[c2].update(common)

    return paradigms


def expected_accuracy(p1: Paradigm, p2: Paradigm) -> float:
    common = p1.stems & p2.stems
    if not common:
        return 0.0
    n1 = len(p1.stems - p2.stems)
    n2 = len(p2.stems - p1.stems)
    p = len(common)
    acc1 = p / (p + n1)
    acc2 = p / (p + n2)
    return 0.5 * (acc1 + acc2)


def merge_paradigms(
    paradigms: list[Paradigm],
    threshold: float,
    max_iters: int,
) -> list[Paradigm]:
    merged = True
    iterations = 0

    while merged and iterations < max_iters:
        merged = False
        iterations += 1
        best_pair: tuple[int, int, float] | None = None

        for i in range(len(paradigms)):
            for j in range(i + 1, len(paradigms)):
                score = expected_accuracy(paradigms[i], paradigms[j])
                if score >= threshold:
                    if not best_pair or score > best_pair[2]:
                        best_pair = (i, j, score)

        if best_pair:
            i, j, _ = best_pair
            first = paradigms[i]
            second = paradigms[j]
            merged_pairs = list({*first.morpheme_cluster_pairs, *second.morpheme_cluster_pairs})
            merged_stems = first.stems | second.stems
            paradigms[i] = Paradigm(merged_pairs, merged_stems)
            paradigms.pop(j)
            merged = True

    return paradigms


def build_paradigm_index(paradigms: Iterable[Paradigm]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for paradigm in paradigms:
        for stem in paradigm.stems:
            for morpheme, _ in paradigm.morpheme_cluster_pairs:
                word = f"{stem}{morpheme}"
                if word not in index:
                    index[word] = (stem, morpheme)
    return index


def serialize_paradigms(paradigms: Iterable[Paradigm]) -> list[dict]:
    payload: list[dict] = []
    for paradigm in paradigms:
        payload.append(
            {
                "morpheme_cluster_pairs": [list(pair) for pair in paradigm.morpheme_cluster_pairs],
                "stems": sorted(paradigm.stems),
            }
        )
    return payload
