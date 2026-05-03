from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

START_TOKEN = "<START>"
END_TOKEN = "<END>"


def build_word_frequencies(token_sequences: Iterable[list[str]]) -> Counter:
    counts: Counter = Counter()
    for tokens in token_sequences:
        counts.update(tokens)
    return counts


def select_vocab(counts: Counter, min_freq: int, max_vocab: int) -> set[str]:
    filtered = [word for word, freq in counts.items() if freq >= min_freq]
    filtered.sort(key=lambda word: counts[word], reverse=True)
    return set(filtered[:max_vocab])


def iter_context_pairs(tokens: list[str]) -> Iterable[tuple[str, str, str]]:
    for index, token in enumerate(tokens):
        left = tokens[index - 1] if index > 0 else START_TOKEN
        right = tokens[index + 1] if index + 1 < len(tokens) else END_TOKEN
        yield token, left, right


def build_context_counts(token_sequences: Iterable[list[str]], vocab: set[str]) -> dict[str, Counter]:
    context_counts: dict[str, Counter] = defaultdict(Counter)
    for tokens in token_sequences:
        for token, left, right in iter_context_pairs(tokens):
            if token not in vocab:
                continue
            context_counts[token][(left, right)] += 1
    return context_counts


def normalize_counts(counts: Counter) -> dict[tuple[str, str], float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def build_context_distributions(context_counts: dict[str, Counter]) -> dict[str, dict[tuple[str, str], float]]:
    distributions: dict[str, dict[tuple[str, str], float]] = {}
    for word, counts in context_counts.items():
        distributions[word] = normalize_counts(counts)
    return distributions


def average_distribution(distributions: Iterable[dict[tuple[str, str], float]]) -> dict[tuple[str, str], float]:
    accumulator: Counter = Counter()
    total = 0
    for dist in distributions:
        total += 1
        for key, value in dist.items():
            accumulator[key] += value
    if total == 0:
        return {}
    return {key: value / total for key, value in accumulator.items()}
