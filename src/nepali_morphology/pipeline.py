from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .clustering import build_cluster_map, cluster_words
from .config import PipelineConfig
from .context import (
    build_context_counts,
    build_context_distributions,
    build_word_frequencies,
    select_vocab,
)
from .data import load_corpus_tokens
from .morpheme import collect_morpheme_stats, compute_morpheme_probabilities, filter_morphemes_by_prob
from .paradigm import (
    build_paradigm_index,
    induce_initial_paradigms,
    merge_paradigms,
    select_top_morphemes,
    serialize_paradigms,
)
from .segment import build_morpheme_dictionary, segment_word


def group_words_by_cluster(cluster_map: dict[str, int]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for word, cluster_id in cluster_map.items():
        grouped[cluster_id].append(word)
    return grouped


def select_sample_words(vocab: set[str], sample_size: int) -> list[str]:
    return sorted(list(vocab))[:sample_size]


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    token_sequences = load_corpus_tokens(config)
    freq = build_word_frequencies(token_sequences)
    vocab = select_vocab(freq, config.min_word_freq, config.max_vocab_size)

    context_counts = build_context_counts(token_sequences, vocab)
    context_distributions = build_context_distributions(context_counts)
    words = list(context_distributions.keys())

    labels = cluster_words(words, context_distributions, config.num_clusters)
    cluster_map = build_cluster_map(words, labels)
    clustered_words = group_words_by_cluster(cluster_map)

    morpheme_stats = collect_morpheme_stats(clustered_words)
    morpheme_probs = compute_morpheme_probabilities(morpheme_stats)
    allowed_morphemes = filter_morphemes_by_prob(morpheme_probs, config.morpheme_prob_threshold)
    cluster_morpheme_stems = select_top_morphemes(
        morpheme_stats, allowed_morphemes, config.max_morphemes_per_cluster
    )

    paradigms = induce_initial_paradigms(cluster_morpheme_stems, config.max_paradigm_iters)
    paradigms = merge_paradigms(paradigms, config.merge_threshold, config.max_paradigm_iters)

    paradigm_index = build_paradigm_index(paradigms)
    morpheme_dict = build_morpheme_dictionary(paradigms)

    samples = select_sample_words(vocab, config.sample_words)
    sample_segmentations = {
        word: segment_word(word, paradigm_index, morpheme_dict, vocab, config.enable_compounds)
        for word in samples
    }

    output_dir = config.output_dir
    ensure_output_dir(output_dir)
    write_json(output_dir / "pos_clusters.json", clustered_words)
    write_json(output_dir / "paradigms.json", serialize_paradigms(paradigms))
    write_json(output_dir / "segmentation_samples.json", sample_segmentations)

    return {
        "vocab_size": len(vocab),
        "clusters": len(clustered_words),
        "paradigms": len(paradigms),
        "sample_segmentations": len(sample_segmentations),
        "output_dir": str(output_dir),
    }
