from __future__ import annotations

from typing import Iterable

from .paradigm import Paradigm


def build_morpheme_dictionary(paradigms: Iterable[Paradigm]) -> list[str]:
    dictionary: set[str] = set()
    for paradigm in paradigms:
        morphemes = [morpheme for morpheme, _ in paradigm.morpheme_cluster_pairs]
        if not morphemes:
            continue
        initials = {m[0] for m in morphemes if m}
        if len(initials) == 1:
            dictionary.add(max(morphemes, key=len))
        else:
            dictionary.update(morphemes)
    return sorted(dictionary, key=len, reverse=True)


def segment_by_paradigm(word: str, paradigm_index: dict[str, tuple[str, str]]) -> list[str] | None:
    if word not in paradigm_index:
        return None
    stem, morpheme = paradigm_index[word]
    if not morpheme:
        return [stem]
    return [stem, morpheme]


def segment_by_morpheme_dict(word: str, morpheme_dict: list[str]) -> list[str]:
    for morpheme in morpheme_dict:
        if word.endswith(morpheme) and len(word) > len(morpheme):
            stem = word[: -len(morpheme)]
            return segment_by_morpheme_dict(stem, morpheme_dict) + [morpheme]
    return [word]


def split_compound_word(word: str, vocab: set[str]) -> list[str]:
    for size in range(len(word) - 1, 0, -1):
        suffix = word[size:]
        if suffix in vocab and len(suffix) > 1:
            prefix = word[:size]
            return split_compound_word(prefix, vocab) + [suffix]
    return [word]


def apply_compound_splitting(segments: list[str], vocab: set[str]) -> list[str]:
    result: list[str] = []
    for segment in segments:
        result.extend(split_compound_word(segment, vocab))
    return result


def segment_word(
    word: str,
    paradigm_index: dict[str, tuple[str, str]],
    morpheme_dict: list[str],
    vocab: set[str],
    enable_compounds: bool,
) -> list[str]:
    paradigm_split = segment_by_paradigm(word, paradigm_index)
    if paradigm_split:
        return paradigm_split
    segments = segment_by_morpheme_dict(word, morpheme_dict)
    if enable_compounds:
        return apply_compound_splitting(segments, vocab)
    return segments
