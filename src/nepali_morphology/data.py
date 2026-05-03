from __future__ import annotations

import re
import string
import unicodedata
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm

from .config import PipelineConfig

NEPALI_DANDA = "\u0964"
NEPALI_DOUBLE_DANDA = "\u0965"
PUNCTUATION = string.punctuation + NEPALI_DANDA + NEPALI_DOUBLE_DANDA


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def basic_cleanup(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?\u0964\u0965]+", text)
    return [part.strip() for part in parts if part.strip()]


def strip_token_punctuation(token: str) -> str:
    return re.sub(rf"^[{re.escape(PUNCTUATION)}]+|[{re.escape(PUNCTUATION)}]+$", "", token)


def tokenize_sentence(sentence: str) -> list[str]:
    tokens = []
    for raw in sentence.split():
        cleaned = strip_token_punctuation(raw)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def tokenize_article(text: str) -> list[str]:
    normalized = normalize_text(text)
    cleaned = basic_cleanup(normalized)
    tokens: list[str] = []
    for sentence in split_sentences(cleaned):
        tokens.extend(tokenize_sentence(sentence))
    return tokens


def iter_articles(dataset, text_column: str) -> Iterable[str]:
    for item in dataset:
        yield item.get(text_column, "")


def load_corpus_tokens(config: PipelineConfig) -> list[list[str]]:
    dataset = load_dataset(config.dataset_id, split=config.dataset_split)
    sequences: list[list[str]] = []
    max_articles = config.max_articles or len(dataset)

    for index, article in enumerate(
        tqdm(iter_articles(dataset, config.text_column), total=max_articles, desc="Loading articles")
    ):
        if config.max_articles and index >= config.max_articles:
            break
        tokens = tokenize_article(article)
        if config.max_tokens_per_article:
            tokens = tokens[: config.max_tokens_per_article]
        if tokens:
            sequences.append(tokens)

    return sequences
