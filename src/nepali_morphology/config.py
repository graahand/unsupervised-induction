from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    dataset_id: str = "IRIIS-RESEARCH/Nepali-Text-Corpus"
    dataset_split: str = "train"
    text_column: str = "Article"
    max_articles: int | None = 2000
    max_tokens_per_article: int | None = 20000
    min_word_freq: int = 5
    max_vocab_size: int = 5000
    num_clusters: int = 60
    context_window: int = 1
    morpheme_prob_threshold: float = 0.1
    merge_threshold: float = 0.75
    max_paradigm_iters: int = 200
    max_morphemes_per_cluster: int = 50
    enable_compounds: bool = True
    output_dir: Path = Path("outputs")
    sample_words: int = 200
    random_state: int = 13
