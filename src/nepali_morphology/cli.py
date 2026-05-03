from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nepali morphology pipeline")
    parser.add_argument("--dataset-id", default=PipelineConfig.dataset_id)
    parser.add_argument("--dataset-split", default=PipelineConfig.dataset_split)
    parser.add_argument("--text-column", default=PipelineConfig.text_column)
    parser.add_argument("--max-articles", type=int, default=PipelineConfig.max_articles)
    parser.add_argument("--max-tokens-per-article", type=int, default=PipelineConfig.max_tokens_per_article)
    parser.add_argument("--min-freq", type=int, default=PipelineConfig.min_word_freq)
    parser.add_argument("--max-vocab", type=int, default=PipelineConfig.max_vocab_size)
    parser.add_argument("--num-clusters", type=int, default=PipelineConfig.num_clusters)
    parser.add_argument("--morpheme-threshold", type=float, default=PipelineConfig.morpheme_prob_threshold)
    parser.add_argument("--merge-threshold", type=float, default=PipelineConfig.merge_threshold)
    parser.add_argument("--max-paradigm-iters", type=int, default=PipelineConfig.max_paradigm_iters)
    parser.add_argument(
        "--max-morphemes-per-cluster",
        type=int,
        default=PipelineConfig.max_morphemes_per_cluster,
    )
    parser.add_argument("--sample-words", type=int, default=PipelineConfig.sample_words)
    parser.add_argument("--output-dir", type=Path, default=PipelineConfig.output_dir)
    parser.add_argument("--no-compounds", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        max_articles=args.max_articles,
        max_tokens_per_article=args.max_tokens_per_article,
        min_word_freq=args.min_freq,
        max_vocab_size=args.max_vocab,
        num_clusters=args.num_clusters,
        morpheme_prob_threshold=args.morpheme_threshold,
        merge_threshold=args.merge_threshold,
        max_paradigm_iters=args.max_paradigm_iters,
        max_morphemes_per_cluster=args.max_morphemes_per_cluster,
        sample_words=args.sample_words,
        output_dir=args.output_dir,
        enable_compounds=not args.no_compounds,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    result = run_pipeline(config)
    print("Pipeline finished")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
