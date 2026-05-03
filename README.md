# Unsupervised Induction
Unsupervised learning based morphological analysis with syntactic and PoS induction.

## Nepali Morphology Pipeline (Unsupervised)

This project implements a function-based, short-function pipeline for:
- Unsupervised syntactic category induction (PoS clusters)
- Morphological paradigm induction
- Paradigm merging and morphology-driven segmentation

The implementation follows the core ideas from the paper you shared: context-based syntactic clustering, morpheme extraction with statistical filtering, cross-cluster merging, and paradigm generalization.

## 1. Environment setup (Python 3.12)

1) Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Dataset setup (IRIIS Nepali Text Corpus)

The pipeline downloads the dataset directly from Hugging Face.
- Dataset: IRIIS-RESEARCH/Nepali-Text-Corpus
- Split: train
- Column used: Article

No manual file conversion is required, but you can limit corpus size to keep experiments fast.

## 3. Run the pipeline

From the project root:

```bash
python -m nepali_morphology.cli \
  --max-articles 2000 \
  --max-vocab 5000 \
  --num-clusters 60 \
  --min-freq 5 \
  --morpheme-threshold 0.1 \
  --merge-threshold 0.75
```

## 4. Output artifacts

Outputs are written to the `outputs/` folder:
- `pos_clusters.json`: words grouped by cluster
- `paradigms.json`: induced paradigms with stems and morpheme/cluster pairs
- `segmentation_samples.json`: sample segmentation results

## 5. Notes on experimentation

- Increase `--max-articles` and `--max-vocab` for better quality but slower clustering.
- The clustering step is O(N^2) in the number of clustered words. Use a manageable `--max-vocab`.
- You can disable compound splitting with `--no-compounds`.

## 6. Example: smaller quick run

```bash
python -m nepali_morphology.cli --max-articles 200 --max-vocab 1200 --num-clusters 30
```
