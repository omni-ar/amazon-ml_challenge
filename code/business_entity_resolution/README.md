# Business Entity Resolution Pipeline

## Overview
This repository implements an end-to-end Machine Learning pipeline for the **Amazon ML Challenge: Business Entity Resolution Challenge**.
Given millions of noisy business entity records across three independent data sources (Source 1 reference, Source 2, and Source 3), the solution identifies all matching records from Source 2 and Source 3 for every Source 1 entity.

## Architecture

1. **Country Partitioning & Normalization**:
   - Entities are cleanly partitioned by country (`US`, `India`, and `France` for zero-shot testing). Cross-country matches are strictly 0.00%.
   - Preprocessing includes Unicode NFKD normalization, stripping Latin accents, preserving native Indic scripts (Devanagari, Tamil, etc.), expanding address abbreviations, and isolating building/house numbers and PIN codes.

2. **Stage 1: Multi-Channel Inverted Index Blocking**:
   - Multi-key inverted index incorporating:
     - Name tokens & 4-character prefixes
     - 8-character glued name prefixes (resolves domain flattening such as `companyname.com`)
     - Distinctive street/locality words
     - House/door numbers + locality tokens
   - Keys are dynamically weighted by Inverse Document Frequency (IDF: $w = \ln(N + 1) - \ln(df)$).
   - Generates the top $K=40$ candidate pool per entity for `candidate_pairs.tsv` with >99.7% true link coverage.

3. **Stage 2: Pairwise Feature Engineering & LightGBM Ranking**:
   - High-throughput parallel feature extraction via `rapidfuzz` in C++ (`workers=-1`):
     - Name similarity: `fuzz.ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio`, `JaroWinkler`
     - Address similarity: `fuzz.ratio`, `token_set_ratio`, `partial_ratio`
     - Numerical overlap: shared digits, first digit equality
     - Graph structure features: candidate degree, rank, relative score
     - Legal suffix consistency and conflict indicators
   - A LightGBM binary classifier predicts link probability.

4. **Stage 3: Decision Rule & F_0.5 Threshold Tuning**:
   - Rank-aware decision thresholds ($t_1, t_2, t_3$) with candidate exclusivity.
   - Specifically tuned on held-out validation data to maximize macro $F_{0.5}$ and protect singletons.

## Environment & Requirements
* Python 3.10+
* Packages:
  ```bash
  pip install -r requirements.txt
  ```

## Reproduction Instructions
From the root directory:
```bash
python code/business_entity_resolution/src/run_pipeline.py
```
This runs the full pipeline:
- Loads the dataset from `student_resource/dataset/`
- Trains the LightGBM classifier on the training partition
- Performs country-by-country candidate generation and inference on the full test set (US, India, and France)
- Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- Runs `student_resource/utils/validate_submission.py` to confirm verification status (`PASS`).
