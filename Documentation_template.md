# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** SayCheese 
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present a high-throughput, multi-stage Entity Resolution (ER) system engineered to resolve noisy, unaligned business records from multiple independent sources against a deduplicated reference source (Source 1). Our solution achieves a **0.9694 Validation Macro $F_{0.5}$ score** (with **99.46% precision** and **98.62% singleton accuracy**) by combining:
1. **Strict Zero-Loss Country Partitioning** that breaks the $O(N^2)$ cross-matching problem into independent locales while generalizing zero-shot to unseen countries (France).
2. **Multi-Channel Inverted-Index Blocking** using dynamic IDF-weighted keys across name n-grams, URL/domain roots, street tokens, and house number/postal code combinations, achieving $>99.7\%$ true-match recall ceiling.
3. **Vectorized C++ Rapidfuzz Feature Extraction & LightGBM Ranking**, followed by a rank-aware, mutually exclusive decision rule mathematically calibrated to maximize the precision-heavy Macro $F_{0.5}$ metric and prevent false merges on singletons.

---

## 2. Methodology

### 2.1 Problem Analysis
In exploratory data analysis across ~24.2 million records across Train and Test sets, we identified several structural properties and noise signatures:
* **Strict Country Boundary (0.00% Cross-Country Matches)**: In the 2.2M ground-truth annotations, 100.00% of matches occur within the same country label. No business entity in India or the US ever resolves across borders.
* **Zero-Shot Test Locale (France)**: While training data only covers `India` and `US`, the test set introduces `France` (15.0% of reference test entities). The pipeline explicitly avoids locale-specific hardcoded filters and uses language-agnostic Unicode decomposition (NFKD) and general address token expansion.
* **Transliteration & Multi-Script Discrepancy**: In Indian records, Source 1 often contains English transliterations (`Raj Investments LLP`, `Ss Food Private Limited`), while Source 2 and 3 records use native Indic scripts (Tamil: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`, Devanagari: `एसएस फूड प्राइवेट लिमिटेड`). However, address numerals (door/plot numbers, phone numbers, PIN codes) remain in Latin numerals across sources, providing an invariant matching anchor.
* **Domain & URL Flattening**: Source 3 frequently formats company names as domain URLs (e.g., `Maure Williams Colombier Inc` $\to$ `maurewilliamscolombier.com`).
* **High Density of Positive Matches**: Only 5.58% of entities are true singletons (0 matches); 94.42% of reference entities possess between 1 and 6 matching records across S2 and S3, with a median of 3 matches.

### 2.2 Solution Strategy
**Approach Type:** Hybrid (Multi-Channel Inverted Index Blocking + Vectorized C++ Feature Engineering + Gradient Boosted Ranking + Rank-Aware Exclusivity Thresholding)  
**Core Innovation:** 
* **Multi-Modal Blocking Keys**: Instead of relying solely on name prefixes, we combine phonetic/token keys with house-number-locality compound keys (`h:num_street`) to retain transliterated records that share zero English name tokens.
* **Rank-Aware Mutual Exclusivity**: We enforce candidate exclusivity (a noisy candidate record can only bind to its highest-scoring reference entity) combined with tiered confidence thresholds ($t_1$ for top-ranked candidate, $t_2$ for second candidate, $t_3$ for subsequent candidates) tuned specifically for the 2:1 precision:recall penalty of $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ pairwise comparison space down to a tractable candidate set without dropping true positives:
* **Country Partitioning**: Hard partition by `US`, `India`, and `France`.
* **Blocking Keys Generated**:
  1. `n:<token>`: Cleaned name core tokens (length $\ge 2$, legal suffixes removed).
  2. `p:<prefix>`: 4-character prefix of significant name tokens (length $\ge 5$).
  3. `w:<prefix>`: 8-character prefix of space-stripped ("glued") names (captures URL flattening and compound words).
  4. `a:<token>`: Distinctive address words (length $\ge 3$, address stop words removed).
  5. `h:<num>_<word>`: Compound key pairing address numbers with distinctive locality words.
* **Dynamic Key Pruning & IDF Weighting**: Keys appearing in $>300$ candidate records in a country are pruned. Remaining keys are weighted by Inverse Document Frequency $w = \ln(N + 1) - \ln(df)$.
* **Candidate Pool**: Candidates are sorted by accumulated IDF score, keeping the top $K=40$ candidates per Source 1 entity.
* **Coverage Guarantee**: On 20,000 ground-truth reference entities, this blocking architecture achieved a **99.755% recall ceiling**, reducing the search space by $>99.999\%$.

---

## 4. Matching Model

### Features Used:
* **Name Similarities (Rapidfuzz C++ parallelized)**:
  - `nm_ratio`: Levenshtein ratio on core business name
  - `nm_tset`: Token Set Ratio (resilient to word insertions/deletions)
  - `nm_tsort`: Token Sort Ratio (resilient to word-order transposition)
  - `nm_partial`: Partial Ratio (substring matching)
  - `nm_jw`: Jaro-Winkler similarity (rewards shared prefixes)
  - `nm_glued`: Ratio on concatenated/glued names (catches URL flattening)
  - `nm_full_tset`: Full raw name token set similarity
* **Address Similarities**:
  - `ad_ratio`, `ad_tset`, `ad_partial`: Levenshtein, token set, and partial ratios on normalized addresses
  - `addr2_empty`: Binary indicator for records with missing addresses
* **Numerical & Exact Token Features**:
  - `num_shared`: Count of overlapping numerical tokens (door numbers, PIN codes)
  - `num_n1`, `num_n2`: Total numbers in reference vs candidate address
  - `num_first_eq`: Equality of the primary building/street number
* **Entity & Graph Metadata**:
  - `legal_eq`, `legal_conf`: Agreement vs conflict of legal business suffixes (`LLC`, `Pvt Ltd`, `SARL`)
  - `b_score`, `b_rank`, `b_rel`: Blocking key score, rank, and relative score
  - `cand_deg`, `cand_rank`, `cand_rel`: Candidate degree and reverse rank
  - `s1_namefreq`: Reference entity name frequency
  - `is_s3`: Source indicator (S3 vs S2)
  - `indic`: Indicator for Indic Unicode script presence

### Model Type & Optimization:
* **Model**: LightGBM Binary Classifier (127 leaves, learning rate 0.05, feature fraction 0.8, early stopping).
* **Threshold Selection**: Tuned via fine-grained grid search on held-out validation data directly optimizing the macro $F_{0.5}$ metric:
  $$\text{Optimal Config:} \quad t_1 = 0.80, \; t_2 = 0.50, \; t_3 = 0.70, \; \text{exclusive} = \text{True}$$

---

## 5. Results & Error Analysis

- **Validation Macro $F_{0.5}$ Score:** **`0.9694`**
  - **Macro Precision:** `0.9946` (99.46%)
  - **Macro Recall:** `0.9339` (93.39%)
  - **Singleton Accuracy:** `0.9862` (98.62%)
- **False Positives (Wrong Merges)**: Only 1.3% of entities produced any false positive. These were almost entirely national franchise chains (e.g. multiple bank branches or retail outlets sharing identical names in neighboring city districts).
- **False Negatives (Missed Matches)**: Missed links predominantly occurred where a secondary record had both an entirely missing address (`nan`) and a severe typo in a short company name.

---

## 6. Conclusion
The proposed architecture provides a scalable, mathematically rigorous solution for entity resolution at the tens-of-millions record scale. By combining locale partitioning, IDF-weighted multi-modal blocking, parallel C++ feature computation, and $F_{0.5}$-calibrated decision thresholding, our pipeline delivers near-perfect precision (>99.4%) and high recall (>93.3%) while operating comfortably within standard hardware constraints.

---

## Appendix

### A. Code Artefacts
All reproduction code is organized in `code/business_entity_resolution/`:
* `src/config.py`: Centralized configuration, hyperparameter specifications, and file paths.
* `src/normalize.py`: Unicode normalization, regex parsing, and address abbreviation standardization.
* `src/blocking.py`: Multi-channel key generator and country-partitioned candidate blocking.
* `src/features.py`: Parallelized Rapidfuzz C++ pairwise feature extractor.
* `src/model.py`: LightGBM training routine, Macro $F_{0.5}$ metric calculator, and decision tuning.
* `src/run_pipeline.py`: Main executable reproducing the complete pipeline and output files.
* `requirements.txt`: Pinned dependencies (`polars`, `pyarrow`, `rapidfuzz`, `lightgbm`, `numpy`, `scipy`).
* `README.md`: Step-by-step reproduction instructions.

Reproduction command:
```bash
python code/business_entity_resolution/src/run_pipeline.py
```
This produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` and automatically validates both against `utils/validate_submission.py`.
