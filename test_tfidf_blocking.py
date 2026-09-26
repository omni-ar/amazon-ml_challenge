import os
import sys
import re
import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\train"

print("--- TESTING TF-IDF / SPARSE MATRIX BLOCKING ---")

# Load 5,000 S1 records
s1_records = []
s1_ids = []
with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 5000:
            break
        parts = line.strip().split("\t")
        s1_ids.append(parts[0])
        s1_records.append({
            "id": parts[0],
            "name": parts[1] if len(parts) > 1 else "",
            "address": parts[2] if len(parts) > 2 else "",
            "country": parts[3] if len(parts) > 3 else ""
        })

# Load ground truth for these 5k
gt_dict = {}
all_true_targets = set()
with open(os.path.join(DATASET_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, tab, rest = line.partition("\t")
        if s1 in s1_ids:
            m = rest.strip().split(",") if rest.strip() else []
            gt_dict[s1] = set(m)
            all_true_targets.update(m)

# Load target records (true targets + 100k noise)
s2_cands = []
s3_cands = []
with open(os.path.join(DATASET_DIR, "train_source2.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.strip().split("\t")
        eid = parts[0]
        if eid in all_true_targets or len(s2_cands) < 50000:
            s2_cands.append((eid, parts[1] if len(parts)>1 else "", parts[2] if len(parts)>2 else "", parts[3] if len(parts)>3 else ""))

with open(os.path.join(DATASET_DIR, "train_source3.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        eid = parts[0]
        if eid in all_true_targets or len(s3_cands) < 50000:
            s3_cands.append((eid, parts[1] if len(parts)>1 else "", parts[2] if len(parts)>2 else "", parts[3] if len(parts)>3 else ""))

all_cands = s2_cands + s3_cands
cand_ids = [c[0] for c in all_cands]
print(f"Target pool size: {len(all_cands):,} (contains {len(all_true_targets)} true matches)")

# Preprocessing text: combine name and address
def prep_text(name, addr):
    # remove domain suffix
    name = re.sub(r'\.(com|in|org|net)', ' ', name, flags=re.IGNORECASE)
    # clean
    text = f"{name} {addr}"
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text

s1_texts = [prep_text(r["name"], r["address"]) for r in s1_records]
cand_texts = [prep_text(c[1], c[2]) for c in all_cands]

print("Fitting TF-IDF Vectorizer...")
t0 = time.time()
# Use word + char_wb 3-4 ngrams to capture typos and transliterations
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 4),
    min_df=2,
    max_features=150000,
    dtype=np.float32
)

# Fit on all candidates
cand_matrix = vectorizer.fit_transform(cand_texts)
s1_matrix = vectorizer.transform(s1_texts)
t_fit = time.time() - t0
print(f"TF-IDF fitted & transformed in {t_fit:.2f} seconds. Vocab size: {cand_matrix.shape[1]:,}")

# Fast Top-K Retrieval using chunked sparse dot products
t0 = time.time()
TOP_K = 25
chunk_size = 500
total_true = 0
found_true = 0

for start in range(0, len(s1_records), chunk_size):
    end = min(start + chunk_size, len(s1_records))
    sub_s1 = s1_matrix[start:end]
    # Dot product: sub_s1 (chunk x V) @ cand_matrix.T (V x C) -> (chunk x C)
    sims = sub_s1.dot(cand_matrix.T)
    
    for row_idx in range(end - start):
        s1_idx = start + row_idx
        row_sims = sims.getrow(row_idx)
        
        # Get top K indices
        if row_sims.nnz > 0:
            top_indices = row_sims.indices[np.argpartition(row_sims.data, -min(TOP_K, row_sims.nnz))[-TOP_K:]]
            retrieved = set(cand_ids[idx] for idx in top_indices)
        else:
            retrieved = set()
            
        true_set = gt_dict.get(s1_ids[s1_idx], set())
        if true_set:
            total_true += len(true_set)
            found_true += len(true_set & retrieved)

t_query = time.time() - t0
print(f"Top-{TOP_K} retrieval finished in {t_query:.2f} seconds ({len(s1_records)/t_query:.0f} entities/sec).")
print(f"Recall on true matches: {found_true:,} / {total_true:,} = {found_true/total_true*100:.2f}%")
