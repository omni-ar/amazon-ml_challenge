import os
import sys
from collections import Counter
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset"

def fast_line_count(filename):
    lines = 0
    with open(filename, 'rb') as f:
        buf = f.raw.read(1024*1024) if hasattr(f, 'raw') else f.read(1024*1024)
        while buf:
            lines += buf.count(b'\n')
            buf = f.read(1024*1024)
    return lines

print("--- FILE SIZES & LINE COUNTS ---")
files = [
    ("train_source1", os.path.join(DATASET_DIR, "train", "train_source1.tsv")),
    ("train_source2", os.path.join(DATASET_DIR, "train", "train_source2.tsv")),
    ("train_source3", os.path.join(DATASET_DIR, "train", "train_source3.tsv")),
    ("train_ground_truth", os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")),
    ("test_source1", os.path.join(DATASET_DIR, "test", "test_source1.tsv")),
    ("test_source2", os.path.join(DATASET_DIR, "test", "test_source2.tsv")),
    ("test_source3", os.path.join(DATASET_DIR, "test", "test_source3.tsv")),
]

for name, path in files:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        lines = fast_line_count(path)
        print(f"{name:20s}: {size_mb:8.1f} MB | {lines:10,d} lines (approx {lines-1:,} records)")
        # Show first 2 records cleanly
        df = pd.read_csv(path, sep="\t", nrows=2)
        for _, r in df.iterrows():
            clean_dict = {k: str(v)[:80] for k, v in dict(r).items()}
            print(f"   sample: {clean_dict}")

print("\n--- GROUND TRUTH ANALYSIS ---")
gt_path = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")
total_s1 = 0
singletons = 0
match_counts = []
source_dist = Counter()

with open(gt_path, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        total_s1 += 1
        s1, tab, rest = line.partition("\t")
        rest = rest.strip()
        if not rest:
            singletons += 1
            match_counts.append(0)
        else:
            ids = rest.split(",")
            match_counts.append(len(ids))
            for mid in ids:
                source_dist[mid[:2]] += 1

print(f"Total Source 1 in Train GT: {total_s1:,}")
print(f"Singletons (0 matches): {singletons:,} ({singletons/total_s1*100:.2f}%)")
print(f"Entities with >= 1 match: {total_s1 - singletons:,} ({(total_s1 - singletons)/total_s1*100:.2f}%)")
print(f"Matches breakdown by prefix: {dict(source_dist)}")
match_dist = Counter(match_counts)
print("Distribution of match counts per S1:")
for count in sorted(match_dist.keys())[:10]:
    print(f"  {count} matches: {match_dist[count]:,} ({match_dist[count]/total_s1*100:.2f}%)")

print("\n--- COUNTRY DISTRIBUTION (SAMPLE S1) ---")
for split in ["train", "test"]:
    s1_path = os.path.join(DATASET_DIR, split, f"{split}_source1.tsv")
    country_counts = Counter()
    with open(s1_path, "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        c_idx = header.index("country")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > c_idx:
                country_counts[parts[c_idx]] += 1
    print(f"{split.upper()} S1 Countries: {dict(country_counts)}")

print("\nDONE!")
