import os
import sys
import polars as pl
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\train"

print("Analyzing token frequency in training source 2/3 (India sample 200k)...")

tokens = []
with open(os.path.join(DATASET_DIR, "train_source2.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 200000:
            break
        parts = line.strip().split("\t")
        if len(parts) > 3 and parts[3] == "India":
            name = parts[1].lower()
            for w in name.split():
                if len(w) >= 3:
                    tokens.append(w)

counts = Counter(tokens)
print(f"Total token occurrences: {len(tokens):,}, Unique tokens: {len(counts):,}")
print("\nTop 30 most frequent business name tokens:")
for w, c in counts.most_common(30):
    print(f"   {w:20s}: {c:,} in 200k sample (projected in 4.7M: {c * 23.5:,.0f})")

above_300 = sum(1 for c in counts.values() if c > 300)
above_300_occ = sum(c for c in counts.values() if c > 300)
print(f"\nNumber of unique tokens with frequency > 300 in 200k sample: {above_300}")
print(f"Number of token occurrences with frequency > 300 in 200k sample: {above_300_occ:,} ({above_300_occ/len(tokens)*100:.2f}%)")
