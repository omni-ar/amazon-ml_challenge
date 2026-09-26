import os
import sys

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset"

print("Checking country consistency between S1 and matched S2/S3...")

# Load country lookup for a sample or streaming
# Let's inspect first 50,000 matches in ground truth
s1_country = {}
print("Loading S1 countries (sample 100k)...")
with open(os.path.join(DATASET_DIR, "train", "train_source1.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 100000:
            break
        parts = line.strip().split("\t")
        s1_country[parts[0]] = parts[3] if len(parts) > 3 else "UNKNOWN"

# Load S2 countries
s2_country = {}
print("Loading S2 countries (sample 200k)...")
with open(os.path.join(DATASET_DIR, "train", "train_source2.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 200000:
            break
        parts = line.strip().split("\t")
        s2_country[parts[0]] = parts[3] if len(parts) > 3 else "UNKNOWN"

# Load S3 countries
s3_country = {}
print("Loading S3 countries (sample 200k)...")
with open(os.path.join(DATASET_DIR, "train", "train_source3.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 200000:
            break
        parts = line.strip().split("\t")
        s3_country[parts[0]] = parts[3] if len(parts) > 3 else "UNKNOWN"

same_country = 0
cross_country = 0
checked = 0

with open(os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, tab, rest = line.partition("\t")
        if s1 not in s1_country:
            continue
        c1 = s1_country[s1]
        matches = rest.strip().split(",") if rest.strip() else []
        for m in matches:
            c2 = s2_country.get(m) or s3_country.get(m)
            if c2:
                checked += 1
                if c1 == c2:
                    same_country += 1
                else:
                    cross_country += 1
                    print(f"CROSS COUNTRY FOUND: {s1} ({c1}) <-> {m} ({c2})")
                    if cross_country > 5:
                        break
        if checked >= 20000:
            break

print(f"Checked matches: {checked:,}")
print(f"Same country matches: {same_country:,} ({same_country/(checked or 1)*100:.2f}%)")
print(f"Cross country matches: {cross_country:,} ({cross_country/(checked or 1)*100:.2f}%)")
