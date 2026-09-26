import os
import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\train"

# Pick first 5 non-singleton S1 IDs from ground truth
s1_targets = {}
with open(os.path.join(DATASET_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, tab, rest = line.partition("\t")
        rest = rest.strip()
        if rest:
            s1_targets[s1] = set(rest.split(","))
            if len(s1_targets) >= 5:
                break

all_target_s2 = set()
all_target_s3 = set()
for targets in s1_targets.values():
    for mid in targets:
        if mid.startswith("S2"):
            all_target_s2.add(mid)
        elif mid.startswith("S3"):
            all_target_s3.add(mid)

# Retrieve records
s1_records = {}
with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in s1_targets:
            s1_records[parts[0]] = parts

s2_records = {}
with open(os.path.join(DATASET_DIR, "train_source2.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in all_target_s2:
            s2_records[parts[0]] = parts
            if len(s2_records) == len(all_target_s2):
                break

s3_records = {}
with open(os.path.join(DATASET_DIR, "train_source3.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in all_target_s3:
            s3_records[parts[0]] = parts
            if len(s3_records) == len(all_target_s3):
                break

print("=== REAL MATCH SAMPLES FROM GROUND TRUTH ===")
for s1, matches in s1_targets.items():
    r1 = s1_records.get(s1, [s1, "N/A", "N/A", "N/A"])
    print("\n" + "="*80)
    print(f"[SOURCE 1 (Reference)]: ID: {r1[0]} | Country: {r1[3]}")
    print(f"   Name   : {r1[1]}")
    print(f"   Address: {r1[2]}")
    print("-" * 40 + " MATCHES: " + "-" * 40)
    for m in sorted(matches):
        if m.startswith("S2"):
            r = s2_records.get(m, [m, "N/A", "N/A", "N/A"])
        else:
            r = s3_records.get(m, [m, "N/A", "N/A", "N/A"])
        print(f"[{m[:2]} MATCH] ID: {r[0]} | Country: {r[3] if len(r)>3 else 'N/A'}")
        print(f"   Name   : {r[1] if len(r)>1 else 'N/A'}")
        print(f"   Address: {r[2] if len(r)>2 else 'N/A'}")

