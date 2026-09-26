import os
import sys
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\train"

# Re-run for missed pairs
s1_dict = {}
with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 5000:
            break
        parts = line.strip().split("\t")
        s1_dict[parts[0]] = {"name": parts[1] if len(parts) > 1 else "", "address": parts[2] if len(parts) > 2 else "", "country": parts[3] if len(parts) > 3 else ""}

gt_dict = {}
all_matched_ids = set()
with open(os.path.join(DATASET_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, tab, rest = line.partition("\t")
        if s1 in s1_dict:
            m = rest.strip().split(",") if rest.strip() else []
            gt_dict[s1] = set(m)
            all_matched_ids.update(m)

all_target_records = {}
for src in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(DATASET_DIR, src), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in all_matched_ids:
                all_target_records[parts[0]] = {"name": parts[1] if len(parts) > 1 else "", "address": parts[2] if len(parts) > 2 else "", "country": parts[3] if len(parts) > 3 else ""}

LEGAL_SUFFIXES = {'inc', 'llc', 'corp', 'ltd', 'limited', 'pvt', 'private', 'co', 'company'}

def clean_name(name):
    name = re.sub(r'\.(com|in|org|net)', ' ', name, flags=re.IGNORECASE)
    name = re.sub(r'[^\w\s]', ' ', name.lower())
    return set([t for t in name.split() if t not in LEGAL_SUFFIXES and len(t) > 1])

def extract_numbers(addr):
    return set(re.findall(r'\b\d{2,}\b', addr))

missed = []
for s1, targets in gt_dict.items():
    r1 = s1_dict[s1]
    s1_t = clean_name(r1["name"])
    s1_n = extract_numbers(r1["address"])
    s1_pre = re.sub(r'\W+', '', r1["name"].lower())[:6]

    for mid in targets:
        if mid not in all_target_records:
            continue
        r2 = all_target_records[mid]
        s2_t = clean_name(r2["name"])
        s2_n = extract_numbers(r2["address"])
        s2_pre = re.sub(r'\W+', '', r2["name"].lower())[:6]

        hit = bool(s1_t & s2_t) or (s1_pre and s2_pre and s1_pre == s2_pre) or bool(s1_n & s2_n)
        if not hit:
            missed.append((s1, r1, mid, r2))

print(f"Total missed in 5000 S1: {len(missed)}")
print("Sample missed pairs:")
for s1, r1, mid, r2 in missed[:10]:
    print("-" * 60)
    print(f"S1: {s1} | Country: {r1['country']}")
    print(f"   Name   : {r1['name']}")
    print(f"   Address: {r1['address']}")
    print(f"Target: {mid} | Country: {r2['country']}")
    print(f"   Name   : {r2['name']}")
    print(f"   Address: {r2['address']}")
