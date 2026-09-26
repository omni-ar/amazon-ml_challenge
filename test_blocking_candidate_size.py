import os
import sys
import re
import time
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\train"

print("--- TESTING CANDIDATE GENERATION ENGINE (RECALL & POOL SIZE) ---")

# Let's take a sample of 5,000 S1 entities (US + India)
s1_records = []
s1_ids = set()
with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 5000:
            break
        parts = line.strip().split("\t")
        eid = parts[0]
        s1_ids.add(eid)
        s1_records.append({
            "id": eid,
            "name": parts[1] if len(parts) > 1 else "",
            "address": parts[2] if len(parts) > 2 else "",
            "country": parts[3] if len(parts) > 3 else ""
        })

# Load ground truth for these 5,000
gt_matches = {}
all_true_targets = set()
with open(os.path.join(DATASET_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, tab, rest = line.partition("\t")
        if s1 in s1_ids:
            targets = rest.strip().split(",") if rest.strip() else []
            gt_matches[s1] = set(targets)
            all_true_targets.update(targets)

print(f"Loaded 5,000 S1 entities. Ground truth has {len(all_true_targets)} true matches.")

# To test realistic blocking, let's load a pool of 100,000 S2 and 100,000 S3 records
# including ALL true targets for our 5k S1
s2_pool = []
s3_pool = []
loaded_targets = set()

print("Loading target pool (including true targets + noise)...")
with open(os.path.join(DATASET_DIR, "train_source2.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.strip().split("\t")
        eid = parts[0]
        if eid in all_true_targets or len(s2_pool) < 100000:
            s2_pool.append({
                "id": eid,
                "name": parts[1] if len(parts) > 1 else "",
                "address": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            })
            if eid in all_true_targets:
                loaded_targets.add(eid)
        if len(s2_pool) >= 100000 and len(loaded_targets) == len(all_true_targets):
            break

with open(os.path.join(DATASET_DIR, "train_source3.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.strip().split("\t")
        eid = parts[0]
        if eid in all_true_targets or len(s3_pool) < 100000:
            s3_pool.append({
                "id": eid,
                "name": parts[1] if len(parts) > 1 else "",
                "address": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            })
            if eid in all_true_targets:
                loaded_targets.add(eid)
        if len(s3_pool) >= 100000 and len(loaded_targets) == len(all_true_targets):
            break

print(f"Candidate pool loaded: {len(s2_pool):,} S2 records, {len(s3_pool):,} S3 records.")
print(f"Verified {len(loaded_targets)} / {len(all_true_targets)} true matches in pool.")

LEGAL_SUFFIXES = {'inc', 'llc', 'corp', 'ltd', 'limited', 'pvt', 'private', 'co', 'company', 'enterprises', 'solutions', 'services', 'sa', 'sas', 'sarl'}
ADDRESS_COMMON = {'street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'drive', 'dr', 'blvd', 'near', 'opp', 'floor', 'plot', 'phase', 'sector', 'block', 'no', 'hno', 'flat', 'city', 'state', 'india', 'us', 'usa', 'france', 'north', 'south', 'east', 'west', 'null', 'nan', 'main'}

def extract_keys(rec):
    keys = []
    # 1. Cleaned name tokens
    name_clean = re.sub(r'\.(com|in|org|net)', ' ', rec["name"], flags=re.IGNORECASE)
    name_clean = re.sub(r'[^\w\s]', ' ', name_clean.lower())
    tokens = [t for t in name_clean.split() if t not in LEGAL_SUFFIXES and len(t) >= 3]
    for t in tokens[:3]: # top 3 tokens
        keys.append(f"tok:{t}")

    # 2. Alphanumeric 5-char prefix
    alphanumeric = re.sub(r'\W+', '', rec["name"].lower())
    if len(alphanumeric) >= 5:
        keys.append(f"pre:{alphanumeric[:5]}")

    # 3. Address numbers
    nums = set(re.findall(r'\b0*(\d{2,})\b', rec["address"]))
    for num in sorted(nums)[:2]:
        keys.append(f"num:{num}")

    return keys

# Build inverted index by country
t0 = time.time()
print("\nBuilding inverted index...")
index = defaultdict(lambda: defaultdict(list))
# index[country][key] -> list of candidate ids

all_candidates = s2_pool + s3_pool
for cand in all_candidates:
    c = cand["country"]
    cid = cand["id"]
    for key in extract_keys(cand):
        index[c][key].append(cid)

t_index = time.time() - t0
print(f"Index built in {t_index:.2f} seconds.")

# Query inverted index for the 5,000 S1 records
t0 = time.time()
candidate_counts = []
total_true = 0
found_true = 0

# Limit candidates per S1 to top K most frequent key matches to keep pool tight
MAX_CANDIDATES = 50

for s1 in s1_records:
    c = s1["country"]
    keys = extract_keys(s1)
    # Count frequency of retrieved candidate IDs
    scores = defaultdict(int)
    for k in keys:
        for cid in index[c].get(k, []):
            scores[cid] += 1

    # Sort by number of matching keys
    sorted_cands = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:MAX_CANDIDATES]
    candidate_set = set(sorted_cands)
    candidate_counts.append(len(candidate_set))

    # Evaluate against GT
    true_set = gt_matches.get(s1["id"], set())
    if true_set:
        total_true += len(true_set)
        found_true += len(true_set & candidate_set)

t_query = time.time() - t0
print(f"Queried 5,000 entities in {t_query:.2f} seconds ({len(s1_records)/t_query:.0f} entities/sec).")
print(f"\n--- RESULTS ---")
print(f"Average candidates per S1: {sum(candidate_counts)/len(candidate_counts):.1f}")
print(f"Median candidates per S1: {sorted(candidate_counts)[len(candidate_counts)//2]}")
print(f"Max candidates per S1: {max(candidate_counts)}")
print(f"Recall on true matches: {found_true:,} / {total_true:,} = {found_true/total_true*100:.2f}%")
