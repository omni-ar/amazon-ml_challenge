import os
import sys
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\train"

# Test on 10,000 S1
s1_dict = {}
with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 10000:
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

LEGAL_SUFFIXES = {'inc', 'llc', 'corp', 'ltd', 'limited', 'pvt', 'private', 'co', 'company', 'enterprises', 'solutions', 'services', 'sa', 'sas', 'sarl'}

ADDRESS_COMMON_WORDS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'drive', 'dr', 'blvd',
    'near', 'opp', 'opposite', 'floor', 'plot', 'phase', 'sector', 'block', 'no',
    'hno', 'flat', 'city', 'state', 'india', 'us', 'usa', 'france', 'north', 'south',
    'east', 'west', 'null', 'nan', 'main'
}

def clean_name(name):
    name = re.sub(r'\.(com|in|org|net)', ' ', name, flags=re.IGNORECASE)
    name = re.sub(r'[^\w\s]', ' ', name.lower())
    return set([t for t in name.split() if t not in LEGAL_SUFFIXES and len(t) > 1])

def clean_address(addr):
    # strip leading zeros from numbers (e.g. 01109 -> 1109, 31D -> 31)
    nums = set(re.findall(r'\b0*(\d{2,})\b', addr))
    # extract distinctive words
    words = re.sub(r'[^\w\s]', ' ', addr.lower()).split()
    dist_words = set([w for w in words if len(w) >= 4 and w not in ADDRESS_COMMON_WORDS and not w.isdigit()])
    return nums, dist_words

total = 0
found = 0
for s1, targets in gt_dict.items():
    r1 = s1_dict[s1]
    s1_t = clean_name(r1["name"])
    s1_nums, s1_words = clean_address(r1["address"])
    s1_pre = re.sub(r'\W+', '', r1["name"].lower())[:5]

    for mid in targets:
        if mid not in all_target_records:
            continue
        total += 1
        r2 = all_target_records[mid]
        s2_t = clean_name(r2["name"])
        s2_nums, s2_words = clean_address(r2["address"])
        s2_pre = re.sub(r'\W+', '', r2["name"].lower())[:5]

        # Check blocking condition:
        # 1. Any shared name token
        # 2. Shared 5-char alphanumeric prefix
        # 3. Shared address numbers
        # 4. At least 2 shared distinctive address words
        hit = (
            bool(s1_t & s2_t) or
            (s1_pre and s2_pre and s1_pre == s2_pre) or
            bool(s1_nums & s2_nums) or
            (len(s1_words & s2_words) >= 2)
        )
        if hit:
            found += 1

print(f"\nTested on {total:,} true pairs across 10,000 S1 records:")
print(f"Recall: {found:,} / {total:,} = {found/total*100:.3f}%")
print(f"Missed: {total - found} pairs ({ (total-found)/total*100:.3f}%)")
