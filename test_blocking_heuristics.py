import os
import sys
import re
from collections import Counter, defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\train"

print("Loading a sample of 20,000 S1 records and their matches to analyze blocking recall...")

# Load first 20,000 S1
s1_dict = {}
with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 20000:
            break
        parts = line.strip().split("\t")
        s1_dict[parts[0]] = {
            "name": parts[1] if len(parts) > 1 else "",
            "address": parts[2] if len(parts) > 2 else "",
            "country": parts[3] if len(parts) > 3 else ""
        }

# Load ground truth for these 20k
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

print(f"Loaded {len(s1_dict)} S1 entities. Total true matches to find: {len(all_matched_ids)}")

# Load the matching S2 and S3 records
s2_matches = {}
with open(os.path.join(DATASET_DIR, "train_source2.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in all_matched_ids:
            s2_matches[parts[0]] = {
                "name": parts[1] if len(parts) > 1 else "",
                "address": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            }

s3_matches = {}
with open(os.path.join(DATASET_DIR, "train_source3.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in all_matched_ids:
            s3_matches[parts[0]] = {
                "name": parts[1] if len(parts) > 1 else "",
                "address": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else ""
            }

all_target_records = {**s2_matches, **s3_matches}
print(f"Retrieved {len(all_target_records)} target records across S2 and S3.")

# Normalization helper
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'llc', 'corp', 'corporation', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'enterprises', 'enterprise', 'services',
    'solutions', 'group', 'holdings', 'industries', 'associates', 'llp', 'gmbh',
    'sa', 'sas', 'sarl', 'cie'
}

def clean_name(name):
    # remove domain suffixes
    name = re.sub(r'\.(com|in|org|net|co|io|biz|info)', ' ', name, flags=re.IGNORECASE)
    # remove punctuation
    name = re.sub(r'[^\w\s]', ' ', name.lower())
    tokens = [t for t in name.split() if t not in LEGAL_SUFFIXES and len(t) > 1]
    return tokens

def extract_numbers(addr):
    # extract all number sequences of length >= 2
    return set(re.findall(r'\b\d{2,}\b', addr))

# Evaluate recall under different blocking keys
# Key 1: Exact first significant name token
# Key 2: Any 2 shared name tokens
# Key 3: Cleaned alphanumeric name prefix (e.g. first 6 chars of stripped name)
# Key 4: Shared number from address + at least one letter/token

name_token_hits = 0
number_hits = 0
prefix_hits = 0
either_hits = 0
total_pairs = 0

diff_script_count = 0
diff_script_found_by_num = 0

for s1, targets in gt_dict.items():
    r1 = s1_dict[s1]
    s1_tokens = set(clean_name(r1["name"]))
    s1_nums = extract_numbers(r1["address"])
    s1_raw_alphanumeric = re.sub(r'\W+', '', r1["name"].lower())
    s1_prefix = s1_raw_alphanumeric[:6] if len(s1_raw_alphanumeric) >= 4 else s1_raw_alphanumeric

    for mid in targets:
        if mid not in all_target_records:
            continue
        total_pairs += 1
        r2 = all_target_records[mid]
        s2_tokens = set(clean_name(r2["name"]))
        s2_nums = extract_numbers(r2["address"])
        s2_raw_alphanumeric = re.sub(r'\W+', '', r2["name"].lower())
        s2_prefix = s2_raw_alphanumeric[:6] if len(s2_raw_alphanumeric) >= 4 else s2_raw_alphanumeric

        # check script difference (e.g. S1 has ASCII, S2 is non-ASCII)
        is_s1_ascii = all(ord(c) < 128 for c in r1["name"])
        is_s2_ascii = all(ord(c) < 128 for c in r2["name"])
        if is_s1_ascii and not is_s2_ascii:
            diff_script_count += 1
            if bool(s1_nums & s2_nums):
                diff_script_found_by_num += 1

        hit_token = bool(s1_tokens & s2_tokens)
        hit_prefix = (s1_prefix and s2_prefix and (s1_prefix == s2_prefix or s1_prefix in s2_raw_alphanumeric or s2_prefix in s1_raw_alphanumeric))
        hit_number = bool(s1_nums and s2_nums and (s1_nums & s2_nums))

        if hit_token:
            name_token_hits += 1
        if hit_prefix:
            prefix_hits += 1
        if hit_number:
            number_hits += 1
        if hit_token or hit_prefix or hit_number:
            either_hits += 1

print("\n--- BLOCKING KEY RECALL ON 20,000 GROUND TRUTH ENTITIES ---")
print(f"Total evaluated true pairs: {total_pairs:,}")
print(f"1. Any shared significant name token: {name_token_hits:,} ({name_token_hits/total_pairs*100:.2f}%)")
print(f"2. Alphanumeric prefix / containment: {prefix_hits:,} ({prefix_hits/total_pairs*100:.2f}%)")
print(f"3. Shared address numbers:            {number_hits:,} ({number_hits/total_pairs*100:.2f}%)")
print(f"Combined (Tokens OR Prefix OR Numbers): {either_hits:,} ({either_hits/total_pairs*100:.2f}%)")
print(f"\nNon-ASCII / Transliterated pairs: {diff_script_count:,}")
if diff_script_count > 0:
    print(f"  Found via address numbers: {diff_script_found_by_num:,} ({diff_script_found_by_num/diff_script_count*100:.2f}%)")
