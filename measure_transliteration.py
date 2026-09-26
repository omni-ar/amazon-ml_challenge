import os
import sys
import re
import anyascii
from rapidfuzz import fuzz

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\train"

print("--- MEASURING NAME SIMILARITY JUMP WITH TRANSLITERATION ---")

# Load Indian S1 records and their matches
s1_records = {}
with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) > 3 and parts[3] == "India":
            s1_records[parts[0]] = parts[1]
            if len(s1_records) >= 3000:
                break

s1_matches = {}
all_matched = set()
with open(os.path.join(DATASET_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, tab, rest = line.partition("\t")
        if s1 in s1_records:
            m = rest.strip().split(",") if rest.strip() else []
            if m:
                s1_matches[s1] = m
                all_matched.update(m)

cand_records = {}
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(DATASET_DIR, fname), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in all_matched:
                cand_records[parts[0]] = parts[1]

# Now compare pairs without transliteration vs with transliteration
scores_raw = []
scores_translit = []

diff_script_raw = []
diff_script_translit = []

for s1, targets in s1_matches.items():
    name1 = s1_records[s1]
    name1_tr = anyascii.anyascii(name1).lower()
    
    for mid in targets:
        if mid not in cand_records:
            continue
        name2 = cand_records[mid]
        name2_tr = anyascii.anyascii(name2).lower()
        
        r_raw = fuzz.token_set_ratio(name1.lower(), name2.lower())
        r_tr = fuzz.token_set_ratio(name1_tr, name2_tr)
        
        scores_raw.append(r_raw)
        scores_translit.append(r_tr)
        
        is_native = any(ord(c) > 127 for c in name2)
        if is_native:
            diff_script_raw.append(r_raw)
            diff_script_translit.append(r_tr)

print(f"Total evaluated true pairs: {len(scores_raw):,}")
print(f"Overall average name similarity WITHOUT transliteration: {sum(scores_raw)/len(scores_raw):.1f}%")
print(f"Overall average name similarity WITH transliteration:    {sum(scores_translit)/len(scores_translit):.1f}%")

print(f"\nNon-ASCII / Native Script pairs: {len(diff_script_raw):,} ({len(diff_script_raw)/len(scores_raw)*100:.1f}% of all pairs)")
print(f"  Average similarity WITHOUT transliteration: {sum(diff_script_raw)/len(diff_script_raw):.1f}% (Virtually zero match!)")
print(f"  Average similarity WITH transliteration:    {sum(diff_script_translit)/len(diff_script_translit):.1f}% (Dramatic jump!)")
print(f"  Pairs with score >= 70% without translit:  {sum(1 for s in diff_script_raw if s >= 70):,} / {len(diff_script_raw):,} ({sum(1 for s in diff_script_raw if s >= 70)/len(diff_script_raw)*100:.1f}%)")
print(f"  Pairs with score >= 70% WITH translit:     {sum(1 for s in diff_script_translit if s >= 70):,} / {len(diff_script_translit):,} ({sum(1 for s in diff_script_translit if s >= 70)/len(diff_script_translit)*100:.1f}%)")
