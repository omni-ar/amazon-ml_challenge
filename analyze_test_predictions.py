import os
import polars as pl

s1_path = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\test\test_source1.tsv"
mr_path = r"c:\Users\arjit\Desktop\ml_challenge\output\matching_results.tsv"
cp_path = r"c:\Users\arjit\Desktop\ml_challenge\output\candidate_pairs.tsv"

print("Reading test_source1 and matching_results...")
s1 = pl.read_csv(s1_path, separator="\t", quote_char=None, infer_schema=False).select(["entity_id", "country"])
mr = pl.read_csv(mr_path, separator="\t", quote_char=None, infer_schema=False)
cp = pl.read_csv(cp_path, separator="\t", quote_char=None, infer_schema=False)

df_mr = s1.join(mr, left_on="entity_id", right_on="source1_entity_id")
df_cp = s1.join(cp, left_on="entity_id", right_on="source1_entity_id")

# Analyze by country
for c in ["France", "US", "India"]:
    sub_mr = df_mr.filter(pl.col("country") == c)
    sub_cp = df_cp.filter(pl.col("country") == c)
    
    # matches count
    match_lens = sub_mr["matched_entity_ids"].map_elements(lambda x: len(x.split(",")) if x else 0, return_dtype=pl.Int32)
    cand_lens = sub_cp["candidate_entity_ids"].map_elements(lambda x: len(x.split(",")) if x else 0, return_dtype=pl.Int32)
    
    empty_matches = (match_lens == 0).sum()
    empty_cands = (cand_lens == 0).sum()
    
    print(f"\n{'='*20} {c.upper()} ({sub_mr.height:,} entities) {'='*20}")
    print(f"Empty candidates (0 candidates found): {empty_cands:,} ({empty_cands/sub_mr.height*100:.2f}%)")
    print(f"Average candidates per S1: {cand_lens.mean():.2f}, Median: {cand_lens.median()}")
    print(f"Empty matches (predicted singleton): {empty_matches:,} ({empty_matches/sub_mr.height*100:.2f}%)")
    print(f"Non-empty matches (>=1 match): {sub_mr.height - empty_matches:,} ({(sub_mr.height - empty_matches)/sub_mr.height*100:.2f}%)")
    print(f"Average matches per S1: {match_lens.mean():.2f}")
    
    # Distribution of match counts
    vc = match_lens.value_counts().sort("count", descending=True).head(7)
    print("Match counts distribution:")
    for row in vc.iter_rows(named=True):
        print(f"   {row['matched_entity_ids']} matches: {row['count']:,} ({row['count']/sub_mr.height*100:.2f}%)")
