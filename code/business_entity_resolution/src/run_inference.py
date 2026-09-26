"""
Inference-only pipeline: loads saved LightGBM model and val_metrics, 
then runs full test inference country-by-country.
Skips training entirely (~100s saved).
"""
import os, sys, gc, time, json, subprocess
import polars as pl
import numpy as np
import lightgbm as lgb

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import Config
from normalize import normalise
from blocking import block_country
from features import pair_features
from model import decide

T0 = time.time()
def log(*msg):
    print(f"[{time.time()-T0:7.1f}s]", *msg, flush=True)

def prepare_split(s1, cand):
    s1 = normalise(s1).with_row_index("idx")
    cand = normalise(cand).with_row_index("idx")
    s1 = s1.with_columns(pl.len().over(["country", "name_core"]).cast(pl.UInt32).alias("s1_namefreq"))
    return s1, cand

def scan_tsv(path):
    return pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8-lossy").with_columns(pl.all().fill_null(""))

def run_inference():
    log("="*70)
    log("INFERENCE-ONLY PIPELINE (reusing saved model)")
    log("="*70)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Load saved model and config
    model_path = os.path.join(Config.WORK_DIR, "lgbm_model.txt")
    metrics_path = os.path.join(Config.WORK_DIR, "val_metrics.json")
    
    model = lgb.Booster(model_file=model_path)
    log(f"Loaded saved model from {model_path}")
    
    with open(metrics_path) as f:
        saved = json.load(f)
    best_cfg = saved["config"]
    log(f"Loaded decision config: {best_cfg}")
    log(f"Saved validation F0.5={saved['metrics']['macroF05']:.4f}, P={saved['metrics']['macroP']:.4f}, R={saved['metrics']['macroR']:.4f}")

    # Full test inference country-by-country
    countries = ["France", "US", "India"]
    all_matching_dfs = []
    all_candidate_dfs = []

    for country in countries:
        log(f"\n>>> PROCESSING COUNTRY: {country.upper()} <<<")
        s1_c = scan_tsv(Config.TEST_S1_PATH).filter(pl.col("country") == country).collect()
        s2_c = scan_tsv(Config.TEST_S2_PATH).filter(pl.col("country") == country).collect()
        s3_c = scan_tsv(Config.TEST_S3_PATH).filter(pl.col("country") == country).collect()
        cand_c = pl.concat([s2_c, s3_c])
        del s2_c, s3_c
        gc.collect()

        log(f"  {country}: {s1_c.height:,} S1 entities, {cand_c.height:,} candidates.")
        s1_c, cand_c = prepare_split(s1_c, cand_c)

        # Blocking
        log(f"  Running blocking for {country}...")
        pairs_c = block_country(s1_c, cand_c, k=Config.K_CANDIDATES, key_cap=Config.KEY_CAP, s1_chunk=Config.S1_CHUNK)
        if pairs_c is None or pairs_c.height == 0:
            log(f"  [WARN] No pairs generated for {country}!")
            continue

        log(f"  Generated {pairs_c.height:,} candidate pairs ({pairs_c.height/s1_c.height:.1f} per S1).")

        # Join entity IDs
        ids1_c = s1_c.select(pl.col("idx").alias("i1"), pl.col("entity_id").alias("s1_id"))
        ids2_c = cand_c.select(pl.col("idx").alias("i2"), pl.col("entity_id").alias("cand_id"))
        pairs_c = pairs_c.join(ids1_c, on="i1").join(ids2_c, on="i2")
        del ids1_c, ids2_c

        # Feature extraction & scoring in chunks
        log(f"  Extracting features and scoring {pairs_c.height:,} pairs...")
        scored_chunks = []
        for i in range(0, pairs_c.height, Config.FEAT_CHUNK):
            ch = pair_features(pairs_c.slice(i, Config.FEAT_CHUNK), s1_c, cand_c, feat_chunk=Config.FEAT_CHUNK)
            p_scores = model.predict(ch.select(Config.FEATURES).to_numpy())
            scored_chunks.append(ch.select("s1_id", "cand_id", pl.Series("p", p_scores)))
            del ch
            gc.collect()

        sc_c = pl.concat(scored_chunks)
        del scored_chunks, pairs_c
        gc.collect()

        # Decision
        matches_c = decide(sc_c, best_cfg["t1"], best_cfg["t2"], best_cfg["t3"], best_cfg["exclusive"])
        log(f"  Matches decided: {matches_c.height:,} links for {s1_c.height:,} entities.")

        # Aggregate
        all_s1_c = s1_c.select(pl.col("entity_id").alias("source1_entity_id"))

        cands_agg = (sc_c.select("s1_id", "cand_id")
                          .group_by("s1_id")
                          .agg(pl.col("cand_id").unique().sort().str.join(",").alias("candidate_entity_ids")))
        out_cands_c = all_s1_c.join(cands_agg.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left").fill_null("")
        all_candidate_dfs.append(out_cands_c)

        matches_agg = (matches_c.group_by("s1_id")
                               .agg(pl.col("cand_id").unique().sort().str.join(",").alias("matched_entity_ids")))
        out_matches_c = all_s1_c.join(matches_agg.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left").fill_null("")
        all_matching_dfs.append(out_matches_c)

        non_empty = (out_matches_c["matched_entity_ids"] != "").mean()
        log(f"  {country} done: {non_empty*100:.1f}% entities matched at least 1 record.")

        del s1_c, cand_c, sc_c, matches_c, out_cands_c, out_matches_c
        gc.collect()

    # Combine & export
    log("\n" + "="*70)
    log("Combining partitions and writing full output files...")
    log("="*70)

    full_matches = pl.concat(all_matching_dfs)
    full_candidates = pl.concat(all_candidate_dfs)

    full_matches.write_csv(Config.MATCHING_OUT, separator="\t", quote_style="never")
    full_candidates.write_csv(Config.CANDIDATE_OUT, separator="\t", quote_style="never")
    full_matches.write_csv(Config.MATCHING_OUT_2, separator="\t", quote_style="never")
    full_candidates.write_csv(Config.CANDIDATE_OUT_2, separator="\t", quote_style="never")

    log(f"Saved {full_matches.height:,} rows to:")
    log(f"  - {Config.MATCHING_OUT}")
    log(f"  - {Config.MATCHING_OUT_2}")
    log(f"Saved {full_candidates.height:,} rows to:")
    log(f"  - {Config.CANDIDATE_OUT}")
    log(f"  - {Config.CANDIDATE_OUT_2}")

    # Validate
    log("\n" + "="*70)
    log("Running official submission validator...")
    log("="*70)
    validator_path = os.path.join(Config.WORKSPACE_ROOT, "student_resource", "utils", "validate_submission.py")
    test_dir = os.path.join(Config.DATASET_DIR, "test")
    cmd2 = [sys.executable, validator_path, "--matching", Config.MATCHING_OUT_2, "--candidate", Config.CANDIDATE_OUT_2, "--test-dir", test_dir]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    print(res2.stdout)
    if res2.stderr:
        print(res2.stderr)
    if res2.returncode == 0:
        log("SUCCESS! Validation PASSED for (2) files!")
    else:
        log(f"Validation FAILED with exit code {res2.returncode}")

    # Package
    log("\n" + "="*70)
    log("Packaging submission(2).zip...")
    log("="*70)
    packager_path = os.path.join(Config.WORKSPACE_ROOT, "package_submission.py")
    subprocess.run([sys.executable, packager_path], check=True)
    log("Pipeline run complete!")

if __name__ == "__main__":
    run_inference()
