import os
import sys
import gc
import time
import json
import subprocess
import polars as pl
import numpy as np

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src directory is in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import Config
from normalize import normalise
from blocking import block_country
from features import pair_features
from model import train_lgbm, tune_decision, decide, macro_f05

T0 = time.time()
def log(*msg):
    print(f"[{time.time()-T0:7.1f}s]", *msg, flush=True)

def id_num(col: str) -> pl.Expr:
    return pl.col(col).str.slice(3).cast(pl.Int64)

def scan_tsv(path: str) -> pl.LazyFrame:
    return pl.scan_csv(
        path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8-lossy"
    ).with_columns(pl.all().fill_null(""))

def read_tsv(path: str) -> pl.DataFrame:
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8-lossy"
    ).with_columns(pl.all().fill_null(""))

def prepare_split(s1: pl.DataFrame, cand: pl.DataFrame):
    s1 = normalise(s1).with_row_index("idx")
    cand = normalise(cand).with_row_index("idx")
    s1 = s1.with_columns(pl.len().over(["country", "name_core"]).cast(pl.UInt32).alias("s1_namefreq"))
    return s1, cand

def run():
    log("="*70)
    log("STARTING BUSINESS ENTITY RESOLUTION PIPELINE")
    log("="*70)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. LOAD & PREPARE TRAINING DATA
    # -------------------------------------------------------------
    log("Step 1: Sampling and preparing training set for model training...")
    gt_raw = read_tsv(Config.TRAIN_GT_PATH)
    gt = (gt_raw.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
               .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "cand_id"}))
    del gt_raw
    
    # Sample ~60,000 S1 records (using deterministic hash)
    # 60,000 S1 gives ~600,000 candidate pairs for training, which fits easily in RAM and trains fast
    SAMPLE_PCT = 3  # 3% of 2.2M is ~66,000 S1 records
    s1_scan = scan_tsv(Config.TRAIN_S1_PATH)
    s1_tr = s1_scan.filter((id_num("entity_id") * 2654435761 % 1000003) % 100 < SAMPLE_PCT).collect()
    
    # Keep matching ground truth candidates and a realistic random sample of all candidates
    # (Without filtering out candidates that matched other S1 entities, providing realistic hard negatives)
    gt_sample = gt.join(s1_tr.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
    want_cands = gt_sample.select(pl.col("cand_id").alias("entity_id"), pl.lit(True).alias("_w"))
    
    rnd_mask = (id_num("entity_id") * 40503 % 1000033) % 100 < SAMPLE_PCT
    cand_tr = (pl.concat([scan_tsv(Config.TRAIN_S2_PATH), scan_tsv(Config.TRAIN_S3_PATH)])
               .join(want_cands.lazy(), on="entity_id", how="left")
               .filter(pl.col("_w").is_not_null() | rnd_mask)
               .drop("_w").collect())
    del want_cands, gt
    gc.collect()
    
    log(f"Training sample loaded: {s1_tr.height:,} S1 entities, {cand_tr.height:,} candidate records.")
    s1_tr, cand_tr = prepare_split(s1_tr, cand_tr)
    
    # -------------------------------------------------------------
    # 2. BLOCKING ON TRAINING SAMPLE
    # -------------------------------------------------------------
    log("Step 2: Running blocking on training sample...")
    parts = []
    for c in s1_tr["country"].unique().to_list():
        s1c = s1_tr.filter(pl.col("country") == c)
        cc = cand_tr.filter(pl.col("country") == c)
        log(f"  Training blocking country={c}: S1={s1c.height:,}, cand={cc.height:,}")
        p = block_country(s1c, cc, k=Config.K_CANDIDATES, key_cap=max(15, int(Config.KEY_CAP * SAMPLE_PCT / 100)))
        if p is not None:
            parts.append(p)
    pairs_tr = pl.concat(parts)
    log(f"Blocking generated {pairs_tr.height:,} candidate pairs ({pairs_tr.height/s1_tr.height:.1f} per S1).")
    
    # Join entity IDs and ground truth labels
    ids1 = s1_tr.select(pl.col("idx").alias("i1"), pl.col("entity_id").alias("s1_id"), pl.col("country"))
    ids2 = cand_tr.select(pl.col("idx").alias("i2"), pl.col("entity_id").alias("cand_id"))
    pairs_tr = pairs_tr.join(ids1, on="i1").join(ids2, on="i2")
    
    lab = gt_sample.with_columns(pl.lit(1, pl.Int8).alias("y"))
    pairs_tr = pairs_tr.join(lab, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("y").fill_null(0))
    rec = pairs_tr["y"].sum() / gt_sample.height
    log(f"Blocking recall on training slice: {rec:.4f} ({pairs_tr['y'].sum():,} / {gt_sample.height:,} true links found)")
    
    # Train / Val Split
    pairs_tr = pairs_tr.with_columns(((id_num("s1_id") * 2246822519 % 1000003) % 100 < 20).alias("is_val"))
    val_s1 = s1_tr.filter(((id_num("entity_id") * 2246822519 % 1000003) % 100 < 20))["entity_id"]
    fit_pairs = pairs_tr.filter(~pl.col("is_val"))
    val_pairs = pairs_tr.filter(pl.col("is_val"))
    del pairs_tr
    gc.collect()
    
    # -------------------------------------------------------------
    # 3. FEATURE EXTRACTION & MODEL TRAINING
    # -------------------------------------------------------------
    log("Step 3: Extracting pairwise features for training and validation...")
    fit_pairs = pair_features(fit_pairs, s1_tr, cand_tr, feat_chunk=Config.FEAT_CHUNK)
    val_pairs = pair_features(val_pairs, s1_tr, cand_tr, feat_chunk=Config.FEAT_CHUNK)
    del s1_tr, cand_tr
    gc.collect()
    
    log(f"Training LightGBM model on {fit_pairs.height:,} training pairs (evaluated on {val_pairs.height:,} val pairs)...")
    model = train_lgbm(fit_pairs, val_pairs, Config.FEATURES, seed=Config.SEED)
    model.save_model(os.path.join(Config.WORK_DIR, "lgbm_model.txt"))
    
    # Evaluate and optimize thresholds on validation set
    val_sc = val_pairs.select("s1_id", "cand_id", pl.Series("p", model.predict(val_pairs.select(Config.FEATURES).to_numpy())))
    gt_val = gt_sample.join(pl.DataFrame({"s1_id": val_s1}), on="s1_id", how="semi")
    (best_m, best_cfg), (glob_m, glob_t) = tune_decision(val_sc, gt_val, val_s1)
    
    log(f"VALIDATION Macro F0.5 Score: {best_m['macroF05']:.4f}")
    log(f"  Precision: {best_m['macroP']:.4f} | Recall: {best_m['macroR']:.4f} | Singleton Accuracy: {best_m['singleton_acc']:.4f}")
    log(f"  Optimal Decision Config: {best_cfg}")
    
    with open(os.path.join(Config.WORK_DIR, "val_metrics.json"), "w") as f:
        json.dump({"metrics": best_m, "config": best_cfg}, f, indent=2)
        
    del fit_pairs, val_pairs, val_sc, gt_val, gt_sample
    gc.collect()
    
    # -------------------------------------------------------------
    # 4. FULL TEST SET INFERENCE (COUNTRY BY COUNTRY)
    # -------------------------------------------------------------
    log("\n" + "="*70)
    log("Step 4: Executing Full Test Inference Partition-by-Partition...")
    log("="*70)
    
    # We process France first, then US, then India
    countries = ["France", "US", "India"]
    all_matching_dfs = []
    all_candidate_dfs = []
    
    for country in countries:
        log(f"\n>>> PROCESSING COUNTRY: {country.upper()} <<<")
        # 1. Scan and filter S1 for this country
        s1_c = scan_tsv(Config.TEST_S1_PATH).filter(pl.col("country") == country).collect()
        # 2. Scan and filter S2 and S3 for this country
        s2_c = scan_tsv(Config.TEST_S2_PATH).filter(pl.col("country") == country).collect()
        s3_c = scan_tsv(Config.TEST_S3_PATH).filter(pl.col("country") == country).collect()
        cand_c = pl.concat([s2_c, s3_c])
        del s2_c, s3_c
        gc.collect()
        
        log(f"  {country}: {s1_c.height:,} S1 entities, {cand_c.height:,} candidates.")
        s1_c, cand_c = prepare_split(s1_c, cand_c)
        
        # 3. Blocking for this country
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
        
        # 4. Feature Extraction & Scoring in Chunks
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
        
        # 5. Apply Tuned Decision Rule
        matches_c = decide(sc_c, best_cfg["t1"], best_cfg["t2"], best_cfg["t3"], best_cfg["exclusive"])
        log(f"  Matches decided: {matches_c.height:,} links for {s1_c.height:,} entities.")
        
        # 6. Aggregate to required TSV format
        all_s1_c = s1_c.select(pl.col("entity_id").alias("source1_entity_id"))
        
        # Candidates
        cands_agg = (sc_c.select("s1_id", "cand_id")
                          .group_by("s1_id")
                          .agg(pl.col("cand_id").unique().sort().str.join(",").alias("candidate_entity_ids")))
        out_cands_c = all_s1_c.join(cands_agg.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left").fill_null("")
        all_candidate_dfs.append(out_cands_c)
        
        # Matches
        matches_agg = (matches_c.group_by("s1_id")
                               .agg(pl.col("cand_id").unique().sort().str.join(",").alias("matched_entity_ids")))
        out_matches_c = all_s1_c.join(matches_agg.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left").fill_null("")
        all_matching_dfs.append(out_matches_c)
        
        non_empty = (out_matches_c["matched_entity_ids"] != "").mean()
        log(f"  {country} done: {non_empty*100:.1f}% entities matched at least 1 record.")
        
        del s1_c, cand_c, sc_c, matches_c, out_cands_c, out_matches_c
        gc.collect()
        
    # -------------------------------------------------------------
    # 5. COMBINE & EXPORT COMPLETE TSVs
    # -------------------------------------------------------------
    log("\n" + "="*70)
    log("Step 5: Combining partitions and writing full output files...")
    log("="*70)
    
    full_matches = pl.concat(all_matching_dfs)
    full_candidates = pl.concat(all_candidate_dfs)
    
    # Save standard files
    full_matches.write_csv(Config.MATCHING_OUT, separator="\t", quote_style="never")
    full_candidates.write_csv(Config.CANDIDATE_OUT, separator="\t", quote_style="never")
    
    # Save (2) files as requested
    full_matches.write_csv(Config.MATCHING_OUT_2, separator="\t", quote_style="never")
    full_candidates.write_csv(Config.CANDIDATE_OUT_2, separator="\t", quote_style="never")
    
    log(f"Saved {full_matches.height:,} rows to:")
    log(f"  - {Config.MATCHING_OUT}")
    log(f"  - {Config.MATCHING_OUT_2}")
    log(f"Saved {full_candidates.height:,} rows to:")
    log(f"  - {Config.CANDIDATE_OUT}")
    log(f"  - {Config.CANDIDATE_OUT_2}")
    
    # -------------------------------------------------------------
    # 6. RUN OFFICIAL VALIDATION SCRIPT
    # -------------------------------------------------------------
    log("\n" + "="*70)
    log("Step 6: Running official submission validator on output files...")
    log("="*70)
    validator_path = os.path.join(Config.WORKSPACE_ROOT, "student_resource", "utils", "validate_submission.py")
    test_dir = os.path.join(Config.DATASET_DIR, "test")
    
    # Validate matching_results(2).tsv and candidate_pairs(2).tsv
    cmd2 = [
        sys.executable, validator_path,
        "--matching", Config.MATCHING_OUT_2,
        "--candidate", Config.CANDIDATE_OUT_2,
        "--test-dir", test_dir
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    print(res2.stdout)
    if res2.stderr:
        print(res2.stderr)
        
    if res2.returncode == 0:
        log("SUCCESS! Official submission validation PASSED for (2) files (Exit code 0)!")
    else:
        log(f"Validation FAILED for (2) files with exit code {res2.returncode}")
        
    # -------------------------------------------------------------
    # 7. PACKAGE SUBMISSION ARCHIVES
    # -------------------------------------------------------------
    log("\n" + "="*70)
    log("Step 7: Packaging submission(2).zip archive...")
    log("="*70)
    packager_path = os.path.join(Config.WORKSPACE_ROOT, "package_submission.py")
    subprocess.run([sys.executable, packager_path], check=True)
    log("Pipeline run complete!")

if __name__ == "__main__":
    run()

