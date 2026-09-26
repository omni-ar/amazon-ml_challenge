import os
import polars as pl
import lightgbm as lgb
import numpy as np

def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> dict:
    """pred/gt: frames (s1_id, cand_id); s1_ids: all S1 ids being evaluated. Returns dict."""
    base = pl.DataFrame({"s1_id": s1_ids})
    nt = gt.group_by("s1_id").agg(pl.len().alias("nt"))
    npd = pred.group_by("s1_id").agg(pl.len().alias("np"))
    tp = pred.join(gt, on=["s1_id", "cand_id"]).group_by("s1_id").agg(pl.len().alias("tp"))
    
    d = (base.join(nt, on="s1_id", how="left").join(npd, on="s1_id", how="left").join(tp, on="s1_id", how="left")
             .fill_null(0).with_columns(pl.col(["nt", "np", "tp"]).cast(pl.Float64)))
             
    d = d.with_columns(
        pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
          .when(pl.col("np") == 0).then(0.0)
          .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp"))))
          .alias("f"),
        pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(None).alias("prec"),
        pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(None).alias("rec")
    )
    single = d.filter(pl.col("nt") == 0)
    return {
        "macroF05": float(d["f"].mean()),
        "macroP": float(d["prec"].mean()) if d["prec"].is_not_null().any() else 0.0,
        "macroR": float(d["rec"].mean()) if d["rec"].is_not_null().any() else 0.0,
        "singleton_acc": float(single["f"].mean()) if single.height else None,
        "pct_entities_with_FP": float((d["np"] > d["tp"]).mean()),
        "n": d.height
    }

def decide(sc: pl.DataFrame, t1: float, t2: float, t3: float, exclusive: bool = True) -> pl.DataFrame:
    """sc: (s1_id, cand_id, p). Exclusivity + rank-aware thresholds."""
    if exclusive and sc.height > 0:
        sc = sc.filter(pl.col("p") >= pl.col("p").max().over("cand_id"))
        
    if sc.height == 0:
        return sc.select("s1_id", "cand_id")
        
    sc = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("r"))
    thr = pl.when(pl.col("r") == 1).then(t1).when(pl.col("r") == 2).then(t2).otherwise(t3)
    return sc.filter(pl.col("p") >= thr).select("s1_id", "cand_id")

def tune_decision(sc: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> tuple:
    best = None
    grid1 = [0.4, 0.5, 0.6, 0.7, 0.8]
    grid2 = [0.5, 0.6, 0.7, 0.8, 0.9]
    grid3 = [0.6, 0.7, 0.8, 0.9, 0.95]
    
    for ex in [True]:
        for t1 in grid1:
            for t2 in grid2:
                for t3 in grid3:
                    if t3 < t2 or t2 < t1 - 0.3:
                        continue
                    m = macro_f05(decide(sc, t1, t2, t3, ex), gt, s1_ids)
                    if best is None or m["macroF05"] > best[0]["macroF05"]:
                        best = (m, dict(t1=t1, t2=t2, t3=t3, exclusive=ex))
                        
    glob = max(((macro_f05(decide(sc, t, t, t, False), gt, s1_ids), t) for t in [0.4, 0.5, 0.6, 0.7, 0.8]),
               key=lambda x: x[0]["macroF05"])
    return best, glob

def train_lgbm(fit_pairs: pl.DataFrame, val_pairs: pl.DataFrame, features: list, seed: int = 42):
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
        seed=seed,
        num_threads=os.cpu_count() or 4
    )
    
    X_train = fit_pairs.select(features).to_numpy()
    y_train = fit_pairs["y"].to_numpy()
    X_val = val_pairs.select(features).to_numpy()
    y_val = val_pairs["y"].to_numpy()
    
    dtr = lgb.Dataset(X_train, label=y_train, feature_name=features, free_raw_data=True)
    dva = lgb.Dataset(X_val, label=y_val, reference=dtr)
    
    model = lgb.train(
        params,
        dtr,
        num_boost_round=1200,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    return model
