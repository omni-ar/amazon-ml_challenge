import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame, feat_chunk: int = 500_000) -> pl.DataFrame:
    """pairs: i1, i2 + blocking cols. Adds comprehensive string and conflict features."""
    if pairs.height == 0:
        return pairs
        
    a = s1.select(
        pl.col("idx").alias("i1"),
        pl.col("name_n").alias("n1"),
        pl.col("name_tr").alias("ntr1"),
        pl.col("name_core").alias("c1"),
        pl.col("name_tr_core").alias("ctr1"),
        pl.col("name_glued").alias("g1"),
        pl.col("legal").alias("l1"),
        pl.col("addr_n").alias("a1"),
        pl.col("nums").alias("u1"),
        pl.col("postal_code").alias("pc1"),
        pl.col("s1_namefreq")
    )
    b = cand.select(
        pl.col("idx").alias("i2"),
        pl.col("name_n").alias("n2"),
        pl.col("name_tr").alias("ntr2"),
        pl.col("name_core").alias("c2"),
        pl.col("name_tr_core").alias("ctr2"),
        pl.col("name_glued").alias("g2"),
        pl.col("legal").alias("l2"),
        pl.col("addr_n").alias("a2"),
        pl.col("nums").alias("u2"),
        pl.col("postal_code").alias("pc2"),
        (pl.col("src") == "S3").alias("is_s3"),
        "indic"
    )
    
    out = []
    for i in range(0, pairs.height, feat_chunk):
        p = pairs.slice(i, feat_chunk).join(a, on="i1", how="left").join(b, on="i2", how="left")
        f = {}
        c1, c2 = p["c1"].to_list(), p["c2"].to_list()
        ctr1, ctr2 = p["ctr1"].to_list(), p["ctr2"].to_list()
        a1, a2 = p["a1"].to_list(), p["a2"].to_list()
        
        # 1. Original name features
        f["nm_ratio"] = cpdist(c1, c2, scorer=fuzz.ratio, workers=-1)
        f["nm_tset"] = cpdist(c1, c2, scorer=fuzz.token_set_ratio, workers=-1)
        f["nm_tsort"] = cpdist(c1, c2, scorer=fuzz.token_sort_ratio, workers=-1)
        f["nm_partial"] = cpdist(c1, c2, scorer=fuzz.partial_ratio, workers=-1)
        f["nm_jw"] = cpdist(c1, c2, scorer=JaroWinkler.normalized_similarity, workers=-1)
        f["nm_glued"] = cpdist(p["g1"].to_list(), p["g2"].to_list(), scorer=fuzz.ratio, workers=-1)
        f["nm_full_tset"] = cpdist(p["n1"].to_list(), p["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        
        # 2. Transliterated name features (recovers Indic cross-script pairs)
        if p["indic"].any():
            f["nm_tr_ratio"] = cpdist(ctr1, ctr2, scorer=fuzz.ratio, workers=-1)
            f["nm_tr_tset"] = cpdist(ctr1, ctr2, scorer=fuzz.token_set_ratio, workers=-1)
            f["nm_tr_tsort"] = cpdist(ctr1, ctr2, scorer=fuzz.token_sort_ratio, workers=-1)
            f["nm_tr_jw"] = cpdist(ctr1, ctr2, scorer=JaroWinkler.normalized_similarity, workers=-1)
        else:
            f["nm_tr_ratio"] = f["nm_ratio"]
            f["nm_tr_tset"] = f["nm_tset"]
            f["nm_tr_tsort"] = f["nm_tsort"]
            f["nm_tr_jw"] = f["nm_jw"]
        
        # 3. Address similarities
        f["ad_ratio"] = cpdist(a1, a2, scorer=fuzz.ratio, workers=-1)
        f["ad_tset"] = cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1)
        f["ad_partial"] = cpdist(a1, a2, scorer=fuzz.partial_ratio, workers=-1)
        
        del c1, c2, ctr1, ctr2, a1, a2
        
        p = p.with_columns([pl.Series(k, v.astype(np.float32)) for k, v in f.items()])
        
        # Maximum name set similarity across raw and transliterated representations
        p = p.with_columns(
            pl.max_horizontal("nm_tset", "nm_tr_tset").alias("nm_max_tset")
        )
        
        # Numerical & Postal code overlaps and conflicts
        p = p.with_columns(
            pl.col("u1").list.set_intersection("u2").list.len().cast(pl.Int16).alias("num_shared"),
            pl.col("u1").list.len().cast(pl.Int16).alias("num_n1"),
            pl.col("u2").list.len().cast(pl.Int16).alias("num_n2"),
            (pl.col("u1").list.first() == pl.col("u2").list.first()).cast(pl.Int8).fill_null(-1).alias("num_first_eq"),
            ((pl.col("u1").list.len() > 0) & (pl.col("u2").list.len() > 0) & (pl.col("u1").list.set_intersection("u2").list.len() == 0)).cast(pl.Float32).alias("num_conflict"),
            ((pl.col("pc1") != "") & (pl.col("pc2") != "") & (pl.col("pc1") == pl.col("pc2"))).cast(pl.Float32).alias("postal_match"),
            ((pl.col("pc1") != "") & (pl.col("pc2") != "") & (pl.col("pc1") != pl.col("pc2"))).cast(pl.Float32).alias("postal_conflict"),
            ((pl.col("a1") != "") & (pl.col("a2") != "")).cast(pl.Float32).alias("addr_both_present"),
            (pl.col("a2") == "").cast(pl.Int8).alias("addr2_empty"),
            ((pl.col("l1") == pl.col("l2")) & (pl.col("l1") != "")).cast(pl.Int8).alias("legal_eq"),
            ((pl.col("l1") != "") & (pl.col("l2") != "") & (pl.col("l1") != pl.col("l2"))).cast(pl.Int8).alias("legal_conf"),
            (pl.col("c2").str.len_chars().cast(pl.Float32) / pl.col("c1").str.len_chars().clip(1)).alias("nm_lenratio"),
            pl.col("is_s3").cast(pl.Int8),
            pl.col("indic").cast(pl.Int8),
        )
        
        # Locality token Jaccard & Interaction between high name and conflicting numbers
        p = p.with_columns(
            ((pl.col("nm_max_tset") >= 80.0) & (pl.col("num_conflict") == 1.0)).cast(pl.Float32).alias("name_high_num_conflict"),
            (pl.col("ad_tset") / 100.0).alias("loc_jaccard"),
        ).drop("n1", "n2", "ntr1", "ntr2", "c1", "c2", "ctr1", "ctr2", "g1", "g2", "l1", "l2", "a1", "a2", "u1", "u2", "pc1", "pc2")
        
        out.append(p)
        
    return pl.concat(out)
