import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame, feat_chunk: int = 500_000) -> pl.DataFrame:
    """pairs: i1, i2 + blocking cols. Adds string-similarity features chunked across parallel workers."""
    if pairs.height == 0:
        return pairs
        
    a = s1.select(
        pl.col("idx").alias("i1"), pl.col("name_n").alias("n1"), pl.col("name_core").alias("c1"),
        pl.col("name_glued").alias("g1"), pl.col("legal").alias("l1"), pl.col("addr_n").alias("a1"),
        pl.col("nums").alias("u1"), pl.col("s1_namefreq")
    )
    b = cand.select(
        pl.col("idx").alias("i2"), pl.col("name_n").alias("n2"), pl.col("name_core").alias("c2"),
        pl.col("name_glued").alias("g2"), pl.col("legal").alias("l2"), pl.col("addr_n").alias("a2"),
        pl.col("nums").alias("u2"), (pl.col("src") == "S3").alias("is_s3"), "indic"
    )
    
    out = []
    for i in range(0, pairs.height, feat_chunk):
        p = pairs.slice(i, feat_chunk).join(a, on="i1", how="left").join(b, on="i2", how="left")
        f = {}
        c1, c2 = p["c1"].to_list(), p["c2"].to_list()
        a1, a2 = p["a1"].to_list(), p["a2"].to_list()
        
        f["nm_ratio"] = cpdist(c1, c2, scorer=fuzz.ratio, workers=-1)
        f["nm_tset"] = cpdist(c1, c2, scorer=fuzz.token_set_ratio, workers=-1)
        f["nm_tsort"] = cpdist(c1, c2, scorer=fuzz.token_sort_ratio, workers=-1)
        f["nm_partial"] = cpdist(c1, c2, scorer=fuzz.partial_ratio, workers=-1)
        f["nm_jw"] = cpdist(c1, c2, scorer=JaroWinkler.normalized_similarity, workers=-1)
        f["nm_glued"] = cpdist(p["g1"].to_list(), p["g2"].to_list(), scorer=fuzz.ratio, workers=-1)
        f["nm_full_tset"] = cpdist(p["n1"].to_list(), p["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        f["ad_ratio"] = cpdist(a1, a2, scorer=fuzz.ratio, workers=-1)
        f["ad_tset"] = cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1)
        f["ad_partial"] = cpdist(a1, a2, scorer=fuzz.partial_ratio, workers=-1)
        del c1, c2, a1, a2
        
        p = p.with_columns([pl.Series(k, v.astype(np.float32)) for k, v in f.items()])
        p = p.with_columns(
            pl.col("u1").list.set_intersection("u2").list.len().cast(pl.Int16).alias("num_shared"),
            pl.col("u1").list.len().cast(pl.Int16).alias("num_n1"),
            pl.col("u2").list.len().cast(pl.Int16).alias("num_n2"),
            (pl.col("u1").list.first() == pl.col("u2").list.first()).cast(pl.Int8).fill_null(-1).alias("num_first_eq"),
            ((pl.col("l1") == pl.col("l2")) & (pl.col("l1") != "")).cast(pl.Int8).alias("legal_eq"),
            ((pl.col("l1") != "") & (pl.col("l2") != "") & (pl.col("l1") != pl.col("l2"))).cast(pl.Int8).alias("legal_conf"),
            (pl.col("a2") == "").cast(pl.Int8).alias("addr2_empty"),
            (pl.col("c2").str.len_chars().cast(pl.Float32) / pl.col("c1").str.len_chars().clip(1)).alias("nm_lenratio"),
            pl.col("is_s3").cast(pl.Int8),
            pl.col("indic").cast(pl.Int8),
        ).drop("n1", "n2", "c1", "c2", "g1", "g2", "l1", "l2", "a1", "a2", "u1", "u2")
        out.append(p)
        
    return pl.concat(out)
