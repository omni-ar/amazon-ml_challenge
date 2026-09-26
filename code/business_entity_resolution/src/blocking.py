import math
import polars as pl
from normalize import ADDR_STOP

def make_keys(df: pl.DataFrame) -> pl.DataFrame:
    """df: idx(u32), name_core, name_glued, addr_n, nums -> (idx, h u64, ty u8)"""
    out = []
    
    # 1. Name tokens
    nt = df.select("idx", pl.col("name_core").str.split(" ").alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= 2)
    out.append(nt.select("idx", (pl.lit("n:") + pl.col("t")).alias("k"), pl.lit(1, pl.UInt8).alias("ty")))
    
    # 2. 4-char token prefixes
    out.append(nt.filter(pl.col("t").str.len_chars() >= 5)
                 .select("idx", (pl.lit("p:") + pl.col("t").str.slice(0, 4)).alias("k"), pl.lit(1, pl.UInt8).alias("ty")))
    
    # 3. 8-char glued name prefixes (captures domain/spacing variations)
    out.append(df.filter(pl.col("name_glued").str.len_chars() >= 8)
                 .select("idx", (pl.lit("w:") + pl.col("name_glued").str.slice(0, 8)).alias("k"), pl.lit(1, pl.UInt8).alias("ty")))
    
    # 4. Distinctive address words
    aw = (df.select("idx", pl.col("addr_n").str.split(" ").alias("w")).explode("w")
            .filter((pl.col("w").str.len_chars() >= 3) & ~pl.col("w").str.contains(r"\d") & ~pl.col("w").is_in(list(ADDR_STOP)))
            .unique())
    out.append(aw.select("idx", (pl.lit("a:") + pl.col("w")).alias("k"), pl.lit(2, pl.UInt8).alias("ty")))
    
    # 5. House number + address word combination
    hn = df.select("idx", pl.col("nums").list.head(4).alias("n")).explode("n").filter(pl.col("n").is_not_null())
    hw = hn.join(aw, on="idx")
    out.append(hw.select("idx", (pl.lit("h:") + pl.col("n") + "_" + pl.col("w")).alias("k"), pl.lit(3, pl.UInt8).alias("ty")))
    
    k = pl.concat(out).select(pl.col("idx"), pl.col("k").hash(seed=0).alias("h"), "ty").unique(["idx", "h"])
    return k

def block_country(s1c: pl.DataFrame, cc: pl.DataFrame, k: int = 40, key_cap: int = 300, s1_chunk: int = 60_000) -> pl.DataFrame:
    """s1c/cc: normalised frames with idx. Returns top-k pairs with blocking features."""
    if cc.height == 0 or s1c.height == 0:
        return None
        
    ck = pl.concat([make_keys(cc.slice(i, 500_000)) for i in range(0, cc.height, 500_000)])
    n = cc.height
    df_ = ck.group_by("h").agg(pl.len().alias("df"))
    df_ = df_.filter(pl.col("df") <= key_cap).with_columns((pl.lit(math.log(n + 1)) - pl.col("df").log()).cast(pl.Float32).alias("w"))
    ck = ck.join(df_.select("h", "w"), on="h")
    
    res = []
    for i in range(0, s1c.height, s1_chunk):
        sk = make_keys(s1c.slice(i, s1_chunk)).rename({"idx": "i1"}).drop("ty")
        j = sk.join(ck.rename({"idx": "i2"}), on="h")
        g = j.group_by("i1", "i2").agg(
            pl.col("w").sum().alias("b_score"),
            pl.col("w").filter(pl.col("ty") == 1).sum().alias("b_name"),
            pl.col("w").filter(pl.col("ty") == 2).sum().alias("b_addr"),
            pl.col("w").filter(pl.col("ty") == 3).sum().alias("b_hnum"),
            pl.len().cast(pl.UInt16).alias("b_nkeys")
        )
        g = g.filter(pl.col("b_score").rank("ordinal", descending=True).over("i1") <= k)
        res.append(g)
        del sk, j
        
    if not res:
        return None
        
    pairs = pl.concat(res)
    # Context features computed on candidate graph
    pairs = pairs.with_columns(
        pl.col("b_score").rank("ordinal", descending=True).over("i1").cast(pl.UInt16).alias("b_rank"),
        pl.len().over("i1").cast(pl.UInt16).alias("n_cands"),
        (pl.col("b_score") / pl.col("b_score").max().over("i1")).alias("b_rel"),
        pl.len().over("i2").cast(pl.UInt16).alias("cand_deg"),
        pl.col("b_score").rank("ordinal", descending=True).over("i2").cast(pl.UInt16).alias("cand_rank"),
        (pl.col("b_score") / pl.col("b_score").max().over("i2")).alias("cand_rel"),
    )
    return pairs
