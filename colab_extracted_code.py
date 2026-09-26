
# ============================== CELL 0 ==============================
!pip install -q gdown
!gdown --id 1ia3JtcAncbM4ZzHCkzHs91XpIP0SoTpi -O /content/dataset.zip

import zipfile
with zipfile.ZipFile('/content/dataset.zip') as z:
    z.extractall('/content/student_resource')

# ============================== CELL 1 ==============================
import os
for root, dirs, files in os.walk('/content/student_resource'):
    level = root.replace('/content/student_resource', '').count(os.sep)
    print('  ' * level + os.path.basename(root) + '/')
    for f in files[:3]:
        print('  ' * (level+1) + f)

# ============================== CELL 2 ==============================
os.environ["ER_BASE"] = "/content/student_resource/student_resource"

# ============================== CELL 3 ==============================
import os
p = '/content/student_resource/student_resource/dataset/train'
print(os.listdir(p))

# ============================== CELL 4 ==============================
import os
os.environ["ER_MODE"] = "dev"
os.environ["ER_BASE"] = "/content/student_resource/student_resource"

# ============================== CELL 5 ==============================
%pip install -q "polars>=1.30" pyarrow "rapidfuzz>=3.9" "lightgbm>=4.3" numpy

# ============================== CELL 6 ==============================
import os, gc, time, math, json, subprocess, sys
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

# ------------------------- CONFIG -------------------------
MODE = os.environ.get("ER_MODE", "dev")          # <-- "dev" first (quick check), then "full" for the real submission
BASE = os.environ.get("ER_BASE", "student_resource")   # folder that contains dataset/ and utils/
DATA = f"{BASE}/dataset"
OUT = os.environ.get("ER_OUT", "output")
WORK = os.environ.get("ER_WORK", "work")          # intermediate files
DEV_PCT = 3            # dev mode: % of S1 entities (and % of unmatched S2/S3) kept
K = 40                 # candidates kept per S1 entity (recall vs cost knob)
KEY_CAP = 300          # drop a blocking key if more than this many S2/S3 records share it (per country)
S1_CHUNK = 60_000      # S1 records per blocking join chunk (lower it if RAM is tight)
FEAT_CHUNK = 2_000_000 # pairs per feature/predict chunk
VAL_PCT = 20           # % of train S1 entities held out for validation
TRAIN_S1_MAX = 500_000 # cap on train-split S1 entities used to fit the model (full mode)
SEED = 42
if MODE == "dev":   # scale the key-frequency cap to the sample size so dev recall is not over-optimistic
    KEY_CAP = max(10, int(KEY_CAP * DEV_PCT / 100))
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
pl.Config.set_tbl_rows(20)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
log("MODE", MODE, "threads", pl.thread_pool_size())

# ============================== CELL 7 ==============================
def read_tsv(path):
    # empty fields are kept as "" (fill_null) - works on old and new polars versions
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       encoding="utf8-lossy").with_columns(pl.all().fill_null(""))

def id_num(col):  # "S2-12345" -> 12345 (used only for deterministic hashing / sampling)
    return pl.col(col).str.slice(3).cast(pl.Int64)

def load_split(split):
    s1 = read_tsv(f"{DATA}/{split}/{split}_source1.tsv")
    s2 = read_tsv(f"{DATA}/{split}/{split}_source2.tsv")
    s3 = read_tsv(f"{DATA}/{split}/{split}_source3.tsv")
    cand = pl.concat([s2, s3])
    return s1, cand

def load_gt():
    gt = read_tsv(f"{DATA}/train/train_ground_truth.tsv")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "cand_id"}))

# ============================== CELL 8 ==============================
LEGAL = ["private limited", "pvt ltd", "pvt", "private", "limited", "ltd", "llp", "llc", "l l c", "inc", "incorporated",
         "corp", "corporation", "co", "company", "plc", "pllc", "lp", "l p", "pc", "p c", "sarl", "sas", "sasu", "sa",
         "eurl", "sci", "snc", "selarl", "gmbh", "fka", "f k a", "dba", "d b a", "mr", "mrs", "ms", "smt", "the", "www", "com"]
LEGAL_RE = r"\b(?:" + "|".join(sorted(LEGAL, key=len, reverse=True)) + r")\b"
LEGAL_KEEP = ["pvt", "private", "limited", "ltd", "llp", "llc", "inc", "incorporated", "corp", "corporation", "co",
              "company", "plc", "pllc", "lp", "pc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc"]
LEGAL_CANON = {"pvt": "pvt", "private": "pvt", "limited": "ltd", "ltd": "ltd", "incorporated": "inc", "inc": "inc",
               "corporation": "corp", "corp": "corp", "company": "co", "co": "co"}
ABBR = {  # address token -> canonical (general street-type vocabulary; US / India / France)
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place",
    "hwy": "highway", "pkwy": "parkway", "ter": "terrace", "trl": "trail", "sq": "square", "ste": "suite", "apt": "apartment",
    "n": "north", "s": "south", "e": "east", "w": "west", "so": "south", "no": "", "nr": "near", "opp": "opposite",
    "r": "rue", "imp": "impasse", "ch": "chemin", "rte": "route", "fl": "floor", "flr": "floor", "bldg": "building",
    "nagr": "nagar", "ngr": "nagar", "sec": "sector", "dist": "district", "po": "po", "null": "", "nan": "", "none": ""}
ADDR_STOP = {"street", "road", "avenue", "boulevard", "drive", "lane", "court", "circle", "place", "highway", "suite",
             "apartment", "unit", "near", "opposite", "floor", "building", "door", "house", "plot", "flat", "the", "and",
             "rue", "de", "du", "la", "le", "des", "po", "box", "sector", "nagar", "road", "main", "cross", "north", "south",
             "east", "west"}

def base_clean(e):
    """lower + strip Latin accents + non-alphanumerics -> space. Keeps Indic scripts intact."""
    return (e.str.to_lowercase().str.normalize("NFKD")
             .str.replace_all(r"[̀-ͯ]", "")
             .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
             .str.strip_chars().str.replace_all(r"\s+", " "))

def normalise(df):
    name = base_clean(pl.col("business_name"))
    addr = base_clean(pl.col("business_address"))
    df = df.with_columns(
        name.alias("name_n"),
        addr.alias("addr_raw_n"),
        pl.col("business_address").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
          .list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("nums"),
        pl.col("business_name").str.contains(r"[ऀ-෿]").alias("indic"),
        pl.col("entity_id").str.slice(0, 2).alias("src"),
    )
    df = df.with_columns(
        pl.col("name_n").str.replace_all(LEGAL_RE, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("name_core"),
        pl.col("name_n").str.extract_all(r"\b(?:" + "|".join(LEGAL_KEEP) + r")\b")
          .list.eval(pl.element().replace(LEGAL_CANON)).list.unique().list.sort().list.join(" ").alias("legal"),
        pl.col("addr_raw_n").str.split(" ").list.eval(pl.element().replace(ABBR)).list.join(" ")
          .str.replace_all(r"\s+", " ").str.strip_chars().alias("addr_n"),
    )
    df = df.with_columns(
        pl.when(pl.col("name_core") == "").then(pl.col("name_n")).otherwise(pl.col("name_core")).alias("name_core"),
    ).with_columns(pl.col("name_core").str.replace_all(" ", "").alias("name_glued"))
    return df.select("entity_id", "country", "src", "name_n", "name_core", "name_glued", "legal", "addr_n", "nums", "indic",
                     "business_name", "business_address")

# ============================== CELL 9 ==============================
KEY_TYPES = {"n": 1, "p": 1, "w": 1, "a": 2, "h": 3}

def make_keys(df):
    """df: idx(u32), name_core, name_glued, addr_n, nums -> (idx, h u64, ty u8)"""
    out = []
    nt = df.select("idx", pl.col("name_core").str.split(" ").alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= 2)
    out.append(nt.select("idx", (pl.lit("n:") + pl.col("t")).alias("k"), pl.lit(1, pl.UInt8).alias("ty")))
    out.append(nt.filter(pl.col("t").str.len_chars() >= 5)
                 .select("idx", (pl.lit("p:") + pl.col("t").str.slice(0, 4)).alias("k"), pl.lit(1, pl.UInt8).alias("ty")))
    out.append(df.filter(pl.col("name_glued").str.len_chars() >= 8)
                 .select("idx", (pl.lit("w:") + pl.col("name_glued").str.slice(0, 8)).alias("k"), pl.lit(1, pl.UInt8).alias("ty")))
    aw = (df.select("idx", pl.col("addr_n").str.split(" ").alias("w")).explode("w")
            .filter((pl.col("w").str.len_chars() >= 3) & ~pl.col("w").str.contains(r"\d") & ~pl.col("w").is_in(list(ADDR_STOP)))
            .unique())
    out.append(aw.select("idx", (pl.lit("a:") + pl.col("w")).alias("k"), pl.lit(2, pl.UInt8).alias("ty")))
    hn = df.select("idx", pl.col("nums").list.head(4).alias("n")).explode("n").filter(pl.col("n").is_not_null())
    hw = hn.join(aw, on="idx")
    out.append(hw.select("idx", (pl.lit("h:") + pl.col("n") + "_" + pl.col("w")).alias("k"), pl.lit(3, pl.UInt8).alias("ty")))
    k = pl.concat(out).select(pl.col("idx"), pl.col("k").hash(seed=0).alias("h"), "ty").unique(["idx", "h"])
    return k

def block_country(s1c, cc, k=K):
    """s1c/cc: normalised frames with idx. Returns top-k pairs with blocking features."""
    ck = pl.concat([make_keys(cc.slice(i, 500_000)) for i in range(0, cc.height, 500_000)]) if cc.height else None
    if ck is None or s1c.height == 0:
        return None
    n = cc.height
    df_ = ck.group_by("h").agg(pl.len().alias("df"))
    df_ = df_.filter(pl.col("df") <= KEY_CAP).with_columns((pl.lit(math.log(n + 1)) - pl.col("df").log()).cast(pl.Float32).alias("w"))
    ck = ck.join(df_.select("h", "w"), on="h")
    res = []
    for i in range(0, s1c.height, S1_CHUNK):
        sk = make_keys(s1c.slice(i, S1_CHUNK)).rename({"idx": "i1"}).drop("ty")
        j = sk.join(ck.rename({"idx": "i2"}), on="h")
        if i == 0: log(f"    first chunk join rows: {j.height:,} (watch RAM; lower S1_CHUNK if huge)")
        g = j.group_by("i1", "i2").agg(
            pl.col("w").sum().alias("b_score"),
            pl.col("w").filter(pl.col("ty") == 1).sum().alias("b_name"),
            pl.col("w").filter(pl.col("ty") == 2).sum().alias("b_addr"),
            pl.col("w").filter(pl.col("ty") == 3).sum().alias("b_hnum"),
            pl.len().cast(pl.UInt16).alias("b_nkeys"))
        g = g.filter(pl.col("b_score").rank("ordinal", descending=True).over("i1") <= k)
        res.append(g); del sk, j
    return pl.concat(res)

def run_blocking(s1, cand):
    """s1/cand normalised with global idx columns. Blocks within each country label."""
    parts = []
    for c in s1["country"].unique().to_list():
        s1c = s1.filter(pl.col("country") == c); cc = cand.filter(pl.col("country") == c)
        log(f"  blocking country={c}: S1={s1c.height:,} cand={cc.height:,}")
        p = block_country(s1c, cc)
        if p is not None: parts.append(p)
        gc.collect()
    pairs = pl.concat(parts)
    # context features computed on the full candidate graph
    pairs = pairs.with_columns(
        pl.col("b_score").rank("ordinal", descending=True).over("i1").cast(pl.UInt16).alias("b_rank"),
        pl.len().over("i1").cast(pl.UInt16).alias("n_cands"),
        (pl.col("b_score") / pl.col("b_score").max().over("i1")).alias("b_rel"),
        pl.len().over("i2").cast(pl.UInt16).alias("cand_deg"),
        pl.col("b_score").rank("ordinal", descending=True).over("i2").cast(pl.UInt16).alias("cand_rank"),
        (pl.col("b_score") / pl.col("b_score").max().over("i2")).alias("cand_rel"),
    )
    return pairs

# ============================== CELL 10 ==============================
def pair_features(pairs, s1, cand):
    """pairs: i1, i2 + blocking cols. Adds string-similarity features (chunked)."""
    a = s1.select(pl.col("idx").alias("i1"), pl.col("name_n").alias("n1"), pl.col("name_core").alias("c1"),
                  pl.col("name_glued").alias("g1"), pl.col("legal").alias("l1"), pl.col("addr_n").alias("a1"),
                  pl.col("nums").alias("u1"), pl.col("s1_namefreq"))
    b = cand.select(pl.col("idx").alias("i2"), pl.col("name_n").alias("n2"), pl.col("name_core").alias("c2"),
                    pl.col("name_glued").alias("g2"), pl.col("legal").alias("l2"), pl.col("addr_n").alias("a2"),
                    pl.col("nums").alias("u2"), (pl.col("src") == "S3").alias("is_s3"), "indic")
    out = []
    for i in range(0, pairs.height, FEAT_CHUNK):
        p = pairs.slice(i, FEAT_CHUNK).join(a, on="i1", how="left").join(b, on="i2", how="left")
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
            pl.col("is_s3").cast(pl.Int8), pl.col("indic").cast(pl.Int8),
        ).drop("n1", "n2", "c1", "c2", "g1", "g2", "l1", "l2", "a1", "a2", "u1", "u2")
        out.append(p)
        log(f"    features {min(i+FEAT_CHUNK, pairs.height):,}/{pairs.height:,}")
    return pl.concat(out)

FEATURES = ["b_score", "b_name", "b_addr", "b_hnum", "b_nkeys", "b_rank", "n_cands", "b_rel", "cand_deg", "cand_rank",
            "cand_rel", "s1_namefreq", "nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "nm_jw", "nm_glued", "nm_full_tset",
            "ad_ratio", "ad_tset", "ad_partial", "num_shared", "num_n1", "num_n2", "num_first_eq", "legal_eq", "legal_conf",
            "addr2_empty", "nm_lenratio", "is_s3", "indic"]
# NOTE: `country` is deliberately NOT a feature (France is unseen in train).

# ============================== CELL 11 ==============================
def macro_f05(pred, gt, s1_ids):
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
        pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(None).alias("rec"))
    single = d.filter(pl.col("nt") == 0)
    return {"macroF05": d["f"].mean(), "macroP": d["prec"].mean(), "macroR": d["rec"].mean(),
            "singleton_acc": single["f"].mean() if single.height else None,
            "pct_entities_with_FP": (d["np"] > d["tp"]).mean(), "n": d.height}

def decide(sc, t1, t2, t3, exclusive=True):
    """sc: (s1_id, cand_id, p). Exclusivity + rank-aware thresholds (t1 for best, t2 for 2nd, t3 for 3rd+)."""
    if exclusive:
        sc = sc.filter(pl.col("p") >= pl.col("p").max().over("cand_id"))
    sc = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("r"))
    thr = pl.when(pl.col("r") == 1).then(t1).when(pl.col("r") == 2).then(t2).otherwise(t3)
    return sc.filter(pl.col("p") >= thr).select("s1_id", "cand_id")

def tune_decision(sc, gt, s1_ids):
    best = None
    grid1 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; grid2 = [0.5, 0.6, 0.7, 0.8, 0.9]; grid3 = [0.6, 0.7, 0.8, 0.9, 0.95]
    for ex in [True]:
        for t1 in grid1:
            for t2 in grid2:
                for t3 in grid3:
                    if t3 < t2 or t2 < t1 - 0.3: continue
                    m = macro_f05(decide(sc, t1, t2, t3, ex), gt, s1_ids)
                    if best is None or m["macroF05"] > best[0]["macroF05"]:
                        best = (m, dict(t1=t1, t2=t2, t3=t3, exclusive=ex))
    # also report the best single global threshold for comparison
    glob = max(((macro_f05(decide(sc, t, t, t, False), gt, s1_ids), t) for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
               key=lambda x: x[0]["macroF05"])
    return best, glob

# ============================== CELL 12 ==============================
log("loading train")
gt = load_gt()
if MODE == "dev":   # lazy scan + filter so a laptop / free Colab only materialises the sample
    scan = lambda f: pl.scan_csv(f"{DATA}/train/{f}", separator="\t", quote_char=None, infer_schema=False,
                                 encoding="utf8-lossy").with_columns(pl.all().fill_null(""))
    s1_tr = scan("train_source1.tsv").filter((id_num("entity_id") * 2654435761 % 1000003) % 100 < DEV_PCT).collect()
    matched_all = gt.select(pl.col("cand_id").alias("entity_id"), pl.lit(True).alias("_m"))
    gt = gt.join(s1_tr.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
    want = gt.select(pl.col("cand_id").alias("entity_id"), pl.lit(True).alias("_w"))
    rnd = (id_num("entity_id") * 40503 % 1000033) % 100 < DEV_PCT
    cand_tr = (pl.concat([scan("train_source2.tsv"), scan("train_source3.tsv")])
               .join(want.lazy(), on="entity_id", how="left").join(matched_all.lazy(), on="entity_id", how="left")
               .filter(pl.col("_w").is_not_null() | (pl.col("_m").is_null() & rnd))
               .drop("_w", "_m").collect())
    del matched_all, want
else:
    s1_tr, cand_tr = load_split("train")
log(f"train S1={s1_tr.height:,} cand={cand_tr.height:,} gt_pairs={gt.height:,}")

def prepare(s1, cand):
    s1 = normalise(s1).with_row_index("idx")
    cand = normalise(cand).with_row_index("idx")
    s1 = s1.with_columns(pl.len().over(["country", "name_core"]).cast(pl.UInt32).alias("s1_namefreq"))
    return s1, cand

s1_tr, cand_tr = prepare(s1_tr, cand_tr); log("normalised")
print(s1_tr.select("business_name", "name_core", "legal", "business_address", "addr_n", "nums").head(5))
print(cand_tr.select("business_name", "name_core", "legal", "business_address", "addr_n", "nums").head(5))

# ============================== CELL 13 ==============================
pairs_tr = run_blocking(s1_tr, cand_tr); log(f"blocking done: {pairs_tr.height:,} pairs")
ids1 = s1_tr.select(pl.col("idx").alias("i1"), pl.col("entity_id").alias("s1_id"), pl.col("country"))
ids2 = cand_tr.select(pl.col("idx").alias("i2"), pl.col("entity_id").alias("cand_id"))
pairs_tr = pairs_tr.join(ids1, on="i1").join(ids2, on="i2")
lab = gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
pairs_tr = pairs_tr.join(lab, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("y").fill_null(0))

# Blocking recall = the ceiling
rec = pairs_tr["y"].sum() / gt.height
ceil = macro_f05(pairs_tr.filter(pl.col("y") == 1).select("s1_id", "cand_id"), gt, s1_tr["entity_id"])
by_c = (gt.join(ids1.select("s1_id", "country"), on="s1_id")
          .join(pairs_tr.filter(pl.col("y") == 1).select("s1_id", "cand_id", "y"), on=["s1_id", "cand_id"], how="left")
          .group_by("country", pl.col("cand_id").str.slice(0, 2).alias("src")).agg(pl.col("y").is_not_null().mean().alias("recall")))
log(f"BLOCKING pair recall={rec:.4f}  ceiling macroF05={ceil['macroF05']:.4f}  pairs/S1={pairs_tr.height/s1_tr.height:.1f}")
print(by_c.sort("country", "src"))

# ============================== CELL 14 ==============================
# split by S1 entity (deterministic hash) -> train / validation
pairs_tr = pairs_tr.with_columns(((id_num("s1_id") * 2246822519 % 1000003) % 100 < VAL_PCT).alias("is_val"))
val_s1 = s1_tr.filter(((id_num("entity_id") * 2246822519 % 1000003) % 100 < VAL_PCT))["entity_id"]
fit_pairs = pairs_tr.filter(~pl.col("is_val"))
if MODE == "full":
    fit_ids = fit_pairs.select("s1_id").unique().sample(n=min(TRAIN_S1_MAX, fit_pairs["s1_id"].n_unique()), seed=SEED)
    fit_pairs = fit_pairs.join(fit_ids, on="s1_id")
val_pairs = pairs_tr.filter(pl.col("is_val"))
del pairs_tr; gc.collect()
fit_pairs = pair_features(fit_pairs, s1_tr, cand_tr); log("fit features done")
val_pairs = pair_features(val_pairs, s1_tr, cand_tr); log("val features done")

# ============================== CELL 15 ==============================
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=os.cpu_count())
dtr = lgb.Dataset(fit_pairs.select(FEATURES).to_numpy(), label=fit_pairs["y"].to_numpy(), feature_name=FEATURES, free_raw_data=True)
dva = lgb.Dataset(val_pairs.select(FEATURES).to_numpy(), label=val_pairs["y"].to_numpy(), reference=dtr)
model = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
model.save_model(f"{WORK}/lgbm_v1.txt")
imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
print("top features:", [(f, int(g)) for f, g in imp[:12]])

# ============================== CELL 16 ==============================
val_sc = val_pairs.select("s1_id", "cand_id", pl.Series("p", model.predict(val_pairs.select(FEATURES).to_numpy())))
gt_val = gt.join(pl.DataFrame({"s1_id": val_s1}), on="s1_id", how="semi")
(best_m, best_cfg), (glob_m, glob_t) = tune_decision(val_sc, gt_val, val_s1)
log(f"VALIDATION  global-threshold t={glob_t}: {glob_m}")
log(f"VALIDATION  best decision {best_cfg}: {best_m}")
json.dump({"mode": MODE, "blocking_recall": rec, "ceiling": ceil, "global": [glob_m, glob_t], "best": [best_m, best_cfg],
           "best_iter": model.best_iteration}, open(f"{WORK}/val_report_{MODE}.json", "w"), indent=1, default=str)
DECISION = best_cfg
del fit_pairs, val_pairs, dtr, dva; gc.collect()

# ============================== CELL 17 ==============================
TEST_DIR = f"{DATA}/test"
if True:
    del s1_tr, cand_tr; gc.collect()
    if MODE == "full":
        s1_te, cand_te = load_split("test")
    else:  # mini test for smoke-testing: 3% of test S1 + 3% of test S2/S3, written as TSVs for the validator
        sc_ = lambda f: pl.scan_csv(f"{DATA}/test/{f}", separator="\t", quote_char=None, infer_schema=False,
                                   encoding="utf8-lossy").with_columns(pl.all().fill_null(""))
        TEST_DIR = f"{WORK}/mini_test"; os.makedirs(TEST_DIR, exist_ok=True)
        smp = (id_num("entity_id") * 2654435761 % 1000003) % 100 < DEV_PCT
        s1_te = sc_("test_source1.tsv").filter(smp).collect()
        s2_te = sc_("test_source2.tsv").filter(smp).collect(); s3_te = sc_("test_source3.tsv").filter(smp).collect()
        for d_, f_ in [(s1_te, "test_source1.tsv"), (s2_te, "test_source2.tsv"), (s3_te, "test_source3.tsv")]:
            d_.write_csv(f"{TEST_DIR}/{f_}", separator="\t", quote_style="never")
        cand_te = pl.concat([s2_te, s3_te]); del s2_te, s3_te
    s1_te, cand_te = prepare(s1_te, cand_te); log(f"test S1={s1_te.height:,} cand={cand_te.height:,}")
    pairs_te = run_blocking(s1_te, cand_te); log(f"test pairs {pairs_te.height:,}")
    pairs_te = (pairs_te.join(s1_te.select(pl.col("idx").alias("i1"), pl.col("entity_id").alias("s1_id")), on="i1")
                        .join(cand_te.select(pl.col("idx").alias("i2"), pl.col("entity_id").alias("cand_id")), on="i2"))
    scored = []
    for i in range(0, pairs_te.height, FEAT_CHUNK * 2):
        ch = pair_features(pairs_te.slice(i, FEAT_CHUNK * 2), s1_te, cand_te)
        scored.append(ch.select("s1_id", "cand_id", pl.Series("p", model.predict(ch.select(FEATURES).to_numpy()))))
        del ch; gc.collect()
    sc_te = pl.concat(scored)
    matches = decide(sc_te, DECISION["t1"], DECISION["t2"], DECISION["t3"], DECISION["exclusive"])
    all_s1 = s1_te.select(pl.col("entity_id").alias("source1_entity_id"))
    def write_lists(pairs_df, colname, path):
        agg = pairs_df.group_by("s1_id").agg(pl.col("cand_id").unique().sort().str.join(",").alias(colname))
        out = all_s1.join(agg.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left").fill_null("")
        assert out.height == s1_te.height and out["source1_entity_id"].n_unique() == out.height
        out.write_csv(path, separator="\t", quote_style="never")
        return out
    write_lists(sc_te.select("s1_id", "cand_id"), "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
    res = write_lists(matches, "matched_entity_ids", f"{OUT}/matching_results.tsv")
    log(f"wrote outputs; S1 with >=1 match: {(res['matched_entity_ids'] != '').mean():.3f}; "
        f"mean matches: {matches.height / s1_te.height:.2f}")
    print(res.join(s1_te.select(pl.col("entity_id").alias("source1_entity_id"), "country"), on="source1_entity_id")
             .group_by("country").agg((pl.col("matched_entity_ids") != "").mean().alias("frac_nonempty")))
    r = subprocess.run([sys.executable, f"{BASE}/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                        "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", TEST_DIR, "--check-ids"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
log("DONE")

# ============================== CELL 18 ==============================
import os
print(os.listdir("output"))

# ============================== CELL 19 ==============================
os.environ["ER_MODE"] = "full"

# ============================== CELL 20 ==============================
import polars as pl
print(pl.scan_csv("/content/student_resource/student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema=False).select(pl.len()).collect())

# ============================== CELL 21 ==============================
import os

# Create the directory
os.makedirs("utils", exist_ok=True)

# Save the validator script
validator_code = """
import argparse
import os
import sys

DELIM = "\\t"
MAX_EXAMPLES = 5
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

def read_ids(path):
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}

def examples(items):
    items = sorted(items)
    shown = ", ".join(items[:MAX_EXAMPLES])
    if len(items) > MAX_EXAMPLES:
        return f"{len(items)} total, e.g. {shown}, ..."
    return shown

def load_match_targets(test_dir, warnings):
    targets = set()
    for name in ("test_source2.tsv", "test_source3.tsv"):
        path = os.path.join(test_dir, name)
        if not os.path.isfile(path):
            warnings.append(
                f"{path} not found — skipping the (optional) check that matched "
                f"IDs exist in the test set. Every other rule is still checked."
            )
            return None
        targets |= read_ids(path)
    return targets

def validate_id_list_file(path, expected_header, col_label, required, valid_ids, errors):
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None

    name = os.path.basename(path)
    mapping = {}
    seen, dup_rows, intra_dupes = set(), set(), set()
    self_matches, wrong_prefix, unknown = set(), set(), set()
    n_rows = empties = 0

    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None
        if DELIM not in header and "," in header:
            errors.append(
                f"{name}: header has no TAB but contains commas — file looks COMMA-separated."
            )
            return None
        cols = [c.strip().lower() for c in header.rstrip("\\n").split(DELIM)]
        if cols != expected_header:
            errors.append(f"{name}: unexpected header {cols}. Expected {expected_header}.")
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row at line {line_num}")
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = rest.rstrip("\\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue
            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)
            id_set = set(ids)
            mapping[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)

    findings = [
        (dup_rows, "{name}: duplicate source1_entity_id row(s): {ex}."),
        (intra_dupes, "{name}: repeated ID inside {col} list for: {ex}."),
        (self_matches, "{name}: {col} contains Source-1 IDs: {ex}."),
        (wrong_prefix, "{name}: {col} contains IDs without S2-/S3- prefix: {ex}."),
        (unknown, "{name}: {col} references IDs not in test Source-2/3 files: {ex}."),
        (required - seen, "{name}: required S1 entity(ies) missing: {ex}."),
        (seen - required, "{name}: row(s) using S1 ID not in test set: {ex}."),
    ]
    for offenders, message in findings:
        if offenders:
            errors.append(message.format(name=name, ex=examples(offenders), col=col_label))

    print(f"  {name}: {n_rows} rows ({empties} empty, {n_rows - empties} non-empty).")
    return mapping

def validate(matching_path, candidate_path, test_dir, check_ids=False):
    errors, warnings = [], []
    source1 = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(source1):
        errors.append(f"Test source1 file not found: {source1}.")
        return errors, warnings
    required = read_ids(source1)
    print(f"  required S1 entities: {len(required)}")

    if check_ids:
        valid_ids = load_match_targets(test_dir, warnings)
        if valid_ids is not None:
            print(f"  valid S2/S3 match IDs: {len(valid_ids)}")
    else:
        valid_ids = None

    matched = validate_id_list_file(matching_path, MATCHING_HEADER, "matched_entity_ids", required, valid_ids, errors)
    candidate = None
    if candidate_path and os.path.isfile(candidate_path):
        candidate = validate_id_list_file(candidate_path, CANDIDATE_HEADER, "candidate_entity_ids", required, valid_ids, errors)

    return errors, warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv")
    parser.add_argument("--candidate", "-c", default=None)
    parser.add_argument("--test-dir", "-t", default="dataset/test")
    parser.add_argument("--check-ids", action="store_true")
    args = parser.parse_args()

    candidate_path = args.candidate or "output/candidate_pairs.tsv"
    errors, warnings = validate(args.matching, candidate_path, args.test_dir, check_ids=args.check_ids)

    for w in warnings: print(f"WARNING: {w}")
    if errors:
        print(f"FAIL — {len(errors)} issue(s):")
        for i, e in enumerate(errors, 1): print(f"  {i}. {e}")
        sys.exit(1)
    print("PASS — no blocking issues found. Safe to submit.")
"""

with open("utils/validate_submission.py", "w", encoding="utf-8") as f:
    f.write(validator_code.strip())

print("Validator file saved to utils/validate_submission.py")

# ============================== CELL 22 ==============================
!python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/student_resource/dataset/test
