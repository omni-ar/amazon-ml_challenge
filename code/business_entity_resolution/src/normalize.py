import polars as pl

LEGAL = [
    "private limited", "pvt ltd", "pvt", "private", "limited", "ltd", "llp", "llc", "l l c",
    "inc", "incorporated", "corp", "corporation", "co", "company", "plc", "pllc", "lp", "l p",
    "pc", "p c", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "selarl", "gmbh",
    "fka", "f k a", "dba", "d b a", "mr", "mrs", "ms", "smt", "the", "www", "com"
]
LEGAL_RE = r"\b(?:" + "|".join(sorted(LEGAL, key=len, reverse=True)) + r")\b"

LEGAL_KEEP = [
    "pvt", "private", "limited", "ltd", "llp", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "plc", "pllc", "lp", "pc",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc"
]

LEGAL_CANON = {
    "pvt": "pvt", "private": "pvt", "limited": "ltd", "ltd": "ltd",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co"
}

ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "cir": "circle", "pl": "place", "hwy": "highway", "pkwy": "parkway",
    "ter": "terrace", "trl": "trail", "sq": "square", "ste": "suite", "apt": "apartment",
    "n": "north", "s": "south", "e": "east", "w": "west", "so": "south", "no": "", "nr": "near",
    "opp": "opposite", "r": "rue", "imp": "impasse", "ch": "chemin", "rte": "route",
    "fl": "floor", "flr": "floor", "bldg": "building", "nagr": "nagar", "ngr": "nagar",
    "sec": "sector", "dist": "district", "po": "po", "null": "", "nan": "", "none": ""
}

ADDR_STOP = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "court", "circle", "place",
    "highway", "suite", "apartment", "unit", "near", "opposite", "floor", "building",
    "door", "house", "plot", "flat", "the", "and", "rue", "de", "du", "la", "le", "des",
    "po", "box", "sector", "nagar", "main", "cross", "north", "south", "east", "west"
}

def base_clean(e: pl.Expr) -> pl.Expr:
    """lower + strip Latin accents + non-alphanumerics -> space. Keeps Indic scripts intact."""
    return (e.str.to_lowercase().str.normalize("NFKD")
             .str.replace_all(r"[̀-ͯ]", "")
             .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
             .str.strip_chars().str.replace_all(r"\s+", " "))

def normalise(df: pl.DataFrame) -> pl.DataFrame:
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
    
    return df.select(
        "entity_id", "country", "src", "name_n", "name_core", "name_glued",
        "legal", "addr_n", "nums", "indic", "business_name", "business_address"
    )
