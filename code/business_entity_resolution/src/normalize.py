import re
import anyascii
import polars as pl

# Multilingual legal terms (English, French, Hindi, Tamil, Telugu, Kannada, Bengali)
LEGAL = [
    # English & French
    "private limited", "pvt ltd", "pvt", "private", "limited", "ltd", "llp", "llc", "l l c",
    "inc", "incorporated", "corp", "corporation", "co", "company", "plc", "pllc", "lp", "l p",
    "pc", "p c", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "selarl", "gmbh",
    "fka", "f k a", "dba", "d b a", "mr", "mrs", "ms", "smt", "the", "www", "com",
    "societe a responsabilite limitee", "societe par actions simplifiee",
    # Hindi / Devanagari
    "प्राइवेट लिमिटेड", "प्राइवेट लि", "प्रा लिमिटेड", "प्रा लि", "लिमिटेड", "प्राइवेट", "लि", "प्रा",
    # Tamil
    "பிரைவேட் லிமிடெட்", "லிமிடெட்", "பிரைவேட்", "எல்எல்பி",
    # Telugu
    "ప్రైవేట్ లిమిటెడ్", "లిమిటెడ్", "ప్రైవేట్",
    # Kannada
    "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "ಲಿಮಿಟೆಡ್", "ಪ್ರೈವೇಟ್",
    # Bengali
    "প্রাইভেট লিমিটেড", "লিমিটেড", "প্রাইভেট",
    # Common prefixes
    "shree", "sri", "shri", "m/s"
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

def transliterate_series(s: pl.Series) -> pl.Series:
    """Fast vectorized transliteration using anyascii."""
    return pl.Series(s.name, [anyascii.anyascii(x).lower() if x else "" for x in s.to_list()])

def extract_postal_code(addr_series: pl.Series, country_series: pl.Series) -> pl.Series:
    """Extract standard postal code: 5 digits for US/France, 6 digits for India."""
    addrs = addr_series.to_list()
    countries = country_series.to_list()
    p_codes = []
    for a, c in zip(addrs, countries):
        a_str = str(a)
        if c == "India":
            # 6-digit PIN code
            m = re.findall(r'\b\d{6}\b', a_str)
            p_codes.append(m[0] if m else "")
        else:
            # 5-digit zip / postal code
            m = re.findall(r'\b\d{5}\b', a_str)
            p_codes.append(m[0] if m else "")
    return pl.Series("postal_code", p_codes)

def normalise(df: pl.DataFrame) -> pl.DataFrame:
    # 1. Base clean
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
    
    # 2. Transliterate name and address to Latin ASCII
    name_tr_series = transliterate_series(df["name_n"])
    addr_tr_series = transliterate_series(df["addr_raw_n"])
    postal_series = extract_postal_code(df["business_address"], df["country"])
    
    df = df.with_columns([
        name_tr_series.alias("name_tr"),
        addr_tr_series.alias("addr_tr"),
        postal_series.alias("postal_code"),
    ])
    
    # 3. Strip legal suffixes from both original and transliterated names
    df = df.with_columns(
        pl.col("name_n").str.replace_all(LEGAL_RE, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("name_core"),
        pl.col("name_tr").str.replace_all(LEGAL_RE, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("name_tr_core"),
        pl.col("name_n").str.extract_all(r"\b(?:" + "|".join(LEGAL_KEEP) + r")\b")
          .list.eval(pl.element().replace(LEGAL_CANON)).list.unique().list.sort().list.join(" ").alias("legal"),
        pl.col("addr_tr").str.split(" ").list.eval(pl.element().replace(ABBR)).list.join(" ")
          .str.replace_all(r"\s+", " ").str.strip_chars().alias("addr_n"),
    )
    
    # 4. Fill fallback core names and create glued names
    df = df.with_columns(
        pl.when(pl.col("name_core") == "").then(pl.col("name_n")).otherwise(pl.col("name_core")).alias("name_core"),
        pl.when(pl.col("name_tr_core") == "").then(pl.col("name_tr")).otherwise(pl.col("name_tr_core")).alias("name_tr_core"),
    ).with_columns([
        pl.col("name_core").str.replace_all(" ", "").alias("name_glued"),
        pl.col("name_tr_core").str.replace_all(" ", "").alias("name_tr_glued"),
    ])
    
    return df.select(
        "entity_id", "country", "src", "name_n", "name_tr", "name_core", "name_tr_core",
        "name_glued", "name_tr_glued", "legal", "addr_n", "nums", "postal_code", "indic",
        "business_name", "business_address"
    )
