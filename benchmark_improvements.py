import os
import sys
import gc
import re
import math
import time
import anyascii
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset\train"

print("="*70)
print("BENCHMARKING PIPELINE ENHANCEMENTS ON REALISTIC VALIDATION SPLIT")
print("="*70)

# Native Indic legal suffixes to strip
NATIVE_LEGAL = [
    "private limited", "pvt ltd", "pvt", "private", "limited", "ltd", "llp", "llc",
    "inc", "incorporated", "corp", "corporation", "co", "company", "plc", "sarl", "sas",
    "sasu", "sa", "gmbh", "m/s",
    # Hindi / Devanagari
    "प्राइवेट लिमिटेड", "प्राइवेट लि", "प्रा लिमिटेड", "प्रा लि", "लिमिटेड", "प्राइवेट",
    # Tamil
    "பிரைவேட் லிமிடெட்", "லிமிடெட்", "பிரைவேட்",
    # Telugu
    "ప్రైవేట్ లిమిటెడ్", "లిమిటెడ్", "ప్రైవేట్",
    # Kannada
    "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "ಲಿಮಿಟೆಡ್", "ಪ್ರೈವೇಟ್",
    # Bengali
    "প্রাইভেট লিমিটেড", "লিমিটেড", "প্রাইভেট",
    # Common prefixes
    "shree", "sri", "shri"
]
NATIVE_LEGAL_RE = r"\b(?:" + "|".join(sorted(NATIVE_LEGAL, key=len, reverse=True)) + r")\b"

def clean_and_transliterate(name_series: pl.Series) -> tuple[list[str], list[str]]:
    """Returns (original_cleaned, transliterated_cleaned)"""
    names = name_series.to_list()
    cleaned = []
    translit = []
    for n in names:
        n_str = str(n).lower()
        # Transliterate to ASCII
        tr = anyascii.anyascii(n_str)
        # Strip punctuation
        n_c = re.sub(r'[^\w\s]', ' ', n_str)
        n_c = re.sub(NATIVE_LEGAL_RE, ' ', n_c, flags=re.IGNORECASE)
        n_c = re.sub(r'\s+', ' ', n_c).strip()
        
        tr_c = re.sub(r'[^\w\s]', ' ', tr)
        tr_c = re.sub(NATIVE_LEGAL_RE, ' ', tr_c, flags=re.IGNORECASE)
        tr_c = re.sub(r'\s+', ' ', tr_c).strip()
        
        cleaned.append(n_c)
        translit.append(tr_c)
    return cleaned, translit

print("Benchmarking script loaded. Ready for execution.")
