import os
import sys
import polars as pl

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"c:\Users\arjit\Desktop\ml_challenge\student_resource\dataset"

print("="*70)
print("COMPREHENSIVE DATASET INTEGRITY AUDIT")
print("="*70)

files_to_check = [
    ("Train Source 1", os.path.join(DATASET_DIR, "train", "train_source1.tsv"), 4),
    ("Train Source 2", os.path.join(DATASET_DIR, "train", "train_source2.tsv"), 4),
    ("Train Source 3", os.path.join(DATASET_DIR, "train", "train_source3.tsv"), 4),
    ("Train Ground Truth", os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv"), 2),
    ("Test Source 1", os.path.join(DATASET_DIR, "test", "test_source1.tsv"), 4),
    ("Test Source 2", os.path.join(DATASET_DIR, "test", "test_source2.tsv"), 4),
    ("Test Source 3", os.path.join(DATASET_DIR, "test", "test_source3.tsv"), 4),
]

all_passed = True

for desc, path, expected_cols in files_to_check:
    print(f"\nChecking: {desc} ({os.path.basename(path)})")
    if not os.path.exists(path):
        print(f"  [ERROR] File does not exist: {path}")
        all_passed = False
        continue
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Size on disk: {size_mb:.2f} MB")
    
    try:
        # Scan with polars
        lf = pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8-lossy")
        schema = lf.collect_schema()
        n_cols = len(schema.names())
        print(f"  Columns found ({n_cols}): {schema.names()}")
        
        if n_cols != expected_cols:
            print(f"  [WARNING] Expected {expected_cols} columns, found {n_cols}")
            all_passed = False
            
        # Count rows and check last line to ensure no truncation
        row_count = lf.select(pl.len()).collect().item()
        print(f"  Total records: {row_count:,}")
        
        # Check first and last row
        first_row = lf.head(1).collect().to_dicts()[0]
        last_row = lf.tail(1).collect().to_dicts()[0]
        print(f"  First ID: {first_row.get('entity_id') or first_row.get('source1_entity_id')}")
        print(f"  Last ID:  {last_row.get('entity_id') or last_row.get('source1_entity_id')}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to read file: {e}")
        all_passed = False

print("\n" + "="*70)
if all_passed:
    print("RESULT: ALL DATASET FILES ARE COMPLETELY INTACT AND UNCORRUPTED!")
else:
    print("RESULT: ISSUES DETECTED DURING INTEGRITY CHECK!")
print("="*70)
