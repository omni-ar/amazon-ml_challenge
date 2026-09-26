import os
import zipfile
import sys

WORKSPACE_ROOT = r"c:\Users\arjit\Desktop\ml_challenge"
ZIP_OUT = os.path.join(WORKSPACE_ROOT, "submission.zip")

print("Packaging submission zip archive...")

files_to_add = [
    # Output files
    (os.path.join(WORKSPACE_ROOT, "output", "matching_results.tsv"), "output/matching_results.tsv"),
    (os.path.join(WORKSPACE_ROOT, "output", "candidate_pairs.tsv"), "output/candidate_pairs.tsv"),
    # Methodology doc
    (os.path.join(WORKSPACE_ROOT, "Documentation_template.md"), "Documentation_template.md"),
    # Code files
    (os.path.join(WORKSPACE_ROOT, "code", "business_entity_resolution", "README.md"), "code/business_entity_resolution/README.md"),
    (os.path.join(WORKSPACE_ROOT, "code", "business_entity_resolution", "requirements.txt"), "code/business_entity_resolution/requirements.txt"),
    (os.path.join(WORKSPACE_ROOT, "code", "business_entity_resolution", "src", "config.py"), "code/business_entity_resolution/src/config.py"),
    (os.path.join(WORKSPACE_ROOT, "code", "business_entity_resolution", "src", "normalize.py"), "code/business_entity_resolution/src/normalize.py"),
    (os.path.join(WORKSPACE_ROOT, "code", "business_entity_resolution", "src", "blocking.py"), "code/business_entity_resolution/src/blocking.py"),
    (os.path.join(WORKSPACE_ROOT, "code", "business_entity_resolution", "src", "features.py"), "code/business_entity_resolution/src/features.py"),
    (os.path.join(WORKSPACE_ROOT, "code", "business_entity_resolution", "src", "model.py"), "code/business_entity_resolution/src/model.py"),
    (os.path.join(WORKSPACE_ROOT, "code", "business_entity_resolution", "src", "run_pipeline.py"), "code/business_entity_resolution/src/run_pipeline.py"),
]

with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for src_path, arc_name in files_to_add:
        if os.path.exists(src_path):
            size_mb = os.path.getsize(src_path) / (1024 * 1024)
            print(f"Adding: {arc_name:45s} ({size_mb:6.1f} MB)...")
            zipf.write(src_path, arc_name)
        else:
            print(f"[ERROR] Missing file: {src_path}")

zip_size_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
print(f"\nSuccessfully generated {ZIP_OUT} ({zip_size_mb:.1f} MB)")
