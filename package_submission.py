import os
import zipfile
import sys

WORKSPACE_ROOT = r"c:\Users\arjit\Desktop\ml_challenge"
ZIP_OUT_2 = os.path.join(WORKSPACE_ROOT, "submission(2).zip")
ZIP_OUT = os.path.join(WORKSPACE_ROOT, "submission.zip")

print("Packaging submission zip archives...")

files_to_add = [
    # Primary output files for submission
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

# If matching_results(2).tsv exists, also package submission(2).zip
extra_2_files = [
    (os.path.join(WORKSPACE_ROOT, "output", "matching_results(2).tsv"), "output/matching_results(2).tsv"),
    (os.path.join(WORKSPACE_ROOT, "output", "candidate_pairs(2).tsv"), "output/candidate_pairs(2).tsv"),
]

def create_zip(target_zip, extra_items=None):
    items = list(files_to_add)
    if extra_items:
        items.extend(extra_items)
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for src_path, arc_name in items:
            if os.path.exists(src_path):
                size_mb = os.path.getsize(src_path) / (1024 * 1024)
                print(f"[{os.path.basename(target_zip)}] Adding: {arc_name:45s} ({size_mb:6.1f} MB)")
                zipf.write(src_path, arc_name)
            else:
                print(f"[{os.path.basename(target_zip)}] [ERROR] Missing file: {src_path}")
    size_mb = os.path.getsize(target_zip) / (1024 * 1024)
    print(f"--> Successfully created {target_zip} ({size_mb:.1f} MB)\n")

create_zip(ZIP_OUT_2, extra_2_files)
create_zip(ZIP_OUT)
