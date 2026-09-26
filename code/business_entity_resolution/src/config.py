import os

class Config:
    WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATASET_DIR = os.path.join(WORKSPACE_ROOT, "student_resource", "dataset")
    OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "output")
    WORK_DIR = os.path.join(WORKSPACE_ROOT, "work")

    TRAIN_S1_PATH = os.path.join(DATASET_DIR, "train", "train_source1.tsv")
    TRAIN_S2_PATH = os.path.join(DATASET_DIR, "train", "train_source2.tsv")
    TRAIN_S3_PATH = os.path.join(DATASET_DIR, "train", "train_source3.tsv")
    TRAIN_GT_PATH = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")

    TEST_S1_PATH = os.path.join(DATASET_DIR, "test", "test_source1.tsv")
    TEST_S2_PATH = os.path.join(DATASET_DIR, "test", "test_source2.tsv")
    TEST_S3_PATH = os.path.join(DATASET_DIR, "test", "test_source3.tsv")

    # Output paths (both standard and (2) as requested)
    MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    MATCHING_OUT_2 = os.path.join(OUTPUT_DIR, "matching_results(2).tsv")
    CANDIDATE_OUT_2 = os.path.join(OUTPUT_DIR, "candidate_pairs(2).tsv")
    ZIP_OUT_2 = os.path.join(WORKSPACE_ROOT, "submission(2).zip")

    # Hyperparameters
    SEED = 42
    K_CANDIDATES = 40          # Top K candidates per S1 entity
    KEY_CAP = 600              # Cap for key frequency (optimal balance between recall and memory)
    S1_CHUNK = 25_000          # S1 records per blocking chunk
    FEAT_CHUNK = 500_000       # Pairs per feature extraction chunk

    # Model features: Enhanced with transliteration and explicit conflict signals
    FEATURES = [
        # Graph & Blocking context
        "b_score", "b_name", "b_addr", "b_hnum", "b_nkeys", "b_rank", "n_cands", "b_rel",
        "cand_deg", "cand_rank", "cand_rel", "s1_namefreq",
        # Original name similarities
        "nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "nm_jw", "nm_glued", "nm_full_tset",
        # Transliterated name similarities (Crucial for Indian scripts & accents)
        "nm_tr_ratio", "nm_tr_tset", "nm_tr_tsort", "nm_tr_jw", "nm_max_tset",
        # Address similarities
        "ad_ratio", "ad_tset", "ad_partial", "addr2_empty", "addr_both_present", "loc_jaccard",
        # Explicit Numerical & Location conflict features (Crucial for Precision)
        "num_shared", "num_n1", "num_n2", "num_first_eq", "num_conflict",
        "postal_match", "postal_conflict", "name_high_num_conflict",
        # Legal suffix indicators
        "legal_eq", "legal_conf", "nm_lenratio", "is_s3", "indic"
    ]
