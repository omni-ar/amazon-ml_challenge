import os

class Config:
    # Base paths
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

    MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    # Hyperparameters
    SEED = 42
    K_CANDIDATES = 40          # Top K candidates per S1 entity
    KEY_CAP = 300              # Cap on key frequency in blocking
    S1_CHUNK = 60_000          # S1 records per blocking chunk
    FEAT_CHUNK = 500_000       # Pairs per feature extraction chunk
    VAL_RATIO = 0.20           # 20% validation split for training
    TRAIN_SAMPLE_S1 = 120_000  # Number of S1 training entities for LightGBM fit

    # Model features
    FEATURES = [
        "b_score", "b_name", "b_addr", "b_hnum", "b_nkeys", "b_rank", "n_cands", "b_rel",
        "cand_deg", "cand_rank", "cand_rel", "s1_namefreq", "nm_ratio", "nm_tset", "nm_tsort",
        "nm_partial", "nm_jw", "nm_glued", "nm_full_tset", "ad_ratio", "ad_tset", "ad_partial",
        "num_shared", "num_n1", "num_n2", "num_first_eq", "legal_eq", "legal_conf",
        "addr2_empty", "nm_lenratio", "is_s3", "indic"
    ]
