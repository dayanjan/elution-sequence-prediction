"""
Configuration for the Elution Sequence Prediction project.
Central place for all parameters — tokenization, model, evaluation.
"""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SEQUENCES = PROJECT_ROOT / "data" / "sequences"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
CHECKPOINTS = OUTPUTS / "checkpoints"

# Path to shared datasets
DATASETS_ROOT = PROJECT_ROOT.parent.parent / "datasets"

# === Our validation datasets ===
OUR_DATASETS = {
    "redhart2": DATASETS_ROOT / "redhart2_rplc",
    "cardiac_arrest": DATASETS_ROOT / "cardiac_arrest_rplc",
    "gvhd": DATASETS_ROOT / "gvhd_rplc",
    "pcos": DATASETS_ROOT / "pcos_rplc",
}

# === Tokenization ===
# Hydrophobicity-driven tokenization — NOT subclass-only
# RP-LC elution order is governed by total hydrophobicity (acyl chain length + unsaturation)

MZ_BIN_WIDTH = 10  # Da — coarse bins for m/z
RT_GAP_BINS = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 15.0, float("inf")]  # seconds
RT_GAP_LABELS = ["co-elute", "0.1-0.5s", "0.5-1s", "1-2s", "2-5s", "5-15s", ">15s"]

INTENSITY_RANK_BINS = [0, 0.01, 0.05, 0.20, 0.50, 1.0]  # percentile
INTENSITY_RANK_LABELS = ["top1%", "top5%", "top20%", "top50%", "low"]

# Special tokens
SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]

# === Model ===
CONTEXT_LENGTH = 64  # number of prior tokens to condition on
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10  # early stopping

# Transformer-specific
NUM_HEADS = 4
FF_DIM = 256

# === Evaluation ===
PREDICTION_HORIZONS = [5, 15, 30]  # seconds
TOP_K = [1, 3, 5]  # for top-k accuracy

# === Train/Val/Test split ===
# Sample-aware split — no sequence leakage
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42
