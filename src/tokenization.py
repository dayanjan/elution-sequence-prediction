"""
Tokenization: Convert raw feature tables into discrete token sequences.

Each detected feature becomes a multi-field token:
  [mz_bin, mass_defect_bin, rt_gap_bin, polarity, intensity_rank, lipid_class?]

Features within a sample are sorted by RT to form elution sequences.
Sequences are bookended with [BOS] and [EOS] special tokens.

All bin widths and boundaries are defined in config.yaml / config.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_SEQUENCES,
    FIGURES,
    MZ_BIN_WIDTH,
    RT_GAP_BINS,
    RT_GAP_LABELS,
    INTENSITY_RANK_BINS,
    INTENSITY_RANK_LABELS,
    SPECIAL_TOKENS,
    RANDOM_SEED,
)
from preprocessing import load_all_datasets

# Load config.yaml for mass_defect_bins and rt_bin_width
_config_path = Path(__file__).parent.parent / "config.yaml"
with open(_config_path) as f:
    _yaml_cfg = yaml.safe_load(f)

MASS_DEFECT_BINS = _yaml_cfg["tokenization"]["mass_defect_bins"]  # 20
RT_BIN_WIDTH = _yaml_cfg["tokenization"]["rt_bin_width"]  # 3 seconds


def compute_mz_bin(mz: pd.Series) -> pd.Series:
    """Bin m/z values into coarse bins (e.g., 10 Da width)."""
    return (mz // MZ_BIN_WIDTH).astype(int)


def compute_mass_defect_bin(mz: pd.Series) -> pd.Series:
    """Bin fractional part of m/z into equal-width bins across [0, 1)."""
    mass_defect = mz - np.floor(mz)
    return (mass_defect * MASS_DEFECT_BINS).astype(int).clip(upper=MASS_DEFECT_BINS - 1)


def compute_rt_gap_bin(rt_gap: pd.Series) -> pd.Series:
    """Bin RT gap from previous feature into categorical bins."""
    return pd.cut(
        rt_gap,
        bins=RT_GAP_BINS,
        labels=RT_GAP_LABELS,
        right=False,
        include_lowest=True,
    )


def compute_intensity_rank(intensity: pd.Series) -> pd.Series:
    """Compute within-sample intensity percentile rank, then bin."""
    # Rank within group (caller should groupby sample first)
    rank_pct = intensity.rank(pct=True)
    # Invert so top intensity = lowest percentile value
    rank_pct = 1 - rank_pct
    return pd.cut(
        rank_pct,
        bins=INTENSITY_RANK_BINS,
        labels=INTENSITY_RANK_LABELS,
        right=True,
        include_lowest=True,
    )


def encode_polarity(polarity: pd.Series) -> pd.Series:
    """Encode polarity as compact token."""
    return polarity.map({"(+) ESI": "pos", "(-) ESI": "neg"}).fillna("unk")


def tokenize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw feature DataFrame into tokenized feature DataFrame.

    Input: long-format DataFrame from preprocessing.load_all_datasets()
    Output: DataFrame with one row per (study, sample_id, feature), sorted by RT,
            with token fields: mz_bin, md_bin, rt_gap_bin, polarity_tok,
            intensity_rank, lipid_class_tok, composite_token
    """
    # Filter to detected features (intensity > 0, valid m/z and RT)
    tok = df[(df.intensity > 0) & df.mz.notna() & df.rt.notna()].copy()

    # === Compute token fields ===
    tok["mz_bin"] = compute_mz_bin(tok["mz"])
    tok["md_bin"] = compute_mass_defect_bin(tok["mz"])
    tok["polarity_tok"] = encode_polarity(tok["polarity"])

    # Lipid class token (available for ~15%)
    tok["class_tok"] = tok["lipid_class"].fillna("UNK").str.replace(" ", "_")

    # === Per-sample operations: sort by RT, compute gaps and intensity ranks ===
    tok = tok.sort_values(["study", "sample_id", "rt", "mz"])

    # RT gap from previous feature within each sample
    tok["rt_gap"] = tok.groupby(["study", "sample_id"])["rt"].diff() * 60  # min -> seconds
    tok["rt_gap"] = tok["rt_gap"].fillna(0).clip(lower=0)
    tok["rt_gap_bin"] = compute_rt_gap_bin(tok["rt_gap"])

    # Intensity rank within each sample
    tok["intensity_rank"] = tok.groupby(["study", "sample_id"])["intensity"].transform(
        lambda x: pd.cut(
            1 - x.rank(pct=True),
            bins=INTENSITY_RANK_BINS,
            labels=INTENSITY_RANK_LABELS,
            right=True,
            include_lowest=True,
        )
    )

    # === Composite token string ===
    # Format: mz{bin}_md{bin}_gap{bin}_pol{p}_int{rank}[_cls{class}]
    tok["composite_token"] = (
        "mz" + tok["mz_bin"].astype(str)
        + "_md" + tok["md_bin"].astype(str)
        + "_gap" + tok["rt_gap_bin"].astype(str)
        + "_" + tok["polarity_tok"]
        + "_" + tok["intensity_rank"].astype(str)
    )

    # Append lipid class when known
    has_class = tok["lipid_class"].notna()
    tok.loc[has_class, "composite_token"] = (
        tok.loc[has_class, "composite_token"] + "_cls" + tok.loc[has_class, "class_tok"]
    )

    # === RT bin (the prediction target) ===
    # This is what the model predicts: which RT bin does the next feature fall into?
    tok["rt_bin"] = (tok["rt"] * 60 / RT_BIN_WIDTH).astype(int)  # RT in min -> bin index

    # === Sequence position within each sample ===
    tok["seq_pos"] = tok.groupby(["study", "sample_id"]).cumcount()

    return tok


def build_sequences(tok: pd.DataFrame) -> dict:
    """
    Build per-sample token sequences from tokenized DataFrame.

    Returns dict: {(study, sample_id): {
        "tokens": [str, ...],         # composite token strings
        "rt_bins": [int, ...],         # target RT bins
        "rt_seconds": [float, ...],    # raw RT in seconds
        "sample_type": str,            # "QC" or "analytical"
        "n_features": int,
    }}
    """
    sequences = {}
    for (study, sample_id), group in tok.groupby(["study", "sample_id"]):
        g = group.sort_values("rt")
        sequences[(study, sample_id)] = {
            "tokens": ["[BOS]"] + g["composite_token"].tolist() + ["[EOS]"],
            "rt_bins": [0] + g["rt_bin"].tolist() + [0],
            "rt_seconds": [0.0] + (g["rt"] * 60).tolist() + [0.0],
            "sample_type": g["sample_type"].iloc[0],
            "n_features": len(g),
        }
    return sequences


def build_vocabulary(sequences: dict) -> dict:
    """
    Build token-to-index vocabulary from all sequences.

    Returns: {token_str: int_index}
    """
    vocab = {}
    # Special tokens first
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab[tok] = i

    # Collect all unique tokens
    all_tokens = set()
    for seq_data in sequences.values():
        all_tokens.update(seq_data["tokens"])

    # Remove special tokens (already added)
    all_tokens -= set(SPECIAL_TOKENS)

    # Sort for reproducibility
    for tok in sorted(all_tokens):
        vocab[tok] = len(vocab)

    return vocab


def print_stats(tok: pd.DataFrame, sequences: dict, vocab: dict):
    """Print comprehensive tokenization statistics."""
    print("=" * 70)
    print("TOKENIZATION STATISTICS")
    print("=" * 70)

    # Token field distributions
    print(f"\n--- Token field ranges ---")
    print(f"  m/z bins:          {tok['mz_bin'].nunique():>6} unique  (range: {tok['mz_bin'].min()}-{tok['mz_bin'].max()})")
    print(f"  Mass defect bins:  {tok['md_bin'].nunique():>6} unique  (range: {tok['md_bin'].min()}-{tok['md_bin'].max()})")
    print(f"  RT gap bins:       {tok['rt_gap_bin'].nunique():>6} unique  ({', '.join(RT_GAP_LABELS)})")
    print(f"  Polarity tokens:   {tok['polarity_tok'].nunique():>6} unique  ({', '.join(tok['polarity_tok'].unique())})")
    print(f"  Intensity ranks:   {tok['intensity_rank'].nunique():>6} unique")
    print(f"  Lipid class tokens:{tok['class_tok'].nunique():>6} unique  ({tok['lipid_class'].notna().sum()}/{len(tok)} have class)")
    print(f"  Composite tokens:  {tok['composite_token'].nunique():>6} unique")

    # RT bin distribution (prediction target)
    print(f"\n--- Prediction target (RT bins, {RT_BIN_WIDTH}s width) ---")
    print(f"  RT bin range: {tok['rt_bin'].min()} to {tok['rt_bin'].max()}")
    print(f"  Unique RT bins: {tok['rt_bin'].nunique()}")
    print(f"  RT range covered: {tok['rt_bin'].min() * RT_BIN_WIDTH}s to {tok['rt_bin'].max() * RT_BIN_WIDTH}s")

    # Vocabulary
    print(f"\n--- Vocabulary ---")
    print(f"  Total vocabulary size: {len(vocab)}")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  Feature tokens: {len(vocab) - len(SPECIAL_TOKENS)}")

    # Sequence statistics
    seq_lens = [s["n_features"] for s in sequences.values()]
    sample_types = [s["sample_type"] for s in sequences.values()]
    n_qc = sum(1 for t in sample_types if t == "QC")
    n_analytical = sum(1 for t in sample_types if t == "analytical")

    print(f"\n--- Sequences ---")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  QC samples: {n_qc}")
    print(f"  Analytical samples: {n_analytical}")
    print(f"  Sequence lengths (features per sample):")
    print(f"    Mean:   {np.mean(seq_lens):.0f}")
    print(f"    Median: {np.median(seq_lens):.0f}")
    print(f"    Min:    {np.min(seq_lens)}")
    print(f"    Max:    {np.max(seq_lens)}")
    print(f"    Std:    {np.std(seq_lens):.0f}")

    # Per-study breakdown
    print(f"\n--- Per-study breakdown ---")
    for study in sorted(set(k[0] for k in sequences)):
        study_seqs = {k: v for k, v in sequences.items() if k[0] == study}
        study_lens = [s["n_features"] for s in study_seqs.values()]
        print(f"  {study}: {len(study_seqs)} sequences, "
              f"mean {np.mean(study_lens):.0f} features/sample "
              f"(range: {np.min(study_lens)}-{np.max(study_lens)})")

    # RT gap distribution
    print(f"\n--- RT gap distribution ---")
    print(tok["rt_gap_bin"].value_counts().sort_index().to_string())

    # Top 20 most common composite tokens
    print(f"\n--- Top 20 most common tokens ---")
    top_tokens = tok["composite_token"].value_counts().head(20)
    for token, count in top_tokens.items():
        print(f"  {token:<60} {count:>6}")


def main():
    np.random.seed(RANDOM_SEED)

    print("Loading datasets...")
    df = load_all_datasets()

    print("\nTokenizing features...")
    tok = tokenize_features(df)

    print("Building sequences...")
    sequences = build_sequences(tok)

    print("Building vocabulary...")
    vocab = build_vocabulary(sequences)

    print_stats(tok, sequences, vocab)

    # Save artifacts
    DATA_SEQUENCES.mkdir(parents=True, exist_ok=True)

    # Save tokenized features
    tok_save = tok[["study", "sample_id", "sample_type", "feature_id", "annotation",
                     "mz", "rt", "intensity", "polarity", "lipid_class",
                     "mz_bin", "md_bin", "rt_gap", "rt_gap_bin", "polarity_tok",
                     "intensity_rank", "class_tok", "composite_token",
                     "rt_bin", "seq_pos"]].copy()
    tok_save.to_parquet(DATA_SEQUENCES / "tokenized_features.parquet", index=False)
    print(f"\nSaved tokenized features: {DATA_SEQUENCES / 'tokenized_features.parquet'}")

    # Save vocabulary
    vocab_df = pd.DataFrame(list(vocab.items()), columns=["token", "index"])
    vocab_df.to_csv(DATA_SEQUENCES / "vocabulary.csv", index=False)
    print(f"Saved vocabulary: {DATA_SEQUENCES / 'vocabulary.csv'}")

    # Save sequence metadata
    seq_meta = []
    for (study, sample_id), sdata in sequences.items():
        seq_meta.append({
            "study": study,
            "sample_id": sample_id,
            "sample_type": sdata["sample_type"],
            "n_features": sdata["n_features"],
            "duration_s": sdata["rt_seconds"][-2] if sdata["n_features"] > 0 else 0,
        })
    seq_meta_df = pd.DataFrame(seq_meta)
    seq_meta_df.to_csv(DATA_SEQUENCES / "sequence_metadata.csv", index=False)
    print(f"Saved sequence metadata: {DATA_SEQUENCES / 'sequence_metadata.csv'}")

    return tok, sequences, vocab


if __name__ == "__main__":
    tok, sequences, vocab = main()
