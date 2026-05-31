"""
PyTorch datasets for elution sequence prediction.

Sliding-window approach: each training example is a fixed-length context window
of token indices, with the target being the next token's m/z bin (and optionally
RT bin, mass defect bin).

Multi-field input: each token position has multiple features
  [mz_bin, md_bin, rt_gap_bin_idx, polarity_idx, intensity_rank_idx]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_SEQUENCES,
    RANDOM_SEED,
    CONTEXT_LENGTH,
    BATCH_SIZE,
    RT_GAP_LABELS,
    INTENSITY_RANK_LABELS,
)


# Categorical encodings (fixed across all runs)
POLARITY_MAP = {"pos": 0, "neg": 1, "unk": 2}
RT_GAP_MAP = {label: i for i, label in enumerate(RT_GAP_LABELS)}
INTENSITY_MAP = {label: i for i, label in enumerate(INTENSITY_RANK_LABELS)}


def load_and_encode(parquet_path=None):
    """Load tokenized features and encode all fields as integers."""
    if parquet_path is None:
        parquet_path = DATA_SEQUENCES / "tokenized_features.parquet"
    tok = pd.read_parquet(parquet_path)
    tok = tok.sort_values(["study", "sample_id", "seq_pos"])

    # Encode categorical fields as integers
    tok["polarity_idx"] = tok["polarity_tok"].map(POLARITY_MAP).fillna(2).astype(int)
    tok["rt_gap_idx"] = tok["rt_gap_bin"].astype(str).map(RT_GAP_MAP).fillna(0).astype(int)
    tok["intensity_idx"] = tok["intensity_rank"].astype(str).map(INTENSITY_MAP).fillna(4).astype(int)

    return tok


def sample_aware_split(tok, test_fraction=0.15, val_fraction=0.15):
    """Split by sample (no leakage). Returns train, val, test DataFrames."""
    rng = np.random.RandomState(RANDOM_SEED)
    samples = tok[["study", "sample_id"]].drop_duplicates().reset_index(drop=True)

    train_keys, val_keys, test_keys = set(), set(), set()

    for study, group in samples.groupby("study"):
        indices = group.index.values.copy()
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))

        for idx in indices[:n_test]:
            test_keys.add((group.loc[idx, "study"], group.loc[idx, "sample_id"]))
        for idx in indices[n_test:n_test + n_val]:
            val_keys.add((group.loc[idx, "study"], group.loc[idx, "sample_id"]))
        for idx in indices[n_test + n_val:]:
            train_keys.add((group.loc[idx, "study"], group.loc[idx, "sample_id"]))

    def mask_for(keys):
        return tok.apply(lambda r: (r["study"], r["sample_id"]) in keys, axis=1)

    return tok[mask_for(train_keys)], tok[mask_for(val_keys)], tok[mask_for(test_keys)]


def build_sample_arrays(df):
    """Convert DataFrame to per-sample arrays of encoded features.

    Returns: list of dicts, each with arrays:
      mz_bin, md_bin, rt_gap_idx, polarity_idx, intensity_idx, rt_bin (target)
    """
    samples = []
    for (study, sid), group in df.groupby(["study", "sample_id"]):
        g = group.sort_values("seq_pos")
        samples.append({
            "study": study,
            "sample_id": sid,
            "mz_bin": g["mz_bin"].values.astype(np.int64),
            "md_bin": g["md_bin"].values.astype(np.int64),
            "rt_gap_idx": g["rt_gap_idx"].values.astype(np.int64),
            "polarity_idx": g["polarity_idx"].values.astype(np.int64),
            "intensity_idx": g["intensity_idx"].values.astype(np.int64),
            "rt_bin": g["rt_bin"].values.astype(np.int64),
        })
    return samples


class ElutionSequenceDataset(Dataset):
    """Sliding-window dataset for next-token prediction.

    Each example: context window of `context_length` multi-field tokens
    Target: m/z_bin of the next token (classification over m/z bins)

    Secondary targets (optional): rt_bin, md_bin of the next token
    """

    def __init__(self, sample_arrays, context_length=CONTEXT_LENGTH,
                 max_mz_bin=120, max_md_bin=20, max_rt_gap=7,
                 max_polarity=3, max_intensity=5):
        self.context_length = context_length
        self.max_mz_bin = max_mz_bin

        # Build flat index: (sample_idx, position) for all valid windows
        self.examples = []
        self.sample_arrays = sample_arrays

        for si, s in enumerate(sample_arrays):
            n = len(s["mz_bin"])
            # Each position from context_length to n-1 is a valid example
            for pos in range(context_length, n):
                self.examples.append((si, pos))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        si, pos = self.examples[idx]
        s = self.sample_arrays[si]

        start = pos - self.context_length
        # Context: previous `context_length` tokens (multi-field)
        ctx_mz = torch.tensor(s["mz_bin"][start:pos], dtype=torch.long)
        ctx_md = torch.tensor(s["md_bin"][start:pos], dtype=torch.long)
        ctx_gap = torch.tensor(s["rt_gap_idx"][start:pos], dtype=torch.long)
        ctx_pol = torch.tensor(s["polarity_idx"][start:pos], dtype=torch.long)
        ctx_int = torch.tensor(s["intensity_idx"][start:pos], dtype=torch.long)

        # Target: next token's m/z bin
        target_mz = torch.tensor(s["mz_bin"][pos], dtype=torch.long)
        # Secondary targets
        target_rt = torch.tensor(s["rt_bin"][pos], dtype=torch.long)
        target_md = torch.tensor(s["md_bin"][pos], dtype=torch.long)

        return {
            "ctx_mz": ctx_mz,
            "ctx_md": ctx_md,
            "ctx_gap": ctx_gap,
            "ctx_pol": ctx_pol,
            "ctx_int": ctx_int,
            "target_mz": target_mz,
            "target_rt": target_rt,
            "target_md": target_md,
        }


def create_dataloaders(context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE,
                       num_workers=0):
    """Full pipeline: load, encode, split, build datasets and dataloaders."""
    print("Loading and encoding features...")
    tok = load_and_encode()

    print("Splitting samples...")
    train_df, val_df, test_df = sample_aware_split(tok)
    print(f"  Train: {train_df[['study','sample_id']].drop_duplicates().shape[0]} samples")
    print(f"  Val:   {val_df[['study','sample_id']].drop_duplicates().shape[0]} samples")
    print(f"  Test:  {test_df[['study','sample_id']].drop_duplicates().shape[0]} samples")

    print("Building sample arrays...")
    train_arrays = build_sample_arrays(train_df)
    val_arrays = build_sample_arrays(val_df)
    test_arrays = build_sample_arrays(test_df)

    # Compute max m/z bin for output layer sizing
    max_mz = max(
        max(s["mz_bin"].max() for s in train_arrays),
        max(s["mz_bin"].max() for s in val_arrays),
        max(s["mz_bin"].max() for s in test_arrays),
    ) + 1  # +1 for 0-indexed

    print(f"  Max m/z bin: {max_mz}")
    print("Building datasets...")

    train_ds = ElutionSequenceDataset(train_arrays, context_length, max_mz_bin=max_mz)
    val_ds = ElutionSequenceDataset(val_arrays, context_length, max_mz_bin=max_mz)
    test_ds = ElutionSequenceDataset(test_arrays, context_length, max_mz_bin=max_mz)

    print(f"  Train examples: {len(train_ds):,}")
    print(f"  Val examples:   {len(val_ds):,}")
    print(f"  Test examples:  {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    meta = {
        "max_mz_bin": max_mz,
        "context_length": context_length,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }

    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    train_loader, val_loader, test_loader, meta = create_dataloaders()
    print(f"\nMeta: {meta}")

    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
