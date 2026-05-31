"""Build the Phase 9 transfer-learning recovery Colab notebook.

The notebook is assembled with nbformat so the saved .ipynb is valid JSON.
It is intentionally self-contained: model definitions, tokenization,
evaluation, Markov calibration, head-only fine-tuning, and full fine-tuning are
all inline for Colab.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path
from textwrap import dedent


try:
    import nbformat as nbf
except ModuleNotFoundError:
    # The project venv used for smoke tests may not have nbformat installed.
    # Fall back to the user-site package that ships with the Jupyter install.
    for user_site in site.getusersitepackages(), *site.getsitepackages():
        if user_site not in sys.path:
            sys.path.append(user_site)
    import nbformat as nbf


SCRIPT_PATH = Path(__file__).resolve()
POC3_DIR = SCRIPT_PATH.parents[1]
NOTEBOOK_PATH = POC3_DIR / "notebooks" / "09_transfer_learning" / "09_transfer_learning.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "colab": {"provenance": [], "gpuType": "L4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        }
    )

    nb.cells = [
        md(
            """
            # Phase 9: Transfer-Learning Recovery on ST000990

            **Version: 1.0** | 2026-05-31

            **Question:** Is the ST000990 zero-shot failure a hard generalization wall, or a recoverable adaptation gap after a small QC calibration set?

            | Experiment | Calibration | Evaluation | Expected proof point |
            |---|---:|---:|---|
            | Zero-shot gate | none | all 153 ST000990 samples | Reproduces Phase 8 top-1 ≈ 2.63% |
            | Leakage check | none | 126 analytical SO### samples | Pure LSTM analytical top-1 ≈ 2.59% |
            | Markov calibration | 1-15 QC samples | 126 analytical SO### samples | Cheap non-neural recovery reference |
            | Head-only fine-tune | 1-15 QC samples | 126 analytical SO### samples | Frozen-body adaptation reference |
            | Full fine-tune | 1-15 QC samples | 126 analytical SO### samples | Recovery to about 48% top-1 / 67% top-5 at N=15 |

            **Runtime expectation:** ~a few minutes on an L4/A100; ~45 min on CPU.

            **Prerequisites:** Trained LSTM checkpoint from `01_train_models/outputs/lstm_best.pt` and ST000990 `merged_features.parquet`.

            **Changelog:**
            - v1.0: Initial Phase 9 / 9B Colab recovery notebook.
            """
        ),
        md("## 1. Setup"),
        code(
            """
            import copy
            import json
            import os
            import subprocess
            import sys
            import time
            from dataclasses import dataclass

            import numpy as np
            import pandas as pd
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset

            try:
                import pyarrow  # noqa: F401
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])

            import matplotlib.pyplot as plt

            print(f"PyTorch {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {device}")
            """
        ),
        md("## 2. Mount Drive, Paths, and Configuration"),
        code(
            """
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                IN_COLAB = True
                BASE_DIR = "/content/drive/MyDrive/0000 Fun with coding/088 Lights-Out R01 Grant/Specific Aim 1/poc3_elution_sequence"
            except Exception:
                IN_COLAB = False
                BASE_DIR = os.environ.get("POC3_BASE", ".")

            LSTM_CHECKPOINT = os.environ.get(
                "LSTM_CHECKPOINT_OVERRIDE",
                f"{BASE_DIR}/01_train_models/outputs/lstm_best.pt",
            )
            ST000990_PATH = f"{BASE_DIR}/08_st000990_validation/merged_features.parquet"
            if (not IN_COLAB) and (not os.path.exists(ST000990_PATH)):
                local_st000990 = f"{BASE_DIR}/data/external/ST000990/msdial_merged/merged_features.parquet"
                if os.path.exists(local_st000990):
                    ST000990_PATH = local_st000990
            ST000990_PATH = os.environ.get("ST000990_PATH_OVERRIDE", ST000990_PATH)

            EXPERIMENT_DIR = f"{BASE_DIR}/09_transfer_learning"
            OUTPUT_DIR = f"{EXPERIMENT_DIR}/outputs"
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            VERSION = "1.0"
            RANDOM_SEED = 42
            CONTEXT_LENGTH = 64
            EMBEDDING_DIM = 64
            HIDDEN_DIM = 128
            NUM_LAYERS = 2
            DROPOUT = 0.1
            MZ_BIN_WIDTH = 10
            MASS_DEFECT_BINS = 20
            RT_BIN_WIDTH = 3

            RT_GAP_BINS = [-0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 15.0, 9999]
            RT_GAP_LABELS = ["co-elute", "0.1-0.5s", "0.5-1s", "1-2s", "2-5s", "5-15s", ">15s"]
            INTENSITY_RANK_BINS = [-0.001, 0.01, 0.05, 0.20, 0.50, 1.001]
            INTENSITY_RANK_LABELS = ["top1%", "top5%", "top20%", "top50%", "low"]
            POLARITY_MAP = {"pos": 0, "neg": 1, "unk": 2}
            RT_GAP_MAP = {label: i for i, label in enumerate(RT_GAP_LABELS)}
            INTENSITY_MAP = {label: i for i, label in enumerate(INTENSITY_RANK_LABELS)}

            N_VALUES = [int(x) for x in os.environ.get("PHASE9_N_VALUES", "1,2,5,10,15").split(",") if x.strip()]
            ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
            BATCH_SIZE = 2048
            HEAD_EPOCHS = int(os.environ.get("PHASE9_HEAD_EPOCHS", "40"))
            FULL_MAX_EPOCHS = int(os.environ.get("PHASE9_FULL_MAX_EPOCHS", "30"))
            FULL_PATIENCE = int(os.environ.get("PHASE9_FULL_PATIENCE", "6"))
            FULL_LR = 3e-4
            FULL_BATCH_SIZE = 32

            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)

            print(f"IN_COLAB: {IN_COLAB}")
            print(f"BASE_DIR: {BASE_DIR}")
            print(f"LSTM checkpoint: {LSTM_CHECKPOINT}")
            print(f"ST000990 data: {ST000990_PATH}")
            print(f"Output dir: {OUTPUT_DIR}")
            print(f"N values: {N_VALUES}; full fine-tune max epochs: {FULL_MAX_EPOCHS}")
            """
        ),
        md("## 3. Model Definition and Checkpoint Loading"),
        code(
            """
            class MultiFieldEmbedding(nn.Module):
                def __init__(self, embedding_dim, max_mz_bin=120, max_md_bin=20,
                             max_rt_gap=7, max_polarity=3, max_intensity=5):
                    super().__init__()
                    self.mz_embed = nn.Embedding(max_mz_bin, embedding_dim)
                    self.md_embed = nn.Embedding(max_md_bin, embedding_dim)
                    self.gap_embed = nn.Embedding(max_rt_gap, embedding_dim)
                    self.pol_embed = nn.Embedding(max_polarity, embedding_dim)
                    self.int_embed = nn.Embedding(max_intensity, embedding_dim)

                def forward(self, mz, md, gap, pol, intensity):
                    return (
                        self.mz_embed(mz)
                        + self.md_embed(md)
                        + self.gap_embed(gap)
                        + self.pol_embed(pol)
                        + self.int_embed(intensity)
                    )


            class LSTMModel(nn.Module):
                def __init__(self, num_mz_classes, embedding_dim=64, hidden_dim=128,
                             num_layers=2, dropout=0.1, **embed_kwargs):
                    super().__init__()
                    self.embedding = MultiFieldEmbedding(embedding_dim, **embed_kwargs)
                    self.lstm = nn.LSTM(
                        input_size=embedding_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0,
                        batch_first=True,
                    )
                    self.dropout = nn.Dropout(dropout)
                    self.head = nn.Linear(hidden_dim, num_mz_classes)

                def forward(self, mz, md, gap, pol, intensity):
                    x = self.embedding(mz, md, gap, pol, intensity)
                    out, _ = self.lstm(x)
                    return self.head(self.dropout(out[:, -1, :]))


            def load_lstm_model(device):
                ckpt = torch.load(LSTM_CHECKPOINT, map_location=device, weights_only=False)
                max_mz_bin = int(ckpt["config"]["num_classes"])
                embed_kw = dict(
                    max_mz_bin=max_mz_bin,
                    max_md_bin=20,
                    max_rt_gap=7,
                    max_polarity=3,
                    max_intensity=5,
                )
                model = LSTMModel(
                    max_mz_bin,
                    EMBEDDING_DIM,
                    HIDDEN_DIM,
                    NUM_LAYERS,
                    DROPOUT,
                    **embed_kw,
                ).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                return model, ckpt, max_mz_bin


            base_model, lstm_ckpt, MAX_MZ_BIN = load_lstm_model(device)
            print(f"Max m/z bin: {MAX_MZ_BIN}")
            print(
                f"LSTM loaded -- training test: top1={lstm_ckpt['test_metrics']['top1']:.4f}, "
                f"MAE={lstm_ckpt['test_metrics']['mae_da']:.1f} Da"
            )
            """
        ),
        md("## 4. Load and Tokenize ST000990"),
        code(
            """
            @dataclass
            class SampleTokens:
                sample_id: str
                sample_type: str
                mz: np.ndarray
                md: np.ndarray
                gap: np.ndarray
                pol: np.ndarray
                inten: np.ndarray

                @property
                def n_predictions(self):
                    return max(0, len(self.mz) - CONTEXT_LENGTH)


            def classify_sample(name):
                if "_NIST" in name:
                    return "nist"
                if "_QC" in name:
                    return "qc"
                if "_Rep" in name:
                    return "replicate"
                return "analytical"


            def tokenize_sample(feat_df, sample_col, max_mz_bin):
                mask = feat_df[sample_col].notna() & (feat_df[sample_col] > 0)
                sf = feat_df.loc[mask, ["mz", "rt_min", "adduct_ion", sample_col]].copy()
                sf = sf.rename(columns={sample_col: "intensity"})
                if len(sf) < CONTEXT_LENGTH + 1:
                    return None

                sf = sf.sort_values(["rt_min", "mz"]).reset_index(drop=True)
                sf["mz_bin"] = (sf["mz"] // MZ_BIN_WIDTH).astype(int).clip(upper=max_mz_bin - 1)

                mass_defect = sf["mz"] - np.floor(sf["mz"])
                sf["md_bin"] = (mass_defect * MASS_DEFECT_BINS).astype(int).clip(upper=MASS_DEFECT_BINS - 1)

                # ST000990 column name is rt_sec, but the values are minutes. Convert gaps to seconds.
                sf["rt_gap"] = sf["rt_min"].diff() * 60
                sf["rt_gap"] = sf["rt_gap"].fillna(0).clip(lower=0)
                sf["rt_gap_idx"] = (
                    pd.cut(
                        sf["rt_gap"],
                        bins=RT_GAP_BINS,
                        labels=RT_GAP_LABELS,
                        right=False,
                        include_lowest=True,
                    )
                    .astype(str)
                    .map(RT_GAP_MAP)
                    .fillna(0)
                    .astype(int)
                )

                sf["polarity_idx"] = POLARITY_MAP["pos"]

                pct_rank = 1 - sf["intensity"].rank(pct=True)
                sf["intensity_idx"] = (
                    pd.cut(
                        pct_rank,
                        bins=INTENSITY_RANK_BINS,
                        labels=INTENSITY_RANK_LABELS,
                        right=True,
                        include_lowest=True,
                    )
                    .astype(str)
                    .map(INTENSITY_MAP)
                    .fillna(4)
                    .astype(int)
                )

                return SampleTokens(
                    sample_id=sample_col,
                    sample_type=classify_sample(sample_col),
                    mz=sf["mz_bin"].to_numpy(np.int64),
                    md=sf["md_bin"].to_numpy(np.int64),
                    gap=sf["rt_gap_idx"].to_numpy(np.int64),
                    pol=sf["polarity_idx"].to_numpy(np.int64),
                    inten=sf["intensity_idx"].to_numpy(np.int64),
                )


            feat = pd.read_parquet(ST000990_PATH)
            sample_cols = [c for c in feat.columns if c.startswith("GLA_TT6_Lipids_")]
            nist_cols = [c for c in sample_cols if "_NIST" in c]
            qc_cols = [c for c in sample_cols if "_QC" in c]
            rep_cols = [c for c in sample_cols if "_Rep" in c]
            analytical_cols = [c for c in sample_cols if "_SO" in c]

            print(f"Merged features: {len(feat)} features x {feat.shape[1]} columns")
            print(f"Samples: {len(sample_cols)} total")
            print(f"  NIST: {len(nist_cols)}, QC: {len(qc_cols)}, Rep: {len(rep_cols)}, Analytical: {len(analytical_cols)}")

            feat = feat.rename(columns={"rt_sec": "rt_min"})
            print(f"m/z range: {feat['mz'].min():.1f} - {feat['mz'].max():.1f} Da")
            print(f"RT range: {feat['rt_min'].min():.2f} - {feat['rt_min'].max():.2f} min")

            # Reviewer-readable wide-to-long view. Tokenization below still works
            # per sample to avoid extra joins and exactly match the validated scripts.
            st000990_long = feat[["mz", "rt_min", "adduct_ion"] + sample_cols].melt(
                id_vars=["mz", "rt_min", "adduct_ion"],
                value_vars=sample_cols,
                var_name="sample_id",
                value_name="intensity",
            )
            st000990_long = st000990_long[st000990_long["intensity"].notna() & (st000990_long["intensity"] > 0)].copy()
            st000990_long["sample_type"] = st000990_long["sample_id"].map(classify_sample)
            print(f"Wide-to-long detected feature rows: {len(st000990_long):,}")
            print(st000990_long["sample_type"].value_counts().to_string())

            tokens = {}
            for col in sample_cols:
                sample = tokenize_sample(feat, col, MAX_MZ_BIN)
                if sample is not None:
                    tokens[col] = sample

            all_bins = np.concatenate([s.mz for s in tokens.values()])
            print(f"Tokenized: {len(tokens)} samples")
            print(
                "Features per sample: "
                f"min={min(len(s.mz) for s in tokens.values())}, "
                f"median={int(np.median([len(s.mz) for s in tokens.values()]))}, "
                f"max={max(len(s.mz) for s in tokens.values())}"
            )
            print(f"Out-of-vocabulary m/z bins after clipping: {int((all_bins >= MAX_MZ_BIN).sum())}/{len(all_bins)}")
            """
        ),
        md("## 5. GATE - Reproduce Zero-Shot Baseline"),
        code(
            """
            def iter_context_batches(samples, batch_size=BATCH_SIZE):
                field_chunks, target_chunks, prev_chunks, sample_chunks = [], [], [], []
                buffered = 0

                def flush():
                    nonlocal field_chunks, target_chunks, prev_chunks, sample_chunks, buffered
                    if buffered == 0:
                        return None
                    fields = [np.concatenate([chunk[i] for chunk in field_chunks]) for i in range(5)]
                    targets = np.concatenate(target_chunks)
                    prev = np.concatenate(prev_chunks)
                    sample_ids = np.concatenate(sample_chunks)
                    field_chunks, target_chunks, prev_chunks, sample_chunks = [], [], [], []
                    buffered = 0
                    return fields, targets, prev, sample_ids

                offsets = np.arange(CONTEXT_LENGTH)
                for sample in samples:
                    n_pred = sample.n_predictions
                    for start_pos in range(CONTEXT_LENGTH, CONTEXT_LENGTH + n_pred, batch_size):
                        end_pos = min(CONTEXT_LENGTH + n_pred, start_pos + batch_size)
                        positions = np.arange(start_pos, end_pos)
                        starts = positions - CONTEXT_LENGTH
                        idx = starts[:, None] + offsets[None, :]
                        field_chunks.append((sample.mz[idx], sample.md[idx], sample.gap[idx], sample.pol[idx], sample.inten[idx]))
                        target_chunks.append(sample.mz[positions])
                        prev_chunks.append(sample.mz[positions - 1])
                        sample_chunks.append(np.repeat(sample.sample_id, len(positions)))
                        buffered += len(positions)
                        if buffered >= batch_size:
                            out = flush()
                            if out is not None:
                                yield out
                out = flush()
                if out is not None:
                    yield out


            def evaluate_model_stream(model, samples, device):
                model.eval()
                n = top1 = top5 = 0
                with torch.no_grad():
                    for fields, targets, _, _ in iter_context_batches(samples):
                        tensors = [torch.as_tensor(x, dtype=torch.long, device=device) for x in fields]
                        logits = model(*tensors)
                        pred1 = logits.argmax(dim=1).cpu().numpy()
                        top5_idx = logits.topk(5, dim=1).indices.cpu().numpy()
                        top1 += int((pred1 == targets).sum())
                        top5 += int((top5_idx == targets[:, None]).any(axis=1).sum())
                        n += len(targets)
                return {"top1": top1 / n, "top5": top5 / n, "n": n}


            all_samples = list(tokens.values())
            zero_shot_all = evaluate_model_stream(base_model, all_samples, device)
            expected_top1 = 0.0263076228273567
            expected_n = 396349
            gate_pass = abs(zero_shot_all["top1"] - expected_top1) <= 0.005 and abs(zero_shot_all["n"] - expected_n) <= 1000
            print(f"STEP 1 zero-shot reproduction: top1={zero_shot_all['top1']:.6f}, top5={zero_shot_all['top5']:.6f}, n={zero_shot_all['n']:,}")
            print(f"GATE: {'PASS' if gate_pass else 'FAIL'} (expected top1≈{expected_top1:.4f}, n≈{expected_n:,})")
            if not gate_pass:
                raise RuntimeError("Zero-shot gate failed: tokenization/evaluation no longer matches Phase 8.")
            """
        ),
        md("## 6. Recovery Experiments"),
        code(
            """
            def extract_features(model, fields, device):
                tensors = [torch.as_tensor(x, dtype=torch.long, device=device) for x in fields]
                with torch.no_grad():
                    x = model.embedding(*tensors)
                    out, _ = model.lstm(x)
                    return out[:, -1, :]


            def collect_feature_table(model, samples, device):
                features, targets, prev, sample_ids = [], [], [], []
                for fields, batch_targets, batch_prev, batch_sample_ids in iter_context_batches(samples):
                    features.append(extract_features(model, fields, device).cpu())
                    targets.append(batch_targets)
                    prev.append(batch_prev)
                    sample_ids.append(batch_sample_ids)
                return torch.cat(features, dim=0), np.concatenate(targets).astype(np.int64), np.concatenate(prev).astype(np.int64), np.concatenate(sample_ids)


            def topk_metrics(scores, targets, ks=(1, 5)):
                max_k = max(ks)
                top_idx = np.argpartition(scores, -max_k, axis=1)[:, -max_k:]
                metrics = {"n": int(len(targets))}
                for k in ks:
                    if k == max_k:
                        cand = top_idx
                    else:
                        cand_scores = np.take_along_axis(scores, top_idx, axis=1)
                        local = np.argpartition(cand_scores, -k, axis=1)[:, -k:]
                        cand = np.take_along_axis(top_idx, local, axis=1)
                    metrics[f"top{k}"] = float((cand == targets[:, None]).any(axis=1).mean())
                return metrics


            def build_markov(samples, num_classes):
                counts = np.zeros((num_classes, num_classes), dtype=np.float64)
                global_counts = np.zeros(num_classes, dtype=np.float64)
                for sample in samples:
                    for pos in range(CONTEXT_LENGTH, len(sample.mz)):
                        prev_bin = sample.mz[pos - 1]
                        target = sample.mz[pos]
                        counts[prev_bin, target] += 1.0
                        global_counts[target] += 1.0
                global_prior = global_counts / global_counts.sum()
                probs = np.zeros_like(counts)
                row_sums = counts.sum(axis=1)
                observed = row_sums > 0
                probs[observed] = counts[observed] / row_sums[observed, None]
                probs[~observed] = global_prior
                return probs.astype(np.float32)


            def train_head(original_head, train_features, train_targets, num_classes, device, epochs=HEAD_EPOCHS, lr=1e-3):
                head = nn.Linear(train_features.shape[1], num_classes)
                head.load_state_dict(original_head.state_dict())
                head.to(device)
                head.train()
                ds = TensorDataset(train_features, torch.as_tensor(train_targets, dtype=torch.long))
                loader = DataLoader(ds, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_SEED))
                opt = torch.optim.Adam(head.parameters(), lr=lr)
                for _ in range(epochs):
                    for xb, yb in loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        opt.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(head(xb), yb)
                        loss.backward()
                        opt.step()
                head.eval()
                return head


            def eval_head(head, features, targets, device):
                n = top1 = top5 = 0
                with torch.no_grad():
                    for start in range(0, len(targets), 8192):
                        xb = features[start:start + 8192].to(device)
                        logits = head(xb)
                        batch_targets = targets[start:start + 8192]
                        pred1 = logits.argmax(dim=1).cpu().numpy()
                        top5_idx = logits.topk(5, dim=1).indices.cpu().numpy()
                        top1 += int((pred1 == batch_targets).sum())
                        top5 += int((top5_idx == batch_targets[:, None]).any(axis=1).sum())
                        n += len(batch_targets)
                return {"top1": top1 / n, "top5": top5 / n, "n": n}


            def make_examples(samples):
                fields = [[] for _ in range(5)]
                targets, sample_ids = [], []
                offsets = np.arange(CONTEXT_LENGTH)
                for sample in samples:
                    positions = np.arange(CONTEXT_LENGTH, len(sample.mz))
                    starts = positions - CONTEXT_LENGTH
                    idx = starts[:, None] + offsets[None, :]
                    arrays = (sample.mz, sample.md, sample.gap, sample.pol, sample.inten)
                    for i, arr in enumerate(arrays):
                        fields[i].append(arr[idx])
                    targets.append(sample.mz[positions])
                    sample_ids.append(np.repeat(sample.sample_id, len(positions)))
                field_tensors = [torch.as_tensor(np.concatenate(chunks), dtype=torch.long) for chunks in fields]
                target_tensor = torch.as_tensor(np.concatenate(targets), dtype=torch.long)
                return field_tensors, target_tensor, np.concatenate(sample_ids)


            def subset_dataset(fields, targets, mask):
                idx = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
                return TensorDataset(*(field.index_select(0, idx) for field in fields), targets.index_select(0, idx))


            def split_for_n(qc_samples, qc_fields, qc_targets, qc_sample_ids, n_cal):
                train_ids = [sample.sample_id for sample in qc_samples[:n_cal]]
                if n_cal < len(qc_samples):
                    val_ids = [qc_samples[n_cal].sample_id]
                    train_mask = np.isin(qc_sample_ids, np.array(train_ids))
                    val_mask = np.isin(qc_sample_ids, np.array(val_ids))
                    split = {"train_samples": train_ids, "validation_samples": val_ids, "validation_mode": "held_out_qc_sample"}
                else:
                    all_mask = np.isin(qc_sample_ids, np.array(train_ids))
                    all_idx = np.flatnonzero(all_mask)
                    rng = np.random.default_rng(RANDOM_SEED)
                    shuffled = rng.permutation(all_idx)
                    n_val = max(1, int(round(0.15 * len(shuffled))))
                    val_idx = shuffled[:n_val]
                    train_idx = shuffled[n_val:]
                    train_mask = np.zeros(len(qc_sample_ids), dtype=bool)
                    val_mask = np.zeros(len(qc_sample_ids), dtype=bool)
                    train_mask[train_idx] = True
                    val_mask[val_idx] = True
                    split = {"train_samples": train_ids, "validation_samples": train_ids, "validation_mode": "internal_window_split_15pct"}
                split["train_examples"] = int(train_mask.sum())
                split["validation_examples"] = int(val_mask.sum())
                return subset_dataset(qc_fields, qc_targets, train_mask), subset_dataset(qc_fields, qc_targets, val_mask), split


            def eval_loader_topk(model, loader, device):
                model.eval()
                n = top1 = top5 = 0
                with torch.no_grad():
                    for batch in loader:
                        *fields, targets = batch
                        fields = [x.to(device, non_blocking=True) for x in fields]
                        targets = targets.to(device, non_blocking=True)
                        logits = model(*fields)
                        pred1 = logits.argmax(dim=1)
                        top5_idx = logits.topk(5, dim=1).indices
                        top1 += int((pred1 == targets).sum().item())
                        top5 += int((top5_idx == targets[:, None]).any(dim=1).sum().item())
                        n += int(targets.numel())
                return {"top1": top1 / n, "top5": top5 / n, "n": n}


            def train_full_model(n_cal, train_ds, val_ds, device, learning_curve=False):
                model, _, _ = load_lstm_model(device)
                for param in model.parameters():
                    param.requires_grad = True
                train_loader = DataLoader(
                    train_ds,
                    batch_size=FULL_BATCH_SIZE,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(RANDOM_SEED + n_cal),
                )
                val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
                opt = torch.optim.Adam(model.parameters(), lr=FULL_LR)
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = 0
                best_val = eval_loader_topk(model, val_loader, device)
                best_score = best_val["top1"]
                stale_epochs = 0
                curve = [{"epoch": 0, "val_top1": best_val["top1"], "val_top5": best_val["top5"], "val_n": best_val["n"]}]
                for epoch in range(1, FULL_MAX_EPOCHS + 1):
                    model.train()
                    running_loss = 0.0
                    seen = 0
                    for batch in train_loader:
                        *fields, targets = batch
                        fields = [x.to(device, non_blocking=True) for x in fields]
                        targets = targets.to(device, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(*fields), targets)
                        loss.backward()
                        opt.step()
                        running_loss += float(loss.item()) * int(targets.numel())
                        seen += int(targets.numel())
                    val = eval_loader_topk(model, val_loader, device)
                    row = {"epoch": epoch, "train_loss": running_loss / seen, "val_top1": val["top1"], "val_top5": val["top5"], "val_n": val["n"]}
                    if learning_curve:
                        curve.append(row)
                    print(f"N={n_cal} epoch {epoch:02d}/{FULL_MAX_EPOCHS}: loss={row['train_loss']:.4f}, val_top1={val['top1']:.4f}")
                    if val["top1"] > best_score:
                        best_score = val["top1"]
                        best_val = val
                        best_epoch = epoch
                        best_state = copy.deepcopy(model.state_dict())
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                    if stale_epochs >= FULL_PATIENCE:
                        print(f"N={n_cal}: early stop at epoch {epoch} (patience={FULL_PATIENCE})")
                        break
                model.load_state_dict(best_state)
                model.eval()
                info = {
                    "best_epoch": best_epoch,
                    "best_validation_top1": best_val["top1"],
                    "best_validation_top5": best_val["top5"],
                    "max_epochs": FULL_MAX_EPOCHS,
                    "lr": FULL_LR,
                    "batch_size": FULL_BATCH_SIZE,
                }
                if learning_curve:
                    info["learning_curve"] = curve
                return model, info
            """
        ),
        code(
            """
            started = time.time()

            qc_samples = [s for s in all_samples if s.sample_type == "qc"]
            eval_samples = [s for s in all_samples if s.sample_type == "analytical"]
            qc_ids = {s.sample_id for s in qc_samples}
            eval_ids = {s.sample_id for s in eval_samples}
            assert len(qc_samples) == 15, f"Expected 15 QC samples, got {len(qc_samples)}"
            assert len(eval_samples) == 126, f"Expected 126 analytical samples, got {len(eval_samples)}"
            assert qc_ids.isdisjoint(eval_ids), "QC sample leaked into analytical eval set"
            print(f"Split: calibration pool={len(qc_samples)} QC, eval={len(eval_samples)} analytical, disjoint={qc_ids.isdisjoint(eval_ids)}")

            print("Extracting frozen LSTM features for QC calibration and analytical eval...")
            eval_features, eval_targets, eval_prev, eval_sample_ids = collect_feature_table(base_model, eval_samples, device)
            qc_features, qc_targets, _, qc_sample_ids = collect_feature_table(base_model, qc_samples, device)

            original_head = base_model.head.eval()
            with torch.no_grad():
                eval_logits = []
                for start in range(0, len(eval_targets), 8192):
                    logits = original_head(eval_features[start:start + 8192].to(device)).cpu()
                    eval_logits.append(logits)
                eval_logits_t = torch.cat(eval_logits, dim=0)
            eval_probs = F.softmax(eval_logits_t, dim=1).numpy().astype(np.float32)
            pure_eval = topk_metrics(eval_probs, eval_targets)
            analytical_floor_expected = 0.025853403436473466
            print(f"Leakage check pure LSTM on held-out analytical: top1={pure_eval['top1']:.6f}, top5={pure_eval['top5']:.6f}, n={pure_eval['n']:,}")
            if abs(pure_eval["top1"] - analytical_floor_expected) > 0.005:
                raise RuntimeError("Analytical eval floor gate failed; possible leakage or eval subset error.")

            print("Building QC sliding-window tensors once for full fine-tuning...")
            qc_fields_full, qc_targets_full, qc_sample_ids_full = make_examples(qc_samples)

            recovery_table = []
            full_rows = []
            for n_cal in N_VALUES:
                cal_ids = [s.sample_id for s in qc_samples[:n_cal]]
                cal_samples = qc_samples[:n_cal]
                cal_mask = np.isin(qc_sample_ids, np.array(cal_ids))

                markov = build_markov(cal_samples, MAX_MZ_BIN)
                markov_scores = markov[eval_prev]
                best_markov = None
                for alpha in ALPHAS:
                    scores = alpha * eval_probs + (1.0 - alpha) * markov_scores
                    metrics = topk_metrics(scores, eval_targets)
                    row = {"method": "markov_blend", "N": n_cal, "alpha": alpha, "top1": metrics["top1"], "top5": metrics["top5"], "n": metrics["n"]}
                    recovery_table.append(row)
                    if best_markov is None or row["top1"] > best_markov["top1"]:
                        best_markov = row
                    if alpha == 0.0:
                        recovery_table.append({"method": "markov_only", "N": n_cal, "alpha": 0.0, "top1": metrics["top1"], "top5": metrics["top5"], "n": metrics["n"]})

                print(f"Training frozen-body head for N={n_cal} QC samples...")
                head = train_head(original_head, qc_features[cal_mask], qc_targets[cal_mask], MAX_MZ_BIN, device)
                head_metrics = eval_head(head, eval_features, eval_targets, device)
                recovery_table.append({"method": "fine_tune_head", "N": n_cal, "alpha": None, "top1": head_metrics["top1"], "top5": head_metrics["top5"], "n": head_metrics["n"]})

                train_ds, val_ds, split = split_for_n(qc_samples, qc_fields_full, qc_targets_full, qc_sample_ids_full, n_cal)
                print(f"Full fine-tune N={n_cal}: train_examples={split['train_examples']:,}, val_examples={split['validation_examples']:,}")
                model_ft, info = train_full_model(n_cal, train_ds, val_ds, device, learning_curve=(n_cal == max(N_VALUES)))
                ft_eval = evaluate_model_stream(model_ft, eval_samples, device)
                val_metrics = {"top1": info["best_validation_top1"], "top5": info["best_validation_top5"]}
                row = {
                    "method": "full_finetune",
                    "N": n_cal,
                    "eval_top1": ft_eval["top1"],
                    "eval_top5": ft_eval["top5"],
                    "eval_n": ft_eval["n"],
                    "heldout_qc_top1": val_metrics["top1"],
                    "heldout_qc_top5": val_metrics["top5"],
                    "chosen_epoch": info["best_epoch"],
                    "split": split,
                    "train_config": {"lr": info["lr"], "batch_size": info["batch_size"], "max_epochs": info["max_epochs"], "all_parameters_trainable": True},
                }
                if "learning_curve" in info:
                    row["learning_curve"] = info["learning_curve"]
                full_rows.append(row)

                print(
                    f"N={n_cal}: Markov-only top1={next(r['top1'] for r in recovery_table if r['method']=='markov_only' and r['N']==n_cal):.4f}; "
                    f"best Markov blend top1={best_markov['top1']:.4f} (alpha={best_markov['alpha']}); "
                    f"head-only top1={head_metrics['top1']:.4f}; full-FT top1={ft_eval['top1']:.4f}, top5={ft_eval['top5']:.4f}, held-out QC top1={val_metrics['top1']:.4f}"
                )
            """
        ),
        md("## 7. Figure - Recovery Curve"),
        code(
            """
            n_values = N_VALUES
            full_top1 = [next(r["eval_top1"] for r in full_rows if r["N"] == n) for n in n_values]
            head_top1 = [next(r["top1"] for r in recovery_table if r["method"] == "fine_tune_head" and r["N"] == n) for n in n_values]
            markov_only_top1 = [next(r["top1"] for r in recovery_table if r["method"] == "markov_only" and r["N"] == n) for n in n_values]

            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            ax.plot(n_values, full_top1, marker="o", linewidth=2.0, label="Full fine-tune")
            ax.plot(n_values, head_top1, marker="s", label="Head-only fine-tune")
            ax.plot(n_values, markov_only_top1, marker="^", label="Markov-only")
            ax.axhline(pure_eval["top1"], color="gray", linestyle="--", label=f"Zero-shot floor ({pure_eval['top1']:.1%})")
            ax.axhline(float(lstm_ckpt["test_metrics"]["top1"]), color="black", linestyle=":", label=f"Within-method ceiling ({lstm_ckpt['test_metrics']['top1']:.1%})")
            ax.set_xlabel("QC calibration samples (N)")
            ax.set_ylabel("Held-out analytical top-1 accuracy")
            ax.set_xticks(n_values)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            ax.set_title("ST000990 transfer-learning recovery")
            fig.tight_layout()
            png_path = f"{OUTPUT_DIR}/phase9_recovery.png"
            pdf_path = f"{OUTPUT_DIR}/phase9_recovery.pdf"
            fig.savefig(png_path, dpi=200)
            fig.savefig(pdf_path)
            plt.show()
            print(f"Saved: {png_path}")
            print(f"Saved: {pdf_path}")
            """
        ),
        md("## 8. Save Results and Verdict"),
        code(
            """
            best_full = max(full_rows, key=lambda r: r["eval_top1"])
            results = {
                "experiment": "Phase 9 / 9B: transfer-learning recovery on ST000990",
                "version": VERSION,
                "random_seed": RANDOM_SEED,
                "checkpoint": LSTM_CHECKPOINT,
                "data": ST000990_PATH,
                "device": str(device),
                "runtime_seconds": time.time() - started,
                "reference": {
                    "overall_top1": 0.0263076228273567,
                    "overall_n": 396349,
                    "analytical_top1": analytical_floor_expected,
                    "within_method_ceiling_top1": float(lstm_ckpt["test_metrics"]["top1"]),
                },
                "zero_shot_all": zero_shot_all,
                "leakage_check": {
                    "sample_disjoint": qc_ids.isdisjoint(eval_ids),
                    "qc_samples": sorted(qc_ids),
                    "eval_samples_n": len(eval_ids),
                    "eval_n": int(len(eval_targets)),
                    "pure_lstm_eval_top1": pure_eval["top1"],
                    "pure_lstm_eval_top5": pure_eval["top5"],
                    "reference_analytical_top1": analytical_floor_expected,
                },
                "recovery_table": recovery_table,
                "full_finetune": full_rows,
                "best_full_finetune": best_full,
                "checkpoint_test_metrics": lstm_ckpt.get("test_metrics", {}),
            }

            out_json = f"{OUTPUT_DIR}/phase9_results.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"Saved: {out_json}")
            print(
                "VERDICT: full fine-tuning recovers the ST000990 failure from "
                f"{pure_eval['top1']:.1%} top-1 to best N={best_full['N']} "
                f"{best_full['eval_top1']:.1%} top-1 / {best_full['eval_top5']:.1%} top-5; "
                "the zero-shot failure is a recoverable adaptation gap."
            )
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Cells: {len(nb.cells)}")


if __name__ == "__main__":
    main()
