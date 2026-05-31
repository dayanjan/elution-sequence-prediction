"""
Phase 9: cross-polarity transfer-learning recovery on ST000990.

This mirrors the Phase 8 ST000990 notebook tokenization/evaluation path, then
tests whether small positive-ESI QC calibration sets recover next-m/z-bin
accuracy on held-out analytical samples.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_PATH = Path(__file__).resolve()
POC3 = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[5]
SRC_DIR = POC3 / "src"
sys.path.insert(0, str(SRC_DIR))

from config import CONTEXT_LENGTH, DROPOUT, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS  # noqa: E402
from models import LSTMModel  # noqa: E402


RANDOM_SEED = 42
MZ_BIN_WIDTH = 10
MASS_DEFECT_BINS = 20
RT_GAP_BINS = [-0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 15.0, 9999]
RT_GAP_LABELS = ["co-elute", "0.1-0.5s", "0.5-1s", "1-2s", "2-5s", "5-15s", ">15s"]
INTENSITY_RANK_BINS = [-0.001, 0.01, 0.05, 0.20, 0.50, 1.001]
INTENSITY_RANK_LABELS = ["top1%", "top5%", "top20%", "top50%", "low"]
POLARITY_MAP = {"pos": 0, "neg": 1, "unk": 2}
RT_GAP_MAP = {label: i for i, label in enumerate(RT_GAP_LABELS)}
INTENSITY_MAP = {label: i for i, label in enumerate(INTENSITY_RANK_LABELS)}

CHECKPOINT = REPO_ROOT / ".claude" / "scratch" / "phase9" / "lstm_best.pt"
REFERENCE_JSON = REPO_ROOT / ".claude" / "scratch" / "phase9" / "st000990_results_reference.json"
ST000990_PATH = POC3 / "data" / "external" / "ST000990" / "msdial_merged" / "merged_features.parquet"
OUTPUT_DIR = POC3 / "outputs" / "phase9_transfer"

N_VALUES = [1, 2, 5, 10, 15]
ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
BATCH_SIZE = 2048


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
    def n_predictions(self) -> int:
        return max(0, len(self.mz) - CONTEXT_LENGTH)


def set_seeds() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def classify_sample(name: str) -> str:
    if "_NIST" in name:
        return "nist"
    if "_QC" in name:
        return "qc"
    if "_Rep" in name:
        return "replicate"
    return "analytical"


def tokenize_sample(feat_df: pd.DataFrame, sample_col: str, max_mz_bin: int) -> SampleTokens | None:
    """Notebook-08-compatible ST000990 tokenization for one sample."""
    mask = feat_df[sample_col].notna() & (feat_df[sample_col] > 0)
    sf = feat_df.loc[mask, ["mz", "rt_min", "adduct_ion", sample_col]].copy()
    sf = sf.rename(columns={sample_col: "intensity"})
    if len(sf) < CONTEXT_LENGTH + 1:
        return None

    sf = sf.sort_values(["rt_min", "mz"]).reset_index(drop=True)
    sf["mz_bin"] = (sf["mz"] // MZ_BIN_WIDTH).astype(int).clip(upper=max_mz_bin - 1)

    mass_defect = sf["mz"] - np.floor(sf["mz"])
    sf["md_bin"] = (mass_defect * MASS_DEFECT_BINS).astype(int).clip(upper=MASS_DEFECT_BINS - 1)

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


def load_model(device: torch.device) -> tuple[LSTMModel, dict, int]:
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
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


def load_tokens(max_mz_bin: int) -> dict[str, SampleTokens]:
    feat = pd.read_parquet(ST000990_PATH)
    sample_cols = [c for c in feat.columns if c.startswith("GLA_TT6_Lipids_")]
    feat = feat.rename(columns={"rt_sec": "rt_min"})

    tokens = {}
    for col in sample_cols:
        sample = tokenize_sample(feat, col, max_mz_bin)
        if sample is not None:
            tokens[col] = sample

    all_bins = np.concatenate([s.mz for s in tokens.values()])
    oov = int((all_bins >= max_mz_bin).sum())
    print(f"Tokenized {len(tokens)} samples; OOV m/z bins after clipping: {oov}/{len(all_bins)}")
    return tokens


def iter_context_batches(samples: list[SampleTokens], batch_size: int = BATCH_SIZE):
    field_chunks = []
    target_chunks = []
    prev_chunks = []
    sample_chunks = []
    buffered = 0

    def flush():
        nonlocal field_chunks, target_chunks, prev_chunks, sample_chunks, buffered
        if buffered == 0:
            return None
        fields = [np.concatenate([chunk[i] for chunk in field_chunks]) for i in range(5)]
        targets = np.concatenate(target_chunks)
        prev = np.concatenate(prev_chunks)
        sample_ids = np.concatenate(sample_chunks)
        field_chunks, target_chunks, prev_chunks, sample_chunks, buffered = [], [], [], [], 0
        return fields, targets, prev, sample_ids

    for sample in samples:
        n_pred = sample.n_predictions
        for start_pos in range(CONTEXT_LENGTH, CONTEXT_LENGTH + n_pred, batch_size):
            end_pos = min(CONTEXT_LENGTH + n_pred, start_pos + batch_size)
            positions = np.arange(start_pos, end_pos)
            starts = positions - CONTEXT_LENGTH
            offsets = np.arange(CONTEXT_LENGTH)
            idx = starts[:, None] + offsets[None, :]
            field_chunks.append(
                (
                    sample.mz[idx],
                    sample.md[idx],
                    sample.gap[idx],
                    sample.pol[idx],
                    sample.inten[idx],
                )
            )
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


def logits_from_fields(model: LSTMModel, fields: list[np.ndarray], device: torch.device) -> torch.Tensor:
    tensors = [torch.as_tensor(x, dtype=torch.long, device=device) for x in fields]
    with torch.no_grad():
        return model(*tensors)


def extract_features(model: LSTMModel, fields: list[np.ndarray], device: torch.device) -> torch.Tensor:
    tensors = [torch.as_tensor(x, dtype=torch.long, device=device) for x in fields]
    with torch.no_grad():
        x = model.embedding(*tensors)
        out, _ = model.lstm(x)
        return out[:, -1, :]


def topk_metrics(scores: np.ndarray, targets: np.ndarray, ks: tuple[int, ...] = (1, 5)) -> dict[str, float | int]:
    max_k = max(ks)
    top_idx = np.argpartition(scores, -max_k, axis=1)[:, -max_k:]
    metrics: dict[str, float | int] = {"n": int(len(targets))}
    for k in ks:
        if k == max_k:
            cand = top_idx
        else:
            cand_scores = np.take_along_axis(scores, top_idx, axis=1)
            local = np.argpartition(cand_scores, -k, axis=1)[:, -k:]
            cand = np.take_along_axis(top_idx, local, axis=1)
        metrics[f"top{k}"] = float((cand == targets[:, None]).any(axis=1).mean())
    return metrics


def evaluate_model_stream(model: LSTMModel, samples: list[SampleTokens], device: torch.device) -> dict[str, float | int]:
    n = 0
    top1 = 0
    top5 = 0
    for fields, targets, _, _ in iter_context_batches(samples):
        logits = logits_from_fields(model, fields, device).cpu().numpy()
        pred1 = logits.argmax(axis=1)
        top5_idx = np.argpartition(logits, -5, axis=1)[:, -5:]
        top1 += int((pred1 == targets).sum())
        top5 += int((top5_idx == targets[:, None]).any(axis=1).sum())
        n += len(targets)
    return {"top1": top1 / n, "top5": top5 / n, "n": n}


def collect_feature_table(
    model: LSTMModel, samples: list[SampleTokens], device: torch.device
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    prev = []
    sample_ids = []
    for fields, batch_targets, batch_prev, batch_sample_ids in iter_context_batches(samples):
        features.append(extract_features(model, fields, device).cpu())
        targets.append(batch_targets)
        prev.append(batch_prev)
        sample_ids.append(batch_sample_ids)
    return (
        torch.cat(features, dim=0),
        np.concatenate(targets).astype(np.int64),
        np.concatenate(prev).astype(np.int64),
        np.concatenate(sample_ids),
    )


def build_markov(samples: list[SampleTokens], num_classes: int) -> np.ndarray:
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


def train_head(
    original_head: nn.Linear,
    train_features: torch.Tensor,
    train_targets: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
) -> nn.Linear:
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


def eval_head(head: nn.Linear, features: torch.Tensor, targets: np.ndarray, device: torch.device) -> dict[str, float | int]:
    n = 0
    top1 = 0
    top5 = 0
    with torch.no_grad():
        for start in range(0, len(targets), 8192):
            xb = features[start : start + 8192].to(device)
            logits = head(xb).cpu().numpy()
            batch_targets = targets[start : start + 8192]
            pred1 = logits.argmax(axis=1)
            top5_idx = np.argpartition(logits, -5, axis=1)[:, -5:]
            top1 += int((pred1 == batch_targets).sum())
            top5 += int((top5_idx == batch_targets[:, None]).any(axis=1).sum())
            n += len(batch_targets)
    return {"top1": top1 / n, "top5": top5 / n, "n": n}


def plot_recovery(results: dict, out_dir: Path) -> None:
    markov_best = []
    markov_only = []
    finetune = []
    for n_cal in N_VALUES:
        rows = [r for r in results["recovery_table"] if r["N"] == n_cal and r["method"] == "markov_blend"]
        best = max(rows, key=lambda r: r["top1"])
        markov_best.append(best["top1"])
        markov_only.append(next(r["top1"] for r in rows if r["alpha"] == 0.0))
        finetune.append(
            next(r["top1"] for r in results["recovery_table"] if r["N"] == n_cal and r["method"] == "fine_tune_head")
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(N_VALUES, markov_best, marker="o", label="Markov blend (best alpha)")
    ax.plot(N_VALUES, finetune, marker="s", label="Fine-tune head")
    ax.plot(N_VALUES, markov_only, marker="^", label="Markov only")
    ax.axhline(results["leakage_check"]["pure_lstm_eval_top1"], color="gray", linestyle="--", label="Zero-shot floor")
    ax.axhline(0.984, color="black", linestyle=":", label="Within-method ceiling")
    ax.set_xlabel("QC calibration samples (N)")
    ax.set_ylabel("Held-out analytical top-1 accuracy")
    ax.set_xticks(N_VALUES)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("ST000990 few-shot transfer recovery")
    fig.tight_layout()
    fig.savefig(out_dir / "phase9_recovery.png", dpi=200)
    fig.savefig(out_dir / "phase9_recovery.pdf")
    plt.close(fig)


def write_summary(results: dict, out_dir: Path) -> None:
    table = results["recovery_table"]
    best = max(table, key=lambda r: r["top1"])

    def best_at_n(n_cal: int) -> dict:
        return max([r for r in table if r["N"] == n_cal], key=lambda r: r["top1"])

    best5 = best_at_n(5)
    best15 = best_at_n(15)
    lines = [
        "# Phase 9 transfer recovery summary",
        "",
        f"- Gate: reproduced zero-shot ST000990 top-1 = {results['zero_shot_all']['top1']:.4f} "
        f"(n = {results['zero_shot_all']['n']:,}); reference = {results['reference']['overall_top1']:.4f}.",
        f"- Leakage check: QC calibration samples and analytical evaluation samples are disjoint = "
        f"{results['leakage_check']['sample_disjoint']}; pure LSTM on analytical top-1 = "
        f"{results['leakage_check']['pure_lstm_eval_top1']:.4f} "
        f"(n = {results['leakage_check']['eval_n']:,}).",
        f"- Best overall recovery: {best['method']} at N={best['N']}"
        f"{'' if best.get('alpha') is None else f', alpha={best['alpha']}'}: "
        f"top-1 = {best['top1']:.4f}, top-5 = {best['top5']:.4f}.",
        f"- Best at N=5: {best5['method']}"
        f"{'' if best5.get('alpha') is None else f' alpha={best5['alpha']}'}: "
        f"top-1 = {best5['top1']:.4f}, top-5 = {best5['top5']:.4f}.",
        f"- Best at N=15: {best15['method']}"
        f"{'' if best15.get('alpha') is None else f' alpha={best15['alpha']}'}: "
        f"top-1 = {best15['top1']:.4f}, top-5 = {best15['top5']:.4f}.",
        "- Calibration pool: first N QC samples (QC01..QC15 order from the parquet columns).",
        "- Evaluation set: 126 analytical SO samples only.",
        "- Markov rows with unseen previous m/z bins use the calibration global next-bin prior.",
        "- Fine-tuning freezes embeddings and LSTM and trains only the final Linear head for 40 CPU epochs.",
        "- The within-method ceiling reference is the checkpoint test top-1, 0.984.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_seeds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, ckpt, num_classes = load_model(device)
    print(f"Loaded checkpoint with {num_classes} m/z classes")
    tokens = load_tokens(num_classes)

    with REFERENCE_JSON.open() as f:
        reference = json.load(f)
    ref_top1 = float(reference["overall_metrics"]["top1"])
    ref_n = int(reference["overall_metrics"]["n"])
    ref_eval_top1 = float(reference["per_type_metrics"]["analytical"]["top1"])

    all_samples = list(tokens.values())
    zero_shot_all = evaluate_model_stream(model, all_samples, device)
    print(f"STEP 1 zero-shot reproduction: top1={zero_shot_all['top1']:.4f}, n={zero_shot_all['n']:,}")
    if abs(zero_shot_all["top1"] - ref_top1) > 0.005 or abs(zero_shot_all["n"] - ref_n) > 1000:
        raise RuntimeError(
            "Step 1 gate failed: reproduced metrics diverge from notebook-08 reference. "
            f"got top1={zero_shot_all['top1']:.6f}, n={zero_shot_all['n']}; "
            f"expected top1={ref_top1:.6f}, n={ref_n}."
        )

    qc_samples = [s for s in all_samples if s.sample_type == "qc"]
    eval_samples = [s for s in all_samples if s.sample_type == "analytical"]
    qc_ids = {s.sample_id for s in qc_samples}
    eval_ids = {s.sample_id for s in eval_samples}
    assert len(qc_samples) == 15, f"Expected 15 QC samples, got {len(qc_samples)}"
    assert len(eval_samples) == 126, f"Expected 126 analytical samples, got {len(eval_samples)}"
    assert qc_ids.isdisjoint(eval_ids), "QC sample leaked into analytical eval set"

    print("Extracting frozen LSTM features for QC calibration and analytical eval...")
    eval_features, eval_targets, eval_prev, eval_sample_ids = collect_feature_table(model, eval_samples, device)
    qc_features, qc_targets, _, qc_sample_ids = collect_feature_table(model, qc_samples, device)

    original_head = model.head.eval()
    with torch.no_grad():
        eval_logits = []
        for start in range(0, len(eval_targets), 8192):
            logits = original_head(eval_features[start : start + 8192].to(device)).cpu()
            eval_logits.append(logits)
        eval_logits_t = torch.cat(eval_logits, dim=0)
    eval_probs = F.softmax(eval_logits_t, dim=1).numpy().astype(np.float32)
    pure_eval = topk_metrics(eval_probs, eval_targets)
    print(
        f"Leakage check pure LSTM on held-out analytical: "
        f"top1={pure_eval['top1']:.4f}, n={pure_eval['n']:,}"
    )
    if abs(pure_eval["top1"] - ref_eval_top1) > 0.005:
        raise RuntimeError(
            "Analytical eval floor gate failed; possible leakage or eval subset error. "
            f"got top1={pure_eval['top1']:.6f}, expected about {ref_eval_top1:.6f}."
        )

    recovery_table = []
    for n_cal in N_VALUES:
        cal_ids = [s.sample_id for s in qc_samples[:n_cal]]
        cal_samples = qc_samples[:n_cal]
        cal_mask = np.isin(qc_sample_ids, np.array(cal_ids))

        markov = build_markov(cal_samples, num_classes)
        markov_scores = markov[eval_prev]
        for alpha in ALPHAS:
            scores = alpha * eval_probs + (1.0 - alpha) * markov_scores
            metrics = topk_metrics(scores, eval_targets)
            recovery_table.append(
                {
                    "method": "markov_blend",
                    "N": n_cal,
                    "alpha": alpha,
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "n": metrics["n"],
                }
            )
            if alpha == 0.0:
                recovery_table.append(
                    {
                        "method": "markov_only",
                        "N": n_cal,
                        "alpha": 0.0,
                        "top1": metrics["top1"],
                        "top5": metrics["top5"],
                        "n": metrics["n"],
                    }
                )

        print(f"Training frozen-body head for N={n_cal} QC samples...")
        head = train_head(original_head, qc_features[cal_mask], qc_targets[cal_mask], num_classes, device)
        ft_metrics = eval_head(head, eval_features, eval_targets, device)
        recovery_table.append(
            {
                "method": "fine_tune_head",
                "N": n_cal,
                "alpha": None,
                "top1": ft_metrics["top1"],
                "top5": ft_metrics["top5"],
                "n": ft_metrics["n"],
            }
        )
        best_markov = max(
            [r for r in recovery_table if r["method"] == "markov_blend" and r["N"] == n_cal],
            key=lambda r: r["top1"],
        )
        print(
            f"N={n_cal}: best Markov blend top1={best_markov['top1']:.4f} "
            f"(alpha={best_markov['alpha']}), fine-tune top1={ft_metrics['top1']:.4f}"
        )

    results = {
        "experiment": "Phase 9: cross-polarity transfer-learning recovery on ST000990",
        "random_seed": RANDOM_SEED,
        "checkpoint": str(CHECKPOINT),
        "data": str(ST000990_PATH),
        "reference": {
            "overall_top1": ref_top1,
            "overall_n": ref_n,
            "analytical_top1": ref_eval_top1,
            "within_method_ceiling_top1": float(ckpt["test_metrics"]["top1"]),
        },
        "zero_shot_all": zero_shot_all,
        "leakage_check": {
            "sample_disjoint": qc_ids.isdisjoint(eval_ids),
            "qc_samples": sorted(qc_ids),
            "eval_samples_n": len(eval_ids),
            "eval_n": int(len(eval_targets)),
            "pure_lstm_eval_top1": pure_eval["top1"],
            "pure_lstm_eval_top5": pure_eval["top5"],
            "reference_analytical_top1": ref_eval_top1,
        },
        "recovery_table": recovery_table,
    }

    with (OUTPUT_DIR / "phase9_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_recovery(results, OUTPUT_DIR)
    write_summary(results, OUTPUT_DIR)
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
