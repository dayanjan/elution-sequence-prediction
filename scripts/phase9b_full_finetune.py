"""
Phase 9B: full-model fine-tune recovery on ST000990.

This extends the gate-validated Phase 9 pipeline without modifying it:
tokenization, sample typing, checkpoint loading, batching, and analytical
evaluation are imported from phase9_transfer.py.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import phase9_transfer as p9


LR = 3e-4
TRAIN_BATCH_SIZE = 32
MAX_EPOCHS = int(os.environ.get("PHASE9B_MAX_EPOCHS", "40"))
N_VALUES = [1, 2, 5, 10, 15]


def make_examples(samples: list[p9.SampleTokens]) -> tuple[list[torch.Tensor], torch.Tensor, np.ndarray]:
    fields = [[] for _ in range(5)]
    targets = []
    sample_ids = []
    offsets = np.arange(p9.CONTEXT_LENGTH)

    for sample in samples:
        positions = np.arange(p9.CONTEXT_LENGTH, len(sample.mz))
        starts = positions - p9.CONTEXT_LENGTH
        idx = starts[:, None] + offsets[None, :]
        arrays = (sample.mz, sample.md, sample.gap, sample.pol, sample.inten)
        for i, arr in enumerate(arrays):
            fields[i].append(arr[idx])
        targets.append(sample.mz[positions])
        sample_ids.append(np.repeat(sample.sample_id, len(positions)))

    field_tensors = [torch.as_tensor(np.concatenate(chunks), dtype=torch.long) for chunks in fields]
    target_tensor = torch.as_tensor(np.concatenate(targets), dtype=torch.long)
    sample_id_arr = np.concatenate(sample_ids)
    return field_tensors, target_tensor, sample_id_arr


def subset_dataset(fields: list[torch.Tensor], targets: torch.Tensor, mask: np.ndarray) -> TensorDataset:
    idx = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
    return TensorDataset(*(field.index_select(0, idx) for field in fields), targets.index_select(0, idx))


def split_for_n(
    qc_samples: list[p9.SampleTokens],
    qc_fields: list[torch.Tensor],
    qc_targets: torch.Tensor,
    qc_sample_ids: np.ndarray,
    n_cal: int,
) -> tuple[TensorDataset, TensorDataset, dict]:
    train_ids = [sample.sample_id for sample in qc_samples[:n_cal]]

    if n_cal < len(qc_samples):
        val_ids = [qc_samples[n_cal].sample_id]
        train_mask = np.isin(qc_sample_ids, np.array(train_ids))
        val_mask = np.isin(qc_sample_ids, np.array(val_ids))
        split = {
            "train_samples": train_ids,
            "validation_samples": val_ids,
            "validation_mode": "held_out_qc_sample",
        }
    else:
        all_mask = np.isin(qc_sample_ids, np.array(train_ids))
        all_idx = np.flatnonzero(all_mask)
        rng = np.random.default_rng(p9.RANDOM_SEED)
        shuffled = rng.permutation(all_idx)
        n_val = max(1, int(round(0.15 * len(shuffled))))
        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]
        train_mask = np.zeros(len(qc_sample_ids), dtype=bool)
        val_mask = np.zeros(len(qc_sample_ids), dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        split = {
            "train_samples": train_ids,
            "validation_samples": train_ids,
            "validation_mode": "internal_window_split_15pct",
        }

    split["train_examples"] = int(train_mask.sum())
    split["validation_examples"] = int(val_mask.sum())
    return subset_dataset(qc_fields, qc_targets, train_mask), subset_dataset(qc_fields, qc_targets, val_mask), split


def eval_loader_topk(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    n = 0
    top1 = 0
    top5 = 0
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


def train_full_model(
    n_cal: int,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    device: torch.device,
    learning_curve: bool,
) -> tuple[torch.nn.Module, dict]:
    model, _, _ = p9.load_model(device)
    for param in model.parameters():
        param.requires_grad = True

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(p9.RANDOM_SEED + n_cal),
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val = eval_loader_topk(model, val_loader, device)
    curve = [{"epoch": 0, "val_top1": best_val["top1"], "val_top5": best_val["top5"], "val_n": best_val["n"]}]

    for epoch in range(1, MAX_EPOCHS + 1):
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
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "val_top1": val["top1"],
            "val_top5": val["top5"],
            "val_n": val["n"],
        }
        if learning_curve:
            curve.append(row)
        print(
            f"N={n_cal} epoch {epoch:02d}/{MAX_EPOCHS}: "
            f"loss={row['train_loss']:.4f}, val_top1={val['top1']:.4f}"
        )

        if val["top1"] > best_val["top1"]:
            best_val = val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    info = {
        "best_epoch": best_epoch,
        "best_validation_top1": best_val["top1"],
        "best_validation_top5": best_val["top5"],
        "max_epochs": MAX_EPOCHS,
        "lr": LR,
        "batch_size": TRAIN_BATCH_SIZE,
    }
    if learning_curve:
        info["learning_curve"] = curve
    return model, info


def phase9_context() -> dict:
    with p9.OUTPUT_DIR.joinpath("phase9_results.json").open(encoding="utf-8") as f:
        phase9 = json.load(f)
    rows = phase9["recovery_table"]
    return {
        "phase9_floor_top1": phase9["leakage_check"]["pure_lstm_eval_top1"],
        "phase9_head_only_best_top1": max(r["top1"] for r in rows if r["method"] == "fine_tune_head"),
        "phase9_markov_only_best_top1": max(r["top1"] for r in rows if r["method"] == "markov_only"),
        "within_method_ceiling_top1": phase9["reference"]["within_method_ceiling_top1"],
        "phase9_recovery_table": rows,
    }


def plot_phase9b(results: dict) -> None:
    p9_rows = results["context"]["phase9_recovery_table"]
    full_rows = results["full_finetune"]
    n_values = [row["N"] for row in full_rows]
    full = [row["eval_top1"] for row in full_rows]
    head = [next(r["top1"] for r in p9_rows if r["method"] == "fine_tune_head" and r["N"] == n) for n in n_values]
    markov = [next(r["top1"] for r in p9_rows if r["method"] == "markov_only" and r["N"] == n) for n in n_values]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(n_values, full, marker="o", linewidth=2.0, label="Full fine-tune")
    ax.plot(n_values, head, marker="s", label="Head-only fine-tune (Phase 9)")
    ax.plot(n_values, markov, marker="^", label="Markov-only (Phase 9)")
    ax.axhline(results["context"]["phase9_floor_top1"], color="gray", linestyle="--", label="Zero-shot floor")
    ax.axhline(results["context"]["within_method_ceiling_top1"], color="black", linestyle=":", label="Within-method ceiling")
    ax.set_xlabel("QC calibration samples (N)")
    ax.set_ylabel("Held-out analytical top-1 accuracy")
    ax.set_xticks(n_values)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("ST000990 full fine-tune recovery")
    fig.tight_layout()
    fig.savefig(p9.OUTPUT_DIR / "phase9b_recovery.png", dpi=200)
    fig.savefig(p9.OUTPUT_DIR / "phase9b_recovery.pdf")
    plt.close(fig)


def write_summary(results: dict) -> None:
    best = max(results["full_finetune"], key=lambda r: r["eval_top1"])
    context = results["context"]
    verdict = (
        "Full fine-tuning shows a recoverable adaptation gap versus cheap methods, but 15 QC samples "
        "do not recover the within-method ceiling."
        if best["eval_top1"] >= 0.3
        else "Full fine-tuning remains far below the within-method ceiling, so this behaves like a hard wall "
        "for 15 QC samples."
    )
    lines = [
        "# Phase 9B full fine-tune summary",
        "",
        f"- Best full fine-tune: N={best['N']}, top-1={best['eval_top1']:.4f}, top-5={best['eval_top5']:.4f}, best epoch={best['chosen_epoch']}.",
        f"- Eval guardrail: same 126 analytical samples, n={results['self_checks']['eval_n']:,}.",
        f"- Leakage guardrail: QC train/calibration and analytical eval disjoint = {results['self_checks']['sample_disjoint']}.",
        f"- Sanity guardrail: 0-epoch eval top-1={results['self_checks']['zero_epoch_eval_top1']:.4f}; Phase 9 floor={context['phase9_floor_top1']:.4f}.",
        f"- Phase 9 cheap-method references: Markov-only best={context['phase9_markov_only_best_top1']:.4f}; head-only best={context['phase9_head_only_best_top1']:.4f}.",
        f"- Within-method ceiling reference: {context['within_method_ceiling_top1']:.4f}.",
        f"- Verdict: {verdict}",
        f"- Training setup: full model unfrozen, Adam lr={LR}, batch={TRAIN_BATCH_SIZE}, max_epochs={MAX_EPOCHS}, seeds torch/numpy=42.",
        f"- N=15 learning curve is stored in phase9b_results.json.",
        f"- Runtime: {results['runtime_seconds'] / 60:.1f} minutes on {results['device']}.",
        "- Small-N runs use the next QC sample as validation; N=15 uses a deterministic 15% internal window split.",
        "- Non-monotone N behavior, if present, is preserved in the JSON and plot.",
    ]
    (p9.OUTPUT_DIR / "SUMMARY_phase9b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    p9.set_seeds()
    p9.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_model, ckpt, num_classes = p9.load_model(device)
    tokens = p9.load_tokens(num_classes)
    all_samples = list(tokens.values())
    qc_samples = [s for s in all_samples if s.sample_type == "qc"]
    eval_samples = [s for s in all_samples if s.sample_type == "analytical"]
    qc_ids = {s.sample_id for s in qc_samples}
    eval_ids = {s.sample_id for s in eval_samples}

    assert len(qc_samples) == 15, f"Expected 15 QC samples, got {len(qc_samples)}"
    assert len(eval_samples) == 126, f"Expected 126 analytical samples, got {len(eval_samples)}"
    assert qc_ids.isdisjoint(eval_ids), "QC sample leaked into analytical eval set"

    zero_epoch = p9.evaluate_model_stream(base_model, eval_samples, device)
    print(
        f"0-epoch sanity on analytical: top1={zero_epoch['top1']:.4f}, "
        f"top5={zero_epoch['top5']:.4f}, n={zero_epoch['n']:,}"
    )
    if zero_epoch["n"] != 319339:
        raise RuntimeError(f"Expected analytical eval n=319,339, got {zero_epoch['n']:,}")

    context = phase9_context()
    if abs(zero_epoch["top1"] - context["phase9_floor_top1"]) > 1e-12:
        raise RuntimeError(
            "0-epoch sanity failed: fresh model eval does not match Phase 9 floor. "
            f"got {zero_epoch['top1']:.12f}, expected {context['phase9_floor_top1']:.12f}"
        )

    print("Building QC sliding-window tensors once for fine-tuning...")
    qc_fields, qc_targets, qc_sample_ids = make_examples(qc_samples)

    full_rows = []
    for n_cal in N_VALUES:
        train_ds, val_ds, split = split_for_n(qc_samples, qc_fields, qc_targets, qc_sample_ids, n_cal)
        print(
            f"Full fine-tune N={n_cal}: train_examples={split['train_examples']:,}, "
            f"val_examples={split['validation_examples']:,}"
        )
        model, info = train_full_model(n_cal, train_ds, val_ds, device, learning_curve=(n_cal == 15))
        eval_metrics = p9.evaluate_model_stream(model, eval_samples, device)
        row = {
            "method": "full_finetune",
            "N": n_cal,
            "eval_top1": eval_metrics["top1"],
            "eval_top5": eval_metrics["top5"],
            "eval_n": eval_metrics["n"],
            "chosen_epoch": info["best_epoch"],
            "best_validation_top1": info["best_validation_top1"],
            "best_validation_top5": info["best_validation_top5"],
            "split": split,
            "train_config": {
                "lr": info["lr"],
                "batch_size": info["batch_size"],
                "max_epochs": info["max_epochs"],
                "all_parameters_trainable": True,
            },
        }
        if "learning_curve" in info:
            row["learning_curve"] = info["learning_curve"]
        full_rows.append(row)
        print(
            f"N={n_cal} eval: top1={row['eval_top1']:.4f}, "
            f"top5={row['eval_top5']:.4f}, chosen_epoch={row['chosen_epoch']}"
        )

    results = {
        "experiment": "Phase 9B: full fine-tune recovery on ST000990",
        "random_seed": p9.RANDOM_SEED,
        "checkpoint": str(p9.CHECKPOINT),
        "data": str(p9.ST000990_PATH),
        "device": str(device),
        "context": {
            "phase9_floor_top1": context["phase9_floor_top1"],
            "phase9_head_only_best_top1": context["phase9_head_only_best_top1"],
            "phase9_markov_only_best_top1": context["phase9_markov_only_best_top1"],
            "within_method_ceiling_top1": context["within_method_ceiling_top1"],
            "phase9_recovery_table": context["phase9_recovery_table"],
        },
        "self_checks": {
            "eval_samples_n": len(eval_samples),
            "eval_n": zero_epoch["n"],
            "sample_disjoint": qc_ids.isdisjoint(eval_ids),
            "qc_samples": [s.sample_id for s in qc_samples],
            "eval_sample_type": "analytical",
            "zero_epoch_eval_top1": zero_epoch["top1"],
            "zero_epoch_eval_top5": zero_epoch["top5"],
            "phase9_floor_top1": context["phase9_floor_top1"],
            "zero_epoch_matches_phase9_floor": abs(zero_epoch["top1"] - context["phase9_floor_top1"]) <= 1e-12,
        },
        "full_finetune": full_rows,
        "runtime_seconds": time.time() - started,
        "secondary_embedding_frozen_run": "skipped_to_bound_cpu_runtime",
        "checkpoint_test_metrics": ckpt.get("test_metrics", {}),
    }

    with (p9.OUTPUT_DIR / "phase9b_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_phase9b(results)
    write_summary(results)
    print(f"Wrote Phase 9B outputs to {p9.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
