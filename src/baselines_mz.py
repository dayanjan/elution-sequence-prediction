"""
Classical baselines for next-m/z-bin prediction.

The task is to predict the m/z bin of the next RT-ordered feature within each
sample. Evaluation uses every adjacent pair in the held-out test samples.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_SEQUENCES, MZ_BIN_WIDTH, OUTPUTS, RANDOM_SEED
from datasets import sample_aware_split


TOP_K = (1, 3, 5, 10)

TARGETS = {
    "random": {"top1_accuracy": 0.011, "mae_da": 197.0},
    "global_frequency": {"top1_accuracy": 0.034, "mae_da": 156.0},
    "same_as_previous": {"top1_accuracy": 0.231, "mae_da": 119.0},
    "markov_order1_mz": {
        "top1_accuracy": 0.251,
        "top3_accuracy": 0.479,
        "top5_accuracy": 0.583,
        "top10_accuracy": 0.723,
        "mae_da": 115.0,
    },
    "joint_rt_mz_markov": {"top1_accuracy": 0.568, "mae_da": 67.0},
    "rt_only_oracle": {"top1_accuracy": 0.141},
}


def load_tokenized_features():
    """Load and sort tokenized feature occurrences."""
    tok = pd.read_parquet(DATA_SEQUENCES / "tokenized_features.parquet")
    return tok.sort_values(["study", "sample_id", "seq_pos"]).reset_index(drop=True)


def build_sequences(df):
    """Return per-sample arrays ordered by RT sequence position."""
    sequences = []
    for (study, sample_id), group in df.groupby(["study", "sample_id"], sort=False):
        g = group.sort_values("seq_pos")
        sequences.append(
            {
                "study": study,
                "sample_id": sample_id,
                "mz_bin": g["mz_bin"].to_numpy(dtype=np.int64),
                "rt_bin": g["rt_bin"].to_numpy(dtype=np.int64),
            }
        )
    return sequences


def adjacent_examples(sequences):
    """Flatten all predictable adjacent next-feature examples."""
    current_mz, prev_mz, true_mz = [], [], []
    current_rt, true_rt = [], []

    for seq in sequences:
        mz = seq["mz_bin"]
        rt = seq["rt_bin"]
        if len(mz) < 2:
            continue
        current_mz.extend(mz[:-1])
        true_mz.extend(mz[1:])
        current_rt.extend(rt[:-1])
        true_rt.extend(rt[1:])
        prev = np.empty(len(mz) - 1, dtype=np.int64)
        prev[0] = -1
        if len(prev) > 1:
            prev[1:] = mz[:-2]
        prev_mz.extend(prev)

    return {
        "current_mz": np.asarray(current_mz, dtype=np.int64),
        "prev_mz": np.asarray(prev_mz, dtype=np.int64),
        "true_mz": np.asarray(true_mz, dtype=np.int64),
        "current_rt": np.asarray(current_rt, dtype=np.int64),
        "true_rt": np.asarray(true_rt, dtype=np.int64),
    }


def sorted_bins(counter, fallback_order):
    """Rank bins by count descending, then bin id ascending, with fallback tail."""
    ranked = [int(k) for k, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]
    seen = set(ranked)
    ranked.extend(int(b) for b in fallback_order if int(b) not in seen)
    return ranked


def topk_matrix_from_rankings(rankings, n, top_k=TOP_K):
    """Convert per-example ranked lists to a dense matrix with max(top_k) columns."""
    max_k = max(top_k)
    out = np.empty((n, max_k), dtype=np.int64)
    for i, ranking in enumerate(rankings):
        if len(ranking) < max_k:
            raise ValueError("ranking is shorter than requested top-k")
        out[i, :] = ranking[:max_k]
    return out


def evaluate(true_bins, ranked_preds, name):
    """Compute top-k accuracy and MAE in Da from ranked m/z-bin predictions."""
    true_bins = np.asarray(true_bins, dtype=np.int64)
    ranked_preds = np.asarray(ranked_preds, dtype=np.int64)
    top1 = ranked_preds[:, 0]
    metrics = {
        "baseline": name,
        "n": int(len(true_bins)),
        "top1_accuracy": float(np.mean(top1 == true_bins)),
        "mae_da": float(np.mean(np.abs(top1 - true_bins)) * MZ_BIN_WIDTH),
    }
    for k in TOP_K:
        metrics[f"top{k}_accuracy"] = float(
            np.mean(np.any(ranked_preds[:, :k] == true_bins[:, None], axis=1))
        )
    return metrics


def training_distribution(train_sequences):
    counts = Counter()
    for seq in train_sequences:
        counts.update(int(x) for x in seq["mz_bin"])
    fallback_order = sorted_bins(counts, [])
    values = np.asarray(fallback_order, dtype=np.int64)
    probs = np.asarray([counts[int(v)] for v in values], dtype=float)
    probs /= probs.sum()
    return counts, fallback_order, values, probs


def random_baseline(true_mz, train_values, train_probs, fallback_order):
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(true_mz)
    ranked = np.empty((n, max(TOP_K)), dtype=np.int64)
    ranked[:, 0] = rng.choice(train_values, size=n, replace=True, p=train_probs)
    for i in range(n):
        tail = [b for b in fallback_order if b != int(ranked[i, 0])]
        ranked[i, 1:] = tail[: max(TOP_K) - 1]
    return ranked


def repeated_ranking(n, ranking):
    return np.tile(np.asarray(ranking[: max(TOP_K)], dtype=np.int64), (n, 1))


def same_as_previous_baseline(current_mz, fallback_order):
    rankings = []
    for curr in current_mz:
        curr = int(curr)
        rankings.append([curr] + [b for b in fallback_order if b != curr])
    return topk_matrix_from_rankings(rankings, len(current_mz))


def fit_markov(train_sequences, order):
    transitions = defaultdict(Counter)
    for seq in train_sequences:
        mz = seq["mz_bin"]
        for i in range(order, len(mz)):
            context = tuple(int(x) for x in mz[i - order : i])
            transitions[context][int(mz[i])] += 1
    return transitions


def markov_rankings(examples, transitions, fallback_order, order):
    rankings = []
    for prev_bin, curr_bin in zip(examples["prev_mz"], examples["current_mz"]):
        if order == 1:
            context = (int(curr_bin),)
        elif int(prev_bin) >= 0:
            context = (int(prev_bin), int(curr_bin))
        else:
            context = None
        rankings.append(sorted_bins(transitions.get(context, Counter()), fallback_order))
    return topk_matrix_from_rankings(rankings, len(examples["true_mz"]))


def linear_extrapolation_baseline(test_sequences, window, fallback_order):
    true, rankings = [], []
    observed_min = min(fallback_order)
    observed_max = max(fallback_order)

    for seq in test_sequences:
        mz = seq["mz_bin"]
        for i in range(len(mz) - 1):
            start = max(0, i - window + 1)
            recent = mz[start : i + 1].astype(float)
            if len(recent) >= 2:
                x = np.arange(len(recent), dtype=float)
                slope, intercept = np.polyfit(x, recent, 1)
                pred = intercept + slope * len(recent)
            else:
                pred = recent[-1]

            center = int(round(np.clip(pred, observed_min, observed_max)))
            ranked = sorted(fallback_order, key=lambda b: (abs(int(b) - center), int(b)))
            rankings.append(ranked)
            true.append(int(mz[i + 1]))

    return np.asarray(true, dtype=np.int64), topk_matrix_from_rankings(rankings, len(true))


def fit_joint_rt_mz_markov(train_sequences):
    transitions = defaultdict(Counter)
    for seq in train_sequences:
        mz = seq["mz_bin"]
        rt = seq["rt_bin"]
        for i in range(len(mz) - 1):
            transitions[(int(rt[i]), int(mz[i]))][int(mz[i + 1])] += 1
    return transitions


def joint_rt_mz_rankings(examples, transitions, fallback_order):
    rankings = []
    for rt_bin, mz_bin in zip(examples["current_rt"], examples["current_mz"]):
        context = (int(rt_bin), int(mz_bin))
        rankings.append(sorted_bins(transitions.get(context, Counter()), fallback_order))
    return topk_matrix_from_rankings(rankings, len(examples["true_mz"]))


def fit_rt_oracle(train_sequences):
    by_rt = defaultdict(Counter)
    for seq in train_sequences:
        for rt_bin, mz_bin in zip(seq["rt_bin"], seq["mz_bin"]):
            by_rt[int(rt_bin)][int(mz_bin)] += 1
    return by_rt


def rt_oracle_rankings(true_rt, by_rt, fallback_order):
    rankings = [sorted_bins(by_rt.get(int(rt_bin), Counter()), fallback_order) for rt_bin in true_rt]
    return topk_matrix_from_rankings(rankings, len(true_rt))


def print_summary(results):
    print("\nNEXT-M/Z-BIN BASELINES")
    print("-" * 94)
    print(
        f"{'Baseline':<28} {'n':>8} {'Top-1':>8} {'Top-3':>8} "
        f"{'Top-5':>8} {'Top-10':>8} {'MAE Da':>8}"
    )
    print("-" * 94)
    for name, metrics in results.items():
        print(
            f"{name:<28} {metrics['n']:>8,d} {metrics['top1_accuracy']:>8.4f} "
            f"{metrics['top3_accuracy']:>8.4f} {metrics['top5_accuracy']:>8.4f} "
            f"{metrics['top10_accuracy']:>8.4f} {metrics['mae_da']:>8.1f}"
        )


def print_target_comparison(results):
    print("\nCOMPUTED VS TARGET")
    print("-" * 86)
    print(f"{'Baseline':<28} {'Metric':<14} {'Computed':>10} {'Target':>10} {'Delta':>10}")
    print("-" * 86)
    for name, targets in TARGETS.items():
        if name not in results:
            continue
        for metric, target in targets.items():
            computed = results[name][metric]
            delta = computed - target
            print(f"{name:<28} {metric:<14} {computed:>10.4f} {target:>10.4f} {delta:>10.4f}")


def main():
    np.random.seed(RANDOM_SEED)

    print("Loading tokenized features...")
    tok = load_tokenized_features()
    print("Creating sample-aware split with src.datasets.sample_aware_split...")
    train_df, val_df, test_df = sample_aware_split(tok)

    train_sequences = build_sequences(train_df)
    test_sequences = build_sequences(test_df)
    examples = adjacent_examples(test_sequences)
    true_mz = examples["true_mz"]

    print(f"Train samples: {len(train_sequences):,}")
    print(f"Val samples:   {val_df[['study', 'sample_id']].drop_duplicates().shape[0]:,}")
    print(f"Test samples:  {len(test_sequences):,}")
    print(f"Test positions: {len(true_mz):,}")

    train_counts, fallback_order, train_values, train_probs = training_distribution(train_sequences)

    results = {}
    results["random"] = evaluate(
        true_mz,
        random_baseline(true_mz, train_values, train_probs, fallback_order),
        "random",
    )
    results["global_frequency"] = evaluate(
        true_mz,
        repeated_ranking(len(true_mz), fallback_order),
        "global_frequency",
    )
    results["same_as_previous"] = evaluate(
        true_mz,
        same_as_previous_baseline(examples["current_mz"], fallback_order),
        "same_as_previous",
    )

    markov1 = fit_markov(train_sequences, order=1)
    markov1_preds = markov_rankings(examples, markov1, fallback_order, order=1)
    results["markov_order1_mz"] = evaluate(true_mz, markov1_preds, "markov_order1_mz")
    results["frequency_conditioned_current_bin"] = evaluate(
        true_mz,
        markov1_preds,
        "frequency_conditioned_current_bin",
    )

    markov2 = fit_markov(train_sequences, order=2)
    results["markov_order2_mz"] = evaluate(
        true_mz,
        markov_rankings(examples, markov2, fallback_order, order=2),
        "markov_order2_mz",
    )

    for window in (5, 10):
        lin_true, lin_preds = linear_extrapolation_baseline(test_sequences, window, fallback_order)
        results[f"linear_mz_extrapolation_w{window}"] = evaluate(
            lin_true,
            lin_preds,
            f"linear_mz_extrapolation_w{window}",
        )

    joint = fit_joint_rt_mz_markov(train_sequences)
    results["joint_rt_mz_markov"] = evaluate(
        true_mz,
        joint_rt_mz_rankings(examples, joint, fallback_order),
        "joint_rt_mz_markov",
    )

    rt_oracle = fit_rt_oracle(train_sequences)
    results["rt_only_oracle"] = evaluate(
        true_mz,
        rt_oracle_rankings(examples["true_rt"], rt_oracle, fallback_order),
        "rt_only_oracle",
    )

    print_summary(results)
    print_target_comparison(results)

    output_path = OUTPUTS / "metrics" / "baseline_mz_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {output_path}")

    return results


if __name__ == "__main__":
    main()
