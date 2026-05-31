"""
Baseline models for elution sequence prediction.

All baselines predict the NEXT RT bin given some form of context.
They serve as performance floors for the neural models.

Baselines:
  1. Random: Uniform random prediction from observed RT bin distribution
  2. Frequency: Always predict the most common next RT bin (global)
  3. Markov order-1: P(next_bin | current_bin)
  4. Markov order-2: P(next_bin | prev_bin, current_bin)
  5. RT-only linear: Predict next RT from recent RT trend (linear extrapolation)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_SEQUENCES, RANDOM_SEED, OUTPUTS


def load_sequences():
    """Load tokenized features and build (study, sample) -> sorted feature list."""
    tok = pd.read_parquet(DATA_SEQUENCES / "tokenized_features.parquet")
    tok = tok.sort_values(["study", "sample_id", "seq_pos"])
    return tok


def train_test_split_samples(tok, test_fraction=0.15, val_fraction=0.15):
    """
    Sample-aware split: no sample appears in both train and test.
    Stratified by study to ensure all cohorts represented.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    samples = tok[["study", "sample_id", "sample_type"]].drop_duplicates()

    train_samples, val_samples, test_samples = [], [], []

    for study, group in samples.groupby("study"):
        indices = group.index.values.copy()
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))

        test_samples.extend(group.loc[indices[:n_test], ["study", "sample_id"]].values.tolist())
        val_samples.extend(group.loc[indices[n_test:n_test + n_val], ["study", "sample_id"]].values.tolist())
        train_samples.extend(group.loc[indices[n_test + n_val:], ["study", "sample_id"]].values.tolist())

    def filter_tok(sample_list):
        keys = set(tuple(s) for s in sample_list)
        mask = tok.apply(lambda r: (r["study"], r["sample_id"]) in keys, axis=1)
        return tok[mask]

    return filter_tok(train_samples), filter_tok(val_samples), filter_tok(test_samples)


def extract_rt_bin_sequences(tok):
    """Extract per-sample RT bin sequences as lists."""
    sequences = {}
    for (study, sample_id), group in tok.groupby(["study", "sample_id"]):
        sequences[(study, sample_id)] = group["rt_bin"].values
    return sequences


def evaluate_predictions(true_bins, pred_bins, pred_probs=None, top_k=(1, 3, 5)):
    """
    Evaluate prediction quality.

    Args:
        true_bins: array of true RT bin indices
        pred_bins: array of predicted RT bin indices (top-1)
        pred_probs: optional (n_samples, n_classes) probability matrix for top-k
        top_k: tuple of k values for top-k accuracy

    Returns:
        dict of metrics
    """
    n = len(true_bins)
    if n == 0:
        return {}

    # Top-1 accuracy
    top1_acc = np.mean(true_bins == pred_bins)

    # MAE in RT bins (× 3s = MAE in seconds)
    mae_bins = np.mean(np.abs(true_bins - pred_bins))

    # Within-window accuracy (prediction within ±k bins)
    within_1 = np.mean(np.abs(true_bins - pred_bins) <= 1)  # ±3s
    within_3 = np.mean(np.abs(true_bins - pred_bins) <= 3)  # ±9s
    within_5 = np.mean(np.abs(true_bins - pred_bins) <= 5)  # ±15s
    within_10 = np.mean(np.abs(true_bins - pred_bins) <= 10)  # ±30s
    within_20 = np.mean(np.abs(true_bins - pred_bins) <= 20)  # ±60s

    metrics = {
        "n_predictions": n,
        "top1_accuracy": float(top1_acc),
        "mae_bins": float(mae_bins),
        "mae_seconds": float(mae_bins * 3),
        "within_3s": float(within_1),
        "within_9s": float(within_3),
        "within_15s": float(within_5),
        "within_30s": float(within_10),
        "within_60s": float(within_20),
    }

    # Top-k accuracy from probability matrix
    if pred_probs is not None:
        for k in top_k:
            if pred_probs.shape[1] >= k:
                top_k_preds = np.argsort(pred_probs, axis=1)[:, -k:]
                top_k_acc = np.mean([t in preds for t, preds in zip(true_bins, top_k_preds)])
                metrics[f"top{k}_accuracy"] = float(top_k_acc)

    return metrics


class RandomBaseline:
    """Predict next RT bin by sampling from the training distribution."""

    def __init__(self):
        self.bin_probs = None
        self.bin_values = None
        self.rng = np.random.RandomState(RANDOM_SEED)

    def fit(self, sequences):
        all_bins = np.concatenate(list(sequences.values()))
        counts = Counter(all_bins)
        total = sum(counts.values())
        self.bin_values = np.array(sorted(counts.keys()))
        self.bin_probs = np.array([counts[b] / total for b in self.bin_values])

    def predict(self, context):
        return self.rng.choice(self.bin_values, p=self.bin_probs)

    def predict_batch(self, true_bins):
        return self.rng.choice(self.bin_values, size=len(true_bins), p=self.bin_probs)


class FrequencyBaseline:
    """Always predict the most common next RT bin (globally or conditioned on current)."""

    def __init__(self):
        self.most_common = None
        self.transition_mode = None

    def fit(self, sequences):
        # Global most common
        all_bins = np.concatenate(list(sequences.values()))
        self.most_common = Counter(all_bins).most_common(1)[0][0]

        # Conditioned: most common next bin given current bin
        self.transition_mode = {}
        for seq in sequences.values():
            for i in range(len(seq) - 1):
                curr, nxt = int(seq[i]), int(seq[i + 1])
                if curr not in self.transition_mode:
                    self.transition_mode[curr] = Counter()
                self.transition_mode[curr][nxt] += 1

        # Convert to most common
        for curr in self.transition_mode:
            self.transition_mode[curr] = self.transition_mode[curr].most_common(1)[0][0]

    def predict_global(self, n):
        return np.full(n, self.most_common)

    def predict_conditioned(self, current_bins):
        return np.array([
            self.transition_mode.get(int(c), self.most_common)
            for c in current_bins
        ])


class MarkovBaseline:
    """Markov chain of order 1 or 2 for RT bin transitions."""

    def __init__(self, order=1):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.fallback = None

    def fit(self, sequences):
        all_bins = np.concatenate(list(sequences.values()))
        self.fallback = Counter(all_bins).most_common(1)[0][0]

        for seq in sequences.values():
            for i in range(self.order, len(seq)):
                context = tuple(int(seq[j]) for j in range(i - self.order, i))
                nxt = int(seq[i])
                self.transitions[context][nxt] += 1

        # Normalize to probabilities
        self.transition_probs = {}
        for context, counts in self.transitions.items():
            total = sum(counts.values())
            self.transition_probs[context] = {
                k: v / total for k, v in counts.items()
            }

    def predict(self, context_bins):
        """Predict next bin given context (tuple of ints)."""
        context = tuple(int(b) for b in context_bins[-self.order:])
        if context in self.transition_probs:
            probs = self.transition_probs[context]
            return max(probs, key=probs.get)
        return self.fallback

    def predict_batch(self, sequences_dict):
        """Predict next bin for each position in each sequence."""
        all_true = []
        all_pred = []
        for seq in sequences_dict.values():
            for i in range(self.order, len(seq)):
                context = tuple(int(seq[j]) for j in range(i - self.order, i))
                true_bin = int(seq[i])
                pred_bin = self.predict(context)
                all_true.append(true_bin)
                all_pred.append(pred_bin)
        return np.array(all_true), np.array(all_pred)


class LinearRTBaseline:
    """Predict next RT bin by linear extrapolation from recent RT values."""

    def __init__(self, window=5):
        self.window = window

    def fit(self, sequences):
        pass  # No training needed

    def predict_batch(self, sequences_dict):
        all_true = []
        all_pred = []
        for seq in sequences_dict.values():
            for i in range(self.window, len(seq)):
                recent = seq[max(0, i - self.window):i].astype(float)
                # Linear extrapolation
                if len(recent) >= 2:
                    x = np.arange(len(recent))
                    slope = np.polyfit(x, recent, 1)[0]
                    pred = recent[-1] + slope
                else:
                    pred = recent[-1]
                all_true.append(int(seq[i]))
                all_pred.append(int(round(pred)))
        return np.array(all_true), np.array(all_pred)


class SameAsPreviousBaseline:
    """Predict next RT bin = current RT bin (naive persistence)."""

    def predict_batch(self, sequences_dict):
        all_true = []
        all_pred = []
        for seq in sequences_dict.values():
            for i in range(1, len(seq)):
                all_true.append(int(seq[i]))
                all_pred.append(int(seq[i - 1]))
        return np.array(all_true), np.array(all_pred)


def main():
    np.random.seed(RANDOM_SEED)

    print("Loading tokenized features...")
    tok = load_sequences()

    print("Splitting samples (train/val/test)...")
    train_tok, val_tok, test_tok = train_test_split_samples(tok)
    print(f"  Train: {train_tok[['study','sample_id']].drop_duplicates().shape[0]} samples")
    print(f"  Val:   {val_tok[['study','sample_id']].drop_duplicates().shape[0]} samples")
    print(f"  Test:  {test_tok[['study','sample_id']].drop_duplicates().shape[0]} samples")

    train_seqs = extract_rt_bin_sequences(train_tok)
    val_seqs = extract_rt_bin_sequences(val_tok)
    test_seqs = extract_rt_bin_sequences(test_tok)

    results = {}

    # === 1. Same-as-previous baseline ===
    print("\n--- Same-as-previous baseline ---")
    sap = SameAsPreviousBaseline()
    true, pred = sap.predict_batch(test_seqs)
    metrics = evaluate_predictions(true, pred)
    results["same_as_previous"] = metrics
    print(f"  Top-1: {metrics['top1_accuracy']:.4f}, MAE: {metrics['mae_seconds']:.1f}s, "
          f"±15s: {metrics['within_15s']:.4f}, ±30s: {metrics['within_30s']:.4f}")

    # === 2. Random baseline ===
    print("\n--- Random baseline ---")
    rand = RandomBaseline()
    rand.fit(train_seqs)
    true_all = np.concatenate([s[1:] for s in test_seqs.values()])
    pred_rand = rand.predict_batch(true_all)
    metrics = evaluate_predictions(true_all, pred_rand)
    results["random"] = metrics
    print(f"  Top-1: {metrics['top1_accuracy']:.4f}, MAE: {metrics['mae_seconds']:.1f}s, "
          f"±15s: {metrics['within_15s']:.4f}, ±30s: {metrics['within_30s']:.4f}")

    # === 3. Frequency baseline (global) ===
    print("\n--- Frequency baseline (global mode) ---")
    freq = FrequencyBaseline()
    freq.fit(train_seqs)
    pred_freq = freq.predict_global(len(true_all))
    metrics = evaluate_predictions(true_all, pred_freq)
    results["frequency_global"] = metrics
    print(f"  Top-1: {metrics['top1_accuracy']:.4f}, MAE: {metrics['mae_seconds']:.1f}s, "
          f"±15s: {metrics['within_15s']:.4f}, ±30s: {metrics['within_30s']:.4f}")

    # === 4. Frequency baseline (conditioned) ===
    print("\n--- Frequency baseline (conditioned on current bin) ---")
    curr_bins = np.concatenate([s[:-1] for s in test_seqs.values()])
    pred_freq_cond = freq.predict_conditioned(curr_bins)
    metrics = evaluate_predictions(true_all, pred_freq_cond)
    results["frequency_conditioned"] = metrics
    print(f"  Top-1: {metrics['top1_accuracy']:.4f}, MAE: {metrics['mae_seconds']:.1f}s, "
          f"±15s: {metrics['within_15s']:.4f}, ±30s: {metrics['within_30s']:.4f}")

    # === 5. Markov order-1 ===
    print("\n--- Markov order-1 ---")
    m1 = MarkovBaseline(order=1)
    m1.fit(train_seqs)
    true_m1, pred_m1 = m1.predict_batch(test_seqs)
    metrics = evaluate_predictions(true_m1, pred_m1)
    results["markov_order1"] = metrics
    print(f"  Top-1: {metrics['top1_accuracy']:.4f}, MAE: {metrics['mae_seconds']:.1f}s, "
          f"±15s: {metrics['within_15s']:.4f}, ±30s: {metrics['within_30s']:.4f}")

    # === 6. Markov order-2 ===
    print("\n--- Markov order-2 ---")
    m2 = MarkovBaseline(order=2)
    m2.fit(train_seqs)
    true_m2, pred_m2 = m2.predict_batch(test_seqs)
    metrics = evaluate_predictions(true_m2, pred_m2)
    results["markov_order2"] = metrics
    print(f"  Top-1: {metrics['top1_accuracy']:.4f}, MAE: {metrics['mae_seconds']:.1f}s, "
          f"±15s: {metrics['within_15s']:.4f}, ±30s: {metrics['within_30s']:.4f}")

    # === 7. Linear RT extrapolation ===
    print("\n--- Linear RT extrapolation (window=5) ---")
    lin = LinearRTBaseline(window=5)
    true_lin, pred_lin = lin.predict_batch(test_seqs)
    metrics = evaluate_predictions(true_lin, pred_lin)
    results["linear_rt_w5"] = metrics
    print(f"  Top-1: {metrics['top1_accuracy']:.4f}, MAE: {metrics['mae_seconds']:.1f}s, "
          f"±15s: {metrics['within_15s']:.4f}, ±30s: {metrics['within_30s']:.4f}")

    # === 8. Linear RT extrapolation (window=10) ===
    print("\n--- Linear RT extrapolation (window=10) ---")
    lin10 = LinearRTBaseline(window=10)
    true_lin10, pred_lin10 = lin10.predict_batch(test_seqs)
    metrics = evaluate_predictions(true_lin10, pred_lin10)
    results["linear_rt_w10"] = metrics
    print(f"  Top-1: {metrics['top1_accuracy']:.4f}, MAE: {metrics['mae_seconds']:.1f}s, "
          f"±15s: {metrics['within_15s']:.4f}, ±30s: {metrics['within_30s']:.4f}")

    # === Summary table ===
    print("\n" + "=" * 90)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Model':<30} {'Top-1':<8} {'MAE(s)':<8} {'±3s':<8} {'±9s':<8} {'±15s':<8} {'±30s':<8} {'±60s':<8}")
    print("-" * 90)
    for name, m in sorted(results.items(), key=lambda x: -x[1].get("within_15s", 0)):
        print(f"  {name:<28} {m['top1_accuracy']:.4f}  {m['mae_seconds']:<7.1f} "
              f" {m['within_3s']:.4f}  {m['within_9s']:.4f}  {m['within_15s']:.4f}  "
              f"{m['within_30s']:.4f}  {m['within_60s']:.4f}")

    # Save results
    results_dir = OUTPUTS / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {results_dir / 'baseline_results.json'}")

    return results


if __name__ == "__main__":
    results = main()
