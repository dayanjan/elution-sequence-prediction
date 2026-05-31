# Phase 4: QC Warm-Up Experiment

**Question:** Does conditioning the LSTM hidden state with QC injection sequences improve next-m/z-bin prediction on analytical samples?

## Experimental Design

Feed QC elution sequences through the LSTM using teacher forcing (ground truth tokens) to build up hidden state, then evaluate on analytical samples starting from this warmed-up state instead of zeros.

### Dose-Response
- N = 0, 2, 4, 6, 8, 10 QC injections before each analytical sample

### Evaluation Modes
| Mode | Description |
|------|-------------|
| Prime-only | Warm up hidden state with QCs, use for first prediction only, then standard sliding window |
| Carry-hidden | Warm up with QCs, then propagate hidden state through the entire analytical sample |

The comparison answers: does the LSTM benefit from maintaining state across the whole analytical run, or is the initial prime sufficient?

### Controls
| Control | Purpose |
|---------|---------|
| Analytical warm-up | Non-QC conditioning (tests if benefit is QC-specific) |
| Cross-cohort QC | QCs from different study (tests cohort specificity) |
| Shuffled QC | Destroy temporal structure (tests if order matters) |

## Data

This notebook does **not** have its own dataset. It loads `tokenized_features.parquet` from `../01_train_models/` — the same 4-cohort training data, but only held-out QC and test-set analytical samples are used here. The train/val/test split is reproduced identically using `RANDOM_SEED = 42`.

## Notebook

`03_qc_warmup.ipynb` (v1.2) — Run on Google Colab (GPU recommended).

**Prerequisites:** Trained LSTM checkpoint must exist at `../01_train_models/outputs/lstm_best.pt`.

## Outputs

| File | Description |
|------|-------------|
| `outputs/qc_warmup_log.csv` | Per-condition progress log (monitor remotely via Google Drive Desktop) |
| `outputs/phase4_qc_warmup.png` | 4-panel figure: dose-response (prime vs carry), condition comparison, position accuracy, carry-hidden delta |
| `outputs/phase4_results.json` | Summary metrics for all conditions (prime-only + carry-hidden) |
| `outputs/phase4_prime_only_detailed.parquet` | Per-prediction results for prime-only mode |
| `outputs/phase4_carry_hidden_detailed.parquet` | Per-prediction results for carry-hidden mode |

## Design Decision

Option B (pragmatic): Use existing trained model without QC-specific fine-tuning. LSTM-only (Transformer has no hidden state to warm up). Both per-cohort and pooled results reported.
