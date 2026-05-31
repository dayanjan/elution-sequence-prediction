# Phase 3: Model Training (LSTM + Transformer)

**Question:** Can neural sequence models (LSTM, Transformer) outperform statistical baselines at predicting the next m/z bin in a reversed-phase lipidomics elution sequence?

## Data

| File | Description |
|------|-------------|
| `tokenized_features.parquet` | Pre-tokenized elution sequences from 4 clinical cohorts (342 samples, ~1.27M features) |

Tokenization scheme: m/z bin (10 Da), mass defect bin (20 bins), RT gap (7 categories), polarity, intensity rank.

## Notebook

`01_train_models.ipynb` (v1.3) — Run on Google Colab with T4 GPU.

**What it does:**
1. Loads tokenized data from Google Drive
2. Splits into train/val/test by sample (stratified by study)
3. Trains LSTM (257K params) and Transformer (122K params) with early stopping
4. Saves checkpoints + training log to `outputs/` after every epoch
5. Reports test-set top-k accuracy and MAE

**Crash resilience:** Per-epoch checkpoints + persistent CSV training log (`outputs/training_log.csv`). Monitor progress from local machine via Google Drive Desktop.

## Outputs

| File | Description |
|------|-------------|
| `outputs/lstm_best.pt` | Best LSTM checkpoint (weights + config + test metrics) |
| `outputs/transformer_best.pt` | Best Transformer checkpoint |
| `outputs/training_log.csv` | Per-epoch metrics with timestamps and GPU memory |
| `outputs/neural_model_results.json` | Summary results for both models |
| `outputs/training_curves.png` | Loss, top-1, top-5 over epochs |

## Baselines to Beat

| Model | Top-1 | Top-5 |
|-------|-------|-------|
| Random | 1.1% | — |
| Same-as-previous | 23.1% | — |
| Joint (RT,m/z) Markov | 56.8% | — |
