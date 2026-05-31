# Phase 6: External Validation on ST003514 (NIST SRM 1950)

**Question:** Does our elution sequence model generalize to an independent dataset from a different instrument and column?

## Experimental Design

| | Training Data | External Validation |
|---|---|---|
| **Source** | 4 clinical cohorts (342 samples) | NIST SRM 1950 (ST003514, 51 samples) |
| **Instrument** | SCIEX TripleTOF 6600+ | Agilent 6545 QTOF |
| **Column** | Waters CSH C18 | Different C18 |
| **Features** | 15,242 across cohorts | 596 unique lipids |

## Data

| File | Description |
|------|-------------|
| `st003514_long.parquet` | Processed ST003514 in unified long format (36,356 rows) |
| `st003514_metabolites.csv` | Feature metadata (name, RT, m/z, ion mode, InChI Key) |

Data was downloaded from Metabolomics Workbench (study ST003514) and processed by `scripts/download_st003514.py` and `scripts/process_st003514.py`.

## Notebook

`02_external_validation.ipynb` — Run on Google Colab (GPU not required but speeds evaluation).

**Prerequisites:** Trained model checkpoints from `01_train_models/01_train_models.ipynb` must exist at `01_train_models/outputs/lstm_best.pt` and `01_train_models/outputs/transformer_best.pt`.

## Outputs

| File | Description |
|------|-------------|
| `outputs/phase6_external_validation.png` | 2-panel figure: accuracy comparison + position-dependent accuracy |
| `outputs/phase6_results.json` | Summary metrics and generalization ratios |
| `outputs/phase6_lstm_detailed.parquet` | Per-prediction LSTM results |
| `outputs/phase6_tfm_detailed.parquet` | Per-prediction Transformer results |

## Expected Outcome

Performance drop expected (different instrument/column/alignment), but model should outperform random baseline (1.1% top-1) if it learned generalizable elution chemistry.
