# Elution Sequence Prediction — Colab Notebooks

Notebooks for the elution sequence prediction preprint. Each experiment lives in its own folder with data, notebook, and outputs.

## Run Order

```
01_train_models/                   (run first — produces model checkpoints)
        |
        v
02_external_validation/            (requires checkpoints from 01)
03_qc_warmup/                      (requires LSTM checkpoint from 01)
```

## Folder Structure

```
poc3_elution_sequence/
|-- 01_train_models/                   Training (LSTM + Transformer)
|   |-- 01_train_models.ipynb
|   |-- tokenized_features.parquet     Training data (4 cohorts, 342 samples)
|   |-- README.md
|   +-- outputs/                       Model checkpoints + training log
|       |-- lstm_best.pt
|       |-- transformer_best.pt
|       +-- training_log.csv           Per-epoch log (monitor from local machine)
|
|-- 02_external_validation/            Phase 6: Cross-platform generalization
|   |-- 02_external_validation.ipynb
|   |-- st003514_long.parquet          NIST SRM 1950 data (Agilent 6545 QTOF)
|   |-- st003514_metabolites.csv       Feature metadata
|   |-- README.md
|   +-- outputs/
|
+-- 03_qc_warmup/                      Phase 4: QC conditioning experiment
    |-- 03_qc_warmup.ipynb
    |-- README.md
    +-- outputs/
```

## Google Drive Location

`My Drive/0000 Fun with coding/088 Lights-Out R01 Grant/Specific Aim 1/poc3_elution_sequence/`

## Notes

- All notebooks are self-contained (model definitions duplicated intentionally for reproducibility)
- Training notebook saves checkpoints to Google Drive after every epoch
- Monitor training remotely via `01_train_models/outputs/training_log.csv` (synced to local machine via Google Drive Desktop)
- GPU runtime required for training; evaluation notebooks work on CPU but GPU is faster
