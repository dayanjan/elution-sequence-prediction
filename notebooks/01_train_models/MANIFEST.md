# Artifact Manifest — Model Training (Phase 3)

**Generated:** 2026-03-09
**Status:** LSTM complete, Transformer in progress (epoch 35/100)

Binary artifacts are stored on Google Drive (not in git). Use the SHA256 hashes below to verify integrity.

## Model Checkpoints

| File | Size | SHA256 | Status |
|------|------|--------|--------|
| `outputs/lstm_best.pt` | 1.0 MB | `8cb759bd5cb5c8d0ed38d14e2f9f360adae09205a00980e4cf1b13a5fe1e580b` | Final (100 epochs, no early stop) |
| `outputs/transformer_best.pt` | 507 KB | `9184a8737351e1b0fd24cafd18892eeca49df0ce4ee3347450535c03f34fddb6` | In progress (best epoch 33/100) |

## Training Data

| File | Size | SHA256 |
|------|------|--------|
| `tokenized_features.parquet` | 15.2 MB | `1b32eb370037fc70dcb56d08599912cf12a4fa7ca1624615b2a4f5c8b8d8c014` |

## Final Test Metrics

### LSTM (256,824 params)

| Metric | Value |
|--------|-------|
| Test top-1 | 98.38% |
| Test top-5 | 99.99% |
| Test MAE | 3.6 Da |
| Best epoch | 100 |
| Training time | 5.1 hours (T4 GPU) |

### Transformer (122K params)

*Awaiting completion — update this section when training finishes.*

## Verification

```bash
# From the Google Drive poc3_elution_sequence/01_train_models/ directory:
sha256sum outputs/lstm_best.pt outputs/transformer_best.pt tokenized_features.parquet
```

## Google Drive Location

`My Drive/0000 Fun with coding/088 Lights-Out R01 Grant/Specific Aim 1/poc3_elution_sequence/01_train_models/`
