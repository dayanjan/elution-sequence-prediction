# Autoregressive Elution Sequence Prediction for Untargeted LC-MS Lipidomics

Code, trained model checkpoints, and reproducibility materials for the preprint:

> **Autoregressive Elution Sequence Prediction for Untargeted LC-MS Lipidomics.**
> Dayanjan S. Wijesinghe. *Preprint, 2026.* (arXiv ID: _to be assigned_)

---

## Overview

Untargeted LC–HRMS routinely detects thousands of molecular features per sample,
yet only 2–20% receive confident structural annotations. A root cause of this
"dark metabolome" is that tandem MS (MS/MS) acquisition remains **reactive**:
instruments select precursor ions only after they appear, with no foreknowledge
of what will elute next.

This work reframes chromatographic elution as an **autoregressive sequence
prediction** task. Because reversed-phase elution order is governed by
hydrophobicity, successive features are not independent draws but elements of a
physically constrained sequence — analogous to tokens in natural language. We
discretize the *m/z* axis into 110 bins and train LSTM and Transformer models to
predict the next eluting *m/z* bin from five per-token features (*m/z* bin, mass
defect, retention-time gap, ionization polarity, intensity rank).

### Headline results

| Result | Value |
|---|---|
| LSTM next-bin accuracy | **98.4% top-1** (99.97% top-5; MAE = 3 Da) |
| Transformer next-bin accuracy | 98.1% top-1 (99.97% top-5; MAE = 4 Da) |
| Ablation: autoregressive sequence context | **+55.5 pp** (no single input feature > 0.2 pp) |
| Cross-platform validation (same chromatography) | retention-time *r* = 0.999 |
| Cross-condition transfer (zero-shot, different polarity) | 2.6% top-1 (catastrophic) |
| Cross-condition transfer (fine-tune on 2–5 QC injections) | **recovers to ~50% top-1** (99.6% on held-out QC) |

Training data: 15,242 consensus features from four clinical lipidomics cohorts
(342 human plasma samples; SCIEX TripleTOF 6600+; Waters CSH C18).

---

## Repository structure

```
.
├── src/                 # Core library (preprocessing, tokenization, datasets, models, training)
├── scripts/             # Phase scripts (baselines, training, ablation, transfer learning)
├── notebooks/           # Colab notebooks (GPU training + transfer-learning recovery)
├── models/              # Trained checkpoints (LSTM, Transformer, ablation variants)
├── outputs/             # Metrics (JSON), result summaries, and analysis figures
├── manuscript/
│   ├── figures/         # Final manuscript figures (PDF/PNG)
│   └── scripts/         # Figure-generation code
├── docs/                # Design notes
├── config.yaml          # Single source of truth for hyperparameters, bins, seeds
├── requirements.txt     # Pinned Python dependencies
└── data/README.md       # Data availability (public datasets + clinical-data access)
```

---

## Reproducing the results

```bash
# 1. Environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Obtain data — see data/README.md
#    Public validation datasets are downloadable from the Metabolomics Workbench.
#    Clinical training cohorts are available under a data-use agreement (see below).

# 3. Run a phase script, e.g. baselines or the transfer-learning recovery
python scripts/<phase_script>.py
```

All randomness is seeded (`random_seed: 42` in `config.yaml`). GPU training and
the transfer-learning recovery experiment are provided as Colab notebooks under
`notebooks/` (verified on NVIDIA T4 / L4 / A100).

---

## Data availability

See [`data/README.md`](data/README.md). In brief: public validation datasets
(Metabolomics Workbench studies ST000983, ST000990, ST003514) are openly
downloadable; the clinical lipidomics feature tables derived from the four REDHART
plasma cohorts contain protected information and are available from the
corresponding author on reasonable request under an appropriate data-use
agreement, consistent with the governing IRB approvals and patient-consent terms.

---

## License

- **Code** (`src/`, `scripts/`, `notebooks/`, `manuscript/scripts/`): MIT — see [`LICENSE`](LICENSE).
- **The associated preprint manuscript and its figures** are released separately under
  **CC BY-NC-ND 4.0** on the preprint server.

---

## Citation

If you use this code or the trained models, please cite the preprint (see
[`CITATION.cff`](CITATION.cff)). The arXiv identifier will be added on posting.

## Contact

Dayanjan S. Wijesinghe — VCU Laboratory of Pharmacometabolomics and Companion
Diagnostics — <wijesingheds@vcu.edu> · ORCID
[0000-0002-2124-5109](https://orcid.org/0000-0002-2124-5109)
