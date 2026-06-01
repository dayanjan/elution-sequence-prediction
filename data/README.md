# Data availability

This repository does **not** redistribute raw mass-spectrometry data or the
clinical-cohort feature tables. The `data/` directory is intentionally excluded
from version control (see the repository `.gitignore`). This document describes
how to obtain each dataset.

## Public validation datasets (openly available)

Downloadable from the **Metabolomics Workbench** (https://www.metabolomicsworkbench.org):

| Study ID  | Role in this work                                         |
|-----------|-----------------------------------------------------------|
| ST000983  | Cross-platform retention-time validation (same chromatography, different MS) |
| ST000990  | Cross-condition (positive-only) transfer-learning evaluation |
| ST003514  | NIST SRM 1950 RP-LC lipidomics (different chromatography)  |

These can be retrieved directly from the Workbench REST API or web interface
using the study IDs above.

## Clinical training cohorts (controlled access)

The model was trained on 15,242 consensus features from four clinical lipidomics
cohorts (342 human plasma samples; SCIEX TripleTOF 6600+; Waters CSH C18):
heart failure (REDHART 1), cardiac arrest, GVHD, and PCOS.

These feature tables derive from banked human plasma and contain protected
information. They are **available from the corresponding author on reasonable
request, under an appropriate data-use agreement**, consistent with the governing
IRB approvals and the patient-consent terms for the banked specimens. Aggregate
dataset characteristics (sample counts, annotated-feature counts, QC counts) are
reported in the manuscript and in `config.yaml`.

## Trained models

Trained model checkpoints are provided directly in this repository under
`models/` (LSTM, Transformer, and the ablation variants), so the published
results can be reproduced without retraining.
