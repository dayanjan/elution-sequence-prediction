# Phase 9 transfer recovery summary

- Gate: reproduced zero-shot ST000990 top-1 = 0.0263 (n = 396,349); reference = 0.0263.
- Leakage check: QC calibration samples and analytical evaluation samples are disjoint = True; pure LSTM on analytical top-1 = 0.0259 (n = 319,339).
- Best overall recovery: fine_tune_head at N=15: top-1 = 0.1598, top-5 = 0.4112.
- Best at N=5: markov_blend alpha=0.0: top-1 = 0.1194, top-5 = 0.3739.
- Best at N=15: fine_tune_head: top-1 = 0.1598, top-5 = 0.4112.
- Calibration pool: first N QC samples (QC01..QC15 order from the parquet columns).
- Evaluation set: 126 analytical SO samples only.
- Markov rows with unseen previous m/z bins use the calibration global next-bin prior.
- Fine-tuning freezes embeddings and LSTM and trains only the final Linear head for 40 CPU epochs.
- The within-method ceiling reference is the checkpoint test top-1, 0.984.
