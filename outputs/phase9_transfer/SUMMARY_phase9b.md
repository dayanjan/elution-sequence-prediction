# Phase 9B full fine-tune summary

- Best full fine-tune: N=15, top-1=0.4796, top-5=0.6730, best epoch=10.
- Eval guardrail: same 126 analytical samples, n=319,339.
- Leakage guardrail: QC train/calibration and analytical eval disjoint = True.
- Sanity guardrail: 0-epoch eval top-1=0.0259; Phase 9 floor=0.0259.
- Phase 9 cheap-method references: Markov-only best=0.1197; head-only best=0.1598.
- Within-method ceiling reference: 0.9838.
- Verdict: Full fine-tuning shows a recoverable adaptation gap versus cheap methods, but 15 QC samples do not recover the within-method ceiling.
- Training setup: full model unfrozen, Adam lr=0.0003, batch=32, max_epochs=12, seeds torch/numpy=42.
- N=15 learning curve is stored in phase9b_results.json.
- Runtime: 43.6 minutes on cpu.
- Small-N runs use the next QC sample as validation; N=15 uses a deterministic 15% internal window split.
- Non-monotone N behavior, if present, is preserved in the JSON and plot.
