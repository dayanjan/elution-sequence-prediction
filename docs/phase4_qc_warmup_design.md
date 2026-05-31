# Phase 4: QC Warm-Up Experiment — Detailed Design

**Date:** 2026-03-08
**Status:** Design review (pre-implementation)

---

## 1. Hypothesis

**Does presenting QC injection sequences as conditioning context improve the model's next-m/z-bin predictions on subsequent analytical samples?**

In standard metabolomics workflows, the instrument runs 8+ QC injections before analytical samples to equilibrate the column and verify system performance. We test whether the LSTM/Transformer can exploit this same pattern: if it "sees" the elution sequences from QC injections first, it should build a stronger prior over what features to expect (and in what order) when analytical samples follow.

---

## 2. What Makes This Non-Trivial

### The consensus-RT constraint
MS-DIAL alignment produces a **single consensus RT per feature across all samples**, not per-injection RT values. This means:
- Every sample has the **same elution order** for any given feature
- The only per-sample variation is **which features are detected** (presence/absence) and **at what intensity**
- We cannot measure per-injection RT drift or re-ordering

### What QC injections DO provide
Despite consensus RT, QC samples differ from analytical samples in measurable ways:
1. **Detection reproducibility:** 90-96% of features appear in ALL QC injections vs. more variable detection in analytical samples (disease-related metabolite changes, matrix effects)
2. **Intensity profiles:** QC pooled plasma has a characteristic intensity distribution different from individual patient samples
3. **Feature set stability:** QC-to-QC Jaccard similarity is 0.959-0.985 — higher than QC-to-analytical

### The core mechanism
The model sees a sliding window of 64 tokens. QC warm-up works by **seeding the model's hidden state** (LSTM) or **attention context** (Transformer) with representative elution patterns before it encounters analytical samples. The question is whether this seeding produces better predictions than cold-start (no prior context from the same batch).

---

## 3. Experimental Protocol

### 3.1 Data partitioning

Use the existing sample-aware split infrastructure, but with a QC-aware modification:

```
For each cohort (cardiac_arrest, gvhd, pcos, redhart2):
  - QC samples: reserved entirely for warm-up conditioning (never in train/val/test)
  - Analytical samples: split into train/val/test as before (70/15/15)
```

**Rationale:** QC samples should NOT appear in the training set for this experiment. The model must learn general elution patterns from analytical samples, then we test whether QC conditioning at inference time improves performance. If QC samples were in training, we couldn't distinguish "the model learned QC-specific patterns during training" from "the model benefits from QC conditioning at inference."

**Important exception:** The model IS pre-trained on the full training set (which currently includes QC samples per the existing split). For Phase 4, we either:
- **(Option A — cleaner):** Retrain without QC samples in training. This isolates the warm-up effect but costs a Colab training run.
- **(Option B — pragmatic):** Use the existing model but exclude QC samples that were in the training set from the conditioning pool. Since we have 34 QC samples across 4 cohorts, and only ~15% would have landed in train, we still have ~29 QC samples available for conditioning.

**Recommendation:** Option A is scientifically cleaner. Option B is acceptable if we're time-constrained, but we must report it as a limitation.

### 3.2 Conditioning procedure

For a given test analytical sample S from cohort C:

**Step 1: Select warm-up QC injections**
- Take the first N QC injections from cohort C (N varies: 0, 2, 4, 6, 8, 10 per config.yaml)
- These are ordered by their injection sequence (sample label order in the Excel file)

**Step 2: Build the conditioning sequence**
- For each QC injection, extract its full elution sequence (all detected features sorted by consensus RT)
- Concatenate QC sequences in injection order, separated by [EOS][BOS] boundaries
- This creates a long conditioning prefix

**Step 3: Run inference on the analytical sample**
- For the LSTM: Feed the QC conditioning sequence through the model to initialize the hidden state. Then run inference on the analytical sample's sequence using this warmed-up hidden state. The LSTM hidden state naturally carries forward from QC context into analytical predictions.
- For the Transformer: Prepend the last K tokens of the QC conditioning sequence to each analytical sample's sliding window (extending the effective context). Since the Transformer has a fixed context window (64 tokens), we replace the first M positions with QC context tokens and use the remaining 64-M positions for the analytical sample's own history.

**Step 4: Predict next-m/z-bin at each position in the analytical sample**
- Evaluate predictions using the same sliding-window approach as standard evaluation

### 3.3 LSTM conditioning (detailed)

The LSTM's hidden state `(h, c)` naturally carries information across time steps. The warm-up procedure:

```
1. Initialize (h, c) = zeros
2. For each QC injection q in [q1, q2, ..., qN]:
     Feed all tokens of q through the LSTM (teacher-forcing with ground truth)
     Carry (h, c) forward — do NOT reset between QC injections
     Optionally insert [EOS][BOS] tokens at injection boundaries
3. Now (h, c) encodes a summary of N QC injection patterns
4. Feed the analytical sample's tokens through the LSTM starting from this (h, c)
5. At each position, record the predicted next-m/z-bin distribution
```

This is analogous to how a language model can be "prompted" — the QC injections are the prompt, the analytical sample is where we measure performance.

**Key detail:** During conditioning, we use teacher forcing (feed ground-truth tokens, not predictions). We are not asking the model to predict QC features — we are using QC features to build context.

### 3.4 Transformer conditioning (detailed)

The Transformer sees a fixed-length context window. Two approaches:

**(A) Extended context (preferred):** Temporarily extend the context window by prepending QC tokens.
- Take the last 32 tokens from the final QC injection
- Prepend them to the analytical sample's 64-token window → 96-token effective context
- Requires extending positional embeddings (or using relative positions)
- Problem: model was trained with context_length=64; 96 is out-of-distribution

**(B) Context injection:** Replace early positions in the 64-token window with QC tokens.
- Take the last M tokens from the final QC injection (M = 16, 32)
- Use them as the first M tokens of the context window
- Remaining 64-M tokens come from the analytical sample's own history
- Problem: analytical sample has less of its own history in context

**Recommendation:** Approach B with M=16 (25% QC context). This stays within the trained context length. The analytical sample still has 48 tokens of its own history, which is substantial.

**Alternative for Transformer:** Process all QC tokens through the Transformer to get a "summary embedding" from the last position's output, then add this as a bias to the analytical sample's embeddings. This avoids context length issues entirely but requires a small architecture modification.

---

## 4. Controls and Comparisons

### 4.1 Primary comparison: warm-up dose-response

| Condition | N QC injections | Description |
|-----------|----------------|-------------|
| Cold start | 0 | Standard inference, no conditioning (baseline) |
| Minimal warm-up | 2 | Minimal conditioning |
| Moderate warm-up | 4 | Moderate conditioning |
| Standard warm-up | 6 | Approaching standard practice |
| Full warm-up | 8 | Standard metabolomics QC protocol |
| Maximum warm-up | 10 | Maximum available (some cohorts have fewer) |

**Expected result:** Monotonically increasing (or plateau) accuracy with more QC injections. A plateau at 6-8 QCs would match metabolomics best practice.

**Note:** GVHD has only 6 QC injections, so conditions with N > 6 use all available QCs for that cohort.

### 4.2 Control: analytical-sample warm-up

To verify that improvement comes from QC *specifically* (not just any prior context):
- Replace QC conditioning with N random analytical samples from the same cohort
- If QC warm-up outperforms analytical warm-up, QC samples provide unique value (more reproducible detection patterns)
- If they're equivalent, the benefit is simply from seeing more data from the same batch

### 4.3 Control: cross-cohort QC warm-up

To test whether QC conditioning is cohort-specific:
- Condition with N QC injections from a *different* cohort
- If within-cohort QC warm-up outperforms cross-cohort, the model learns cohort-specific patterns (column condition, mobile phase batch)
- If they're equivalent, the model learns general elution priors

### 4.4 Control: shuffled QC warm-up

To test whether QC *sequence order* matters:
- Shuffle the token order within QC injections before conditioning
- If shuffled QC performs worse than ordered QC, the sequential structure is important
- If equivalent, the model only uses the feature presence/absence signal, not the ordering

---

## 5. Metrics

### 5.1 Primary metrics (same as Phase 3)

All measured on analytical test samples only:

| Metric | What it measures |
|--------|-----------------|
| **Top-1 accuracy** (next m/z bin, 110 classes) | Primary performance metric |
| **Top-5 accuracy** | Acquisition-relevant: can we cover the correct answer in 5 guesses? |
| **MAE in Da** | When wrong, how far off? |
| **Cross-entropy loss** | Calibration of the probability distribution |

### 5.2 Stratified analysis

Report metrics broken down by:
1. **Cohort** — does warm-up help more in some cohorts?
2. **Position in sequence** — does warm-up help more at the beginning of the analytical sample (where cold-start is worst)?
3. **Transition vs. same-bin** — does warm-up help specifically at RT bin transitions?
4. **Feature detection frequency** — does warm-up help more for rare features (detected in few QCs) vs. common features?

### 5.3 Warm-up curve

Plot: X-axis = number of QC injections (0-10), Y-axis = top-1 accuracy on analytical test samples. Include error bars across cohorts. This is the key figure for the paper.

### 5.4 Position-dependent improvement

Plot: X-axis = position in analytical sample sequence (0 to ~4000), Y-axis = accuracy improvement over cold start. Hypothesis: improvement is largest at early positions (where cold-start has minimal context) and diminishes as the analytical sample builds its own context.

---

## 6. Handling Consensus RT

### The problem
MS-DIAL alignment means all samples share the same feature-level RT. The elution "sequence" within a sample is therefore determined by:
1. Which features are detected (presence/absence based on intensity > 0)
2. Their consensus RT order (fixed)

Two samples from the same cohort differ only in which features drop out (below detection threshold) — not in the ordering of detected features.

### Implications for the experiment
1. **The model does NOT learn RT drift** from QC conditioning — there is no per-sample RT to learn
2. **What the model CAN learn from QC conditioning:**
   - The **feature vocabulary** of this batch (which m/z bins are populated)
   - The **detection density** at each RT region (how many features per time bin)
   - **Intensity-dependent detection patterns** (features that are borderline-detectable show up in some QC injections but not others)
3. **The sequence is essentially a feature checklist** with a fixed order — QC conditioning tells the model "here are the features this instrument/method/column detects, and here's their intensity profile"

### Is this still interesting?
**Yes, for two reasons:**
1. **Practical relevance:** In real LC-MS acquisition, you DO run QC injections first. If the model can extract useful priors from aligned QC data (even with consensus RT), that's directly applicable. Per-injection RT data would make it stronger, but this is the data format people actually have.
2. **Detection pattern learning:** Even without per-sample RT, knowing which features are detected in QC samples narrows the prediction space. If the model sees that features in m/z bins 45-55 are consistently detected in QC injections, it should assign higher probability to those bins in analytical samples.

### What we explicitly DO NOT claim
- We do NOT claim the model learns RT drift or correction from QC conditioning
- We DO NOT claim per-injection RT modeling (that requires raw peak tables, not aligned data)
- We DO claim the model learns batch-specific detection priors from QC conditioning

---

## 7. Minimum Code Required

### What we can reuse (no changes needed)
- `src/preprocessing.py` — data loading (already separates QC vs. analytical via `sample_type`)
- `src/tokenization.py` — tokenization (works on any sample)
- `src/config.py` — constants
- `src/models.py` — LSTM and Transformer architectures (forward pass unchanged)
- `src/train.py` — training loop (if retraining without QC)

### What needs modification
1. **`src/datasets.py`** — add a function to:
   - Exclude QC samples from train/val/test splits
   - Build separate QC sample arrays per cohort for conditioning
   - Build "conditioned" test examples that include QC prefix

2. **New: `src/qc_warmup.py`** — the experiment script:
   - Load trained model checkpoint
   - For each N in [0, 2, 4, 6, 8, 10]:
     - For each test analytical sample:
       - Build QC conditioning sequence (first N QCs from same cohort)
       - Run LSTM with warm-up procedure (feed QC → carry hidden state → evaluate analytical)
       - Record per-position predictions
     - Aggregate metrics
   - Run control experiments (analytical warm-up, cross-cohort, shuffled)
   - Save results to `outputs/metrics/qc_warmup_results.json`
   - Generate warm-up curve figure

### Estimated scope
- `datasets.py` modifications: ~50 lines
- `qc_warmup.py` new script: ~300-400 lines
- Can run locally (CPU) — inference only, no GPU needed
- Estimated runtime: ~10-20 minutes for all conditions (342 samples x 6 N-values x 4 controls)

---

## 8. Expected Outcomes and Interpretation

### If warm-up helps significantly (>2% top-1 improvement at N=8):
- **Paper story:** "QC conditioning acts as in-context learning — the model adapts its predictions to the specific instrument configuration and batch by observing reproducible QC patterns before analytical samples"
- **Grant story:** Directly supports SA1.3 (closed-loop optimization) — the system uses early QC runs to calibrate itself

### If warm-up helps only at early positions:
- **Paper story:** "QC conditioning accelerates convergence — the model reaches peak performance faster in the analytical sequence because it starts with an informed hidden state rather than zeros"
- **Grant story:** Still valuable — reduces the number of analytical samples needed before the model is effective

### If warm-up doesn't help:
- **Paper story:** "The 64-token sliding window provides sufficient local context; batch-level information from QC injections is redundant once the model has seen ~64 features from the current sample"
- **Grant story:** This is actually fine — it means the model generalizes without needing batch-specific conditioning, which is BETTER for a deployed system (no warm-up required)
- **Still publishable:** A negative result is informative — it tells us the model's predictions are driven by local sequence structure, not batch-level patterns

### If analytical warm-up matches QC warm-up:
- **Interpretation:** The benefit is from seeing more data from the same batch, not from QC-specific reproducibility. QC samples aren't special for conditioning — any same-batch data works.

---

## 9. Open Questions for PI Review

1. **Option A vs B for training data?** Retrain without QC samples (cleaner but costs Colab time) vs. use existing model (faster but less clean)?

2. **LSTM vs Transformer priority?** The LSTM hidden-state warm-up is more natural and easier to implement. The Transformer context injection is more awkward. Should we do LSTM-only for Phase 4 and add Transformer if results are promising?

3. **Cohort-level or pooled analysis?** Each cohort has different numbers of QC injections (6-10). Should we report results per-cohort (more granular, smaller N) or pooled across cohorts (more power, averages over cohort differences)?

4. **Is the consensus-RT limitation worth discussing explicitly in the paper?** It's scientifically honest but might give reviewers an easy target ("you didn't test with real per-injection RT data").

5. **Should we include a "simulated RT jitter" experiment?** Add random per-sample RT noise (e.g., uniform ±5s) to break the consensus alignment and test whether the model is robust to realistic RT variability. This would partially address the consensus-RT limitation.
