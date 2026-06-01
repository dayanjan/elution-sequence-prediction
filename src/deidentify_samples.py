"""
De-identify sample IDs in the clinical feature tables.

The raw vendor sample labels embed patient initials (of the form
"<n> <initials> <timepoint>"), which are potentially identifying. This replaces
every sample_id with an anonymized,
study-scoped sequential code (e.g. "redhart2_0001") so that feature tables shared
on request under the data-use agreement carry no direct identifiers.

KEY GUARANTEE -- split invariance. The model's sample-aware train/val/test split
(src/datasets.sample_aware_split) shuffles the POSITIONAL indices of the
lexicographically-sorted (study, sample_id) pairs with a fixed seed; it never keys
off the ID strings themselves. We therefore assign anonymized IDs in each study's
lexicographic sort order as zero-padded sequential codes, which re-sort into the
identical order -> the seeded shuffle yields the identical split -> every reported
metric is bit-identical (sample_id is used only for grouping/splitting, never as a
model feature/token). `verify_split_invariance` asserts this empirically before any
file is written.

The raw data under datasets/ retains the original labels and is the DUA-protected
master; the anonymization is deterministic from this script, so the mapping is
recoverable by the data custodian without persisting any PHI here.

Usage:
    python src/deidentify_samples.py --verify        # prove split invariance, no writes
    python src/deidentify_samples.py --export        # write separate *_deidentified copies
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # poc3_elution_sequence
sys.path.insert(0, str(ROOT / "src"))

# Local, gitignored, DUA-protected feature tables that carry sample_id.
TARGETS = [
    ROOT / "data/sequences/tokenized_features.parquet",
    ROOT / "data/sequences/sequence_metadata.csv",
]


def build_anon_map(sample_ids_by_study):
    """Order-preserving per-study anonymization map.

    For each study, sort the unique sample_ids lexicographically (matching
    datasets.load_and_encode's sort) and assign {study}_{NNNN} in that order, so
    the anon IDs re-sort into the identical positional order.
    """
    amap = {}
    for study, sids in sample_ids_by_study.items():
        for i, sid in enumerate(sorted(map(str, sids)), start=1):
            amap[(study, sid)] = f"{study}_{i:04d}"
    return amap


def _sample_ids_by_study(df):
    out = {}
    for study, g in df.groupby("study"):
        out[study] = sorted(g["sample_id"].astype(str).unique())
    return out


def verify_split_invariance(parquet_path):
    """Run the real sample_aware_split on original vs anonymized IDs; assert the
    train/val/test assignment is identical (mapped back to original IDs)."""
    from datasets import load_and_encode, sample_aware_split

    tok = load_and_encode(parquet_path)
    amap = build_anon_map(_sample_ids_by_study(tok))

    def split_assignment(t):
        tr, va, te = sample_aware_split(t)
        def keyset(d):
            return set(map(tuple, d[["study", "sample_id"]].drop_duplicates().values))
        return keyset(tr), keyset(va), keyset(te)

    tr0, va0, te0 = split_assignment(tok)

    tok_anon = tok.copy()
    tok_anon["sample_id"] = [amap[(s, str(sid))] for s, sid in
                             zip(tok_anon["study"], tok_anon["sample_id"])]
    tr1, va1, te1 = split_assignment(tok_anon)

    # Map the original split's IDs through amap and compare to the anon split.
    def via_map(keyset):
        return set((s, amap[(s, str(sid))]) for s, sid in keyset)

    ok = (via_map(tr0) == tr1) and (via_map(va0) == va1) and (via_map(te0) == te1)
    print(f"  original split  : train={len(tr0)} val={len(va0)} test={len(te0)}")
    print(f"  anonymized split: train={len(tr1)} val={len(va1)} test={len(te1)}")
    print(f"  SPLIT INVARIANT : {ok}")
    if not ok:
        raise SystemExit("ABORT: anonymization changes the split; do not apply.")
    return amap


def export_deidentified(amap):
    """Write a SEPARATE de-identified copy of each feature table (non-destructive).

    The canonical local files (the model pipeline's input) are left untouched; the
    `*_deidentified` copies are the artifacts to share on request under the DUA.
    Both originals and copies remain gitignored (individual-level clinical data).
    """
    for path in TARGETS:
        if not path.exists():
            print(f"  skip (missing): {path}")
            continue
        out = path.with_name(f"{path.stem}_deidentified{path.suffix}")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        before = df["sample_id"].astype(str).iloc[0]
        df["sample_id"] = [amap.get((s, str(sid)), str(sid)) for s, sid in
                           zip(df["study"], df["sample_id"])]
        after = df["sample_id"].iloc[0]
        if path.suffix == ".parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        print(f"  wrote {out.name}: e.g. {before!r} -> {after!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true",
                    help="write separate *_deidentified copies (non-destructive)")
    ap.add_argument("--verify", action="store_true", help="prove split invariance only")
    args = ap.parse_args()

    canonical = ROOT / "data/sequences/tokenized_features.parquet"
    print("Verifying split invariance on", canonical.name)
    amap = verify_split_invariance(canonical)

    if args.export:
        print("Writing de-identified copies (canonical files untouched)...")
        export_deidentified(amap)
        print("Done. Share the *_deidentified tables; both stay gitignored (DUA).")
    else:
        print("(verify-only; no files written. Re-run with --export to write copies.)")


if __name__ == "__main__":
    main()
