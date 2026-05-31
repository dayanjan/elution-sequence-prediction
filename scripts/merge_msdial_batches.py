"""Merge MS-DIAL alignment results from 3 batches into a unified feature table.

Matches features across batches by m/z + RT proximity using MS-DIAL's own
alignment tolerances (0.025 Da m/z, 3.0 s RT = 0.05 min).

Outputs:
  - merged_features.parquet: unified feature table (feature_id, mz, rt_sec,
    adduct, per-sample abundances for all 153 samples)
  - merge_report.txt: statistics on matching

Usage:
    .venv/Scripts/python scripts/merge_msdial_batches.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# === Paths ===
ST000990_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "ST000990"
BATCH_DIRS = {
    "batch1": ST000990_DIR / "msdial_output_subset" / "results",
    "batch2": ST000990_DIR / "msdial_output_batch2" / "results",
    "batch3": ST000990_DIR / "msdial_output_batch3" / "results",
}
OUTPUT_DIR = ST000990_DIR / "msdial_merged"

# Matching tolerances (same as MS-DIAL alignment params)
MZ_TOL = 0.025   # Da
RT_TOL = 3.0     # seconds


def parse_mztabm(mztabm_path):
    """Parse mzTabM file, return feature table (SMF) and sample mapping."""
    smf_rows = []
    sml_rows = []
    sfh_header = None
    smh_header = None
    assay_map = {}  # assay[N] -> sample name

    with open(mztabm_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t")
            if parts[0] == "MTD" and "assay[" in parts[1] and "-ms_run_ref" not in parts[1]:
                # e.g., "assay[1]\tGLA_TT6_Lipids_NIST1"
                idx = parts[1]  # "assay[1]"
                assay_map[idx] = parts[2]
            elif parts[0] == "SFH":
                sfh_header = parts
            elif parts[0] == "SMF":
                smf_rows.append(parts)
            elif parts[0] == "SMH":
                smh_header = parts
            elif parts[0] == "SML":
                sml_rows.append(parts)

    # Build SMF DataFrame
    smf_df = pd.DataFrame(smf_rows, columns=sfh_header)
    smf_df["exp_mass_to_charge"] = pd.to_numeric(smf_df["exp_mass_to_charge"])
    smf_df["retention_time_in_seconds"] = pd.to_numeric(smf_df["retention_time_in_seconds"])

    # Extract per-sample abundances
    abundance_cols = [c for c in smf_df.columns if c.startswith("abundance_assay")]
    for col in abundance_cols:
        smf_df[col] = pd.to_numeric(smf_df[col], errors="coerce")

    # Map abundance columns to sample names
    rename = {}
    for col in abundance_cols:
        # "abundance_assay[1]" -> "assay[1]"
        assay_key = col.replace("abundance_", "")
        if assay_key in assay_map:
            rename[col] = assay_map[assay_key]

    smf_df = smf_df.rename(columns=rename)

    # Also get annotations from SML
    sml_df = pd.DataFrame(sml_rows, columns=smh_header) if sml_rows and smh_header else None

    return smf_df, sml_df, assay_map


def match_features(ref_mz, ref_rt, query_mz, query_rt, mz_tol, rt_tol):
    """Match query features to reference features by m/z + RT proximity.

    Returns array of reference indices for each query feature (-1 if no match).
    Uses a KD-tree on normalized coordinates for efficiency.
    """
    # Normalize: scale RT so that rt_tol maps to same distance as mz_tol
    rt_scale = mz_tol / rt_tol

    ref_coords = np.column_stack([ref_mz, ref_rt * rt_scale])
    query_coords = np.column_stack([query_mz, query_rt * rt_scale])

    tree = cKDTree(ref_coords)
    # Search within mz_tol (since RT is scaled to same units)
    distances, indices = tree.query(query_coords, k=1, distance_upper_bound=mz_tol)

    # Verify both tolerances are met individually
    matches = np.full(len(query_mz), -1, dtype=int)
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if idx < len(ref_mz):  # valid match found
            if (abs(query_mz[i] - ref_mz[idx]) <= mz_tol and
                abs(query_rt[i] - ref_rt[idx]) <= rt_tol):
                matches[i] = idx

    return matches


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Merging MS-DIAL Batches")
    print("=" * 60)

    # Parse all batches
    batch_data = {}
    for name, results_dir in BATCH_DIRS.items():
        mztabm_files = list(results_dir.glob("*.mzTabM"))
        if not mztabm_files:
            print(f"WARNING: No mzTabM in {results_dir}")
            continue
        print(f"\nParsing {name}: {mztabm_files[0].name}")
        smf_df, sml_df, assay_map = parse_mztabm(mztabm_files[0])
        batch_data[name] = {
            "smf": smf_df,
            "sml": sml_df,
            "assay_map": assay_map,
            "n_features": len(smf_df),
            "n_samples": len(assay_map),
        }
        print(f"  Features: {len(smf_df)}, Samples: {len(assay_map)}")

    # Use batch1 as reference (has NIST + QC + Rep + analytical)
    ref = batch_data["batch1"]["smf"]
    ref_mz = ref["exp_mass_to_charge"].values
    ref_rt = ref["retention_time_in_seconds"].values

    # Get sample columns from batch1
    sample_cols_b1 = [c for c in ref.columns
                      if c.startswith("GLA_TT6_Lipids_")]

    print(f"\n{'='*60}")
    print(f"Reference (batch1): {len(ref)} features, {len(sample_cols_b1)} samples")
    print(f"Matching tolerance: {MZ_TOL} Da, {RT_TOL} s")
    print(f"{'='*60}")

    # Initialize merged table with batch1 features
    merged = ref[["SMF_ID", "exp_mass_to_charge", "retention_time_in_seconds",
                   "adduct_ion"] + sample_cols_b1].copy()
    merged = merged.rename(columns={
        "exp_mass_to_charge": "mz",
        "retention_time_in_seconds": "rt_sec",
    })

    # Track new features from other batches
    new_features = []
    report_lines = []

    for batch_name in ["batch2", "batch3"]:
        if batch_name not in batch_data:
            continue

        bdata = batch_data[batch_name]["smf"]
        query_mz = bdata["exp_mass_to_charge"].values
        query_rt = bdata["retention_time_in_seconds"].values

        # Get sample columns
        sample_cols = [c for c in bdata.columns if c.startswith("GLA_TT6_Lipids_")]

        # Match to reference
        matches = match_features(ref_mz, ref_rt, query_mz, query_rt, MZ_TOL, RT_TOL)

        n_matched = np.sum(matches >= 0)
        n_new = np.sum(matches < 0)
        print(f"\n{batch_name}: {len(query_mz)} features -> {n_matched} matched, {n_new} new")
        report_lines.append(f"{batch_name}: {len(query_mz)} features -> {n_matched} matched, {n_new} new")

        # Add matched sample abundances to merged table
        for col in sample_cols:
            merged[col] = np.nan  # initialize

        for i, ref_idx in enumerate(matches):
            if ref_idx >= 0:
                for col in sample_cols:
                    merged.loc[merged.index[ref_idx], col] = bdata.iloc[i][col]

        # Collect unmatched features as new rows
        unmatched_mask = matches < 0
        if n_new > 0:
            new_rows = bdata.loc[unmatched_mask, ["exp_mass_to_charge",
                                                    "retention_time_in_seconds",
                                                    "adduct_ion"] + sample_cols].copy()
            new_rows = new_rows.rename(columns={
                "exp_mass_to_charge": "mz",
                "retention_time_in_seconds": "rt_sec",
            })
            new_features.append(new_rows)

    # Append new features
    if new_features:
        new_df = pd.concat(new_features, ignore_index=True)

        # Deduplicate new features across batches 2 and 3
        if len(new_features) > 1:
            # Simple: keep all for now (they're from different batches with different samples)
            pass

        # Assign new feature IDs
        max_id = merged["SMF_ID"].astype(int).max()
        new_df["SMF_ID"] = range(max_id + 1, max_id + 1 + len(new_df))

        merged = pd.concat([merged, new_df], ignore_index=True)
        print(f"\nAdded {len(new_df)} new features from batches 2+3")
        report_lines.append(f"New features added: {len(new_df)}")

    # Sort by RT
    merged = merged.sort_values("rt_sec").reset_index(drop=True)
    merged["feature_id"] = range(len(merged))
    # Drop the original SMF_ID (mixed types) — feature_id replaces it
    if "SMF_ID" in merged.columns:
        merged = merged.drop(columns=["SMF_ID"])
    # Defragment the DataFrame
    merged = merged.copy()

    # Count sample columns and detection rates
    all_sample_cols = [c for c in merged.columns if c.startswith("GLA_TT6_Lipids_")]
    detection_rate = merged[all_sample_cols].notna().sum(axis=1).mean() / len(all_sample_cols)

    print(f"\n{'='*60}")
    print(f"MERGED RESULT")
    print(f"  Total features: {len(merged)}")
    print(f"  Total samples:  {len(all_sample_cols)}")
    print(f"  m/z range:      {merged['mz'].min():.2f} – {merged['mz'].max():.2f}")
    print(f"  RT range:       {merged['rt_sec'].min():.1f} – {merged['rt_sec'].max():.1f} s")
    print(f"  Mean detection: {detection_rate:.1%} of samples per feature")
    print(f"{'='*60}")

    # Save
    out_parquet = OUTPUT_DIR / "merged_features.parquet"
    merged.to_parquet(out_parquet, index=False)
    print(f"\nSaved: {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB)")

    # Also save a smaller summary CSV (no per-sample abundances)
    summary = merged[["feature_id", "mz", "rt_sec", "adduct_ion"]].copy()
    summary["n_detected"] = merged[all_sample_cols].notna().sum(axis=1)
    summary["mean_abundance"] = merged[all_sample_cols].mean(axis=1)
    summary_path = OUTPUT_DIR / "merged_features_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Save report
    report_path = OUTPUT_DIR / "merge_report.txt"
    with open(report_path, "w") as f:
        f.write("MS-DIAL Batch Merge Report\n")
        f.write(f"{'='*40}\n")
        f.write(f"Tolerances: {MZ_TOL} Da, {RT_TOL} s\n\n")
        for name, bdata in batch_data.items():
            f.write(f"{name}: {bdata['n_features']} features, {bdata['n_samples']} samples\n")
        f.write(f"\n")
        for line in report_lines:
            f.write(f"{line}\n")
        f.write(f"\nMerged: {len(merged)} features, {len(all_sample_cols)} samples\n")
        f.write(f"Mean detection rate: {detection_rate:.1%}\n")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
