"""Process ST003514 (NIST SRM 1950) into the same long-format as our 4 cohorts.

Reads:
  - data/external/st003514_metabolites.csv  (feature metadata: name, RT, m/z, ion_mode)
  - data/external/st003514_data_pos.csv     (sample x metabolite intensity matrix, pos mode)
  - data/external/st003514_data_neg.csv     (sample x metabolite intensity matrix, neg mode)
  - data/external/st003514_samples.csv      (sample metadata)

Outputs:
  - data/external/st003514_long.parquet     (unified long-format DataFrame matching preprocessing.py schema)

Key differences from our 4 cohorts:
  - Instrument: Agilent 6545 QTOF (ours: SCIEX 6600+)
  - Column: different C18 (ours: Waters CSH C18)
  - Alignment: different MS-DIAL version and settings
  - All samples are NIST SRM 1950 pooled plasma (no disease cohort variation)
"""

import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "external"


def load_metabolite_metadata():
    """Load metabolite list with RT, m/z, ion mode."""
    df = pd.read_csv(DATA_DIR / "st003514_metabolites.csv")
    print(f"  Metabolite metadata: {len(df)} entries")
    print(f"    Pos: {(df.ion_mode == 'pos').sum()}, Neg: {(df.ion_mode == 'neg').sum()}")
    return df


def load_intensity_matrix(ion_mode):
    """Load sample x metabolite intensity matrix for one ion mode."""
    path = DATA_DIR / f"st003514_data_{ion_mode}.csv"
    df = pd.read_csv(path)
    print(f"  {ion_mode.upper()} intensity matrix: {df.shape[0]} samples x {df.shape[1] - 2} metabolites")
    return df


def parse_lipid_name(name):
    """Parse ST003514 lipid name into class, carbons, unsaturation.

    ST003514 uses format like: PC(16:0/18:1), TG(52:3), Cer(d18:1/16:0)
    """
    if not name or not isinstance(name, str):
        return None, None, None

    # Extract class (everything before first parenthesis)
    class_match = re.match(r"^([A-Za-z\-]+)", name)
    lipid_class = class_match.group(1) if class_match else None

    # Try summary notation: CLASS(C:U)
    summary = re.search(r"\((\d+):(\d+)\)", name)
    if summary:
        return lipid_class, int(summary.group(1)), int(summary.group(2))

    # Try sn-position or sphingoid base notation: (d18:1/16:0), (16:0_18:1)
    sn_matches = re.findall(r"[d]?(\d+):(\d+)", name)
    if sn_matches:
        total_c = sum(int(m[0]) for m in sn_matches)
        total_u = sum(int(m[1]) for m in sn_matches)
        return lipid_class, total_c, total_u

    return lipid_class, None, None


def melt_intensity_matrix(intensity_df, metabolite_meta, ion_mode):
    """Convert wide intensity matrix to long format, merging in metabolite metadata."""
    # Columns: Samples, Class, then metabolite names
    sample_col = intensity_df.columns[0]  # "Samples"
    class_col = intensity_df.columns[1]   # "Class"
    metabolite_cols = intensity_df.columns[2:]

    # Build sample metadata
    sample_info = intensity_df[[sample_col, class_col]].copy()
    sample_info.columns = ["sample_id", "sample_class"]

    # Determine sample_type (QC-like = NIST pooled, analytical = others)
    # In ST003514: "Factor:NIST" samples are pooled QC, rest are analytical
    sample_info["sample_type"] = sample_info["sample_class"].apply(
        lambda x: "QC" if "NIST" in str(x) else "analytical"
    )

    # Melt to long format
    melted = intensity_df.melt(
        id_vars=[sample_col, class_col],
        value_vars=metabolite_cols,
        var_name="metabolite_name",
        value_name="intensity",
    )
    melted.columns = ["sample_id", "sample_class", "metabolite_name", "intensity"]

    # Merge sample type
    melted["sample_type"] = melted["sample_class"].apply(
        lambda x: "QC" if "NIST" in str(x) else "analytical"
    )

    # Look up RT and m/z from metabolite metadata
    # Match by name and ion_mode
    meta_mode = metabolite_meta[metabolite_meta.ion_mode == ion_mode].copy()

    # The intensity matrix metabolite names may not exactly match the metadata names
    # Try direct match first
    meta_lookup = meta_mode.set_index("name")[["rt", "mz", "refmet_name", "inchi_key"]].to_dict("index")

    rts, mzs, annotations, inchi_keys = [], [], [], []
    for name in melted["metabolite_name"]:
        info = meta_lookup.get(name, {})
        rts.append(info.get("rt"))
        mzs.append(info.get("mz"))
        annotations.append(info.get("refmet_name", name))
        inchi_keys.append(info.get("inchi_key"))

    melted["rt"] = rts
    melted["mz"] = mzs
    melted["annotation"] = annotations
    melted["inchi_key"] = inchi_keys
    melted["ion_mode"] = ion_mode

    return melted


def main():
    print("=" * 60)
    print("Processing ST003514 into unified long format")
    print("=" * 60)

    # Load metadata
    print("\n1. Loading metabolite metadata...")
    meta = load_metabolite_metadata()

    # Load intensity matrices
    print("\n2. Loading intensity matrices...")
    pos_df = load_intensity_matrix("pos")
    neg_df = load_intensity_matrix("neg")

    # Melt to long format
    print("\n3. Converting to long format...")
    pos_long = melt_intensity_matrix(pos_df, meta, "pos")
    neg_long = melt_intensity_matrix(neg_df, meta, "neg")
    print(f"  Pos long: {len(pos_long)} rows")
    print(f"  Neg long: {len(neg_long)} rows")

    # Combine
    combined = pd.concat([pos_long, neg_long], ignore_index=True)

    # Parse lipid names
    print("\n4. Parsing lipid names...")
    parsed = combined["metabolite_name"].apply(parse_lipid_name)
    combined["lipid_class"] = [p[0] for p in parsed]
    combined["total_carbons"] = [p[1] for p in parsed]
    combined["total_unsat"] = [p[2] for p in parsed]

    # Map to preprocessing.py schema
    print("\n5. Mapping to unified schema...")
    unified = pd.DataFrame({
        "study": "st003514",
        "sample_id": combined["sample_id"],
        "sample_type": combined["sample_type"],
        "polarity": combined["ion_mode"].map({"pos": "(+) ESI", "neg": "(-) ESI"}),
        "feature_id": combined["metabolite_name"],
        "annotation": combined["annotation"],
        "lipid_class": combined["lipid_class"],
        "total_carbons": combined["total_carbons"],
        "total_unsat": combined["total_unsat"],
        "inchi_key": combined["inchi_key"],
        "adduct": None,
        "mz": combined["mz"],
        "rt": combined["rt"],
        "intensity": combined["intensity"].fillna(0).astype(float),
        "is_istd": combined["metabolite_name"].str.contains(r"\(d\d+\)", na=False)
                   & combined["metabolite_name"].str.contains("d7|d9", na=False),
    })

    # Filter to rows with valid RT and m/z
    valid = unified.mz.notna() & unified.rt.notna()
    print(f"  Valid RT+m/z: {valid.sum()}/{len(unified)} ({valid.mean()*100:.1f}%)")
    unified_valid = unified[valid].copy()

    # Filter to detected features (intensity > 0)
    detected = unified_valid[unified_valid.intensity > 0]
    print(f"  Detected (intensity > 0): {len(detected)}/{len(unified_valid)}")

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"  Samples: {detected.sample_id.nunique()}")
    print(f"  Features: {detected.feature_id.nunique()}")
    print(f"  QC samples: {(detected.drop_duplicates('sample_id').sample_type == 'QC').sum()}")
    print(f"  Analytical samples: {(detected.drop_duplicates('sample_id').sample_type == 'analytical').sum()}")
    print(f"  Lipid classes: {detected.lipid_class.nunique()}")
    print(f"  RT range: {detected.rt.min():.2f} - {detected.rt.max():.2f} min")
    print(f"  m/z range: {detected.mz.min():.1f} - {detected.mz.max():.1f}")
    print(f"\n  Lipid class distribution:")
    for cls, count in detected.drop_duplicates("feature_id").lipid_class.value_counts().head(15).items():
        print(f"    {cls:<15} {count:>4}")

    # Save
    out_path = DATA_DIR / "st003514_long.parquet"
    unified_valid.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  Shape: {unified_valid.shape}")

    return unified_valid


if __name__ == "__main__":
    main()
