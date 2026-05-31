"""
Preprocessing: Load all 4 West Coast lipidomics datasets into a unified long-format DataFrame.

Handles column name differences across datasets:
- Cardiac Arrest: 'Annotation', separate pos/neg sheets, no 'ESI mode' column
- GVHD: 'Annotation', combined sheet, has 'ESI mode'
- PCOS: 'Annotation', combined sheet, has 'ESI mode'
- REDHART 2: 'Metabolite name' (not 'Annotation'), combined sheet, has 'ESI mode'

Output columns:
  study, sample_id, sample_type, polarity, feature_id, annotation, inchi_key,
  adduct, mz, rt, intensity, is_istd
"""

import re
from pathlib import Path

import openpyxl
import pandas as pd


DATASETS_ROOT = Path(__file__).parent.parent.parent.parent / "datasets"

DATASET_FILES = {
    "cardiac_arrest": {
        "path": DATASETS_ROOT / "cardiac_arrest_rplc" / "mx 292009 Warncke_CSH-QTOF MSMS_lipidomics_01-2017_submit.xlsx",
        "sheets": {
            "CSH-posESI QTOF MSMS": "(+) ESI",
            "CSH-negative ESI-QTOF MSMS": "(-) ESI",
        },
        "ann_col": "Annotation",
        "meta_cols_end": 7,  # columns 0-6 are metadata, 7+ are samples
    },
    "gvhd": {
        "path": DATASETS_ROOT / "gvhd_rplc" / "mx 322917_Warncke_Project_1_CSH-QTOF MS_lipidomics_05-2017_submit.xlsx",
        "sheets": {"Submit": None},  # polarity from ESI mode column
        "ann_col": "Annotation",
        "meta_cols_end": 8,
    },
    "pcos": {
        "path": DATASETS_ROOT / "pcos_rplc" / "864 PCOS - mx 323067_864_LCMS_project 2_CSH-QTOF MS_lipidomics_06-2017_submit.xlsx",
        "sheets": {"Submit": None},
        "ann_col": "Annotation",
        "meta_cols_end": 8,
    },
    "redhart2": {
        "path": DATASETS_ROOT / "redhart2_rplc" / "mx 323914_Warncke_lipidomics_CSH-QTOF MS_project #5_05-2017_submit.xlsx.xlsx",
        "sheets": {"Submit": None},
        "ann_col": "Metabolite name",
        "meta_cols_end": 8,
    },
}


def parse_lipid_name(annotation: str) -> dict:
    """Parse lipid annotation to extract class, total carbons, unsaturation.

    Examples:
        'PC (34:2)' -> {'lipid_class': 'PC', 'total_carbons': 34, 'total_unsat': 2}
        'TG (52:3)' -> {'lipid_class': 'TG', 'total_carbons': 52, 'total_unsat': 3}
        'CE (22:1) iSTD' -> {'lipid_class': 'CE', 'total_carbons': 22, 'total_unsat': 1}
        'Ceramide (d18:1/16:0)' -> {'lipid_class': 'Ceramide', 'total_carbons': 34, 'total_unsat': 1}
    """
    if not annotation or not isinstance(annotation, str):
        return {"lipid_class": None, "total_carbons": None, "total_unsat": None}

    # Strip leading number prefix (e.g., "1_CE (22:1) iSTD" -> "CE (22:1) iSTD")
    ann = re.sub(r"^\d+_", "", annotation.strip())

    # Extract class name (everything before the first parenthesis or space+digit)
    class_match = re.match(r"([A-Za-z\-]+(?:\s[A-Za-z]+)?)", ann)
    lipid_class = class_match.group(1).strip() if class_match else None

    # Try summary notation: CLASS (C:U)
    summary = re.search(r"\((\d+):(\d+)\)", ann)
    if summary:
        return {
            "lipid_class": lipid_class,
            "total_carbons": int(summary.group(1)),
            "total_unsat": int(summary.group(2)),
        }

    # Try sn-position notation: CLASS (C1:U1/C2:U2) or (d18:1/16:0)
    sn_matches = re.findall(r"[d]?(\d+):(\d+)", ann)
    if sn_matches:
        total_c = sum(int(m[0]) for m in sn_matches)
        total_u = sum(int(m[1]) for m in sn_matches)
        return {
            "lipid_class": lipid_class,
            "total_carbons": total_c,
            "total_unsat": total_u,
        }

    return {"lipid_class": lipid_class, "total_carbons": None, "total_unsat": None}


def load_dataset(study: str) -> pd.DataFrame:
    """Load a single dataset into long-format DataFrame."""
    info = DATASET_FILES[study]
    wb = openpyxl.load_workbook(str(info["path"]), read_only=True)
    all_records = []

    for sheet_name, forced_polarity in info["sheets"].items():
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        # Find header row (contains 'Identifier')
        header_idx = None
        for i, row in enumerate(rows):
            if row and any(str(c).strip() == "Identifier" for c in row if c):
                header_idx = i
                break
        if header_idx is None:
            continue

        header = [str(c).strip() if c else "" for c in rows[header_idx]]

        # Build column index map
        col_map = {}
        for j, h in enumerate(header):
            if h and h not in col_map:
                col_map[h] = j

        ann_col_idx = col_map.get(info["ann_col"], col_map.get("Annotation", 1))
        inchi_col_idx = col_map.get("InChI Key", None)
        adduct_col_idx = col_map.get("Species", None)
        mz_col_idx = col_map.get("m/z", None)
        rt_col_idx = col_map.get("RT", None)
        esi_col_idx = col_map.get("ESI mode", None)

        # Sample metadata from rows 0 (Label) and 1 (Sample #)
        label_row = rows[0]
        sample_type_row = rows[1]
        meta_end = info["meta_cols_end"]

        # Build sample info for each data column
        sample_cols = []
        for j in range(meta_end, len(label_row) if label_row else 0):
            label = str(label_row[j]).strip() if label_row[j] else f"sample_{j}"
            stype_raw = str(sample_type_row[j]).strip() if j < len(sample_type_row) and sample_type_row[j] else ""
            stype = "QC" if stype_raw == "QC" else "analytical"
            sample_cols.append((j, label, stype))

        # Extract feature data
        for row in rows[header_idx + 1:]:
            if not row or not row[0]:
                continue

            annotation = str(row[ann_col_idx]).strip() if ann_col_idx < len(row) and row[ann_col_idx] else ""
            inchi = str(row[inchi_col_idx]).strip() if inchi_col_idx and inchi_col_idx < len(row) and row[inchi_col_idx] else ""
            adduct = str(row[adduct_col_idx]).strip() if adduct_col_idx and adduct_col_idx < len(row) and row[adduct_col_idx] else ""

            # m/z: may have multiple values separated by _, take first
            mz_raw = str(row[mz_col_idx]) if mz_col_idx and mz_col_idx < len(row) and row[mz_col_idx] else ""
            try:
                mz = float(mz_raw.split("_")[0].strip())
            except (ValueError, IndexError):
                mz = None

            # RT: may have multiple values, take first
            rt_raw = str(row[rt_col_idx]) if rt_col_idx and rt_col_idx < len(row) and row[rt_col_idx] else ""
            try:
                rt = float(rt_raw.split("_")[0].strip())
            except (ValueError, IndexError):
                rt = None

            # Polarity
            if forced_polarity:
                polarity = forced_polarity
            elif esi_col_idx and esi_col_idx < len(row) and row[esi_col_idx]:
                pol_str = str(row[esi_col_idx]).strip()
                polarity = "(+) ESI" if "+" in pol_str or "pos" in pol_str.lower() else "(-) ESI"
            else:
                polarity = "unknown"

            is_istd = "iSTD" in annotation if annotation else False
            feature_id = str(row[0]).strip() if row[0] else ""

            # Parse lipid name
            parsed = parse_lipid_name(annotation)

            # Per-sample intensities
            for col_idx, sample_label, sample_type in sample_cols:
                intensity = row[col_idx] if col_idx < len(row) else None
                try:
                    intensity = float(intensity) if intensity else 0.0
                except (ValueError, TypeError):
                    intensity = 0.0

                all_records.append({
                    "study": study,
                    "sample_id": sample_label,
                    "sample_type": sample_type,
                    "polarity": polarity,
                    "feature_id": feature_id,
                    "annotation": annotation if annotation else None,
                    "lipid_class": parsed["lipid_class"],
                    "total_carbons": parsed["total_carbons"],
                    "total_unsat": parsed["total_unsat"],
                    "inchi_key": inchi if inchi else None,
                    "adduct": adduct if adduct else None,
                    "mz": mz,
                    "rt": rt,
                    "intensity": intensity,
                    "is_istd": is_istd,
                })

    wb.close()
    return pd.DataFrame(all_records)


def load_all_datasets() -> pd.DataFrame:
    """Load all 4 datasets into a single unified DataFrame."""
    dfs = []
    for study in DATASET_FILES:
        print(f"Loading {study}...")
        df = load_dataset(study)
        print(f"  {len(df)} records ({df['feature_id'].nunique()} features x {df['sample_id'].nunique()} samples)")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} records")
    return combined


if __name__ == "__main__":
    df = load_all_datasets()
    print(df.head(20))
    print(df.dtypes)
    print(df.describe())
