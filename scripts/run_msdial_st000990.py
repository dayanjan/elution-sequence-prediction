"""Process ST000990 raw .wiff files through MS-DIAL console app.

Pipeline:
1. Locate .wiff files from extracted ST000990 archive
2. Separate positive and negative ESI files (if applicable)
3. Run MS-DIAL console app (MSDIALCUI.exe) with RP-LC lipidomics parameters
4. Parse output feature tables into a unified parquet file
5. Ready for tokenization and model prediction

Usage:
    .venv/Scripts/python scripts/run_msdial_st000990.py

Prerequisites:
    - ST000990.zip extracted to data/external/ST000990/
    - MS-DIAL console installed at tools/msdial/console/
"""

import os
import sys
import subprocess
import glob
import shutil
import pandas as pd
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
MSDIAL_EXE = PROJECT_ROOT / "tools" / "msdial" / "console" / "MSDIALCUI.exe"
ST000990_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "ST000990"
PARAMS_DIR = ST000990_DIR / "msdial_params"
OUTPUT_DIR = ST000990_DIR / "msdial_output"


def find_wiff_files():
    """Find all .wiff files in the extracted ST000990 directory."""
    wiff_files = list(ST000990_DIR.rglob("*.wiff"))
    # Exclude .wiff.scan files (index files, not data)
    wiff_files = [f for f in wiff_files if not str(f).endswith(".wiff.scan")]
    return sorted(wiff_files)


def separate_by_polarity(wiff_files):
    """Separate .wiff files into pos/neg based on filename patterns.

    Cajka & Fiehn naming: files typically contain 'pos' or 'neg' in name.
    If not separable by name, process all files together.
    """
    pos_files = [f for f in wiff_files if "pos" in f.name.lower()]
    neg_files = [f for f in wiff_files if "neg" in f.name.lower()]

    if pos_files or neg_files:
        # Some files may match neither — add to both
        unmatched = [f for f in wiff_files
                     if f not in pos_files and f not in neg_files]
        if unmatched:
            print(f"  Warning: {len(unmatched)} files don't match pos/neg pattern:")
            for f in unmatched[:5]:
                print(f"    {f.name}")
        return pos_files, neg_files
    else:
        # Can't separate — Cajka data may use CSH_pos / CSH_neg folders
        # Check parent directory names
        pos_files = [f for f in wiff_files if "pos" in str(f.parent).lower()]
        neg_files = [f for f in wiff_files if "neg" in str(f.parent).lower()]
        if pos_files or neg_files:
            return pos_files, neg_files

        print("  Cannot determine polarity from filenames. Processing all as single batch.")
        return wiff_files, []


def prepare_input_folder(wiff_files, folder_name):
    """Create a folder with symlinks/copies of .wiff files for MS-DIAL input."""
    input_dir = OUTPUT_DIR / folder_name / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    for wiff in wiff_files:
        dest = input_dir / wiff.name
        if not dest.exists():
            # Copy .wiff and its .wiff.scan companion if it exists
            shutil.copy2(wiff, dest)
            scan_file = Path(str(wiff) + ".scan")
            if scan_file.exists():
                shutil.copy2(scan_file, input_dir / scan_file.name)

    return input_dir


def run_msdial(input_dir, output_dir, param_file, label=""):
    """Run MS-DIAL console app on a folder of .wiff files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(MSDIAL_EXE),
        "lcms",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-m", str(param_file),
    ]

    print(f"\n{'='*60}")
    print(f"Running MS-DIAL {label}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Params: {param_file}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout
    )

    if result.returncode != 0:
        print(f"MS-DIAL FAILED (exit code {result.returncode})")
        print(f"STDOUT:\n{result.stdout[:2000]}")
        print(f"STDERR:\n{result.stderr[:2000]}")
        return False

    print(f"MS-DIAL completed successfully")
    if result.stdout:
        # Print last 20 lines of output
        lines = result.stdout.strip().split("\n")
        for line in lines[-20:]:
            print(f"  {line}")

    return True


def find_output_tables(output_dir):
    """Find MS-DIAL output alignment/peak tables."""
    patterns = ["*Height*.txt", "*Area*.txt", "*AlignResult*.txt", "*.msdial"]
    found = []
    for pattern in patterns:
        found.extend(output_dir.rglob(pattern))
    return found


def main():
    print("=" * 60)
    print("ST000990 Processing Pipeline")
    print("MS-DIAL Console → Feature Table → Tokenize → Predict")
    print("=" * 60)

    # Check prerequisites
    if not MSDIAL_EXE.exists():
        print(f"ERROR: MS-DIAL not found at {MSDIAL_EXE}")
        sys.exit(1)

    # Find .wiff files
    print(f"\nSearching for .wiff files in {ST000990_DIR}...")
    wiff_files = find_wiff_files()

    if not wiff_files:
        print("ERROR: No .wiff files found. Is ST000990.zip fully extracted?")
        print("  Expected: ST000990/ -> 7z archive -> .wiff files")
        # List what we have
        for f in sorted(ST000990_DIR.rglob("*"))[:20]:
            print(f"  {f.relative_to(ST000990_DIR)}")
        sys.exit(1)

    print(f"Found {len(wiff_files)} .wiff files:")
    for f in wiff_files[:5]:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({size_mb:.1f} MB)")
    if len(wiff_files) > 5:
        print(f"  ... and {len(wiff_files) - 5} more")

    # Separate by polarity
    print("\nSeparating by ESI polarity...")
    pos_files, neg_files = separate_by_polarity(wiff_files)
    print(f"  Positive ESI: {len(pos_files)} files")
    print(f"  Negative ESI: {len(neg_files)} files")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process positive ESI
    if pos_files:
        input_dir = prepare_input_folder(pos_files, "positive")
        param_file = PARAMS_DIR / "lcms_dda_pos.txt"
        out_dir = OUTPUT_DIR / "positive" / "results"
        success = run_msdial(input_dir, out_dir, param_file, label="(Positive ESI)")
        if success:
            tables = find_output_tables(out_dir)
            print(f"  Output tables: {[t.name for t in tables]}")

    # Process negative ESI
    if neg_files:
        input_dir = prepare_input_folder(neg_files, "negative")
        param_file = PARAMS_DIR / "lcms_dda_neg.txt"
        out_dir = OUTPUT_DIR / "negative" / "results"
        success = run_msdial(input_dir, out_dir, param_file, label="(Negative ESI)")
        if success:
            tables = find_output_tables(out_dir)
            print(f"  Output tables: {[t.name for t in tables]}")

    # If no polarity separation, process all together
    if not pos_files and not neg_files:
        input_dir = prepare_input_folder(wiff_files, "all")
        param_file = PARAMS_DIR / "lcms_dda_pos.txt"  # default to positive
        out_dir = OUTPUT_DIR / "all" / "results"
        success = run_msdial(input_dir, out_dir, param_file, label="(All files)")
        if success:
            tables = find_output_tables(out_dir)
            print(f"  Output tables: {[t.name for t in tables]}")

    print("\n" + "=" * 60)
    print("Pipeline complete. Check output tables above.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
