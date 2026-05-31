"""Process a representative subset of ST000990 .wiff files through MS-DIAL.

Selects ~50 files: all NIST (2) + all QC (15) + all Rep (10) + 23 evenly-spaced
analytical samples. This is sufficient for external validation while being more
crash-resilient than running all 153 files.

Usage:
    .venv/Scripts/python scripts/run_msdial_st000990_subset.py

Prerequisites:
    - ST000990 extracted to data/external/ST000990/
    - MS-DIAL console installed at tools/msdial/console/
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[5]
MSDIAL_EXE = PROJECT_ROOT / "tools" / "msdial" / "console" / "MSDIALCUI.exe"
ST000990_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "ST000990"
RAW_DATA_DIR = ST000990_DIR / "ST000990" / "GLA STUDY TripleTOF 6600" / "Raw Data"
PARAMS_DIR = ST000990_DIR / "msdial_params"
OUTPUT_DIR = ST000990_DIR / "msdial_output_subset"
LOG_FILE = OUTPUT_DIR / "msdial_stdout.log"


def select_subset():
    """Select ~50 representative .wiff files."""
    all_wiff = sorted(f for f in RAW_DATA_DIR.iterdir()
                      if f.suffix == ".wiff" and not str(f).endswith(".wiff.scan"))

    nist = [f for f in all_wiff if "_NIST" in f.name]
    qc = [f for f in all_wiff if "_QC" in f.name]
    rep = [f for f in all_wiff if "_Rep" in f.name]
    analytical = [f for f in all_wiff if "_SO" in f.name]

    # Take every ~5th analytical sample for even coverage
    step = max(1, len(analytical) // 23)
    analytical_subset = analytical[::step][:23]

    subset = sorted(nist + qc + rep + analytical_subset, key=lambda f: f.name)

    print(f"Selected {len(subset)} files:")
    print(f"  NIST references: {len(nist)}")
    print(f"  QC injections:   {len(qc)}")
    print(f"  Replicates:      {len(rep)}")
    print(f"  Analytical:      {len(analytical_subset)} / {len(analytical)}")

    return subset


def prepare_input_folder(wiff_files):
    """Copy .wiff + .wiff.scan files to a clean input folder."""
    input_dir = OUTPUT_DIR / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    for wiff in wiff_files:
        dest = input_dir / wiff.name
        if not dest.exists():
            shutil.copy2(wiff, dest)
            # Copy companion .wiff.scan if it exists
            scan = Path(str(wiff) + ".scan")
            if scan.exists():
                shutil.copy2(scan, input_dir / scan.name)
            # Copy companion .wiff.1.~idx2 if it exists
            idx = wiff.parent / (wiff.name + ".1.~idx2")
            if idx.exists():
                shutil.copy2(idx, input_dir / idx.name)
            print(f"  Copied: {wiff.name}")

    return input_dir


def main():
    print("=" * 60)
    print("ST000990 Subset Processing (Option 1: ~50 files)")
    print("=" * 60)

    if not MSDIAL_EXE.exists():
        print(f"ERROR: MS-DIAL not found at {MSDIAL_EXE}")
        sys.exit(1)

    if not RAW_DATA_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DATA_DIR}")
        sys.exit(1)

    # Select subset
    subset = select_subset()

    # Prepare input folder
    print(f"\nPreparing input folder...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_dir = prepare_input_folder(subset)

    # Run MS-DIAL
    param_file = PARAMS_DIR / "lcms_dda_pos.txt"
    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(MSDIAL_EXE),
        "lcms",
        "-i", str(input_dir),
        "-o", str(results_dir),
        "-m", str(param_file),
    ]

    print(f"\n{'='*60}")
    print(f"Starting MS-DIAL on {len(subset)} .wiff files...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {results_dir}")
    print(f"  Params: {param_file}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {LOG_FILE}")
    print(f"{'='*60}\n")

    # Run with stdout/stderr to log file, no timeout (let it run)
    with open(LOG_FILE, "w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    print(f"PID: {process.pid}")
    print(f"MS-DIAL launched in foreground. Waiting for completion...")
    print(f"Monitor progress: tail -f '{LOG_FILE}'")

    returncode = process.wait()

    if returncode == 0:
        print(f"\nMS-DIAL completed successfully!")
        # List output files
        for f in sorted(results_dir.iterdir()):
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"\nMS-DIAL FAILED (exit code {returncode})")
        print(f"Check log: {LOG_FILE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
