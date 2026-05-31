"""Process remaining ST000990 analytical samples in two batches.

Batch 1 (subset) already completed: 2 NIST + 15 QC + 10 Rep + 23 analytical = 50 files.
This script runs the remaining 103 analytical samples in two batches of ~52.

Per-sample .mdpeak files are used for model validation (not alignment tables),
so separate batch alignments don't matter.

Usage:
    .venv/Scripts/python scripts/run_msdial_st000990_batches.py

Prerequisites:
    - Batch 1 (subset) already completed in msdial_output_subset/
    - MS-DIAL console installed at tools/msdial/console/
"""

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
BATCH1_INPUT = ST000990_DIR / "msdial_output_subset" / "input"


def get_remaining_analytical():
    """Find analytical .wiff files not already in batch 1."""
    all_analytical = sorted(
        f for f in RAW_DATA_DIR.iterdir()
        if f.suffix == ".wiff" and "_SO" in f.name
    )

    done = set(f.name for f in BATCH1_INPUT.iterdir() if f.suffix == ".wiff")
    remaining = [f for f in all_analytical if f.name not in done]

    print(f"Total analytical: {len(all_analytical)}")
    print(f"Already processed: {len(done & {f.name for f in all_analytical})}")
    print(f"Remaining: {len(remaining)}")

    return remaining


def copy_files(wiff_files, input_dir):
    """Copy .wiff + companion files to input folder."""
    input_dir.mkdir(parents=True, exist_ok=True)
    for wiff in wiff_files:
        dest = input_dir / wiff.name
        if not dest.exists():
            shutil.copy2(wiff, dest)
            for ext in [".scan", ".1.~idx2"]:
                companion = wiff.parent / (wiff.name + ext)
                if companion.exists():
                    shutil.copy2(companion, input_dir / companion.name)
    return input_dir


def run_batch(batch_name, wiff_files):
    """Run MS-DIAL on a batch of files."""
    output_dir = ST000990_DIR / f"msdial_output_{batch_name}"
    input_dir = output_dir / "input"
    results_dir = output_dir / "results"
    log_file = output_dir / "msdial_stdout.log"

    print(f"\n{'='*60}")
    print(f"Batch: {batch_name} ({len(wiff_files)} files)")
    print(f"{'='*60}")

    # Copy files
    print("Copying files...")
    copy_files(wiff_files, input_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    param_file = PARAMS_DIR / "lcms_dda_pos.txt"
    cmd = [
        str(MSDIAL_EXE), "lcms",
        "-i", str(input_dir),
        "-o", str(results_dir),
        "-m", str(param_file),
    ]

    print(f"  Input:  {input_dir}")
    print(f"  Output: {results_dir}")
    print(f"  Log:    {log_file}")
    print(f"Launching MS-DIAL...")

    with open(log_file, "w") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    print(f"  PID: {process.pid}")
    returncode = process.wait()

    if returncode == 0:
        print(f"Batch {batch_name} completed successfully!")
        for f in sorted(results_dir.iterdir()):
            if f.suffix in ('.mdalign', '.mzTabM'):
                size_mb = f.stat().st_size / 1e6
                print(f"  {f.name} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"Batch {batch_name} FAILED (exit code {returncode})")
        print(f"Check log: {log_file}")
        return False


def main():
    print("=" * 60)
    print("ST000990 Remaining Batches (103 analytical samples)")
    print("=" * 60)

    if not MSDIAL_EXE.exists():
        print(f"ERROR: MS-DIAL not found at {MSDIAL_EXE}")
        sys.exit(1)

    remaining = get_remaining_analytical()
    if not remaining:
        print("No remaining files to process!")
        sys.exit(0)

    # Split into two batches
    mid = len(remaining) // 2
    batch2_files = remaining[:mid]
    batch3_files = remaining[mid:]

    print(f"\nBatch 2: {len(batch2_files)} files (SO{batch2_files[0].name.split('_SO')[1][:3]}–SO{batch2_files[-1].name.split('_SO')[1][:3]})")
    print(f"Batch 3: {len(batch3_files)} files (SO{batch3_files[0].name.split('_SO')[1][:3]}–SO{batch3_files[-1].name.split('_SO')[1][:3]})")

    # Run batch 2
    success = run_batch("batch2", batch2_files)
    if not success:
        print("\nBatch 2 failed. Stopping.")
        sys.exit(1)

    # Run batch 3
    success = run_batch("batch3", batch3_files)
    if not success:
        print("\nBatch 3 failed.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("All batches complete!")
    print(f"  Batch 1 (subset): msdial_output_subset/results/")
    print(f"  Batch 2:          msdial_output_batch2/results/")
    print(f"  Batch 3:          msdial_output_batch3/results/")
    print(f"Use per-sample .mdpeak files for model validation.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
