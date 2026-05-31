"""Download ST003514 (NIST SRM 1950 lipidomics) from Metabolomics Workbench.

Downloads:
1. Metabolite list with RT, m/z from the study page (HTML scraping)
2. Quantitative data (corrected areas) per sample via REST API
3. Sample metadata via REST API
4. Chemical identifiers (InChI Key, SMILES) via RefMet API

Outputs:
- data/external/st003514_metabolites.csv  (name, RT, m/z, ion_mode, refmet, inchi_key, smiles)
- data/external/st003514_intensities.csv  (sample x metabolite matrix)
- data/external/st003514_samples.csv      (sample metadata)
"""
import os
import sys
import json
import time
import re
import urllib.request
import urllib.parse

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "external")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REST_BASE = "https://www.metabolomicsworkbench.org/rest"
STUDY_ID = "ST003514"


def fetch_json(url):
    """Fetch JSON from URL with retry."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  FAILED: {e}")
                return None


def fetch_html(url):
    """Fetch HTML from URL."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def download_sample_metadata():
    """Download sample factors/metadata."""
    print("Downloading sample metadata...")
    data = fetch_json(f"{REST_BASE}/study/study_id/{STUDY_ID}/factors")
    if not data:
        return

    rows = []
    for key, sample in data.items():
        rows.append({
            "sample_id": sample["local_sample_id"],
            "mb_sample_id": sample["mb_sample_id"],
            "source": sample["sample_source"],
            "factor": sample["factors"].replace("Factor:", ""),
            "raw_files": sample.get("raw_data", ""),
        })

    # Write CSV manually (no pandas dependency)
    out_path = os.path.join(OUTPUT_DIR, "st003514_samples.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("sample_id,mb_sample_id,source,factor,raw_files\n")
        for r in rows:
            f.write(f"{r['sample_id']},{r['mb_sample_id']},{r['source']},{r['factor']},{r['raw_files']}\n")
    print(f"  Saved {len(rows)} samples to {out_path}")
    return rows


def download_metabolite_list():
    """Scrape metabolite list with RT and m/z from study page."""
    print("Downloading metabolite list (RT + m/z)...")

    metabolites = []

    for analysis_id, ion_mode in [("AN005769", "pos"), ("AN005770", "neg")]:
        url = (
            f"https://www.metabolomicsworkbench.org/data/"
            f"show_metabolites_by_study.php?STUDY_ID={STUDY_ID}"
            f"&ANALYSIS_ID={analysis_id}&RESULT_TYPE=1"
        )
        print(f"  Fetching {ion_mode} mode ({analysis_id})...")
        html = fetch_html(url)

        # Parse table rows — look for rows with metabolite data
        # Pattern: <td>name</td><td>refmet</td><td>RT</td><td>m/z</td><td>ME_ID</td>
        # The table structure varies, so we use a flexible regex
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)

        count = 0
        for row in rows:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            # Clean HTML tags from cells
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]

            if len(cells) < 4:
                continue

            # Try to identify the pattern — we need name, refmet, RT, m/z
            # Look for a cell that looks like an RT (float < 20) and m/z (float > 100)
            name = None
            refmet = None
            rt = None
            mz = None
            me_id = None

            for i, cell in enumerate(cells):
                if cell.startswith("ME") and cell[2:].isdigit():
                    me_id = cell
                elif re.match(r"^\d+\.\d+$", cell):
                    val = float(cell)
                    if val < 25 and rt is None:
                        rt = val
                    elif val > 50:
                        mz = val

            # Name is typically the first non-numeric, non-empty cell
            for cell in cells:
                if cell and not re.match(r"^\d", cell) and not cell.startswith("ME") and len(cell) > 2:
                    if name is None:
                        name = cell
                    elif refmet is None and cell != name:
                        refmet = cell

            if name and rt is not None and mz is not None:
                metabolites.append({
                    "name": name,
                    "refmet_name": refmet or name,
                    "rt": rt,
                    "mz": mz,
                    "ion_mode": ion_mode,
                    "me_id": me_id or "",
                    "analysis_id": analysis_id,
                })
                count += 1

        print(f"    Parsed {count} metabolites from {ion_mode} mode")
        time.sleep(1)

    # Write CSV
    out_path = os.path.join(OUTPUT_DIR, "st003514_metabolites.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("name,refmet_name,rt,mz,ion_mode,me_id,analysis_id\n")
        for m in metabolites:
            f.write(f"{m['name']},{m['refmet_name']},{m['rt']},{m['mz']},{m['ion_mode']},{m['me_id']},{m['analysis_id']}\n")
    print(f"  Saved {len(metabolites)} total metabolites to {out_path}")
    return metabolites


def download_refmet_identifiers(metabolites):
    """Resolve RefMet names to InChI Key and SMILES via RefMet API."""
    print("Resolving chemical identifiers via RefMet API...")

    unique_names = list(set(m["refmet_name"] for m in metabolites))
    print(f"  {len(unique_names)} unique RefMet names to resolve")

    name_to_ids = {}
    resolved = 0

    for i, name in enumerate(unique_names):
        encoded = urllib.parse.quote(name)
        url = f"{REST_BASE}/refmet/name/{encoded}/all"
        data = fetch_json(url)

        if data and isinstance(data, dict) and "inchi_key" in data:
            name_to_ids[name] = {
                "inchi_key": data.get("inchi_key", ""),
                "smiles": data.get("smiles", ""),
                "formula": data.get("formula", ""),
                "exactmass": data.get("exactmass", ""),
                "super_class": data.get("super_class", ""),
                "main_class": data.get("main_class", ""),
                "sub_class": data.get("sub_class", ""),
            }
            resolved += 1

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(unique_names)} processed, {resolved} resolved")
        time.sleep(0.2)  # Be nice to the API

    print(f"  Resolved {resolved}/{len(unique_names)} RefMet names")

    # Merge back into metabolites
    for m in metabolites:
        ids = name_to_ids.get(m["refmet_name"], {})
        m["inchi_key"] = ids.get("inchi_key", "")
        m["smiles"] = ids.get("smiles", "")
        m["formula"] = ids.get("formula", "")
        m["exactmass"] = ids.get("exactmass", "")
        m["super_class"] = ids.get("super_class", "")
        m["main_class"] = ids.get("main_class", "")
        m["sub_class"] = ids.get("sub_class", "")

    # Rewrite CSV with identifiers
    out_path = os.path.join(OUTPUT_DIR, "st003514_metabolites.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("name,refmet_name,rt,mz,ion_mode,me_id,analysis_id,inchi_key,smiles,formula,exactmass,super_class,main_class,sub_class\n")
        for m in metabolites:
            # Escape commas in SMILES
            smiles = m.get("smiles", "").replace(",", ";")
            f.write(
                f"{m['name']},{m['refmet_name']},{m['rt']},{m['mz']},{m['ion_mode']},"
                f"{m['me_id']},{m['analysis_id']},{m.get('inchi_key','')},{smiles},"
                f"{m.get('formula','')},{m.get('exactmass','')},{m.get('super_class','')},"
                f"{m.get('main_class','')},{m.get('sub_class','')}\n"
            )
    print(f"  Updated metabolite CSV with chemical identifiers")
    return name_to_ids


def fetch_tsv(url):
    """Fetch TSV data from URL with retry."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  FAILED: {e}")
                return None


def download_quantitative_data():
    """Download quantitative data (corrected areas) via REST API (returns TSV)."""
    print("Downloading quantitative data...")

    for analysis_id, ion_mode in [("AN005769", "pos"), ("AN005770", "neg")]:
        url = f"{REST_BASE}/study/analysis_id/{analysis_id}/datatable"
        print(f"  Fetching {ion_mode} data table ({analysis_id})...")
        tsv_text = fetch_tsv(url)

        if not tsv_text:
            print(f"    Failed to download {ion_mode} data")
            continue

        # Convert TSV to CSV
        lines = tsv_text.strip().split("\n")
        out_path = os.path.join(OUTPUT_DIR, f"st003514_data_{ion_mode}.csv")
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                # Split on tab, rejoin with comma (quote fields containing commas)
                fields = line.split("\t")
                csv_fields = []
                for field in fields:
                    if "," in field or '"' in field:
                        csv_fields.append('"' + field.replace('"', '""') + '"')
                    else:
                        csv_fields.append(field)
                f.write(",".join(csv_fields) + "\n")
        print(f"    Saved {len(lines)} rows to {out_path}")
        time.sleep(1)


def main():
    print("=" * 60)
    print("Downloading ST003514 (NIST SRM 1950 Lipidomics)")
    print("=" * 60)

    # Step 1: Sample metadata
    samples = download_sample_metadata()

    # Step 2: Metabolite list with RT/m/z
    metabolites = download_metabolite_list()

    # Step 3: Chemical identifiers from RefMet
    if metabolites:
        download_refmet_identifiers(metabolites)

    # Step 4: Quantitative data
    download_quantitative_data()

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
