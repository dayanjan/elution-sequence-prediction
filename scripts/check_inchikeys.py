"""Check InChI Key validity and identify problematic keys."""
import sys, re
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from preprocessing import load_all_datasets

df = load_all_datasets()
ann = df[df["annotation"].notna() & (df["annotation"] != "")].copy()
feat = ann.groupby("feature_id").first().reset_index()
feat = feat[feat["inchi_key"].notna() & (feat["inchi_key"] != "")]
feat = feat[~feat["is_istd"]]

def is_valid_inchikey(k):
    if pd.isna(k) or not k:
        return False
    k = str(k).strip().rstrip("?")
    return bool(re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", k))

feat["valid_ik"] = feat["inchi_key"].apply(is_valid_inchikey)
feat["has_question"] = feat["inchi_key"].str.contains(r"\?", na=False)
feat["has_underscore"] = feat["inchi_key"].str.startswith("_", na=False)
feat["has_multi"] = feat["inchi_key"].str.contains(r"[A-Z]{14}-[A-Z]{10}-[A-Z].*[A-Z]{14}", na=False)

print(f"Total features with InChI Keys: {len(feat)}")
print(f"Valid format: {feat['valid_ik'].sum()}")
print(f"Has trailing ?: {feat['has_question'].sum()}")
print(f"Starts with _: {feat['has_underscore'].sum()}")
print(f"Multiple concatenated keys: {feat['has_multi'].sum()}")
print()

invalid = feat[~feat["valid_ik"]]
print(f"Invalid InChI Keys ({len(invalid)}):")
for _, row in invalid.head(20).iterrows():
    ik = str(row["inchi_key"])
    print(f"  {row['annotation']:40s} | {ik[:70]}")

print()
# Clean: strip trailing ? and take first key if concatenated
def clean_inchikey(k):
    k = str(k).strip().rstrip("?")
    # If starts with _, remove it
    if k.startswith("_"):
        k = k[1:]
    # If multiple keys concatenated, take the first valid one
    matches = re.findall(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", k)
    return matches[0] if matches else None

feat["clean_ik"] = feat["inchi_key"].apply(clean_inchikey)
cleaned = feat["clean_ik"].notna().sum()
print(f"After cleaning: {cleaned}/{len(feat)} valid ({cleaned/len(feat)*100:.1f}%)")
still_bad = feat[feat["clean_ik"].isna()]
if len(still_bad) > 0:
    print(f"\nStill invalid ({len(still_bad)}):")
    for _, row in still_bad.iterrows():
        print(f"  {row['annotation']:40s} | {row['inchi_key']}")
