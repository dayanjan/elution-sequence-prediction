"""
Systematic analysis of all candidate features for RT prediction / tokenization.
Tests: m/z, mass defect, Kendrick defect, ECN, unsaturation ratio,
       adduct type, polarity, intensity, and multivariate combinations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import load_all_datasets


def main():
    df = load_all_datasets()
    feat = df.drop_duplicates(subset=["study", "feature_id"]).copy()

    # ALL features
    af = feat.copy()
    af["mass_defect"] = af.mz - np.floor(af.mz)
    af["log_mz"] = np.log10(af.mz)
    af["kendrick_mass"] = af.mz * 14.0 / 14.01565
    af["kendrick_defect"] = np.ceil(af.kendrick_mass) - af.kendrick_mass

    # Mean intensity per feature
    feat_int = df.groupby(["study", "feature_id"])["intensity"].mean().reset_index()
    feat_int.columns = ["study", "feature_id", "mean_intensity"]
    af = af.merge(feat_int, on=["study", "feature_id"])
    af["log_intensity"] = np.log10(af.mean_intensity.clip(lower=1))

    # ANNOTATED features
    ann = af[(af.total_carbons.notna()) & (~af.is_istd) & (af.lipid_class.notna())].copy()
    ann["ecn"] = ann.total_carbons - 2 * ann.total_unsat
    ann["unsat_ratio"] = ann.total_unsat / ann.total_carbons.replace(0, np.nan)

    print(f"All features: {len(af)}")
    print(f"Annotated (with carbon info): {len(ann)}")
    print()

    # === Univariate correlations ===
    print("=" * 70)
    print("UNIVARIATE CORRELATIONS WITH RT")
    print("=" * 70)
    print()
    print(f"{'Feature':<28} {'Pearson r':<12} {'p-value':<15} {'n':<8} {'Scope'}")
    print("-" * 78)

    # All features
    for col, label in [
        ("mz", "m/z"),
        ("log_mz", "log10(m/z)"),
        ("mass_defect", "Mass defect"),
        ("kendrick_defect", "Kendrick mass defect"),
        ("log_intensity", "log10(mean intensity)"),
    ]:
        mask = af[col].notna() & af["rt"].notna()
        if mask.sum() > 10:
            r, p = stats.pearsonr(af.loc[mask, col], af.loc[mask, "rt"])
            print(f"  {label:<26} {r:>+.4f}      {p:.2e}       {mask.sum():<8} all features")

    print()

    # Annotated features
    for col, label in [
        ("total_carbons", "Total carbons"),
        ("total_unsat", "Total unsaturation"),
        ("ecn", "ECN (C - 2*U)"),
        ("unsat_ratio", "Unsat ratio (U/C)"),
        ("mz", "m/z"),
        ("mass_defect", "Mass defect"),
        ("kendrick_defect", "Kendrick mass defect"),
    ]:
        mask = ann[col].notna() & ann["rt"].notna()
        if mask.sum() > 10:
            r, p = stats.pearsonr(ann.loc[mask, col], ann.loc[mask, "rt"])
            print(f"  {label:<26} {r:>+.4f}      {p:.2e}       {mask.sum():<8} annotated")

    # === Polarity ===
    print()
    print("=" * 70)
    print("ESI POLARITY EFFECT ON RT")
    print("=" * 70)
    for pol in sorted(af.polarity.unique()):
        subset = af[af.polarity == pol]
        print(f"  {pol}: n={len(subset)}, mean RT={subset.rt.mean():.2f}, "
              f"median={subset.rt.median():.2f}, std={subset.rt.std():.2f}")

    # Test if polarity predicts RT
    pos = af[af.polarity == "(+) ESI"]["rt"]
    neg = af[af.polarity == "(-) ESI"]["rt"]
    if len(pos) > 10 and len(neg) > 10:
        t, p = stats.ttest_ind(pos, neg)
        print(f"  t-test pos vs neg: t={t:.2f}, p={p:.2e}")

    # === Adduct type ===
    print()
    print("=" * 70)
    print("ADDUCT TYPE EFFECT ON RT")
    print("=" * 70)
    ann_adduct = ann[ann.adduct.notna()].copy()
    ann_adduct["primary_adduct"] = ann_adduct.adduct.str.split("_").str[0].str.strip()

    print("Distribution:")
    print(ann_adduct.primary_adduct.value_counts().head(10).to_string())
    print()

    print("Mean RT by adduct (n >= 10):")
    adduct_rt = ann_adduct.groupby("primary_adduct")["rt"].agg(["mean", "std", "count"])
    adduct_rt = adduct_rt[adduct_rt["count"] >= 10].sort_values("mean")
    print(adduct_rt.to_string())

    # After controlling for ECN+class, does adduct still matter?
    print()
    coeffs = np.polyfit(ann.ecn, ann.rt, 1)
    ann["rt_pred_ecn"] = np.polyval(coeffs, ann.ecn)
    class_offsets = ann.groupby("lipid_class").apply(
        lambda g: (g.rt - np.polyval(coeffs, g.ecn)).mean()
    )
    ann["class_offset"] = ann.lipid_class.map(class_offsets)
    ann["rt_residual_full"] = ann.rt - ann.rt_pred_ecn - ann.class_offset

    # Merge back adduct info
    ann_res = ann[ann.adduct.notna()].copy()
    ann_res["primary_adduct"] = ann_res.adduct.str.split("_").str[0].str.strip()

    print("Adduct RT residuals (after ECN + class correction):")
    adduct_res = ann_res.groupby("primary_adduct")["rt_residual_full"].agg(["mean", "std", "count"])
    adduct_res = adduct_res[adduct_res["count"] >= 10].sort_values("mean")
    for idx, row in adduct_res.iterrows():
        sig = ""
        subset = ann_res[ann_res.primary_adduct == idx]["rt_residual_full"]
        if len(subset) >= 5:
            t, p = stats.ttest_1samp(subset, 0)
            sig = f"p={p:.2e}" if p < 0.05 else "n.s."
        print(f"  {idx:<20} {row['mean']:>+.3f} min ({row['mean']*60:>+.1f}s)  "
              f"std={row['std']:.3f}  n={int(row['count'])}  {sig}")

    # === Multivariate R² ===
    print()
    print("=" * 70)
    print("MULTIVARIATE R-SQUARED (ANNOTATED FEATURES)")
    print("=" * 70)

    y = ann.rt.values
    ss_tot = ((y - y.mean()) ** 2).sum()
    class_dummies = pd.get_dummies(ann.lipid_class, prefix="cls")

    models = []

    def test_model(name, X):
        beta = lstsq(X, y, rcond=None)[0]
        pred = X @ beta
        r2 = 1 - ((y - pred) ** 2).sum() / ss_tot
        rmse = np.sqrt(((y - pred) ** 2).mean())
        mae = np.abs(y - pred).mean()
        models.append({"name": name, "r2": r2, "rmse_min": rmse, "mae_min": mae})
        print(f"  {name:<45} R²={r2:.4f}  RMSE={rmse:.3f} min ({rmse*60:.0f}s)  MAE={mae:.3f} min ({mae*60:.0f}s)")

    test_model("ECN only", np.column_stack([ann.ecn, np.ones(len(ann))]))
    test_model("Total carbons only", np.column_stack([ann.total_carbons, np.ones(len(ann))]))
    test_model("Total unsaturation only", np.column_stack([ann.total_unsat, np.ones(len(ann))]))
    test_model("ECN + unsat", np.column_stack([ann.ecn, ann.total_unsat, np.ones(len(ann))]))
    test_model("Head group class only", np.column_stack([class_dummies.values, np.ones(len(ann))]))
    test_model("ECN + class", np.column_stack([ann.ecn, class_dummies.values, np.ones(len(ann))]))
    test_model("ECN + class + unsat", np.column_stack([ann.ecn, ann.total_unsat, class_dummies.values, np.ones(len(ann))]))
    test_model("ECN + class + unsat + log(m/z)", np.column_stack([ann.ecn, ann.total_unsat, np.log10(ann.mz), class_dummies.values, np.ones(len(ann))]))
    test_model("ECN + class + unsat + mass_defect", np.column_stack([ann.ecn, ann.total_unsat, ann.mass_defect, class_dummies.values, np.ones(len(ann))]))
    test_model("Carbons + unsat + class", np.column_stack([ann.total_carbons, ann.total_unsat, class_dummies.values, np.ones(len(ann))]))

    # All-features models (no annotation needed)
    print()
    print("=" * 70)
    print("MULTIVARIATE R-SQUARED (ALL FEATURES — no annotation needed)")
    print("=" * 70)

    mask = af.mz.notna() & af.rt.notna()
    aff = af[mask].copy()
    y_all = aff.rt.values
    ss_tot_all = ((y_all - y_all.mean()) ** 2).sum()

    def test_model_all(name, X):
        beta = lstsq(X, y_all, rcond=None)[0]
        pred = X @ beta
        r2 = 1 - ((y_all - pred) ** 2).sum() / ss_tot_all
        rmse = np.sqrt(((y_all - pred) ** 2).mean())
        mae = np.abs(y_all - pred).mean()
        print(f"  {name:<45} R²={r2:.4f}  RMSE={rmse:.3f} min ({rmse*60:.0f}s)  MAE={mae:.3f} min ({mae*60:.0f}s)")

    test_model_all("m/z only", np.column_stack([aff.mz, np.ones(len(aff))]))
    test_model_all("m/z + mass_defect", np.column_stack([aff.mz, aff.mass_defect, np.ones(len(aff))]))
    test_model_all("m/z + mass_defect + kendrick", np.column_stack([aff.mz, aff.mass_defect, aff.kendrick_defect, np.ones(len(aff))]))
    test_model_all("m/z + log_intensity", np.column_stack([aff.mz, aff.log_intensity, np.ones(len(aff))]))
    test_model_all("m/z + mass_defect + log_intensity", np.column_stack([aff.mz, aff.mass_defect, aff.log_intensity, np.ones(len(aff))]))

    pol_dummies = pd.get_dummies(aff.polarity, prefix="pol")
    test_model_all("m/z + mass_defect + polarity", np.column_stack([aff.mz, aff.mass_defect, pol_dummies.values, np.ones(len(aff))]))
    test_model_all("m/z + mass_defect + polarity + intensity", np.column_stack([aff.mz, aff.mass_defect, pol_dummies.values, aff.log_intensity, np.ones(len(aff))]))


if __name__ == "__main__":
    main()
