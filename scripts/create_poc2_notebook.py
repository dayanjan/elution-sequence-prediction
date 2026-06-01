"""Create the POC #2 RT prediction Colab notebook."""
import json
import os

cells = []

def md(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})

def code(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    cells.append({"cell_type": "code", "metadata": {}, "source": lines, "outputs": [], "execution_count": None})


# Cell 1: Header
md("""# POC #2: RT Prediction with Class-Specific Models
**Version:** v1.0 (2026-03-08)

**Goal:** Train RF/GBM to predict retention time from molecular descriptors. Compare global vs class-specific models using leave-one-cohort-out cross-validation.

**Key question:** Can we predict RT from structure well enough to support MS/MS scheduling (within +/-1 min)?

**Changelog:**
- v1.0: Initial notebook — RDKit descriptors, RF/GBM, leave-one-cohort-out CV""")

# Cell 2: Setup & Drive mount
code("""import os
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/0000 Fun with coding/088 Lights-Out R01 Grant/Specific Aim 1/poc2_rt_prediction"
os.chdir("/content")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Setup complete")""")

# Cell 3: Load data
code("""# Load annotated features
feat = pd.read_parquet(f"{DRIVE_DIR}/annotated_features.parquet")
print(f"Loaded {len(feat)} annotated features from {feat['study'].nunique()} cohorts")
print(f"RT range: {feat['rt'].min():.2f} - {feat['rt'].max():.2f} min")
print(f"Lipid classes: {feat['lipid_class'].nunique()}")
print(f"Per cohort: {feat['study'].value_counts().to_dict()}")""")

# Cell 4: Feature engineering
code("""# Feature engineering from available data
feat_eng = feat.copy()
feat_eng['log_mz'] = np.log10(feat_eng['mz'])
feat_eng['mass_defect'] = feat_eng['mz'] - np.floor(feat_eng['mz'])
feat_eng['ecn'] = feat_eng['total_carbons'] - 2 * feat_eng['total_unsat']
feat_eng['unsat_ratio'] = feat_eng['total_unsat'] / feat_eng['total_carbons'].replace(0, np.nan)
feat_eng['carbon_mz_ratio'] = feat_eng['total_carbons'] / feat_eng['mz']
feat_eng['polarity_num'] = (feat_eng['polarity'] == 'pos').astype(int)

# Encode lipid class
le_class = LabelEncoder()
feat_eng['class_encoded'] = le_class.fit_transform(feat_eng['lipid_class'].fillna('Unknown'))

feature_cols = ['mz', 'log_mz', 'mass_defect', 'total_carbons', 'total_unsat',
                'ecn', 'unsat_ratio', 'carbon_mz_ratio', 'polarity_num', 'class_encoded']

print("Engineered features:")
for c in feature_cols:
    valid = feat_eng[c].notna().sum()
    print(f"  {c}: {valid}/{len(feat_eng)} valid ({valid/len(feat_eng)*100:.0f}%)")

feat_clean = feat_eng.dropna(subset=feature_cols + ['rt']).copy()
print(f"\\nClean dataset: {len(feat_clean)} features (dropped {len(feat_eng) - len(feat_clean)})")""")

# Cell 5: Resolve InChI Keys via PubChem
code("""from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import urllib.request
import json
import time

# Resolve InChI Keys to SMILES via PubChem REST API
unique_keys = feat_clean['inchi_key'].unique()
print(f"Resolving {len(unique_keys)} unique InChI Keys via PubChem...")

key_to_smiles = {}
failed = 0

for i, k in enumerate(unique_keys):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{k}/property/CanonicalSMILES/JSON"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            key_to_smiles[k] = smiles
    except:
        failed += 1
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(unique_keys)} resolved, {failed} failed")
    time.sleep(0.25)

print(f"\\nResolved: {len(key_to_smiles)}/{len(unique_keys)} ({len(key_to_smiles)/len(unique_keys)*100:.0f}%)")
print(f"Failed: {failed}")""")

# Cell 6: Compute RDKit descriptors
code("""def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        'MolLogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'MolWt': Descriptors.MolWt(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
    }

desc_dict = {}
for k, smi in key_to_smiles.items():
    d = compute_descriptors(smi)
    if d:
        desc_dict[k] = d

print(f"Computed descriptors for {len(desc_dict)} compounds")

# Merge into feature table
desc_df = pd.DataFrame.from_dict(desc_dict, orient='index')
desc_df.index.name = 'inchi_key'
feat_rdkit = feat_clean.merge(desc_df, left_on='inchi_key', right_index=True, how='left')

rdkit_cols = list(desc_df.columns)
has_rdkit = feat_rdkit[rdkit_cols[0]].notna().sum()
print(f"Features with RDKit descriptors: {has_rdkit}/{len(feat_rdkit)} ({has_rdkit/len(feat_rdkit)*100:.0f}%)")""")

# Cell 7: Leave-one-cohort-out CV
code("""basic_features = ['mz', 'log_mz', 'mass_defect', 'total_carbons', 'total_unsat',
                  'ecn', 'unsat_ratio', 'carbon_mz_ratio', 'polarity_num']
class_features = basic_features + ['class_encoded']
rdkit_features = ['MolLogP', 'TPSA', 'MolWt', 'HBD', 'HBA', 'RotBonds',
                  'RingCount', 'FractionCSP3', 'HeavyAtomCount']
all_features = class_features + rdkit_features

# Check if RDKit descriptors are available
if 'MolLogP' in feat_rdkit.columns and feat_rdkit['MolLogP'].notna().sum() > 100:
    df_model = feat_rdkit.copy()
    feature_sets = {
        'Basic (m/z + carbons)': basic_features,
        'Basic + class': class_features,
        'Basic + RDKit': basic_features + rdkit_features,
        'All features': all_features,
    }
    print("Using RDKit descriptors")
else:
    df_model = feat_clean.copy()
    feature_sets = {
        'Basic (m/z + carbons)': basic_features,
        'Basic + class': class_features,
    }
    print("RDKit descriptors not available, using engineered features only")

cohorts = sorted(df_model['study'].unique())
print(f"Cohorts: {cohorts}")
print(f"Total features: {len(df_model)}")

# Leave-one-cohort-out CV
results = []

for model_name, model_cls in [('RF', RandomForestRegressor), ('GBM', GradientBoostingRegressor)]:
    for feat_name, feat_cols in feature_sets.items():
        for test_cohort in cohorts:
            train = df_model[df_model['study'] != test_cohort].copy()
            test = df_model[df_model['study'] == test_cohort].copy()

            valid_cols = [c for c in feat_cols if c in train.columns]
            train = train.dropna(subset=valid_cols + ['rt'])
            test = test.dropna(subset=valid_cols + ['rt'])

            if len(train) < 20 or len(test) < 10:
                continue

            X_train, y_train = train[valid_cols].values, train['rt'].values
            X_test, y_test = test[valid_cols].values, test['rt'].values

            if model_name == 'RF':
                model = model_cls(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            else:
                model = model_cls(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            within_1min = np.mean(np.abs(y_test - y_pred) < 1.0) * 100

            results.append({
                'model': model_name, 'features': feat_name, 'test_cohort': test_cohort,
                'n_train': len(train), 'n_test': len(test),
                'MAE_min': mae, 'R2': r2, 'within_1min_pct': within_1min,
            })

results_df = pd.DataFrame(results)
print(f"\\nCompleted {len(results_df)} experiments")
print(results_df.groupby(['model', 'features'])[['MAE_min', 'R2', 'within_1min_pct']].mean().round(3))""")

# Cell 8: Class-specific models
code("""major_classes = df_model['lipid_class'].value_counts()
major_classes = major_classes[major_classes >= 30].index.tolist()
print(f"Major classes (n>=30): {major_classes}")

class_results = []
best_feat_cols = class_features
valid_cols = [c for c in best_feat_cols if c in df_model.columns]

for test_cohort in cohorts:
    train = df_model[df_model['study'] != test_cohort].copy()
    test = df_model[df_model['study'] == test_cohort].copy()

    # Global model
    train_g = train.dropna(subset=valid_cols + ['rt'])
    test_g = test.dropna(subset=valid_cols + ['rt'])

    if len(train_g) < 20 or len(test_g) < 10:
        continue

    rf_global = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_global.fit(train_g[valid_cols].values, train_g['rt'].values)

    for cls in major_classes:
        train_cls = train[train['lipid_class'] == cls].dropna(subset=valid_cols + ['rt'])
        test_cls = test[test['lipid_class'] == cls].dropna(subset=valid_cols + ['rt'])

        if len(train_cls) < 10 or len(test_cls) < 3:
            continue

        # Global predictions on this class
        pred_global = rf_global.predict(test_cls[valid_cols].values)
        mae_global = mean_absolute_error(test_cls['rt'].values, pred_global)

        # Class-specific model
        rf_class = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf_class.fit(train_cls[valid_cols].values, train_cls['rt'].values)
        pred_class = rf_class.predict(test_cls[valid_cols].values)
        mae_class = mean_absolute_error(test_cls['rt'].values, pred_class)

        class_results.append({
            'test_cohort': test_cohort, 'lipid_class': cls,
            'n_test': len(test_cls),
            'MAE_global': mae_global, 'MAE_class_specific': mae_class,
            'improvement_pct': (mae_global - mae_class) / mae_global * 100,
        })

class_df = pd.DataFrame(class_results)
print("\\nGlobal vs Class-Specific RF (MAE in minutes):")
summary = class_df.groupby('lipid_class').agg({
    'n_test': 'sum',
    'MAE_global': 'mean',
    'MAE_class_specific': 'mean',
    'improvement_pct': 'mean',
}).round(3)
summary = summary.sort_values('n_test', ascending=False)
print(summary)
print(f"\\nOverall improvement: {class_df['improvement_pct'].mean():.1f}%")""")

# Cell 9: Feature importance
code("""best_feat_cols = all_features if 'MolLogP' in df_model.columns and df_model['MolLogP'].notna().sum() > 100 else class_features
valid_cols = [c for c in best_feat_cols if c in df_model.columns]

df_all = df_model.dropna(subset=valid_cols + ['rt'])
rf_full = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_full.fit(df_all[valid_cols].values, df_all['rt'].values)

importances = pd.Series(rf_full.feature_importances_, index=valid_cols).sort_values(ascending=True)
print("Feature importances (RF):")
for f, v in importances.items():
    print(f"  {f:25s}: {v:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Feature Importance (MDI)')
ax.set_title('RF Feature Importance for RT Prediction', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
os.makedirs(f'{DRIVE_DIR}/outputs', exist_ok=True)
plt.savefig(f'{DRIVE_DIR}/outputs/feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")
plt.close()""")

# Cell 10: Predicted vs actual scatter
code("""fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)

best_feat_cols = class_features
valid_cols = [c for c in best_feat_cols if c in df_model.columns]

for idx, test_cohort in enumerate(cohorts):
    ax = axes[idx]
    train = df_model[df_model['study'] != test_cohort].dropna(subset=valid_cols + ['rt'])
    test = df_model[df_model['study'] == test_cohort].dropna(subset=valid_cols + ['rt'])

    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(train[valid_cols].values, train['rt'].values)
    y_pred = rf.predict(test[valid_cols].values)
    y_true = test['rt'].values

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    ax.scatter(y_true, y_pred, s=10, alpha=0.5, c='steelblue')
    ax.plot([0, 13], [0, 13], 'k--', lw=1, alpha=0.5)
    ax.fill_between([0, 13], [-1, 12], [1, 14], alpha=0.1, color='green', label='+/-1 min')

    nice = {'cardiac_arrest': 'Cardiac Arrest', 'gvhd': 'GVHD', 'pcos': 'PCOS', 'redhart2': 'REDHART 1'}
    ax.set_title(f"{nice.get(test_cohort, test_cohort)}\\n(n={len(test)})", fontsize=10)
    ax.text(0.05, 0.95, f"MAE={mae:.2f} min\\nR2={r2:.3f}", transform=ax.transAxes,
            va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if idx == 0:
        ax.set_ylabel('Predicted RT (min)')
    ax.set_xlabel('Actual RT (min)')

fig.suptitle('Leave-One-Cohort-Out RT Prediction (RF, Basic + Class features)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE_DIR}/outputs/predicted_vs_actual_rt.png', dpi=300, bbox_inches='tight')
print("Saved: predicted_vs_actual_rt.png")
plt.close()""")

# Cell 11: Save results
code("""results_df.to_csv(f'{DRIVE_DIR}/outputs/loco_cv_results.csv', index=False)
if len(class_df) > 0:
    class_df.to_csv(f'{DRIVE_DIR}/outputs/class_specific_results.csv', index=False)

print("=" * 70)
print("POC #2: RT PREDICTION RESULTS SUMMARY")
print("=" * 70)

print("\\n--- Leave-One-Cohort-Out CV (averaged across 4 folds) ---")
summary = results_df.groupby(['model', 'features']).agg({
    'MAE_min': ['mean', 'std'],
    'R2': ['mean', 'std'],
    'within_1min_pct': ['mean', 'std'],
}).round(3)
print(summary)

if len(class_df) > 0:
    print("\\n--- Global vs Class-Specific (RF) ---")
    print(f"Global MAE: {class_df['MAE_global'].mean():.3f} min")
    print(f"Class-specific MAE: {class_df['MAE_class_specific'].mean():.3f} min")
    print(f"Improvement: {class_df['improvement_pct'].mean():.1f}%")

print("\\n--- Key Takeaway ---")
best = results_df.groupby(['model', 'features'])['MAE_min'].mean().idxmin()
best_mae = results_df.groupby(['model', 'features'])['MAE_min'].mean().min()
best_within = results_df.groupby(['model', 'features'])['within_1min_pct'].mean().max()
print(f"Best model: {best[0]} with {best[1]}")
print(f"Cross-cohort MAE: {best_mae:.2f} min")
print(f"Within +/-1 min window: {best_within:.0f}%")
print("\\nResults saved to Drive.")""")


# Assemble notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}
    },
    "cells": cells
}

out_path = os.path.join(
    r"C:\Users\wijesingheds\Documents\04 Fun with Coding\2026-03-05-LightsOut-R01",
    r"inputs\02_preliminary_data\analyses\poc3_elution_sequence\gdrive\poc2_rt_prediction",
    "02_rt_prediction.ipynb"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Saved: {out_path}")
