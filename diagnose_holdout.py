import pandas as pd
import joblib
import numpy as np

# Load everything
model = joblib.load('final_lasso_model.joblib')
model_features = joblib.load('model_features.joblib')
df_holdout = pd.read_csv("/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/results/final_feature_matrix_gc_corrected_50000.tsv", sep="\t")

print("--- DIAGNOSTIK ---")

# 1. Bins Check
df_holdout["bin_id"] = df_holdout["chrom"] + "_" + df_holdout["start"].astype(str)
used_metrics = list(set([f.split('_chr')[0] for f in model_features]))
X_holdout_raw = df_holdout.pivot(index="sample", columns="bin_id", values=used_metrics)
X_holdout_raw.columns = [f"{m}_{bid}" for m, bid in X_holdout_raw.columns]

overlap = set(model_features).intersection(set(X_holdout_raw.columns))
print(f"Binstruktur: Modell hat {len(model_features)} Features.")
print(f"Holdout hat {len(X_holdout_raw.columns)} Features.")
print(f"Überschneidung: {len(overlap)} Features stimmen überein.")

if len(overlap) < len(model_features) * 0.5:
    print("❌ WARNUNG: Weniger als 50% der Bins stimmen überein! Hier liegt das Problem.")
else:
    print("✅ Bins scheinen korrekt zu matchen.")

# 2. Labels Check
group_info = df_holdout[['sample', 'group']].drop_duplicates()
print("\nGefundene Gruppen im Holdout:")
print(group_info['group'].value_counts())

y_holdout = (group_info.set_index('sample').loc[X_holdout_raw.index, "group"].str.lower() != "healthy").astype(int).values
print(f"Labels berechnet: {np.sum(y_holdout == 0)} Healthy, {np.sum(y_holdout == 1)} Cancer.")

# 3. Model Weights Check
model_step = model.named_steps['stable_model']
coefs = model_step.coef_[0]
active_indices = np.where(coefs != 0)[0]
active_features = [model_features[i] for i in active_indices]
print(f"\nModell nutzt {len(active_features)} aktive Features (Gewichte != 0).")

# Check if these active features are present in holdout
active_overlap = set(active_features).intersection(set(X_holdout_raw.columns))
print(f"Von den {len(active_features)} wichtigen Features sind {len(active_overlap)} im Holdout-Set vorhanden.")

# 4. Score Distribution
X_holdout = X_holdout_raw.reindex(columns=model_features, fill_value=0)
probs = model.predict_proba(X_holdout)[:, 1]
print("\nScore-Verteilung (Wahrscheinlichkeit für Krebs):")
print(f"Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}")
