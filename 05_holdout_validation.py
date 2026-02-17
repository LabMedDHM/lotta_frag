#!/usr/bin/env python
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Configuration
MODEL_PATH = 'final_lasso_model.joblib'
FEATURES_PATH = 'model_features.joblib'
HOLDOUT_MATRIX_PATH = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/results/final_feature_matrix_gc_corrected_50000.tsv"
BIN_SIZE = 50000

print("--- Final Holdout Validation ---")

# 1. Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    print(f"ERROR: Model or Feature list not found!")
    print(f"Please run the following in your Training Notebook (04):")
    print(f"  joblib.dump(stable_pipeline, 'final_lasso_model.joblib')")
    print(f"  joblib.dump(X_train.columns.tolist(), 'model_features.joblib')")
    exit(1)

# 2. Load the stored Model and Feature names
print("Loading saved pipeline and features...")
model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)

# NEW: Print model parameters for confirmation
model_step = model.named_steps['stable_model']
print(f"\n--- Model Parameters ---")
print(f"Used C-value (c_1se): {model_step.C}")
print(f"Penalty:              {model_step.penalty}")
print(f"Active Features:      {np.sum(model_step.coef_ != 0)} / {len(model_features)}")
print(f"------------------------\n")

# Detect which metrics are needed (e.g., 'mean', 'median', 'stdev' detected from feature names)
# Feature names in the model look like 'metric_chr1_900000'
used_metrics = list(set([f.split('_chr')[0] for f in model_features]))
print(f"Metrics required by model: {used_metrics}")

# 3. Load Holdout Data
print("Loading holdout matrix...")
if not os.path.exists(HOLDOUT_MATRIX_PATH):
    print(f"ERROR: Holdout matrix not found at {HOLDOUT_MATRIX_PATH}")
    exit(1)

df_holdout = pd.read_csv(HOLDOUT_MATRIX_PATH, sep="\t")

# 4. Prepare Holdout Features (Pivoting)
print("Formatting holdout features to match model structure...")
# Add bin_id (format chrom_start) like in your helper_functions
df_holdout["bin_id"] = df_holdout["chrom"] + "_" + df_holdout["start"].astype(str)

# Pivot the long-format holdout data into the wide format the model expects
X_holdout_raw = df_holdout.pivot(index="sample", columns="bin_id", values=needed_metrics)

# Flatten columns to match 'metric_chrom_start' (e.g., 'mean_chr1_900000')
X_holdout_raw.columns = [f"{m}_{bid}" for m, bid in X_holdout_raw.columns]

# IMPORTANT: Align columns with the model features (handle missing bins and order)
X_holdout = X_holdout_raw.reindex(columns=model_features, fill_value=0)
print(f"Feature alignment complete. Shape: {X_holdout.shape}")

# 5. Extract Ground Truth (y_holdout)
# We use the 'group' column which contains the folder names (healthy, pancreatic, etc.)
group_info = df_holdout[['sample', 'group']].drop_duplicates().set_index('sample')
# 1 if cancer, 0 if healthy
y_holdout = (group_info.loc[X_holdout.index, "group"].str.lower() != "healthy").astype(int).values

# 6. Prediction
print(f"Running prediction on {len(X_holdout)} samples...")
probs = model.predict_proba(X_holdout)[:, 1]

# 7. Evaluation (AUC Score)
auc_score = roc_auc_score(y_holdout, probs)
print(f"\n" + "="*40)
print(f"🔥 FINAL HOLDOUT AUC: {auc_score:.4f}")
print("="*40 + "\n")

# 8. Save Results
output_dir = "holdout_preprocessing/results"
os.makedirs(output_dir, exist_ok=True)
results_df = pd.DataFrame({
    'sample': X_holdout.index,
    'prediction_score': probs,
    'true_label': y_holdout
})
results_df.to_csv(f"{output_dir}/holdout_validation_results.csv", index=False)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_holdout, probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkred', lw=3, label=f'Holdout ROC (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final Holdout Validation (Blind Test)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig("holdout_preprocessing/plots/holdout_final_roc_joblib.png")
print("Detailed results and plots saved in 'holdout_preprocessing/'.")
