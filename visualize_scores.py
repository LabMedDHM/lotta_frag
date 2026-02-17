import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Model and Data
model = joblib.load('final_lasso_model.joblib')
model_features = joblib.load('model_features.joblib')
df_holdout = pd.read_csv("/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/results/final_feature_matrix_gc_corrected_50000.tsv", sep="\t")

# 2. Prepare Features
df_holdout["bin_id"] = df_holdout["chrom"] + "_" + df_holdout["start"].astype(str)
used_metrics = list(set([f.split('_chr')[0] for f in model_features]))
X_raw = df_holdout.pivot(index="sample", columns="bin_id", values=used_metrics)
X_raw.columns = [f"{m}_{bid}" for m, bid in X_raw.columns]
X_holdout = X_raw.reindex(columns=model_features, fill_value=0)

# 3. Get Labels
group_info = df_holdout[['sample', 'group']].drop_duplicates().set_index('sample')
y_true = (group_info.loc[X_holdout.index, "group"].str.lower() != "healthy").astype(int).values
cancer_type = group_info.loc[X_holdout.index, "group"].values

# 4. Predict
probs = model.predict_proba(X_holdout)[:, 1]

# 5. Create Comparison DataFrame
analysis_df = pd.DataFrame({
    'sample': X_holdout.index,
    'prob_cancer': probs,
    'is_cancer': y_true,
    'type': cancer_type
}).sort_values('prob_cancer')

print("\n--- INDIVIDUAL SAMPLE SCORES ---")
print(analysis_df.to_string(index=False))

# 6. Plot distributions
plt.figure(figsize=(10, 6))
for label, group in analysis_df.groupby('is_cancer'):
    name = "Cancer" if label == 1 else "Healthy"
    plt.hist(group['prob_cancer'], bins=15, alpha=0.5, label=name)

plt.axvline(0.5, color='black', linestyle='--')
plt.title("Distribution of Predicted Probabilities on Holdout Set")
plt.xlabel("Probability of being CANCER")
plt.ylabel("Count")
plt.legend()
plt.savefig("holdout_preprocessing/plots/score_distribution.png")
print("\nPlot saved to holdout_preprocessing/plots/score_distribution.png")
