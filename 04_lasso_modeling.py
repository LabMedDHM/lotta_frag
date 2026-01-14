import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import importlib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Import config
import config
importlib.reload(config)
from config import BIN_SIZE as bin_size
from config import ANALYSIS_MODE as analysis_mode
from config import SPECIFIC_GROUP as specific_group
from config import STRATIFY_BY as stratify

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

print(f"Analysis Mode: {analysis_mode}")
print(f"Specific Group: {specific_group}")
print(f"Bin Size: {bin_size}")

# 1. Loading of Dataframes
matrix_path = f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_gc_corrected_{bin_size}.tsv"
df = pd.read_csv(matrix_path, sep="\t")

clinical_path = "/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/filtered_clinical_characteristics.csv"
clinical_df_raw = pd.read_csv(clinical_path, sep=";")

if analysis_mode == "specific_vs_healthy":
    clinical_df = clinical_df_raw[
            (clinical_df_raw["Patient Type"] == specific_group) |
            (clinical_df_raw["Patient Type"].str.lower() == "healthy")
        ].copy()
else:
    clinical_df = clinical_df_raw.copy()

if stratify == "Gender":
    clinical_df = clinical_df[clinical_df["Gender"].isin(["M", "F"])]

# Balancing: Sample as many Healthy as there are Cancer samples
cancer_df = clinical_df[clinical_df["Patient Type"].str.lower() != "healthy"]
healthy_df = clinical_df[clinical_df["Patient Type"].str.lower() == "healthy"]
n_cancer = len(cancer_df)

if len(healthy_df) > n_cancer:
    healthy_df = healthy_df.sample(n=n_cancer, random_state=42)
clinical_df = pd.concat([cancer_df, healthy_df]).copy()

valid_samples = clinical_df["Extracted_ID"].unique()
df = df[df["sample"].isin(valid_samples)].copy()
df["bin_id"] = df["chrom"] + "_" + df["start"].astype(str)

print(f"Number of Samples in Matrix: {df['sample'].nunique()}")
print(f"Number of Bins per Sample: {len(df) / df['sample'].nunique()}")

# 2. Pipeline for LASSO
C_values = np.logspace(-2, 2, 50)
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('lasso_cv', LogisticRegressionCV(
        Cs=C_values,
        penalty='l1',
        solver='liblinear',
        cv=5,
        scoring='roc_auc',
        max_iter=10000,
        random_state=42
    ))
])

# 3. General Function for LASSO performance
def run_lasso_for_metrics(df, clinical_df, metrics, pipeline):
    pivot_df = df.pivot(
        index="sample",
        columns="bin_id",
        values=list(metrics)
    )
    pivot_df.columns = [
        f"{metric}_{bin_id}" for metric, bin_id in pivot_df.columns
    ]

    y = []
    strata = []
    for sample_id in pivot_df.index:
        row = clinical_df[clinical_df["Extracted_ID"] == sample_id].iloc[0]
        is_healthy = row["Patient Type"].lower() == "healthy"
        target_val = 0 if is_healthy else 1
        y.append(target_val)
        if stratify == "Gender":
            strata.append(row["Gender"])
        else:
            strata.append(target_val)

    y = np.array(y)
    X = pivot_df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=strata,
        random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    lasso_model = pipeline.named_steps['lasso_cv']
    n_selected = np.sum(lasso_model.coef_[0] != 0)

    return {
        "metrics": metrics,
        "n_metrics": len(metrics),
        "n_features": X.shape[1],
        "n_selected_features": int(n_selected),
        "roc_auc": auc_score,
        "best_C": lasso_model.C_[0]
    }, X_train, X_test, y_train, y_test

# 4. Feature Selection for LASSO (combinations of metrics)
metrics_list = ["mean", "median", "stdev", "wps_value", "min", "max"]
metrics_results = []

print("Evaluating metric combinations...")
for r in range(1, len(metrics_list) + 1):
    for combination in itertools.combinations(metrics_list, r):
        res, _, _, _, _ = run_lasso_for_metrics(df, clinical_df, combination, pipeline)
        metrics_results.append(res)

metrics_with_wps = []
for res in metrics_results:
    if any("wps_value" in m for m in res["metrics"]):
        metrics_with_wps.append(res)

metrics_df = pd.DataFrame(metrics_with_wps).sort_values("roc_auc", ascending=False)
output_csv = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/dataframes_notebooklasso_metrics_results.csv"
metrics_df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")

# 5. Influence of metric selection on model performance
plt.figure(figsize=(10, 6))
metrics_df.groupby("n_metrics")["roc_auc"].mean().plot(marker='o')
plt.title("Mean ROC AUC vs Number of Metrics")
plt.ylabel("ROC AUC")
plt.xlabel("Number of Metrics")
plt.grid(True)
plt.savefig("plots/mean_roc_auc_vs_n_metrics.png")
plt.close()

# 6. Re-training with best metrics and plotting
best_metrics = metrics_df.iloc[0]['metrics']
print(f"Best metrics: {best_metrics}")

res, X_train, X_test, y_train, y_test = run_lasso_for_metrics(df, clinical_df, best_metrics, pipeline)

# Training vs Test ROC
y_prob_train = pipeline.predict_proba(X_train)[:, 1]
y_prob_test = pipeline.predict_proba(X_test)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
auc_train = roc_auc_score(y_train, y_prob_train)
auc_test = roc_auc_score(y_test, y_prob_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC (AUC = {auc_train:.2f})')
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC (AUC = {auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Training vs. Test ROC Performance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("plots/roc_performance.png")
plt.close()

# 7. Important Features
lasso_model = pipeline.named_steps['lasso_cv']
coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lasso_model.coef_[0]
})
important_features = coef_df[coef_df["Coefficient"] != 0].sort_values(by="Coefficient", ascending=False)
important_features.to_csv("plots/important_features.csv", index=False)

print(f"Number of Important Features: {len(important_features)}")
print("Top 10 Features:")
print(important_features.head(10))

# 8. Feature Stability Analysis (Cross-Validation)
from cv_lasso_single_fold import cross_validation, analyze_feature_stability, plot_roc_curves, plot_auc_boxplot

print("Running 5-Fold Cross-Validation for Stability...")
cv_results = cross_validation(df, clinical_df, best_metrics, n_splits=5, stratify_col=stratify)

# Save CV plots
plt.figure()
plot_roc_curves(cv_results)
plt.savefig("plots/cv_roc_curves.png")
plt.close()

plt.figure()
plot_auc_boxplot(cv_results)
plt.savefig("plots/cv_auc_boxplot.png")
plt.close()

stability_df = analyze_feature_stability(cv_results)
stability_df.to_csv("plots/feature_stability.csv", index=False)

print("Background analysis complete.")
