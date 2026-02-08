import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import importlib
import config

importlib.reload(config)
from config import BIN_SIZE as bin_size
from config import ANALYSIS_MODE as analysis_mode
from config import SPECIFIC_GROUP as specific_group
from config import STRATIFY_BY as stratify

print(analysis_mode)
print(specific_group)
print(bin_size)

clinical_path = "/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/filtered_clinical_characteristics.csv"
clinical_df_raw = pd.read_csv(clinical_path, sep=";")

if analysis_mode == "specific_vs_healthy":
    clinical_df = clinical_df_raw[
        (clinical_df_raw["Patient Type"] == specific_group)
        | (clinical_df_raw["Patient Type"].str.lower() == "healthy")
    ].copy()
else:
    clinical_df = clinical_df_raw.copy()

clinical_df = clinical_df[
    clinical_df["Age at Diagnosis"].notna() & clinical_df["Gender"].notna()
]
clinical_df = clinical_df[clinical_df["Gender"].isin(["M", "F"])]

cancer_df = clinical_df[clinical_df["Patient Type"].str.lower() != "healthy"]
print(f"Cancer samples: {cancer_df.shape[0]}")
healthy_df = clinical_df[clinical_df["Patient Type"].str.lower() == "healthy"]
print(f"Healthy samples (before balancing): {healthy_df.shape[0]}")
n_cancer = len(cancer_df)

healthy_df = healthy_df.sample(n=n_cancer, random_state=42)
clinical_df = pd.concat([cancer_df, healthy_df]).copy()
print(f"Final dataset: {len(clinical_df)} samples (balanced)")

clinical_df["AgeGroup"] = pd.cut(
    clinical_df["Age at Diagnosis"], bins=[0, 69, 120], labels=["<70", "70+"]
)

X = clinical_df[["Age at Diagnosis", "Gender"]].copy()
X["Gender"] = (X["Gender"] == "M").astype(int)
y = (clinical_df["Patient Type"].str.lower() != "healthy").astype(int).values

if stratify == "Gender+Age":
    strata = (
        clinical_df["Gender"].astype(str) + "_" + clinical_df["AgeGroup"].astype(str)
    ).values
elif stratify == "Gender":
    strata = clinical_df["Gender"].astype(str).values
else:
    strata = y

assert X.shape[0] == len(y) == len(strata)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=strata, random_state=42
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

pipeline.fit(X_train, y_train)

y_prob_test = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob_test)

print(f"\nAUC-Wert (Alter und Geschlecht): {auc:.4f}")

model = pipeline.named_steps["model"]
age_coef = model.coef_[0][0]
gender_coef = model.coef_[0][1]
intercept = model.intercept_[0]

print(f"\nKoeffizienten:")
print(f"Alter: {age_coef:.4f}")
print(f"Geschlecht (M=1, F=0): {gender_coef:.4f}")
print(f"Intercept: {intercept:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_prob_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.title("ROC-Kurve - Nur Alter und Geschlecht")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
