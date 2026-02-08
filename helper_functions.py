from config import STRATIFY_BY 
from cv_lasso_single_fold import cross_validation, analyze_feature_stability
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import itertools
import importlib
from sklearn.base import clone
import config
importlib.reload(config)
from config import BIN_SIZE, ANALYSIS_MODE, SPECIFIC_GROUP, STRATIFY_BY
import seaborn as sns
from matplotlib.colors import ListedColormap


def preprocess_data(df, clinical_df, STRATIFY_BY, metrics):
    pivot_df = df.pivot(index="sample", columns="bin_id", values=list(metrics))
    pivot_df.columns = [f"{metric}_{bin_id}" for metric, bin_id in pivot_df.columns]

    # Align clinical data exactly to X: Gleiche reihenfolge wie pivot_df
    clinical_df_sub = (
        clinical_df
        .set_index("Extracted_ID")
        .loc[pivot_df.index]
    )

    # Labels: binäre Klassifikation
    y = (clinical_df_sub["Patient Type"].str.lower() != "healthy").astype(int).values
    
    if STRATIFY_BY == "Gender+Age":
        strata = (
            clinical_df_sub["Gender"].astype(str)
            + "_"
            + clinical_df_sub["AgeGroup"].astype(str)
        ).values

    elif STRATIFY_BY == "Gender":
        strata = (
            clinical_df_sub["Gender"].astype(str)
        ).values

    else:
        strata = y

    X = pivot_df

    # Safety check (very recommended)
    n_nans = X.isna().sum().sum()
    if n_nans > 0:
        print(f"{n_nans} NaNs found in dataframe")

    else:
        print("No NaNs in dataframe")
    assert X.shape[0] == len(y) == len(strata)
    strata_counts = pd.Series(strata).value_counts()
    rare_strata = strata_counts[strata_counts == 1]
    if len(rare_strata) > 0:
        print("WARNING: These strata only have 1 sample (train_test_split will fail):")
        print(rare_strata)
        rare_samples = [pivot_df.index[i] for i, s in enumerate(strata) if s in rare_strata.index]
        print("Samples in these rare strata:", rare_samples)

    # Split in Training and Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=0.2, 
        stratify=strata , 
        random_state=42
    )
    return X_train, X_test, y_train, y_test


def get_stable_pipeline(c_1se):
    stable_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('stable_model', LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C=c_1se, 
        max_iter=10000, 
        random_state=42
    ))
])
    return stable_pipeline

def get_simple_pipeline():
    C_values = np.logspace(-4, 4, 50)
    pipeline = Pipeline([
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
    return pipeline

def get_fast_pipeline():
    fast_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso_cv', LogisticRegressionCV(
        Cs=15, 
        cv=2, 
        penalty='l1', 
        solver='liblinear', 
        scoring='roc_auc', 
        max_iter=2000, 
        random_state=42
    ))
])
    return fast_pipeline


def calculate_cs(simple_pipeline):
    lasso_cv = simple_pipeline.named_steps['lasso_cv']

    mean_scores = np.mean(lasso_cv.scores_[1], axis=0)
    best_idx = np.argmax(mean_scores)
    best_score = mean_scores[best_idx]
    best_c = float(lasso_cv.Cs_[best_idx])

    std_scores = np.std(lasso_cv.scores_[1], axis=0)
    sem_scores = std_scores / np.sqrt(5)
    threshold = best_score - sem_scores[best_idx]
    idx_1se = np.where(mean_scores >= threshold)[0][0]
    c_1se = float(lasso_cv.Cs_[idx_1se])

    return best_c, c_1se    


def calculate_stability(X_train, y_train, pipeline, stable_pipeline):
    cv_results = cross_validation(X_train, y_train, pipeline, n_folds=5)
    stability_df = analyze_feature_stability(cv_results)

    stable_feature_names = set(stability_df[stability_df['Frequency'] == 5]['Feature'])
    n_stable = len(stable_feature_names)

    pars_feature_names = set(X_train.columns[stable_pipeline.named_steps['model'].coef_[0] != 0])
    pars_overlap = pars_feature_names.intersection(stable_feature_names)
    pars_stability_ratio = len(pars_overlap) / len(pars_feature_names) if len(pars_feature_names) > 0 else 0.0
    n_pars = len(pars_feature_names)

    simple_feature_names = set(X_train.columns[lasso_cv.coef_[0] != 0])
    simple_overlap = simple_feature_names.intersection(stable_feature_names)
    simple_stability_ratio = len(simple_overlap) / len(simple_feature_names) if len(simple_feature_names) > 0 else 0.0
    n_simple = len(simple_feature_names)
    
    c_values = [res.get('best_C', np.nan) for res in cv_results]
    c_values = [c for c in c_values if c > 0 and not np.isnan(c)]
    c_variation = np.std(np.log10(c_values)) if len(c_values) > 0 else np.nan

    cv_auc = np.mean([e['auc'] for e in cv_results])

    return n_stable, n_simple, n_pars, simple_stability_ratio, pars_stability_ratio, c_variation, cv_auc
