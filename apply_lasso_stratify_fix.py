import json
import re

def patch_notebook(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    # 1. Ensure stratify is imported (already seems to be there but lets be sure and fix the typo in cell 2)
    # Cell 2 typo fix "clincal" -> "clinical" and handle both cases
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'clinical_df_raw = pd.read_csv' in ''.join(cell['source']):
            src = ''.join(cell['source'])
            # Fix if block and typo
            updated_src = src.replace('clinical_df = clincal_df', 'clinical_df = clinical_df')
            # Ensure the M/F filter is clean
            pattern = r'if stratify =="Gender":\s+clinical_df = clinical_df\[clinical_df\["Gender"\]\.isin\(\["M", "F"\]\)\]'
            if 'stratify =="Gender"' in updated_src and '.isin' not in updated_src:
                 updated_src = re.sub(r'if stratify =="Gender":.*?else:', 'if stratify == "Gender":\n    clinical_df = clinical_df[clinical_df["Gender"].isin(["M", "F"])]\nelse:', updated_src, flags=re.DOTALL)
            
            cell['source'] = [l + '\n' if not l.endswith('\n') else l for l in updated_src.splitlines()]

    # 2. Patch run_lasso_for_metrics
    new_func_body = """def run_lasso_for_metrics(df, clinical_df, metrics, pipeline):
    # Pivot
    pivot_df = df.pivot(
        index="sample",
        columns="bin_id",
        values=list(metrics)
    )
    pivot_df.columns = [
        f"{metric}_{bin_id}" for metric, bin_id in pivot_df.columns
    ]

    # Labels and Stratification
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

    print(f"Number Cancer: {sum(y)}")
    print(f"Number Healthy: {len(y) - sum(y)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=strata,
        random_state=42
    )

    # Fit
    pipeline.fit(X_train, y_train)

    # Predict
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    # Koeffizienten
    lasso_model = pipeline.named_steps['lasso_cv']
    n_selected = np.sum(lasso_model.coef_[0] != 0)

    return {
        "metrics": metrics,
        "n_metrics": len(metrics),
        "n_features": X.shape[1],
        "n_selected_features": int(n_selected),
        "roc_auc": auc_score,
        "best_C": lasso_model.C_[0]
    }"""

    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'def run_lasso_for_metrics' in ''.join(cell['source']):
            cell['source'] = [l + '\n' if not l.endswith('\n') else l for l in new_func_body.splitlines()]
            break

    # 3. Patch Re-training cell
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'print(f"Re-training model with best metrics:' in ''.join(cell['source']):
            src = ''.join(cell['source'])
            
            replacement = """y = []
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
X = pivot_df"""
            
            pattern = r'y = \[\].*?X = pivot_df'
            src = re.sub(pattern, replacement, src, flags=re.DOTALL)
            cell['source'] = [l + '\n' if not l.endswith('\n') else l for l in src.splitlines()]
            break

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

patch_notebook('/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/04_lasso_modeling.ipynb')
