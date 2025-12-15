import json
import numpy as np # Import locally not needed but good for syntax highlighting in idea
# The notebook file path
notebook_path = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/04_lasso_modeling.ipynb"

try:
    with open(notebook_path, "r") as f:
        nb = json.load(f)

    # 1. Fix the loop
    loop_fix_found = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "run_lasso_for_metrics(df, clinical_df, combination, combination, pipeline)" in source:
                new_source = source.replace(
                    "run_lasso_for_metrics(df, clinical_df, combination, combination, pipeline)",
                    "run_lasso_for_metrics(df, clinical_df, combination, pipeline)"
                )
                # Split keeping ends to preserve newlines structure if possible, though join/split usually suffices
                cell["source"] = new_source.splitlines(keepends=True)
                loop_fix_found = True
                print("Fixed the 'run_lasso_for_metrics' call loop.")
                break

    if not loop_fix_found:
        print("Warning: Could not find the loop to fix. It might have been fixed already.")

    # 2. Inject re-training logic and ROC plot
    # Look for the cell that does: pipeline = best_result["pipeline"]
    plot_fix_found = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if 'pipeline = best_result["pipeline"]' in source:
                 
                 new_code = [
                    "# --- Re-train on Best Metrics to get Model and Predictions ---\n",
                    "# We need to re-run the pipeline on the best combination to get the fitted model and test data\n",
                    "# run_lasso_for_metrics does not return the pipeline, so we perform the split and fit manually here.\n",
                    "\n",
                    "print(f\"Re-training model with best metrics: {best_metrics}\")\n",
                    "\n",
                    "# Recalculate Pivot for best metrics\n",
                    "# (Logic copied from run_lasso_for_metrics but keeping the objects)\n",
                    "\n",
                    "# Pivot\n",
                    "pivot_df = df.pivot(\n",
                    "    index=\"sample\",\n",
                    "    columns=\"bin_id\",\n",
                    "    values=list(best_metrics)\n",
                    ")\n",
                    "pivot_df.columns = [\n",
                    "    f\"{metric}_{bin_id}\" for metric, bin_id in pivot_df.columns\n",
                    "]\n",
                    "\n",
                    "# Labels\n",
                    "y = []\n",
                    "strata = []\n",
                    "for sample_id in pivot_df.index:\n",
                    "    row = clinical_df[clinical_df[\"Extracted_ID\"] == sample_id].iloc[0]\n",
                    "    is_healthy = row[\"Patient Type\"].lower() == \"healthy\"\n",
                    "    y.append(0 if is_healthy else 1)\n",
                    "    strata.append(row[\"Gender\"])\n",
                    "\n",
                    "y = np.array(y)\n",
                    "X = pivot_df\n",
                    "\n",
                    "# Split\n",
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    X, y,\n",
                    "    test_size=0.2,\n",
                    "    stratify=strata,\n",
                    "    random_state=42\n",
                    ")\n",
                    "\n",
                    "# Fit\n",
                    "pipeline.fit(X_train, y_train)\n",
                    "\n",
                    "# Predict\n",
                    "y_prob = pipeline.predict_proba(X_test)[:, 1]\n",
                    "auc_score = roc_auc_score(y_test, y_prob)\n",
                    "print(f\"Confirmed ROC AUC on test set: {auc_score:.4f}\")\n",
                    "\n",
                    "lasso_cv = pipeline.named_steps['lasso_cv']\n",
                    "\n",
                    "# --- ROC Curve Plot ---\n",
                    "fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')\n",
                    "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
                    "plt.xlim([0.0, 1.0])\n",
                    "plt.ylim([0.0, 1.05])\n",
                    "plt.xlabel('False Positive Rate')\n",
                    "plt.ylabel('True Positive Rate')\n",
                    "plt.title('Receiver Operating Characteristic (Best Model)')\n",
                    "plt.legend(loc=\"lower right\")\n",
                    "plt.grid(True)\n",
                    "plt.show()\n",
                    "\n",
                    "# --- Lasso Parameter Tuning Plot ---\n",
                    "mean_scores = np.mean(lasso_cv.scores_[1], axis=0)\n",
                    "std_scores = np.std(lasso_cv.scores_[1], axis=0)\n",
                    "cs = lasso_cv.Cs_\n",
                    "best_idx = np.argmax(mean_scores)\n",
                    "best_c = cs[best_idx]\n",
                    "\n",
                    "plt.figure(figsize=(10,6))\n",
                    "plt.semilogx(cs, mean_scores, marker='o', label='Mean CV Score (ROC AUC)')\n",
                    "plt.fill_between(cs, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color='gray', label='Std Dev')\n",
                    "plt.axvline(best_c, linestyle='--', color='r', label=f'Best C = {best_c:.2e}')\n",
                    "plt.title(\"Lasso Parameter Tuning\")\n",
                    "plt.xlabel(\"C (Inverse Regularization Strength)\")\n",
                    "plt.ylabel(\"CV Score (ROC AUC)\")\n",
                    "plt.legend()\n",
                    "plt.grid(True)\n",
                    "plt.show()\n"
                 ]
                 cell["source"] = new_code
                 plot_fix_found = True
                 print("Updated plotting logic.")
                 break

    if not plot_fix_found:
        print("Warning: Could not find the plotting cell to fix.")

    if loop_fix_found or plot_fix_found:
        with open(notebook_path, "w") as f:
            json.dump(nb, f, indent=1)
        print("Notebook saved successfully.")
    else:
        print("No changes made.")

except Exception as e:
    print(f"Error patching notebook: {e}")
