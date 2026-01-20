import json
import os

NOTEBOOK_PATH = '/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/05_lasso_pan_cancer.ipynb'

print(f"Reading notebook: {NOTEBOOK_PATH}")
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the Advanced Pan-Cancer Loop Code
# This version iterates Cancer Types -> Metric Combinations to find the absolute best for each.
advanced_loop_code = r"""# 8. Systematic Pan-Cancer Analysis (Auto-Generated)
# This cell checks all cancer types against healthy controls.
# For each cancer type, it iterates through ALL metric combinations to find the best one.

if bin_size == 50000:
    import pandas as pd
    import numpy as np
    import itertools
    
    print(f"\n=== Starting Advanced Pan-Cancer Analysis (Bin Size: {bin_size}) ===")
    
    # Identify all available cancer types in the raw clinical data
    all_cancer_types = sorted([ct for ct in clinical_df_raw['Patient Type'].unique() if str(ct).lower() != 'healthy'])
    analysis_targets = ['All'] + all_cancer_types
    
    # Define Metrics to iterate over
    # (Same list as used in step 4 or loaded from config)
    metrics_pool = [
        "mean", "median", "stdev", "wps_value", "min", "max"
    ]
    
    summary_results = []

    for target in analysis_targets:
        print(f"\n##################################################")
        print(f"--- Processing Target: {target} ---")
        print(f"##################################################")
        
        # A. Filter Clinical Data for this target
        if target == 'All':
            iter_clinical = clinical_df_raw.copy()
        else:
            iter_clinical = clinical_df_raw[
                (clinical_df_raw['Patient Type'] == target) |
                (clinical_df_raw['Patient Type'].str.lower() == 'healthy')
            ].copy()
            
        # B. Gender Stratification
        if stratify == 'Gender':
            iter_clinical = iter_clinical[iter_clinical['Gender'].isin(['M', 'F'])]
            
        # C. Balancing
        cancer_subset = iter_clinical[iter_clinical['Patient Type'].str.lower() != 'healthy']
        healthy_subset = iter_clinical[iter_clinical['Patient Type'].str.lower() == 'healthy']
        
        n_cancer = len(cancer_subset)
        if n_cancer == 0:
            print(f"[SKIP] No cancer samples found for {target}.")
            continue
            
        # Sample healthy to match cancer count
        if len(healthy_subset) > n_cancer:
            healthy_subset = healthy_subset.sample(n=n_cancer, random_state=42)
        elif len(healthy_subset) == 0:
             print(f"[SKIP] No healthy samples found for {target}.")
             continue
            
        balanced_clinical = pd.concat([cancer_subset, healthy_subset])
        print(f"  > Balanced Dataset: {n_cancer} Cancer vs {len(healthy_subset)} Healthy")
        
        # D. Filter Feature Matrix (df)
        valid_ids = balanced_clinical['Extracted_ID'].unique()
        iter_df = df[df['sample'].isin(valid_ids)].copy()
        
        if iter_df.empty:
             print(f"[WARN] Feature matrix empty for these IDs.")
             continue

        # E. Metric Optimization Loop
        # We need to find the best metric combo for THIS specific cancer type
        print(f"  > Optimizing metrics for {target}...")
        
        best_cancer_auc = -1.0
        best_cancer_result = None
        
        # Iterate all combinations (1 to len)
        # To save time, you might want to limit 'r' if needed, but for 6 metrics it's fast enough (63 combos).
        for r in range(1, len(metrics_pool) + 1):
            for combination in itertools.combinations(metrics_pool, r):
                try:
                    # Run Pipeline
                    # Note: We rely on 'run_lasso_for_metrics' returning a dict with 'roc_auc'
                    res = run_lasso_for_metrics(iter_df, balanced_clinical, combination, pipeline)
                    
                    current_auc = res.get('roc_auc', 0)
                    
                    if current_auc > best_cancer_auc:
                        best_cancer_auc = current_auc
                        best_cancer_result = res
                        best_cancer_result['Cancer_Type'] = target
                        best_cancer_result['Best_Metrics_Combo'] = str(combination)
                        
                except Exception as e:
                    # Occasional failures (e.g. convergence) shouldn't stop the loop
                    pass
        
        if best_cancer_result:
            print(f"  >>> BEST Result for {target}: AUC={best_cancer_auc:.4f} using {best_cancer_result['Best_Metrics_Combo']}")
            
            # Enrich result with metadata
            best_cancer_result['N_Cancer'] = n_cancer
            best_cancer_result['N_Healthy'] = len(healthy_subset)
            
            # Rename for display clarity
            summary_entry = {
                'Cancer_Type': best_cancer_result['Cancer_Type'],
                'Best_AUC': best_cancer_result['roc_auc'],
                'Best_Metrics': best_cancer_result['Best_Metrics_Combo'],
                'Best_C': best_cancer_result['best_C'],
                'Num_Features': best_cancer_result['n_selected_features'],
                'N_Samples': n_cancer + len(healthy_subset)
            }
            summary_results.append(summary_entry)
        else:
            print(f"  [FAIL] Could not find a valid model for {target}.")

    # F. Final Output & Saving
    if summary_results:
        final_results_df = pd.DataFrame(summary_results).sort_values(by='Best_AUC', ascending=False)
        
        print("\n\n================================================")
        print("=== FINAL PAN-CANCER RESULTS (OPTIMIZED) ===")
        print("================================================")
        display(final_results_df)
        
        save_path = f"/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/lasso_results_{bin_size}_optimized.csv"
        final_results_df.to_csv(save_path, index=False)
        print(f"\nSaved to: {save_path}")
        
        # Visualization
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=final_results_df, x='Cancer_Type', y='Best_AUC', palette='viridis')
            plt.title(f'Optimized Lasso AUC by Cancer Type (Bin: {bin_size})', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1.05)
            plt.axhline(0.5, color='red', linestyle='--')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")
    else:
        print("No results to show.")

else:
    print(f"Bin Size is {bin_size}, skipping 50k specific analysis.")
"""

# Replace the LAST cell if it looks like the old Pan-Cancer loop, or append if not.
# We'll check the cell content.
updated = False
if len(nb['cells']) > 0:
    last_cell_source = "".join(nb['cells'][-1]['source'])
    if "Pan-Cancer Analysis Loop" in last_cell_source:
        print("Replacing existing Pan-Cancer Analysis cell with Advanced version...")
        nb['cells'][-1]['source'] = advanced_loop_code.splitlines(True)
        updated = True

if not updated:
    print("Appending new Advanced Pan-Cancer Analysis cell...")
    # Add a markdown cell introducing it if we are appending fresh
    if "Systematic Pan-Cancer Analysis" not in str(nb['cells']):
         nb['cells'].append({
           "cell_type": "markdown",
           "metadata": {},
           "source": ["# 8. Systematic Pan-Cancer Analysis (Advanced)\n", "Iterates all cancer types and optimizes metrics for each."]
        })
    
    nb['cells'].append({
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": advanced_loop_code.splitlines(True)
    })

print(f"Writing modified notebook to: {NOTEBOOK_PATH}")
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Done.")
