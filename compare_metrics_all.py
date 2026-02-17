import pandas as pd
import numpy as np

# Paths
train_path = "/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_gc_corrected_50000.tsv"
holdout_path = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/results/final_feature_matrix_gc_corrected_50000.tsv"

metrics = ['mean', 'median', 'stdev', 'min', 'max', 'wps_value']

def analyze_dataset(path, name):
    print(f"\nAnalyzing {name}...")
    # Load data
    df = pd.read_csv(path, sep="\t", usecols=['sample', 'group'] + metrics)
    
    # Split into Healthy and Cancer
    df['status'] = df['group'].apply(lambda x: 'Healthy' if x.lower() == 'healthy' else 'Cancer')
    
    results = {}
    for status in ['Healthy', 'Cancer']:
        subset = df[df['status'] == status]
        if len(subset) == 0:
            continue
        
        stat_dict = {}
        for m in metrics:
            stat_dict[m] = {
                "mean": subset[m].mean(),
                "std": subset[m].std(),
                "n": len(subset)
            }
        results[status] = stat_dict
    
    return results

# Perform analysis
train_stats = analyze_dataset(train_path, "TRAINING")
holdout_stats = analyze_dataset(holdout_path, "HOLDOUT")

# Print Comparison Table
print("\n" + "="*80)
print(f"{'Metric':12} | {'Train Healthy':15} | {'Holdout Healthy':15} | {'Train Cancer':15} | {'Holdout Cancer':15}")
print("-" * 80)

for m in metrics:
    th = train_stats['Healthy'][m]['mean']
    hh = holdout_stats['Healthy'][m]['mean']
    tc = train_stats['Cancer'][m]['mean']
    hc = holdout_stats['Cancer'][m]['mean']
    print(f"{m:12} | {th:15.4f} | {hh:15.4f} | {tc:15.4f} | {hc:15.4f}")

print("="*80)

# Calculate Shifts
print("\n--- SHIFT ANALYSIS (Holdout vs Training) ---")
for m in metrics:
    m_holdout = holdout_stats['Healthy'][m]['mean']
    m_train = train_stats['Healthy'][m]['mean']
    shift_h = ((m_holdout - m_train) / m_train) * 100
    print(f"Shift in {m:12}: {shift_h:+.2f}%")
print("\n--- COHEN'S D (Train Cancer vs Holdout Cancer) ---")

for m in metrics:
    mean1 = train_stats['Cancer'][m]['mean']
    std1 = train_stats['Cancer'][m]['std']
    n1 = train_stats['Cancer'][m]['n']
    
    mean2 = holdout_stats['Cancer'][m]['mean']
    std2 = holdout_stats['Cancer'][m]['std']
    n2 = holdout_stats['Cancer'][m]['n']
    
    # pooled std
    pooled_std = np.sqrt(
        ((n1 - 1)*std1**2 + (n2 - 1)*std2**2) / (n1 + n2 - 2)
    )
    
    d = (mean1 - mean2) / pooled_std
    
    print(f"{m:12}: d = {d:.3f}")
