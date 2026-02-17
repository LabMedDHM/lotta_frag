import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
train_path = "/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_gc_corrected_50000.tsv"
holdout_path = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/results/final_feature_matrix_gc_corrected_50000.tsv"

print("Loading data for distribution comparison...")
# Load only relevant columns to save memory
cols = ['sample', 'group', 'wps_value']
df_train = pd.read_csv(train_path, sep="\t", usecols=cols)
df_hold = pd.read_csv(holdout_path, sep="\t", usecols=cols)

# Label datasets
df_train['dataset'] = 'Training'
df_hold['dataset'] = 'Holdout'

# Simplify groups to Cancer vs Healthy
df_train['status'] = df_train['group'].apply(lambda x: 'Healthy' if x.lower() == 'healthy' else 'Cancer')
df_hold['status'] = df_hold['group'].apply(lambda x: 'Healthy' if x.lower() == 'healthy' else 'Cancer')

# Combine for plotting
combined = pd.concat([df_train, df_hold], ignore_index=True)
combined['label'] = combined['dataset'] + " - " + combined['status']

print("Generating distribution plot...")
plt.figure(figsize=(12, 7))

# Set color palette
palette = {
    'Training - Healthy': '#3498db',
    'Training - Cancer': '#2980b9',
    'Holdout - Healthy': '#e74c3c',
    'Holdout - Cancer': '#c0392b'
}

# Kernel Density Estimate Plot
sns.kdeplot(data=combined, x='wps_value', hue='label', palette=palette, common_norm=False, fill=True, alpha=0.3)

plt.title("Comparison of WPS Value Distributions: Training vs. Holdout", fontsize=15)
plt.xlabel("WPS Value (Adjusted)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axvline(combined[combined['dataset']=='Training']['wps_value'].mean(), color='#2980b9', linestyle='--', label='Mean Training')
plt.axvline(combined[combined['dataset']=='Holdout']['wps_value'].mean(), color='#c0392b', linestyle='--', label='Mean Holdout')

os.makedirs("holdout_preprocessing/plots", exist_ok=True)
plot_path = "holdout_preprocessing/plots/distribution_comparison.png"
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to {plot_path}")

# Calculate summary stats for the user
stats = combined.groupby('label')['wps_value'].agg(['mean', 'std', 'median']).reset_index()
print("\n--- Summary Statistics ---")
print(stats.to_string(index=False))
