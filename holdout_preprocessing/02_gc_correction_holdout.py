import pandas as pd
from pyfaidx import Fasta
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import os


# Paths
bin_size = 50000
INPUT_DIR = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/results"
genome_fasta = "/labmed/workspace/lotta/data/hg38.fa"
input_path = os.path.join(INPUT_DIR, f"final_feature_matrix_blacklist_filtered_{bin_size}.tsv")
output_path = os.path.join(INPUT_DIR, f"final_feature_matrix_gc_corrected_{bin_size}.tsv")

if not os.path.exists(input_path):
    print(f"Error: Run 01_blacklist_holdout.py first. Missing: {input_path}")
    exit(1)

print(f"Loading filtered matrix from {input_path}...")
df = pd.read_csv(input_path, sep="\t")

df["start"] = df["bin"] * bin_size
df["end"] = df["start"] + bin_size

print(f"Loading genome fasta from {genome_fasta}...")
genome = Fasta(genome_fasta)

def calc_gc(chrom, start, end):
    if chrom not in genome:
        return np.nan
    seq = genome[chrom][start:end].seq.upper()
    if len(seq) == 0:
        return np.nan
    return (seq.count("G") + seq.count("C")) / len(seq)

print("Calculating GC-content per bin...")
# Optimization: Calculate unique bins first
unique_bins = df[["chrom", "bin", "start", "end"]].drop_duplicates()
unique_bins["GC"] = unique_bins.apply(lambda row: calc_gc(str(row["chrom"]), int(row["start"]), int(row["end"])), axis=1)

# Merge back
df = df.merge(unique_bins[["chrom", "bin", "GC"]], on=["chrom", "bin"], how="left")

# Identify coverage columns
coverage_cols = [
    col for col in df.columns
    if any(x in col.lower() for x in ["mean", "median", "stdev", "min", "max", "wps_value"])
]

print(f"Coverage columns to correct: {coverage_cols}")
gc_corrected_df = df.copy()

for sample in df["sample"].unique():
    print(f"Processing sample: {sample}")
    subset_idx = df[df["sample"] == sample].index
    subset = df.loc[subset_idx]
    
    # Fill NaNs in GC if any (shouldn't happen for autosomes)
    valid_gc_mask = subset["GC"].notna()
    if not valid_gc_mask.any():
        continue

    for col in coverage_cols:
        y = subset.loc[valid_gc_mask, col].values
        x = subset.loc[valid_gc_mask, "GC"].values
        
        if len(y) < 2:
            continue
            
        # Locally Estimated Scatterplot Smoothing
        loess_fit = lowess(endog=y, exog=x, frac=0.75, return_sorted=False)
        corrected = y - loess_fit + np.median(y)
        gc_corrected_df.loc[subset.loc[valid_gc_mask].index, col] = corrected

# Reorder columns to match training matrix exactly
column_order = ["sample", "group", "chrom", "bin", "mean", "median", "stdev", "min", "max", "wps_value", "start", "end", "GC"]
# Ensure all columns exist before reordering
column_order = [c for c in column_order if c in gc_corrected_df.columns]
gc_corrected_df = gc_corrected_df[column_order]

# Cleanup and save
gc_corrected_df.to_csv(output_path, sep="\t", index=False)
print(f"GC-corrected holdout matrix saved to: {output_path}")
