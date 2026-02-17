import os
import pandas as pd
import numpy as np
import pyBigWig
import math
import glob


# Paths
bin_size = 50000
BASE_DIR = "/labmed/workspace/lotta/finaletoolkit/carsten/outputs_holdout"
FRAG_INTERVAL_DIR = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/frag_intervals"
OUTPUT_DIR = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def discover_samples(base_dir):
    samples = []
    # Search for .adjust_wps.bw files
    bw_files = glob.glob(os.path.join(base_dir, "**", "*.adjust_wps.bw"), recursive=True)
    for f in bw_files:
        sample_id = os.path.basename(f).replace(".adjust_wps.bw", "")
        group_folder = os.path.basename(os.path.dirname(f))
        samples.append({
            "sample": sample_id,
            "path": f,
            "group": group_folder
        })
    return pd.DataFrame(samples)

def get_binned_wps(bw_path, bin_size):
    bw = pyBigWig.open(bw_path)
    chroms = bw.chroms()
    results = []
    autosomes = [f"chr{i}" for i in range(1, 23)]
    for chrom in autosomes:
        if chrom not in chroms: continue
        intervals = bw.intervals(chrom)
        if not intervals: continue
        df_chrom = pd.DataFrame(intervals, columns=["start", "end", "value"])
        df_chrom["bin"] = df_chrom["start"] // bin_size
        binned = df_chrom.groupby("bin")["value"].mean().reset_index()
        binned["chrom"] = chrom
        results.append(binned)
    bw.close()
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def get_binned_frags(sample_id, bin_size):
    # Try to find frag interval file
    pattern = os.path.join(FRAG_INTERVAL_DIR, "**", f"{sample_id}.frag_length_intervals.bed")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
        
    df = pd.read_csv(files[0], sep="\t", header=None,
                     names=["chrom", "start", "stop", "name", "mean", "median", "stdev", "min", "max"])
    df = df.iloc[1:].copy() # Skip header if present
    num_cols = ["mean", "median", "stdev", "min", "max"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df["start"] = pd.to_numeric(df["start"], errors='coerce')
    df = df.dropna(subset=num_cols + ["start"]) # Remove any rows that failed conversion
    
    df["bin"] = df["start"].astype(int) // bin_size
    
    # Process only autosomes
    autosomes = [f"chr{i}" for i in range(1, 23)]
    df = df[df["chrom"].isin(autosomes)]
    
    binned = df.groupby(['chrom', 'bin']).agg({
        "mean": "mean", "median": "mean", "stdev": "mean", "min": "mean", "max": "mean"
    }).reset_index()
    return binned

# 1. Discover samples
samples_df = discover_samples(BASE_DIR)
print(f"Discovered {len(samples_df)} holdout samples.")

all_combined_data = []

# 2. Process each sample
for _, row in samples_df.iterrows():
    print(f"Processing {row['sample']}...")
    
    # Get WPS
    wps_binned = get_binned_wps(row['path'], bin_size)
    if wps_binned.empty:
        print(f"  Warning: No WPS data for {row['sample']}")
        continue
    wps_binned.rename(columns={"value": "wps_value"}, inplace=True)
    
    # Get Frags
    frag_binned = get_binned_frags(row['sample'], bin_size)
    
    if frag_binned is not None:
        merged = pd.merge(frag_binned, wps_binned, on=["chrom", "bin"], how="inner")
        print(f"  Fetched WPS and Fragment metrics.")
    else:
        merged = wps_binned
        print(f"  Warning: No Fragment metrics found for {row['sample']}. Using WPS only.")
        # Add placeholders for frag metrics so columns match the training matrix
        for col in ["mean", "median", "stdev", "min", "max"]:
            merged[col] = 0
            
    merged["sample"] = row["sample"]
    merged["group"] = row["group"]
    merged["start"] = merged["bin"] * bin_size
    all_combined_data.append(merged)

# 3. Save combined long-format TSV (input for 01_blacklist)
merged_df = pd.concat(all_combined_data, ignore_index=True)
output_tsv = os.path.join(OUTPUT_DIR, f"final_feature_matrix_{bin_size}.tsv")
merged_df.to_csv(output_tsv, sep="\t", index=False)

print(f"Holdout long-format matrix saved to: {output_tsv}")
