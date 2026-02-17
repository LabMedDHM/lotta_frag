import pandas as pd
import os


# Paths
bin_size = 50000
INPUT_DIR = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/results"
feature_matrix = os.path.join(INPUT_DIR, f"final_feature_matrix_{bin_size}.tsv")
blacklist_file = "/labmed/workspace/lotta/data/wgEncodeDacMapabilityConsensusExcludable.bed.gz"
output_file = os.path.join(INPUT_DIR, f"final_feature_matrix_blacklist_filtered_{bin_size}.tsv")

if not os.path.exists(feature_matrix):
    print(f"Error: Run 00_preprocessing_holdout.py first. Missing: {feature_matrix}")
    exit(1)

print(f"Loading feature matrix from {feature_matrix}...")
df = pd.read_csv(feature_matrix, sep="\t")

# Check if 'bin' exists (it should from 00_holdout)
if "bin" not in df.columns:
    print("Error: 'bin' column missing in matrix.")
    exit(1)

df["start"] = df["bin"] * bin_size
df["end"] = df["start"] + bin_size

print(f"Loading blacklist from {blacklist_file}...")
blacklist = pd.read_csv(blacklist_file, sep="\t", header=None,
                        names=["chrom", "start", "end", "name", "score", "strand"])

affected_bins = set()
print("Identifying blacklisted bins...")
for chrom in df["chrom"].unique():
    bl_chr = blacklist[blacklist["chrom"] == chrom]
    bins_chr = df[df["chrom"] == chrom]
    
    for _, row in bl_chr.iterrows():
        overlap = bins_chr[
            (bins_chr["start"] < row["end"]) & (bins_chr["end"] > row["start"])
        ]
        for bin_num in overlap["bin"]:
            affected_bins.add((chrom, bin_num))

print(f"Filtering {len(affected_bins)} bins...")
filtered_df = df[~df.apply(lambda x: (x["chrom"], x["bin"]) in affected_bins, axis=1)].copy()
filtered_df = filtered_df.drop(columns=["start", "end"])

filtered_df.to_csv(output_file, sep="\t", index=False)

print(f"Original bins: {len(df)}")
print(f"Filtered bins: {len(filtered_df)}")
print(f"Removed bins: {len(df) - len(filtered_df)}")
print(f"Blacklisted matrix saved to: {output_file}")
