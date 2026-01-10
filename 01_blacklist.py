import pandas as pd
from config import BIN_SIZE as bin_size


feature_matrix = f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_{bin_size}.tsv"
blacklist_file = "/labmed/workspace/lotta/data/wgEncodeDacMapabilityConsensusExcludable.bed.gz"
output_file = f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_blacklist_filtered_{bin_size}.tsv"

df = pd.read_csv(feature_matrix, sep="\t")
df["start"] = df["bin"] * bin_size
df["end"] = df["start"] + bin_size

blacklist = pd.read_csv(blacklist_file, sep="\t", header=None,
                        names=["chrom", "start", "end", "name", "score", "strand"])

affected_bins = set()

for chrom in blacklist["chrom"].unique():
    bl_chr = blacklist[blacklist["chrom"] == chrom]
    bins_chr = df[df["chrom"] == chrom]
    
    for _, row in bl_chr.iterrows():
        overlap = bins_chr[
            (bins_chr["start"] < row["end"]) & (bins_chr["end"] > row["start"])
        ]
        for bin_num in overlap["bin"]:
            affected_bins.add((chrom, bin_num))

filtered_df = df[~df.apply(lambda x: (x["chrom"], x["bin"]) in affected_bins, axis=1)].copy()
filtered_df = filtered_df.drop(columns=["start", "end"])

filtered_df.to_csv(output_file, sep="\t", index=False)

print(f"Original bins: {len(df)}")
print(f"Filtered bins: {len(filtered_df)}")
print(f"Removed bins: {len(df) - len(filtered_df)}")
