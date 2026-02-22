import pandas as pd
from pyfaidx import Fasta
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from config import BIN_SIZE as bin_size


genome_fasta = "/labmed/workspace/lotta/data/hg38.fa"
input_path = f"/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/dataframes_holdout/final_feature_matrix_blacklist_filtered_{bin_size}.tsv"
output_path = f"/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/dataframes_holdout/final_feature_matrix_gc_corrected_{bin_size}.tsv"

df = pd.read_csv(input_path, sep="\t")

df["start"] = df["bin"] * bin_size
df["end"] = df["start"] + bin_size

genome = Fasta(genome_fasta)

def calc_gc(chrom, start, end):
    seq = genome[chrom][start:end].seq.upper()
    if len(seq) == 0:
        return np.nan
    return (seq.count("G") + seq.count("C")) / len(seq)

print("Calculating GC-content…")
df["GC"] = df.apply(lambda row: calc_gc(str(row["chrom"]), int(row["start"]), int(row["end"])), axis=1)

df.to_csv(input_path, sep="\t", index=False)
print(df.head())

coverage_cols = [
    col for col in df.columns
    if any(x in col.lower() for x in ["mean", "median", "stdev", "min", "max", "wps_value"])
]

print(f"Coverage columns to correct: {coverage_cols}")
gc_corrected_df = df.copy()

for sample in df["sample"].unique():
    print(f"Processing sample: {sample}")

    subset = df[df["sample"] == sample]

    for col in coverage_cols:

        y = subset[col].values
        x = subset["GC"].values
        #Locally Estimated Scatterplot Smoothing
        loess_fit = lowess(endog=y, exog=x, frac=0.75, return_sorted=False)
        corrected = y - loess_fit + np.median(y)
        gc_corrected_df.loc[subset.index, col] = corrected

gc_corrected_df.to_csv(output_path, sep="\t", index=False)
print(f"GC-corrected matrix saved to: {output_path}")
print(gc_corrected_df.head())
