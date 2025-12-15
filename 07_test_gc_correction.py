import pandas as pd
import numpy as np
from pyfaidx import Fasta
from statsmodels.nonparametric.smoothers_lowess import lowess
'''
# Parameter für Test
GENOME_FASTA = "/labmed/workspace/lotta/data/hg38.fa"
CHROM = "chr1"
BIN_SIZE_TEST = 50000
SAMPLE_NAME = "TEST1"
NUM_BINS = 10

# Erstelle zufällige Features für 10 Bins
np.random.seed(42)

test_df = pd.DataFrame({
    "sample": [SAMPLE_NAME]*NUM_BINS,
    "chrom": [CHROM]*NUM_BINS,
    "bin": list(range(NUM_BINS)),
    "mean": np.random.randint(140, 170, size=NUM_BINS),
    "median": np.random.randint(135, 165, size=NUM_BINS),
    "stdev": np.random.randint(15, 25, size=NUM_BINS),
    "min": np.random.randint(110, 140, size=NUM_BINS),
    "max": np.random.randint(170, 200, size=NUM_BINS)
})

# Lade Genome
genome = Fasta(GENOME_FASTA)

# GC-Berechnung pro Test-Bin
gc_values = []
for i in range(NUM_BINS):
    start = i * BIN_SIZE_TEST
    end = start + BIN_SIZE_TEST
    seq = genome[CHROM][start:end].seq.upper()
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    gc_values.append(gc)

test_df['gc'] = gc_values
print("Test-GC-Werte pro Bin:\n", test_df[['bin','gc']])


# LOESS-Korrektur Funktion
def gc_correct(column, gc_values, frac=0.75):
    log_col = np.log2(column + 1)
    fit = lowess(log_col, gc_values, frac=frac, return_sorted=False)
    resid = log_col - fit
    median_raw = np.median(log_col)
    corrected_log = resid + median_raw
    corrected = np.power(2, corrected_log) - 1
    corrected[corrected < 0] = 0
    return corrected


# Test LOESS-Korrektur auf 'mean'
test_df['mean_gc'] = gc_correct(test_df['mean'].values, np.array(gc_values))
test_df['median_gc'] = gc_correct(test_df['median'].values, np.array(gc_values))

print("\nVorher/Nachher Vergleich:")
print(test_df[['bin','mean','mean_gc','median','median_gc']])'''
# Korrigierte Matrix laden
df = pd.read_csv("/labmed/workspace/lotta/finaletoolkit/dataframes_notebook/final_feature_matrix_gc_corrected.tsv", sep="\t")

# Nur GC-korrigierte Spalten behalten
df_gc_only = df[["sample", "group", "chrom", "bin",
                 "mean_gc", "median_gc", "stdev_gc", "min_gc", "max_gc", "wps_value_gc"]].copy()

# Spalten umbenennen (ohne _gc)
df_gc_only.rename(columns={
    "mean_gc": "mean",
    "median_gc": "median",
    "stdev_gc": "stdev",
    "min_gc": "min",
    "max_gc": "max",
    "wps_value_gc": "wps_value"
}, inplace=True)

# Neue saubere Matrix speichern
df_gc_only.to_csv("/labmed/workspace/lotta/finaletoolkit/dataframes_notebook/final_feature_matrix_gc_corrected.tsv", sep="\t", index=False)

