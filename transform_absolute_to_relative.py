#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

FRAG_METRICS = ["mean", "median", "stdev", "min", "max"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True, help="Input GC-corrected long TSV")
    ap.add_argument("--out_tsv", required=True, help="Output long TSV with ratios/log-ratios (same columns)")
    ap.add_argument("--mode", choices=["logratio", "ratio"], default="logratio",
                    help="Use log(x/mean_sample_metric) or x/mean_sample_metric for fragment metrics")
    ap.add_argument("--wps", choices=["keep", "center"], default="center",
                    help="wps_value handling: keep as-is or center per sample (recommended)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_tsv, sep="\t")

    # Expected columns (we won't drop any of them)
    needed = ["sample", "group", "chrom", "bin", "start", "end", "GC", "wps_value"] + FRAG_METRICS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Ensure numeric
    for c in FRAG_METRICS + ["wps_value", "start", "end", "bin", "GC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Treat sentinel -1 as missing for frag metrics
    df[FRAG_METRICS] = df[FRAG_METRICS].replace(-1, np.nan)

    # --- Fragment metrics: per-sample normalization per metric ---
    for m in FRAG_METRICS:
        denom = df.groupby("sample")[m].transform("mean").replace(0, np.nan)

        if args.mode == "ratio":
            df[m] = df[m] / denom
        else:
            # log-ratio: log(x) - log(denom)
            x = df[m].where(df[m] > 0, np.nan)
            df[m] = np.log(x) - np.log(denom)

    # --- WPS handling ---
    if args.wps == "center":
        wps_mean = df.groupby("sample")["wps_value"].transform("mean")
        df["wps_value"] = df["wps_value"] - wps_mean

    # Keep exact same column order as your example
    out_cols = ["sample", "group", "chrom", "bin",
                "mean", "median", "stdev", "min", "max",
                "wps_value", "start", "end", "GC"]

    df[out_cols].to_csv(args.out_tsv, sep="\t", index=False)
    print("Wrote:", args.out_tsv)
    print("Shape:", df[out_cols].shape)
    print("Mode:", args.mode, "| WPS:", args.wps)

if __name__ == "__main__":
    main()
