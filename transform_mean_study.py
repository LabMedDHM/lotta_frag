import argparse
import numpy as np
import pandas as pd

def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True, help="Input GC-corrected long TSV")
    ap.add_argument("--out_tsv", required=True, help="Output long TSV with transformed values corrected by study mean")
    ap.add_argument("--meta_tsv", required=True, help="Matrix storing study information")
    args = ap.parse_args()
    # Liste der Metriken, die wir korrigieren wollen
    metrics = ["mean", "median", "stdev", "min", "max", "wps_value"]
    
    df = pd.read_csv(args.in_tsv, sep="\t")
    corrected_df = df.copy()

    meta = pd.read_csv(args.meta_tsv, sep=";")
    meta_df = meta.copy()

    df_combined = pd.merge (
        df,
        meta_df[["Extracted_ID", "Study"]],
        left_on="sample",
        right_on="Extracted_ID",
        how="left"
    )

    # 1. Berechne die Healthy-Referenz pro Studie
    # Wir filtern nur die 'Healthy' Proben und gruppieren nach der Spalte 'Study'
    study_references = df_combined[df_combined["group"].str.lower() == "healthy"].groupby("Study")[metrics].mean()
    
    print("Gefundene Healthy-Referenzen pro Studie:")
    print(study_references)
    corrected_df = df_combined.copy()

    # 2. Korrektur anwenden
    for study in corrected_df["Study"].unique():
            if pd.isna(study): continue # Falls eine Probe keine Study-Zuordnung hat
            
            if study in study_references.index:
                # Maske für alle Proben dieser Studie
                mask = (corrected_df["Study"] == study)
                # Subtrahiere den Healthy-Mittelwert dieser Studie
                corrected_df.loc[mask, metrics] = corrected_df.loc[mask, metrics] - study_references.loc[study].values
            else:
                # Notlösung für Studien ohne Healthies
                global_ref = study_references.mean()
                mask = (corrected_df["Study"] == study)
                corrected_df.loc[mask, metrics] = corrected_df.loc[mask, metrics] - global_ref.values
                print(f"Warnung: {study} hat keine Healthies. Nutze globalen Healthy-Schnitt.")
    
    corrected_df.to_csv(args.out_tsv, sep="\t", index=False)
    print("Wrote:", args.out_tsv)
    print("Shape:", corrected_df.shape)


if __name__ == "__main__":
    main()
