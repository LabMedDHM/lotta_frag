import argparse
import numpy as np
import pandas as pd

def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True, help="Input GC-corrected long TSV")
    ap.add_argument("--out_tsv", required=True, help="Output long TSV with transformed values corrected by study mean")
    ap.add_argument("--meta_tsv", required=True, help="Matrix storing study information")
    ap.add_argument("--reference_csv", required=False, help="Loading csv with study means")
    ap.add_argument("--save_reference_csv", required=False, help="Saving csv with study means")
    args = ap.parse_args()


    metrics = ["mean", "median", "stdev", "min", "max", "wps_value"]
    study_references_loaded = None
    if args.reference_csv:
        study_references_loaded = pd.read_csv(args.reference_csv, index_col=0)
        print("Loaded study references from:", args.reference_csv)
        print(study_references_loaded)
    
    df = pd.read_csv(args.in_tsv, sep="\t")

    
    if "Study" in df.columns:
        print("'Study'-column already in Input – no merge needed.")
        df_combined = df.copy()
    else:
        meta = pd.read_csv(args.meta_tsv, sep=";")
        df_combined = pd.merge(
            df,
            meta[["Extracted_ID", "Study"]],
            left_on="sample",
            right_on="Extracted_ID",
            how="left"
        )

    before = df_combined['sample'].nunique()
    df_combined = df_combined[df_combined['Study'] != 'Sun']
    after = df_combined['sample'].nunique()

    print(f"Removed Sun samples: {before - after}")
    print("Studies left:", df_combined['Study'].unique())

    print("Spalten in df_combined:", df_combined.columns.tolist())
    print(df_combined[["sample", "Study", "group"]].drop_duplicates("sample").head())

    if study_references_loaded is not None:
        study_references = study_references_loaded
    else:
        study_references = df_combined[df_combined["group"].str.lower() == "healthy"].groupby("Study")[metrics].mean()
        if args.save_reference_csv:
            study_references.to_csv(args.save_reference_csv)
            print("Saved study references to:", args.save_reference_csv)
        print("Calculated Healthy references per study:")
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

    print(corrected_df[
    corrected_df["group"].str.lower() == "healthy"].groupby("Study")[metrics].mean())


if __name__ == "__main__":
    main()
