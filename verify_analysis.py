import pandas as pd

try:
    # Load clinical data
    clinical_df = pd.read_csv("/labmed/workspace/lotta/finaletoolkit/dataframes_notebook/1-Tabelle 1.csv", sep=";", skiprows=1)

    # Filter for rows where 'Patient' column contains '/'
    filtered_clinical_df = clinical_df[clinical_df['Patient'].astype(str).str.contains('/', na=False)].copy()

    # Extract the ID part after the '/'
    filtered_clinical_df['Extracted_ID'] = filtered_clinical_df['Patient'].astype(str).str.split('/').str[1]

    print("Filtered Clinical Data (First 5 rows):")
    print(filtered_clinical_df[['Patient', 'Extracted_ID']].head())
    print(f"\nNumber of clinical samples after filtering: {len(filtered_clinical_df)}")

    # Save the filtered clinical table
    output_path = "/labmed/workspace/lotta/finaletoolkit/dataframes_notebook/filtered_clinical_characteristics.csv"
    filtered_clinical_df.to_csv(output_path, index=False)
    print(f"\nSaved filtered clinical table to: {output_path}")

    # Load GC corrected feature matrix
    matrix_df = pd.read_csv("/labmed/workspace/lotta/finaletoolkit/dataframes_notebook/final_feature_matrix_gc_corrected.tsv", sep="\t")

    # Get unique IDs from the matrix
    matrix_ids = set(matrix_df['sample'].unique())
    print(f"Number of unique samples in GC corrected matrix: {len(matrix_ids)}")

    # Get the set of extracted clinical IDs
    clinical_ids = set(filtered_clinical_df['Extracted_ID'].unique())

    # Calculate overlaps and differences
    intersection = matrix_ids.intersection(clinical_ids)
    matrix_only = matrix_ids - clinical_ids
    clinical_only = clinical_ids - matrix_ids

    print("--- Sample Count Analysis ---")
    print(f"Samples in BOTH Matrix and Clinical table: {len(intersection)}")
    print(f"Samples ONLY in Matrix (Missing from Clinical): {len(matrix_only)}")
    print(f"Samples ONLY in Clinical (Missing from Matrix): {len(clinical_only)}")

    print(f"\nTotal Unique Samples in Matrix: {len(matrix_ids)} (Should be {len(intersection)} + {len(matrix_only)}) ")
    print(f"Total Unique Samples in Clinical: {len(clinical_ids)} (Should be {len(intersection)} + {len(clinical_only)})")

    print("\n--- Details ---")
    print("Samples in GC corrected matrix but NOT in the filtered clinical table:")
    print(sorted(list(matrix_only)))

    if clinical_only:
        print("\nSamples in Clinical table but NOT in the matrix (This explains the count discrepancy):")
        print(sorted(list(clinical_only)))

    # Get cancer types (group) for missing IDs (Matrix Only)
    if matrix_only:
        missing_info = matrix_df[matrix_df['sample'].isin(matrix_only)][['sample', 'group']].drop_duplicates()
        print("\nCancer types for samples missing from clinical table:")
        print(missing_info.to_string(index=False))

except Exception as e:
    print(f"An error occurred: {e}")
