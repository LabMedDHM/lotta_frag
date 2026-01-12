import json
import os

nb_path = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/04_lasso_modeling.ipynb"

with open(nb_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Cell 2 (Loading)
        if "matrix_path = f\"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_gc_corrected_{bin_size}.tsv\"" in source:
            new_source = [
                "print(analysis_mode)\n",
                "print(specific_group)\n",
                "\n",
                "matrix_path = f\"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_gc_corrected_{bin_size}.tsv\"\n",
                "df = pd.read_csv(matrix_path, sep=\"\\t\")\n",
                "\n",
                "clinical_path = \"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/filtered_clinical_characteristics.csv\"\n",
                "clinical_df_raw = pd.read_csv(clinical_path)\n",
                "\n",
                "if analysis_mode == \"specific_vs_healthy\":\n",
                "    clinical_df = clinical_df_raw[\n",
                "            (clinical_df_raw[\"Patient Type\"] == specific_group) |\n",
                "            (clinical_df_raw[\"Patient Type\"].str.lower() == \"healthy\")\n",
                "        ].copy()\n",
                "else:\n",
                "    clinical_df = clinical_df_raw.copy()\n",
                "\n",
                "# Balancing: Sample as many Healthy as there are Cancer samples\n",
                "cancer_df = clinical_df[clinical_df[\"Patient Type\"].str.lower() != \"healthy\"]\n",
                "healthy_df = clinical_df[clinical_df[\"Patient Type\"].str.lower() == \"healthy\"]\n",
                "n_cancer = len(cancer_df)\n",
                "if len(healthy_df) > n_cancer:\n",
                "    healthy_df = healthy_df.sample(n=n_cancer, random_state=42)\n",
                "clinical_df = pd.concat([cancer_df, healthy_df]).copy()\n",
                "\n",
                "valid_samples = clinical_df[\"Extracted_ID\"].unique()\n",
                "df = df[df[\"sample\"].isin(valid_samples)].copy()\n",
                "\n",
                "print(f\"Number of Samples in Matrix: {df['sample'].nunique()}\")\n",
                "print(f\"Number of Bins per Sample: {len(df) / df['sample'].nunique()}\")"
            ]
            cell["source"] = new_source
            
        # Metrics cell
        if "metrics = [" in source and "mean_gc_corrected" in source:
            source = source.replace("mean_gc_corrected", "mean")
            source = source.replace("median_gc_corrected", "median")
            source = source.replace("stdev_gc_corrected", "stdev")
            source = source.replace("wps_value_gc_corrected", "wps_value")
            source = source.replace("min_gc_corrected", "min")
            source = source.replace("max_gc_corrected", "max")
            # Split back into lines
            cell["source"] = [line + ("\n" if i < len(source.splitlines())-1 else "") for i, line in enumerate(source.splitlines())]

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
