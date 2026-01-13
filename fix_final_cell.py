import json

def patch_notebook(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    # Targeted replacement based on the current content
    search_text = "# Jetzt mit klinischen Daten mergen"
    
    new_source = [
        "# Merge outliers with test_results to get the probabilities\n",
        "outliers_meta = pd.concat([fn_proben, fp_proben])\n",
        "if not outliers_meta.empty:\n",
        "    ausreisser_klinik = clinical_df.merge(outliers_meta[['Sample_ID', 'Probability_Cancer', 'True_Label']], \n",
        "                                         left_on='Extracted_ID', right_on='Sample_ID')\n",
        "    print(f\"Gefundene Ausreißer mit Schwellenwert (FN < 0.3, FP > 0.7): {len(ausreisser_klinik)}\")\n",
        "    print(ausreisser_klinik[['Extracted_ID', 'Patient Type', 'Gender', 'Probability_Cancer']])\n",
        "else:\n",
        "    print(\"Keine Ausreißer mit den aktuellen Schwellenwerten gefunden.\")\n"
    ]

    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell_content = "".join(cell['source'])
            if search_text in cell_content:
                # Replace the whole cell source or just the part?
                # Let's replace the whole source from 'fn_proben =' onwards to be safe
                cell['source'] = [
                    "# Falsch-Negative (Krebs als gesund vorhergesagt)\n",
                    "fn_proben = test_results[(test_results['True_Label'] == 1) & (test_results['Probability_Cancer'] < 0.3)]\n",
                    "\n",
                    "# Falsch-Positive (Gesund als Krebs vorhergesagt)\n",
                    "fp_proben = test_results[(test_results['True_Label'] == 0) & (test_results['Probability_Cancer'] > 0.7)]\n",
                    "\n",
                    ] + new_source
                found = True
                break
    
    if found:
        with open(filepath, 'w') as f:
            json.dump(nb, f, indent=1)
        print("Final cell patched successfully.")
    else:
        print("Target cell not found.")

patch_notebook('/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/04_lasso_modeling.ipynb')
