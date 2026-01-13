import json
import re

def patch_notebook(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        # Fix Cell 22 plotting and savefig
        if cell['cell_type'] == 'code' and 'plot_roc_curves(cv_results)' in ''.join(cell['source']):
            src = ''.join(cell['source'])
            # Move savefig before show()
            if 'plt.savefig' in src and 'plt.show()' in src:
                # Find the plt.show() and plt.savefig lines
                lines = cell['source']
                savefig_line = -1
                show_line = -1
                for i, line in enumerate(lines):
                    if 'plt.savefig' in line: savefig_line = i
                    if 'plt.show()' in line: show_line = i
                
                if savefig_line > show_line:
                    line_to_move = lines.pop(savefig_line)
                    # Insert it before the show_line
                    lines.insert(show_line, line_to_move)
                cell['source'] = lines

        # Fix outlier detection at the end
        if cell['cell_type'] == 'code' and 'Probability_Cancer' in ''.join(cell['source']) and 'test_results' in ''.join(cell['source']):
             src = ''.join(cell['source'])
             # Relax thresholds: 0.1 -> 0.2, 0.9 -> 0.8
             updated_src = src.replace('< 0.1', '< 0.3').replace('> 0.9', '> 0.7')
             # Add Probability_Cancer to the final print for clarity
             updated_src = updated_src.replace("print(ausreisser_klinik[['Extracted_ID', 'Patient Type', 'Gender']])", "print(ausreisser_klinik[['Extracted_ID', 'Patient Type', 'Gender', 'Probability_Cancer']])")
             
             # Need to make sure Probability_Cancer is in ausreisser_klinik
             # Logic: ausreisser_klinik = clinical_df[clinical_df['Extracted_ID'].isin(...)]
             # We should merge it instead to keep the probabilities
             merge_logic = """
# Merge outliers with test_results to get the probabilities
outliers_meta = pd.concat([fn_proben, fp_proben])
ausreisser_klinik = clinical_df.merge(outliers_meta[['Sample_ID', 'Probability_Cancer', 'True_Label']], 
                                     left_on='Extracted_ID', right_on='Sample_ID')
print(f"Gefundene Ausreißer mit Schwellenwert (FN < 0.3, FP > 0.7): {len(ausreisser_klinik)}")
print(ausreisser_klinik[['Extracted_ID', 'Patient Type', 'Gender', 'Probability_Cancer']])
"""
             # Replace the final block
             pattern = r'# Jetzt mit klinischen Daten mergen.*?Stadium hinzu"'
             updated_src = re.sub(pattern, merge_logic, updated_src, flags=re.DOTALL)
             cell['source'] = [l + '\n' if not l.endswith('\n') else l for l in updated_src.splitlines()]

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

patch_notebook('/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/04_lasso_modeling.ipynb')
