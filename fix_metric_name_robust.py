import json
import re

def patch_notebook(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        # Fix the metric matching logic
        if cell['cell_type'] == 'code' and 'metrics_with_wps = []' in ''.join(cell['source']):
            src = ''.join(cell['source'])
            # Replace the problematic if check with a more robust one
            # Find the line like: if "wps_value" in metric['metrics']:
            # Replace it with: if any("wps" in m for m in metric['metrics']):
            pattern = r'if ".*?" in metric\[(\'|")metrics(\'|")\]:'
            replacement = 'if any("wps_value" in m for m in metric["metrics"]):'
            updated_src = re.sub(pattern, replacement, src)
            cell['source'] = [l + '\n' if not l.endswith('\n') else l for l in updated_src.splitlines()]
            
        # Also fix the typo in cell 2 if it persists
        if cell['cell_type'] == 'code' and 'clinical_df = clincal_df' in ''.join(cell['source']):
            src = ''.join(cell['source'])
            updated_src = src.replace('clinical_df = clincal_df', 'clinical_df = clinical_df')
            cell['source'] = [l + '\n' if not l.endswith('\n') else l for l in updated_src.splitlines()]

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

patch_notebook('/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/04_lasso_modeling.ipynb')
