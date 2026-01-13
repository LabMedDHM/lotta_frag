import json

def patch_notebook(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'metrics_with_wps = []' in ''.join(cell['source']):
            src = ''.join(cell['source'])
            # Target the specific line
            updated_src = src.replace('"wps_value_gc_corrected"', '"wps_value"')
            cell['source'] = [l + '\n' if not l.endswith('\n') else l for l in updated_src.splitlines()]
            break

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

patch_notebook('/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/04_lasso_modeling.ipynb')
