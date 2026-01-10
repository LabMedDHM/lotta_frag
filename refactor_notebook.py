import json

notebook_path = '/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/04_lasso_modeling.ipynb'

def read_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_nb(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

nb = read_nb(notebook_path)

# 1. Find and Remove the cells we added previously (ID: feature_stability_md, feature_stability_code)
# Filter cells where 'id' is NOT in the list
nb['cells'] = [cell for cell in nb['cells'] if cell.get('id') not in ['feature_stability_md', 'feature_stability_code']]

# 2. Find the "Selected Features" cell
selected_features_cell_index = -1
for idx, cell in enumerate(nb['cells']):
    # Check source
    source_str = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '# 6. Selected Features' in source_str:
        selected_features_cell_index = idx
        break

if selected_features_cell_index != -1:
    # 3. Update the content
    new_source = [
        "# 6. Selected Features & Stability Analysis\n",
        "from cv_lasso_single_fold import cross_validation, analyze_feature_stability, plot_roc_curves, plot_auc_boxplot\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "# --- A. Single Model Analysis (Current Split) ---\n",
        "# Zugriff auf das LogisticRegressionCV-Modell in der Pipeline\n",
        "lasso_model = pipeline.named_steps['lasso_cv']\n",
        "\n",
        "coef_df = pd.DataFrame({\n",
        "    \"Feature\": X.columns,\n",
        "    \"Coefficient\": lasso_model.coef_[0]\n",
        "})\n",
        "# Filtere Features, die NICHT 0 sind\n",
        "important_features = coef_df[coef_df[\"Coefficient\"] != 0].sort_values(by=\"Coefficient\", ascending=False)\n",
        "\n",
        "print(f\"Number of Important Features (Single Model): {len(important_features)}\")\n",
        "print(\"\\nTop Features (Single Model - Positive = Indikative for Cancer):\")\n",
        "print(important_features.head(20))\n",
        "\n",
        "\n",
        "# --- B. Feature Stability Analysis (Cross-Validation) ---\n",
        "print(\"\\n\" + \"=\"*50 + \"\\nRunning 5-Fold Cross-Validation for Feature Stability...\\n\")\n",
        "\n",
        "# X and y should be available from previous cells. \n",
        "# We use X (full pivot_df before split if available, or regenerate if needed).\n",
        "# Assuming X and y are the full datasets as defined before train_test_split.\n",
        "\n",
        "# Re-verify label consistency\n",
        "cv_results = cross_validation(X, y, pipeline, n_folds=5)\n",
        "\n",
        "# Plotte Performance\n",
        "plot_roc_curves(cv_results)\n",
        "plot_auc_boxplot(cv_results)\n",
        "\n",
        "# Feature Stability Analyse\n",
        "stability_df = analyze_feature_stability(cv_results)\n",
        "print(\"\\nTop Stable Features (Selected across multiple folds):\")\n",
        "print(stability_df.head(20))\n",
        "\n",
        "# Histogram\n",
        "plt.figure(figsize=(8, 4))\n",
        "stability_df['Frequency'].value_counts().sort_index().plot(kind='bar')\n",
        "plt.title('Feature Selection Frequency across 5 Folds')\n",
        "plt.xlabel('Number of Folds')\n",
        "plt.ylabel('Number of Features')\n",
        "plt.grid(axis='y', alpha=0.3)\n",
        "plt.show()\n"
    ]
    nb['cells'][selected_features_cell_index]['source'] = new_source
    print("Notebook updated: Merged stability analysis into cell 6.")
else:
    print("Error: Could not find 'Selected Features' cell.")

write_nb(nb, notebook_path)
