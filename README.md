# Fragmentomics Analysis Pipeline

This repository contains a set of scripts for analyzing fragmentomics data (WPS, fragment length profiles) to classify cancer types against healthy controls.

## Overview

The pipeline processes genomic data across multiple "bin sizes" (e.g., 5kb, 10kb, 50kb, 100kb), performs GC correction, and trains Lasso models to identify stable features for cancer detection. It includes steps for data exploration, preprocessing, modeling, and rigorous validation on holdout sets.

## Pipeline Steps

The scripts are numbered based on their typical execution order:

1.  **`00_preprocessing_exploration.py`**: Loads raw bedgraph/BigWig data (WPS and fragment intervals), performs initial binning, and creates the baseline feature matrix.
2.  **`01_blacklist.py`**: Filters out regions from the "blacklist" (e.g., genomic regions with known artifacts or high noise).
3.  **`02_gc_correction.py`**: Applies GC-bias correction to the feature matrix to ensure that differences in GC content across samples do not bias the modeling results.
4.  **`03_plotting_gc_correction.py`**: Generates diagnostic plots (PCA, UMAP, Mean vs GC) to verify the effectiveness of the GC correction.
5.  **`04_lasso_modeling.ipynb` / `.py`**: The core modeling script. It performs:
    *   Data preprocessing and train/test splitting (80/20).
    *   Hyperparameter tuning (Lasso regularization parameter `C`) using nested cross-validation.
    *   Stability analysis using a "1SE rule" to find the most parsimonious model.
    *   Feature selection and model evaluation (AUC, ROC curves).
    *   *Alternative:* **`04_lasso_modeling_age_gender.ipynb`** performs modeling using only age and gender for comparison.
6.  **`05_holdout_validation.ipynb`**: Validates the final trained model on a completely independent holdout dataset. It calculates final AUC, Sensitivity, Specificity, and generates ROC curves.
7.  **`06_patients_characteristics.ipynb`**: Analyzes the clinical metadata of the cohort (age, gender, cancer stages).
8.  **`08_archived_frag_bins.ipynb`**: Contains historical/archived code for fragment-based bin analysis.

## Validation and Metrics

Additional scripts for deeper analysis of model performance and data consistency:

*   **`compare_metrics_all.py`**: Compares summary statistics (mean, std, etc.) between the training and holdout datasets to detect distribution shifts.
*   **`visualize_scores.py`**: Generates probability distribution plots for individual samples on the holdout set using the saved model (`final_lasso_model.joblib`).
*   **`compare_distributions.py`**: Compares the distribution of features between different groups.

## Configuration

All configuration is centralized in `config.py`. Key parameters include:

*   `BIN_SIZE`: The genomic window size for analysis (e.g., `50000`).
*   `ANALYSIS_MODE`: Either `"all_vs_healthy"` or `"specific_vs_healthy"`.
*   `SPECIFIC_GROUP`: The cancer type to focus on if in `"specific_vs_healthy"` mode.
*   `STRATIFY_BY`: How to stratify the train/test split (e.g., `"Gender+Age"`, `"Gender"`, or `None`).

## How to Run

You can run the initial preprocessing steps (00, 01, 02) across multiple bin sizes using the orchestration script:

```bash
python3 run_pipeline.py
```

This script will automatically update `config.py` for each bin size and execute the scripts listed in its `SCRIPTS` array. Subsequent modeling and validation are typically performed in the respective notebooks.

## Core Modules

*   **`helper_functions.py`**: Contains reusable logic for data loading, pivoting, robust interpolation of NaNs, and preprocessing pipelines.
*   **`cv_lasso_single_fold.py`**: Implements the custom cross-validation logic, the 1SE rule calculation, and coefficient stability tracking across folds.

## Data and Outputs

*   **Input Data**: Located in `/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/`.
*   **Holdout Data**: Preprocessed data for validation is in `holdout_preprocessing/`.
*   **Plots**: Saved in `/labmed/workspace/lotta/finaletoolkit/outputs/plots/<bin_size>/`.
*   **Tables**: Saved in `/labmed/workspace/lotta/finaletoolkit/outputs/tables/<bin_size>/`.
