# Fragmentomics Analysis Pipeline

This repository contains a set of scripts for analyzing fragmentomics data (WPS, fragment length profiles) to classify cancer types against healthy controls.

## Overview

The pipeline processes genomic data across multiple "bin sizes" (e.g., 5kb, 10kb, 50kb, 100kb), performs GC correction, and trains Lasso models to identify stable features for cancer detection.

## Pipeline Steps

The scripts are numbered based on their typical execution order:

1.  **`00_preprocessing_exploration.py`**: Loads raw bedgraph/BigWig data (WPS and fragment intervals), performs initial binning, and creates the baseline feature matrix.
2.  **`01_blacklist.py`**: Filters out regions from the "blacklist" (e.g., genomic regions with known artifacts or high noise).
3.  **`02_gc_correction.py`**: Applies GC-bias correction to the feature matrix to ensure that differences in GC content across samples do not bias the modeling results.
4.  **`03_plotting_gc_correction.py`**: Generates diagnostic plots (PCA, UMAP, Mean vs GC) to verify the effectiveness of the GC correction.
5.  **`04_lasso_modeling.py`**: The core modeling script. It performs:
    *   Data preprocessing and train/test splitting (80/20).
    *   Hyperparameter tuning (Lasso regularization parameter `C`) using cross-validation.
    *   Stability analysis using a "1SE rule" to find the most parsimonious model.
    *   Feature selection and model evaluation (AUC, ROC curves).
6.  **`05_ensemble_learning.ipynb`**: Exploratory notebook for combining different models.
7.  **`06_patients_characteristics.ipynb`**: Analyzes the clinical metadata of the cohort (age, gender, cancer stages).

## Configuration

All configuration is centralized in `config.py`. Key parameters include:

*   `BIN_SIZE`: The genomic window size for analysis.
*   `ANALYSIS_MODE`: Either `"all_vs_healthy"` or `"specific_vs_healthy"`.
*   `SPECIFIC_GROUP`: The cancer type to focus on if in `"specific_vs_healthy"` mode (e.g., `"Pancreatic Cancer"`).
*   `STRATIFY_BY`: How to stratify the train/test split (e.g., `"Gender+Age"`, `"Gender"`, or `None`).

## How to Run

You can run the entire pipeline (or specific steps) across multiple bin sizes using the orchestration script:

```bash
python3 run_pipeline.py
```

This script will automatically update `config.py` for each bin size and execute the scripts listed in its `SCRIPTS` array.

## Core Modules

*   **`helper_functions.py`**: Contains helper functions for data loading, preprocessing pipelines, and calculating model metrics.
*   **`cv_lasso_single_fold.py`**: Contains the logic for cross-validation loops, the 1SE rule calculation, and feature stability analysis across folds.

## Data and Outputs

*   **Input Data**: Located in `/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/`.
*   **Plots**: Saved in `/labmed/workspace/lotta/finaletoolkit/outputs/plots/<bin_size>/`.
*   **Tables**: Saved in `/labmed/workspace/lotta/finaletoolkit/outputs/tables/<bin_size>/`.
