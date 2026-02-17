#!/bin/bash
# Preprocessing for Holdout Dataset
set -e

DIR="/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing"

echo "Step 0: Generating Fragment length intervals..."
#python3 $DIR/run_holdout_frag_intervals.py

echo "Step 1: Merging WPS and Fragment metrics..."
python3 $DIR/00_preprocessing_holdout.py

echo "Step 2: Blacklist Filtering..."
python3 $DIR/01_blacklist_holdout.py

echo "Step 3: GC Correction (LOWESS)..."
python3 $DIR/02_gc_correction_holdout.py

echo "🎉 Preprocessing for Holdout Dataset complete!"
echo "Final matrix: $DIR/results/final_feature_matrix_gc_corrected_50000.tsv"
