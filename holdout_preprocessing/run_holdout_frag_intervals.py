#!/usr/bin/env python3
import os
import subprocess
import glob
from pathlib import Path

# Paths
BASE_DIR = "/labmed/workspace/lotta/finaletoolkit"
DATA_DIR = os.path.join(BASE_DIR, "data_holdout")
OUTPUT_DIR = "/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/frag_intervals"
CONDA_ENV = "finaletoolkit_workflow"

# Reference
INTERVALS = "/labmed/workspace/lotta/finaletoolkit/carsten/outputs/full/intervals_min5000_autosomes_5kb.bed"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_holdout_samples():
    frag_files = glob.glob(os.path.join(DATA_DIR, "**", "*.frag.gz"), recursive=True)
    samples = []
    for f in frag_files:
        sample_id = os.path.basename(f).replace(".frag.gz", "")
        subfolder = os.path.basename(os.path.dirname(f))
        samples.append((sample_id, subfolder, f))
    return samples

def run_frag_intervals(sample_id, subfolder, input_file):
    print(f"--- Generating fragment intervals for {sample_id} ---")
    dest_dir = os.path.join(OUTPUT_DIR, subfolder)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    output_file = os.path.join(dest_dir, f"{sample_id}.frag_length_intervals.bed")
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"  ✅ already exists.")
        return

    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "finaletoolkit", "frag-length-intervals",
        input_file,
        INTERVALS,
        "-o", output_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"  ✅ Success.")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e}")

def main():
    samples = get_holdout_samples()
    print(f"Found {len(samples)} holdout samples to check/process.")
    
    for sample_id, subfolder, input_file in samples:
        run_frag_intervals(sample_id, subfolder, input_file)

if __name__ == "__main__":
    main()
