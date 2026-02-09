#!/usr/bin/env python3
import subprocess
import time
import sys
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

BIN_SIZES = [5000, 10000, 50000, 100000]
SCRIPTS = [
    "04_lasso_modeling.py"

]

BASE_DIR = Path("/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts")
CONFIG_FILE = BASE_DIR / "config.py"

# =========================
# HELPERS
# =========================

def set_bin_size(bin_size):
    """Overwrite BIN_SIZE in config.py"""
    lines = CONFIG_FILE.read_text().splitlines()
    new_lines = []

    for line in lines:
        if line.startswith("BIN_SIZE"):
            new_lines.append(f"BIN_SIZE = {bin_size}")
        else:
            new_lines.append(line)

    CONFIG_FILE.write_text("\n".join(new_lines))


def run_script(script_name, bin_size):
    script_path = BASE_DIR / script_name
    log_prefix = f"[BIN {bin_size} | {script_name}]"

    print(f"{log_prefix} START")
    start = time.time()

    result = subprocess.run(
        ["python3", script_path.name],
        cwd=BASE_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    runtime = time.time() - start

    if result.returncode != 0:
        print(f"{log_prefix} FAILED after {runtime:.1f}s")
        sys.exit(1)

    print(f"{log_prefix} DONE in {runtime:.1f}s\n")


# =========================
# MAIN
# =========================

def main():
    print("==== Fragmentomics pipeline started ====\n")

    for bin_size in BIN_SIZES:
        print(f"\n===== RUNNING BIN SIZE {bin_size} =====\n")

        set_bin_size(bin_size)
        time.sleep(1)  # make sure filesystem syncs

        for script in SCRIPTS:
            run_script(script, bin_size)

    print("\n==== ALL BIN SIZES FINISHED SUCCESSFULLY ====")


if __name__ == "__main__":
    main()
