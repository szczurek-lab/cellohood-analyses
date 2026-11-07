import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_PATH = "single_dlpfc_script_split_deepset.py"

THRESHOLDS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
RUNS = range(10)

# Fixed default hyperparameters, matching master_script_split_transformer_threshold.py
# exactly (num_heads is unused by the DeepSet encoder but kept for CLI/path parity).
NUM_HEADS = 4
LAYER_SIZE = 128
INTERMEDIATE_LAYER_SIZE = 256
LATENT_SIZE = 64

MAX_RETRIES = 5
MAX_WORKERS = 1  # sequential on one GPU (CUDA_VISIBLE_DEVICES=2 set inside the single script)


def run_job(threshold: float, run_nb: int) -> bool:
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--threshold", str(threshold),
        "--run_nb", str(run_nb),
        "--num_heads", str(NUM_HEADS),
        "--layer_size", str(LAYER_SIZE),
        "--intermediate_layer_size", str(INTERMEDIATE_LAYER_SIZE),
        "--latent_size", str(LATENT_SIZE),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[START] threshold={threshold} run={run_nb} attempt={attempt}")
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            print(f"[OK] threshold={threshold} run={run_nb}")
            return True
        print(f"[FAIL] threshold={threshold} run={run_nb} attempt={attempt}")

    print(f"[GIVE UP] threshold={threshold} run={run_nb}")
    return False


def main():
    jobs = [(t, r) for t in THRESHOLDS for r in RUNS]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_job, t, r) for t, r in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    success_count = sum(results)
    print(f"Finished. Success: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
