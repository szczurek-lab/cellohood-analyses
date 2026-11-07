import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_PATH = "single_dlpfc_script_benchmark_split_reversed.py"

THRESHOLDS = [4.0, 5.0, 6.0]
RUNS = range(5)

MAX_RETRIES = 5
MAX_WORKERS = 1   # start low; raise only if stable


HYPER_PARAMS = [
        (4, 128, 256, 64),
        (4, 128, 256, 128),
        (4, 128, 512, 64),
        (4, 128, 512, 128),
        (8, 128, 256, 64),
        (8, 128, 256, 128),
        (8, 128, 512, 64),
        (8, 128, 512, 128),
        (4, 256, 512, 128),
        (4, 256, 1024, 128),
        (4, 256, 512, 256),
        (4, 256, 1024, 256),
        (8, 256, 512, 128),
        (8, 256, 1024, 128),
        (8, 256, 512, 256),
        (8, 256, 1024, 256),
        (4, 64, 128, 32),
        (4, 64, 128, 64),
        (4, 64, 256, 32),
        (4, 64, 128, 64),
        (8, 64, 128, 32),
        (8, 64, 128, 64),
        (8, 64, 256, 32),
        (8, 64, 128, 64),
]


def run_job(threshold: int, run_nb: int, num_heads: int, layer_size: int, intermediate_layer_size: int, latent_size: int) -> bool:
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--threshold",
        str(threshold),
        "--run_nb",
        str(run_nb),
        "--num_heads",
        str(num_heads),
        "--layer_size",
        str(layer_size),
        "--intermediate_layer_size",
        str(intermediate_layer_size),
        "--latent_size",
        str(latent_size),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[START] threshold={threshold} run={run_nb} attempt={attempt}, nh={num_heads},ls={layer_size},ils={intermediate_layer_size}, ls={latent_size}")

        result = subprocess.run(cmd, env=env)

        if result.returncode == 0:
            print(f"[OK] threshold={threshold} run={run_nb} nh={num_heads},ls={layer_size},ils={intermediate_layer_size}, ls={latent_size}")
            return True

        print(f"[FAIL] threshold={threshold} run={run_nb} attempt={attempt} nh={num_heads},ls={layer_size},ils={intermediate_layer_size}, ls={latent_size}")

    print(f"[GIVE UP] threshold={threshold} run={run_nb} nh={num_heads},ls={layer_size},ils={intermediate_layer_size}, ls={latent_size}")
    return False


def main():
    jobs = [(t, r) for t in THRESHOLDS for r in RUNS]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for num_heads, layer_size, intermediate_layer_size, latent_size in HYPER_PARAMS:
            futures.extend([executor.submit(run_job, t, r, num_heads, layer_size, intermediate_layer_size, latent_size) for t, r in jobs])

        for future in as_completed(futures):
            results.append(future.result())

    success_count = sum(results)
    print(f"Finished. Success: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
