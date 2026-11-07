#!/bin/bash
for run in 0 1 2 3 4; do
  echo "=== Starting run $run at $(date) ==="
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /home/marmarmarmar/immucan/bin/python single_dlpfc_script_split_nopermmatch.py --threshold 5.0 --run_nb $run --num_heads 4 --layer_size 128 --intermediate_layer_size 256 --latent_size 64
  echo "=== Finished run $run at $(date) ==="
done
echo "ALL RUNS COMPLETE"
