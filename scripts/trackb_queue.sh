#!/usr/bin/env bash
# Track B runs, strictly one GPU job at a time (ComputeGuard's lock enforces it). Run after a clean boot.
set -euo pipefail
cd "$(dirname "$0")/.."
O=/mnt/data/jepamath/trackb
nvidia-smi --query-gpu=name,temperature.gpu,power.draw --format=csv,noheader
# run 2: L2 + pooled loss (tests the median/"no object" explanation of the add/delete asymmetry)
python -m jepatrain.train --out $O/run2_l2pool --loss l2 --pool-weight 1.0 > $O/run2_l2pool.log 2>&1
# run 1b: the V-JEPA-style L1 baseline, restarted after the GPU hang killed run 1
python -m jepatrain.train --out $O/run1b_l1 > $O/run1b_l1.log 2>&1
# re-evaluate every checkpoint (adds the pooled reveal score), then attentive probes on the final ones
for r in run2_l2pool run1b_l1; do python -m jepatrain.reeval --run $O/$r >> $O/$r.log 2>&1; done
for r in run2_l2pool run1b_l1; do python -m jepatrain.probe --run $O/$r --weights weights_030000.pt >> $O/$r.log 2>&1; done
