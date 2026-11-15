#!/usr/bin/env bash
# Every experiment of the report, in order. Finished runs are skipped, so the script can be restarted.
# The weight files (final_weights.pt) are not in the repository. --restore-weights retrains the runs whose
# weights the validation and plotting scripts read (main, noise, noise_lr) and writes only the weight file.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
W=${WORKERS:-4}
$PY -m fedavg.sweep --name tune --workers $W            # learning rate x batch size (24 runs)
$PY -m fedavg.sweep --name ce --workers $W              # C x E, 3 seeds (60 runs)
$PY -m fedavg.sweep --name main --workers $W --restore-weights            # chosen hyperparameters, 5 seeds
$PY -m fedavg.sweep --name checks --workers $W          # structure-changing validation runs
$PY -m fedavg.sweep --name noise --workers $W --restore-weights           # Part 2, 13 noise scales x 3 seeds
$PY -m fedavg.sweep --name noise_variants --workers $W  # Part 2 edge cases
$PY -m fedavg.sweep --name noise_lr --workers $W --restore-weights        # Part 2, learning rate re-tuned per noise scale
for s in 0 1 2; do                                      # Part 1.2, Ray actors, 4 clients per round
  [ -f results/ray/seed$s/final_weights.pt ] || $PY -m fedavg.run --out results/ray/seed$s \
      --clients-per-round 4 --E 5 --B 20 --lr 0.5 --rounds 300 --seed $s --ray --verbose 50
done
$PY -m fedavg.validate                                  # validation experiments and their figures
$PY -m fedavg.make_plots                                # figures of the report
