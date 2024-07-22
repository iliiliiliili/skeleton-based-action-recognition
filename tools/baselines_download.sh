#!/bin/bash -i

echo "monid"
rsync -avz --prune-empty-dirs --include "*/" $monid:~/har/skeleton-based-action-recognition/runs/baselines ./runs
echo "cerid"
rsync -avz --prune-empty-dirs --include "*/"  $cerid:~/har/skeleton-based-action-recognition/runs/baselines ./runs
echo "scyid"
rsync -avz --prune-empty-dirs --include "*/"  $scyid:~/har/skeleton-based-action-recognition/runs/baselines ./runs
# echo "charid"
# rsync -avz --prune-empty-dirs --include "*/"  $charid:~/har/skeleton-based-action-recognition/runs ./
echo "titid"
rsync -avz --prune-empty-dirs --include "*/"  $titid:~/har/skeleton-based-action-recognition/runs/baselines ./runs
echo "cycid"
rsync -avz --prune-empty-dirs --include "*/"  $cycid:~/har/skeleton-based-action-recognition/runs/baselines ./runs

# conda activate vgcn
# python tools/collect_results.py