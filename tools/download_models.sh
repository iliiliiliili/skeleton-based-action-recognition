#!/bin/bash -i

echo "monid"
rsync -avz $monid:~/har/skeleton-based-action-recognition/runs ./
echo "cerid"
rsync -avz $cerid:~/har/skeleton-based-action-recognition/runs ./
echo "scyid"
rsync -avz $scyid:~/har/skeleton-based-action-recognition/runs ./
echo "charid"
rsync -avz $charid:~/har/skeleton-based-action-recognition/runs ./
echo "titid"
rsync -avz $titid:~/har/skeleton-based-action-recognition/runs ./
echo "cycid"
rsync -avz $cycid:~/har/skeleton-based-action-recognition/runs ./

conda activate vgcn
python tools/collect_results.py