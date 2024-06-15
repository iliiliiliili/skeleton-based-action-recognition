#!/bin/bash -i

echo "monid"
rsync -avz --prune-empty-dirs --include "*/"  --include="*.test.result" --exclude="*" $monid:~/har/skeleton-based-action-recognition/runs ./
echo "cerid"
rsync -avz --prune-empty-dirs --include "*/"  --include="*.test.result" --exclude="*" $cerid:~/har/skeleton-based-action-recognition/runs ./
echo "scyid"
rsync -avz --prune-empty-dirs --include "*/"  --include="*.test.result" --exclude="*" $scyid:~/har/skeleton-based-action-recognition/runs ./
echo "charid"
rsync -avz --prune-empty-dirs --include "*/"  --include="*.test.result" --exclude="*" $charid:~/har/skeleton-based-action-recognition/runs ./
echo "titid"
rsync -avz --prune-empty-dirs --include "*/"  --include="*.test.result" --exclude="*" $titid:~/har/skeleton-based-action-recognition/runs ./

conda activate vgcn
python tools/collect_results.py