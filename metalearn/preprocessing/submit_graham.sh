#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --array=0-{ntasks}
date
SECONDS=0
which python
cd $HOME/scratch/few_shot_regression/utils/preprocessing/
python preprocessing_ucidatasets.py --server graham --part $SLURM_ARRAY_TASK_ID
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
