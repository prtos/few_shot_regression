#!/bin/bash
{computer_configuration}
#SBATCH --mem=12000M
#SBATCH --time={time}
#SBATCH --account=rrg-corbeilj-ac
#SBATCH --array=0-{ntasks}
date
SECONDS=0
which python
# source $HOME/venv3/bin/activate
cd $HOME/scratch/few_shot_regression/
python {main_file} --part $SLURM_ARRAY_TASK_ID --out {out} --algos {algos} --dataset {dataset}
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
