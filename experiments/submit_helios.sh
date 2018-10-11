#!/bin/bash
#PBS -N expts_mhc_pan
#PBS -A nne-790-aa
#PBS -l nodes=1:gpus=1
#PBS -l walltime={time}
#PBS -l feature=k80
#PBS -V
#PBS -t [0-{ntasks}]
#PBS -o stdout%I.out
#PBS -e stderr%I.err

date
SECONDS=0
which python
cd $HOME/scratch/few_shot_regression/
python {main_file} --part $SLURM_ARRAY_TASK_ID --out {out} --algos {algos} --dataset {dataset}
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
