#!/bin/bash
#PBS -N expts_mhc_pan
#PBS -A nne-790-aa
#PBS -l nodes=1:gpus=1
#PBS -l walltime=12:00:00
#PBS -l feature=k80
#PBS -V
#PBS -t [12,61-80]
#PBS -o stdout%I.out
#PBS -e stderr%I.err

date
which python
cd $HOME/scratch/few_shot_regression/
python main.py $MOAB_JOBARRAYINDEX -1
date
