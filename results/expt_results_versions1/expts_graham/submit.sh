#!/bin/bash
#SBATCH --job-name=test2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000M
#SBATCH --time=12:00:00
#SBATCH --account=def-corbeilj

date
which python
pyenv3
cd $HOME/scratch/few_shot_regression/
python main.py
date
