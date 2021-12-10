#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -c 4
#SBATCH --partition=t4v2
#SBATCH --mail-user=aalaagib@aimsammi.org
python main.py --dataset=adult_income

# python main.py --dataset=compas

# python main.py --dataset=default_credit

# python main.py --dataset=marketing