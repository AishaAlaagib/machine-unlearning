#!/bin/bash
#SBATCH --ntasks=3
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -c 4
#SBATCH --partition=t4v2

set -eou pipefail
IFS=$'\n\t'

datasets=(new_adult_income adult_income compas default_credit marketing )
rseed=(0)

for dataset in "${datasets[@]}" 
    do
        for r in ${rseed[@]}
            do
                # preparing the dataset
                python prepare_data.py --dataset=$dataset --rseed=$r
            done
    done
