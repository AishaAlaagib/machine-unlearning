#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH -o slurm-%j.out  # Write the log on scratch



# datasets=(new_adult_income adult_income compas default_credit marketing )
dataset=$1
rseed=$2

for dataset in "${datasets[@]}" 
    do
        
                # preparing the dataset
        python ./datasets/prepare_data.py --dataset=$dataset --rseed=$rseed
            
    done
