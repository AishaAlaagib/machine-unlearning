#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=3:00:00                                   # The job will run for 3 hours
#SBATCH -o slurm-%j.out  # Write the log on scratch



python main.py --dataset=adult_income

# python main.py --dataset=new_adult_income


python main.py --dataset=compas

python main.py --dataset=default_credit

python main.py --dataset=marketing