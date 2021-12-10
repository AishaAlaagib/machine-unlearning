#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -c 4
#SBATCH --partition=t4v2
#SBATCH --mail-user=aalaagib@aimsammi.org
#SBATCH --mail-type=ALL
#SBATCH --error=R-%x.%j.err

# datasets=(german_credit)
# models=(AdaBoost DNN RF XgBoost)
# rseed=(0)

# for dataset in "${datasets[@]}" 
#     do
#         for r in ${rseed[@]}
#             do
#                 for model in "${models[@]}" 
#                     do	
#                         # pretraining the black-box model
#                         python train_models.py --dataset=$dataset --model_class=$model --nbr_evals=10 --rseed=$r
#                     done
#             done
#     done


# # Other analysis

# ## summary of black-box models, results save in latex tables: in results/latex/perfs.tex
# # compute the unfairness of the black-box
python unfairness_result.py 

