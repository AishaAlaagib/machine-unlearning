#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH -o slurm-%j.out  # Write the log on scratch

datasets=(marketing )
models=(NN )
rseed=(0 1 2 3 4 5 6 7 8 9)
requests=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95)


# for dataset in "${datasets[@]}" 
#     do
#         for r in ${rseed[@]}
#             do
#                 for req in ${requests[@]}
#                     do
#                         for model in "${models[@]}" 
#                             do	
#                                 # pretraining the black-box model
#                                 python train_models.py --dataset=$dataset --model_class=$model --nbr_evals=10 --rseed=$r --requests=$req
#                                 # getting the predictions for the suing group and the test set
#         #                         sleep 2
#         #                         python get_labels.py --dataset=$dataset --model_class=$model --rseed=$r
#                             done
#                     done
#                 # getting the true labels for the suing group and the test set
# #                 python get_true_labels.py --dataset=$dataset --rseed=$r
#             done
#     done

# Other analysis

# ## summary of black-box models, results save in latex tables: in results/latex/perfs.tex
# # compute the unfairness of the black-box
# # python unfairness_result.py 

# python summary.py 
# # python latex_summary.py
python visualization.py