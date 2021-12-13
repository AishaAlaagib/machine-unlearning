#!/bin/bash
#SBATCH --ntasks=3
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -c 4
#SBATCH --partition=t4v2
#SBATCH --mail-user=aalaagib@aimsammi.org
#SBATCH --mail-type=ALL

datasets=(Adult_income )
models=(DNN )
rseed=(0)
requests=(327 1636 3272 4909 6545 8181 9817 11453 13089 14726 16362 17998 19634 21270 22906 24543 26179 27815 29451 31087)
# requests=(42 206 412 618 824 1030 1236 1442 1648 1854 2060 2266 2678 2884 2513 3090 3296 3502 3708 3914) # compas

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
# python latex_summary.py
python visualization.py