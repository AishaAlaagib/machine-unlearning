#!/bin/bash
#SBATCH --ntasks=3
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -c 4
#SBATCH --partition=t4v2
#SBATCH --mail-user=aalaagib@aimsammi.org
#SBATCH --mail-type=ALL

datasets=(new_adult_income )
models=(DNN AdaBoost )
rseed=(0)

for dataset in "${datasets[@]}" 
    do
        for r in ${rseed[@]}
            do
                for model in "${models[@]}" 
                    do	
                        # pretraining the black-box model
                        python train_models.py --dataset=$dataset --model_class=$model --nbr_evals=10 --rseed=$r
                        # getting the predictions for the suing group and the test set
#                         sleep 2
#                         python get_labels.py --dataset=$dataset --model_class=$model --rseed=$r
                    done
                # getting the true labels for the suing group and the test set
#                 python get_true_labels.py --dataset=$dataset --rseed=$r
            done
    done

# # Other analysis

# ## summary of black-box models, results save in latex tables: in results/latex/perfs.tex
# # compute the unfairness of the black-box
# # python unfairness_result.py 
python summary.py
python latex_summary.py
