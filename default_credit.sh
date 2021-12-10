#!/bin/bash
#SBATCH --ntasks=3
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --partition=t4v2

# requests=( 327 1636 3272 4909 6545 8181 9817 11453 13089 14726 16362 17998 19634 21270 22906) # adult_income 
# requests=(42 206 412 618 824 1030 1236 1442 1648 1854 2060 2266 2678 2884) # compas
# requests=( 275 1380 2759 4138 5517 6897 8276 9655 11034 12414 13793 15172 16552 17931 19310 ) # marketing
requests=( 14063)  # default_credit
# for r in "${requests[@]}"; do
    # prepare the dataset 
    # bash  datasets/data.sh

    #get the shards and the requests for all the datasets
#     bash  example-scripts/init.sh 5 $r default_credit

#     bash  example-scripts/train.sh 5 $r default_credit


#     bash  example-scripts/predict.sh 5 $r default_credit


#     bash  example-scripts/data.sh 5 $r default_credit


# done 
python visulaization.py 
