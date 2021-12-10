#!/bin/bash
#SBATCH --ntasks=3
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --partition=t4v2

# requests=( 327 1636 3272 4909 6545 8181 9817 11453 13089 14726 16362 17998 19634 21270 24543 26179 2781529451 31087 ) # adult_income 
# requests=(42 206 412 618 824 1030 1236 1442 1648 1854 2060 2266 2678 2884) # compas
# requests=(  26190 ) #(275 1380 2759 4138 5517 6897 8276 9655 11034 12414 13793 15172 16552 17931 19310 20691 22071 23450 24830 ) #marketing
# requests=(15067 16072 17076 18081 19085) #(200 1005 2009 3014 4019 5023 6029 7031 8036 9040 10045 11049 12054 13058 14063 )  # default_credit
requests=(1310 6554 13109 19664 26219 32773 39328 45883 52438 58992 65547 72102 78657 85211)

# datasets=( new_adult_income)

# for dataset in "${datasets[@]}"; do

#     for r in "${requests[@]}"; do
#         # prepare the dataset 
#         # bash  datasets/data.sh

#         #get the shards and the requests for all the datasets
# #         bash  example-scripts/init.sh 5 $r $dataset

# #         bash  example-scripts/train.sh 5 $r $dataset


# #         bash  example-scripts/predict.sh 5 $r $dataset


#         bash  example-scripts/data.sh 5 $r $dataset


#     done 
# done
python visulaization.py 
