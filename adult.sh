#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH -o slurm-%j.out  # Write the log on scratch

requests=(3272) # (6545 9817 13089 16362 19634  22906  26100  29451 31087) # adult_income 
# requests=(412 824 1236 1648 2060 2476 2884 3296 3708 3914) # compas
# requests=( 275 1380 2759 4138 5517 6897 8276 9655 11034 12414 13793 15172 16552 17931 19310 ) # marketing
# requests=(200 1005 2009 3014 4019 5023 6029 7031 8036 9040 10045 11049 12054 13058 14063)  # default_credit
datasets=(adult_income) #(compas marketing default_credit)
rseeds=(0 )
for r in "${requests[@]}"; do
    # prepare the dataset 
    for rseed in "${rseeds[@]}"; do
        bash  datasets/data.sh adult_income $rseed

#     #     #get the shards and the requests for all the datasets
#         bash  example-scripts/init.sh 5 $r adult_income $rseed

#         bash  example-scripts/train.sh 5 $r adult_income $rseed


#         bash  example-scripts/predict.sh 5 $r adult_income $rseed


#         bash  example-scripts/data.sh 5 $r adult_income $rseed

    done
done 
# python summary.py
# python visulaization.py 
