#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH -o slurm-%j.out  # Write the log on scratch

# requests=(3272 6545 9817 13089 16362 19634  22906  26100  29451 31087) # adult_income 
# requests=(412 824 1236 1648 2060 2476 2884 3296 3708) # compas
requests=(1607 5517 8276 11034 13793 16552 19310 22069 24828 26207 ) # marketing
# requests=(200 1005 2009 3014 4019 5023 6029 7031 8036 9040 10045 11049 12054 13058 14063)  # default_credit
datasets=(adult_income compas marketing default_credit)
rseeds=(0 1 2 3 4 5 6 7 8 9)
for r in "${requests[@]}"; do
    # prepare the dataset 
    for rseed in "${rseeds[@]}"; do
        bash  datasets/data.sh marketing $rseed

    #     #get the shards and the requests for all the datasets
        bash  example-scripts/init.sh 5 $r marketing $rseed

        bash  example-scripts/train.sh 5 $r marketing $rseed


        bash  example-scripts/predict.sh 5 $r marketing $rseed


        bash  example-scripts/data.sh 5 $r marketing $rseed

    done
done 
# python visulaization.py 
# python summary.py