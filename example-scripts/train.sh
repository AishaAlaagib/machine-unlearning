#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
req_per=$2 
dataset=$3

for i in $(seq 0 "$((${shards}-1))"); do
    for j in "${req_per[@]}" ; do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}))"
        r=$((${j}*${shards}/5))
        python sisas.py --model purchase --train --slices 1 --dataset datasets/${dataset}/datasetfile --label "${r}" --epochs 20 --batch_size 16 --learning_rate 0.001 --optimizer sgd --chkpt_interval 1 --container "${shards}" --shard "${i}" --data ${dataset} --per ${req_per}
    done
done




# if  [[ $dataset == "compas" ]];   then 

#     for i in $(seq 0 "$((${shards}-1))"); do
#         for j in "${req_per[@]}" ; do
#             echo "shard: $((${i}+1))/${shards}, requests: $((${j}))"
#             r=$((${j}*${shards}/5))
#             python sisas.py --model purchase --train --slices 1 --dataset datasets/${dataset}/datasetfile --label "${r}" --epochs 20 --batch_size 16 --learning_rate 0.001 --optimizer sgd --chkpt_interval 1 --container "${shards}" --shard "${i}" --data ${dataset} --per ${req_per}
#         done
#     done  
# fi 


# if  [[ $dataset == "default_credit" ]]; then 

#     for i in $(seq 0 "$((${shards}-1))"); do
#         for j in {0..200}; do
#             echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/201"
#             r=$((${j}*${shards}/5))
#             python sisas.py --model purchase --train --slices 1 --dataset datasets/${dataset}/datasetfile --label "${r}" --epochs 20 --batch_size 16 --learning_rate 0.001 --optimizer sgd --chkpt_interval 1 --container "${shards}" --shard "${i}" --data ${dataset} --per ${req_per}
#         done
#     done
# fi


# if  [[ $dataset == "marketing" ]]; then 

#     for i in $(seq 0 "$((${shards}-1))"); do
#         for j in {0..276}; do
#             echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/277"
#             r=$((${j}*${shards}/5))
#             python sisas.py --model purchase --train --slices 1 --dataset datasets/${dataset}/datasetfile --label "${r}" --epochs 20 --batch_size 16 --learning_rate 0.001 --optimizer sgd --chkpt_interval 1 --container "${shards}" --shard "${i}" --data ${dataset} --per ${req_per}
#         done
#     done

# fi




