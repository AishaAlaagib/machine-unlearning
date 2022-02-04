#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
req_per=$2 
dataset=$3
seed=$4
for i in $(seq 0 "$((${shards}-1))"); do
    for j in "${req_per[@]}" ; do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}))"
        r=$((${j}*${shards}/5))
        python sisas.py --model purchase --train --slices 1 --dataset datasets/${dataset}/datasetfile --label "${r}" --epochs 20 --batch_size 16 --learning_rate 0.001 --optimizer sgd --chkpt_interval 1 --container "${shards}" --shard "${i}" --data ${dataset} --per ${req_per} --rseed ${seed}
    done
done







