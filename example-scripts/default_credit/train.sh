#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
datasets=(default_credit )
rseed=(0)

for dataset in "${datasets[@]}"; do

    for i in $(seq 0 "$((${shards}-1))"); do
        for j in {0..200}; do
            echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/201"
            r=$((${j}*${shards}/5))
            python sisa.py --model purchase --train --slices 1 --dataset datasets/${dataset}/datasetfile --label "${r}" --epochs 20 --batch_size 16 --learning_rate 0.001 --optimizer sgd --chkpt_interval 1 --container "${shards}" --shard "${i}" --data ${dataset}
        done
    done
done