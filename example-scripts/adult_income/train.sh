#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
datasets=(adult_income )
rseed=(0)

for dataset in "${datasets[@]}"; do

    for i in $(seq 0 "$((${shards}-1))"); do
        for j in {0..327}; do
            echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/328"
            r=$((${j}*${shards}/5))
            python sisa.py --model purchase --train --slices 1 --dataset datasets/${dataset}/datasetfile --label "${r}" --epochs 20 --batch_size 16 --learning_rate 0.001 --optimizer sgd --chkpt_interval 1 --container "${shards}" --shard "${i}" --data ${dataset}
        done
    done
done
