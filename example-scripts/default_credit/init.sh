#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1

datasets=(default_credit )
rseed=(0)

for dataset in "${datasets[@]}"; do
    
    if [[ ! -d "containers/$dataset/${shards}" ]] ; then

        mkdir -p  "containers/$dataset/${shards}"
        mkdir -p "containers/$dataset/${shards}/cache"
        mkdir -p "containers/$dataset/${shards}/times"
        mkdir -p "containers/$dataset/${shards}/outputs"
        echo 0 > "containers/$dataset/${shards}/times/null.time"
    fi

    python distribution.py --shards "${shards}" --distribution uniform --container "${shards}" --dataset datasets/$dataset/datasetfile --label 0

    for j in {1..2009}; do
        r=$((${j}*${shards}/5))
        python distribution.py --requests "${r}" --distribution uniform --container "${shards}" --dataset datasets/$dataset/datasetfile --label "${r}"
    done
done

