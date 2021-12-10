#!/bin/bash
set -eou pipefail
IFS=$'\n\t'
 
shards=$1
req_per=$2
dataset=$3
req=50
    
if [[ ! -d "./results/${req}" ]] ; then

mkdir -p "./results/${req}"
fi

if [[ ! -f "./results/${req}/$dataset.csv" ]]; then
    echo "nb_shards,nb_requests,accuracy, SP,PE,EOpp,EOdds,retraining_time" > ./results/${req}/$dataset.csv
fi

for j in "${req_per[@]}" ; do
    r=$((${j}*${shards}/5))
    unf_res=$(python sisa_unfairness.py --strategy uniform --container "${shards}" --shards "${shards}" --dataset datasets/${dataset}/datasetfile --data ${dataset} --label "${r}" --per $req_per)
    cat containerss/$req_per/${dataset}/"${shards}"/times/shard-*:"${r}".time > "containerss/$req_per/${dataset}/${shards}/times/times"
    time=$(python time.py --data ${dataset} --per $req_per --container "${shards}" | awk -F ',' '{print $1}')
    echo "${shards}, ${r}, ${unf_res[*]} ${time}" >> ./results/${req}/$dataset.csv
done
