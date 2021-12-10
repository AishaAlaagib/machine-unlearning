#!/bin/bash
set -eou pipefail
IFS=$'\n\t'

shards=$1

if [[ ! -f adult_income-report_10%.csv ]]; then
    echo "nb_shards,nb_requests,accuracy, SP,PE,EOpp,EOdds,retraining_time" > adult_income.csv
fi

for j in {0..3273}; do
    r=$((${j}*${shards}/5))
    unf_res=$(python sisa_unfairness.py --strategy uniform --container "${shards}" --shards "${shards}" --dataset datasets/adult_income/datasetfile --label "${r}")
    cat containers/"${shards}"/times/shard-*:"${r}".time > "containers/${shards}/times/times"
    time=$(python time.py --container "${shards}" | awk -F ',' '{print $1}')
    echo "${shards},${r},${unf_res},${time}" >> adult_income.csv
done
