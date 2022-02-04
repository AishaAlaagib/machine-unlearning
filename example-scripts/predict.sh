
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
        python sisas.py --model purchase --test --dataset datasets/${dataset}/datasetfile --label "${r}" --batch_size 16 --container "${shards}" --shard "${i}" --data ${dataset} --per ${req_per} --rseed ${seed}
    done
done



    
#     if  [[ $dataset == "compas" ]];   then 
    
#         for i in $(seq 0 "$((${shards}-1))"); do
#             for j in "${req_per[@]}" ; do
#                 echo "shard: $((${i}+1))/${shards}, requests: $((${j}))"
#                 r=$((${j}*${shards}/5))
#                 python sisas.py --model purchase --test --dataset datasets/${dataset}/datasetfile --label "${r}" --batch_size 16 --container "${shards}" --shard "${i}" --data ${dataset} --per ${req_per}
#             done
#         done  
#     fi 
    
    
#     if  [[ $dataset == "default_credit" ]]; then 
    
#         for i in $(seq 0 "$((${shards}-1))"); do
#             for j in {0..200}; do
#                 echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/201"
#                 r=$((${j}*${shards}/5))
#                 python sisas.py --model purchase --test --dataset datasets/${dataset}/datasetfile --label "${r}" --batch_size 16 --container "${shards}" --shard "${i}" --data ${dataset} --per ${req_per}
#             done
#         done
#     fi
    
    
#     if  [[ $dataset == "marketing" ]]; then 
    
#         for i in $(seq 0 "$((${shards}-1))"); do
#             for j in {0..276}; do
#                 echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/277"
#                 r=$((${j}*${shards}/5))
#                 python sisas.py --model purchase --test --dataset datasets/${dataset}/datasetfile --label "${r}" --batch_size 16 --container "${shards}" --shard "${i}" --data ${dataset} --per ${req_per}
#             done
#         done
        
#     fi



    
# done

