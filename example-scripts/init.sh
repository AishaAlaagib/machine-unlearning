#SBATCH --array=1,2,3,4
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -c 4
#SBATCH --partition=t4v2
#SBATCH --mail-user=aalaagib@aimsammi.org
#SBATCH --mail-type=ALL


set -eou pipefail
IFS=$'\n\t'

shards=$1
req_per=$2 
dataset=$3
seed=$4

# for dataset in "${datasets[@]}"; do
    
if [[ ! -d "containerss/${req_per}/$seed/$dataset/${shards}" ]] ; then

    mkdir -p "containerss/${req_per}/$seed/$dataset/${shards}"
    mkdir -p "containerss/${req_per}/$seed/$dataset/${shards}/cache"
    mkdir -p "containerss/${req_per}/$seed/$dataset/${shards}/times"
    mkdir -p "containerss/${req_per}/$seed/$dataset/${shards}/outputs"
    echo 0 > "containerss/${req_per}/$seed/$dataset/${shards}/times/null.time"
fi




for j in "${req_per[@]}"  ; do
    r=$((${j}*${shards}/5))
    python distribution.py --shards "${shards}" --distribution uniform --container "${shards}" --dataset datasets/${dataset}/datasetfile --data ${dataset} --per ${req_per} --rseed ${seed} --label 0
    python distribution.py --requests "${r}" --distribution uniform --container "${shards}" --dataset datasets/${dataset}/datasetfile --data ${dataset} --per ${req_per} --rseed ${seed} --label "${r}"
done

# fi


# if  [[ $dataset == "compas" ]];   then 

#     for j in "${req_per[@]}" ; do
#         r=$((${j}*${shards}/5))
#         python distribution.py --shards "${shards}" --distribution uniform --container "${shards}" --dataset datasets/${dataset}/datasetfile --data ${dataset} --per ${req_per} --label 0
#         python distribution.py --requests "${r}" --distribution uniform --container "${shards}" --dataset datasets/${dataset}/datasetfile --data ${dataset} --per ${req_per} --label "${r}"

#     done        
# fi 


# if  [[ $dataset == "default_credit" ]]; then 

#     for j in {1..200}  ; do
#         r=$((${j}*${shards}/5))
#         python distribution.py --shards "${shards}" --distribution uniform --container "${shards}" --dataset datasets/${dataset}/datasetfile --data ${dataset} --per ${req_per} --label 0
#         python distribution.py --requests "${r}" --distribution uniform --container "${shards}" --dataset datasets/${dataset}/datasetfile --data ${dataset} --per ${req_per} --label "${r}"
#     done   
# fi


# if  [[ $dataset == "marketing" ]];
# then 
#     for j in {1..276}  ; do
#         r=$((${j}*${shards}/5))
#         python distribution.py --shards "${shards}" --distribution uniform --container "${shards}" --dataset datasets/${dataset}/datasetfile --data ${dataset} --per ${req_per} --label 0
#         python distribution.py --requests "${r}" --distribution uniform --container "${shards}" --dataset datasets/${dataset}/datasetfile --data ${dataset} --per ${req_per} --label "${r}"
#     done   
# fi

#done        
    
    
    

