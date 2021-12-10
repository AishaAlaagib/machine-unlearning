# data preprocess 'adult income, compas, default credit, marketing'
1. sbatch dataset_main.sh 

# train the 'DNN, RF, AdaBoost, XgBoost' 
# measure the unfairness for all the datasets
2. sbatch train_models.sh # comment the train model