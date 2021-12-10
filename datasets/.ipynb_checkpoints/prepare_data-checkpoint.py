from __future__ import print_function

# utils
import pickle
import argparse
import os
import numpy as np 
from torch.nn import Module, Linear
from torch.nn.functional import tanh
import pandas as pd 


from functools import partial
from urllib.request import urlretrieve
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
dataset_dict = {
    1 : 'adult_income',
    2 : 'compas',
    3 : 'default_credit',
    4 : 'marketing'
}

data_dict = {
    'adult_income'      : ('adult_income', 'income'),
    'compas'            : ('compas', 'two_year_recid'),
    'default_credit'    : ('default_credit', 'DEFAULT_PAYEMENT'),
    'marketing'         : ('marketing', 'subscribed') 
}  
subgroup_dict = {
    'adult_income'      : ('gender_Female', 'gender_Male'),
    'compas'            : ('race_African-American', 'race_Caucasian'),
    'default_credit'    : ('SEX_Female', 'SEX_Male'),
    'marketing'         : ('age_age:not30-60', 'age_age:30-60')      
}

def prepare_data(data, rseed):

    dataset, decision = data_dict[data]
    min_grp, maj_grp = subgroup_dict[data]
    datadir = './preprocessed/{}/'.format(dataset)      
    
    #filenames
    suffix = 'OneHot'
    train_file      = '{}{}_train{}_{}.csv'.format(datadir, dataset, suffix, rseed)
    test_file       = '{}{}_test{}_{}.csv'.format(datadir, dataset, suffix, rseed)

    # load dataframe
    df_train    = pd.read_csv(train_file)
    df_test     = pd.read_csv(test_file)

    # prepare the data
    scaler = StandardScaler()
    ## training set
    y_train = df_train[decision]
    
    maj_features_train = df_train[maj_grp]
    min_features_train = df_train[min_grp]
    
    X_train = df_train.drop(labels=[decision], axis = 1)
    
    X_train = scaler.fit_transform(X_train)
    ### cast
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    maj_train  = np.asarray(maj_features_train).astype(np.float32)
    min_train  = np.asarray(min_features_train).astype(np.float32)
    
    ## test set
    y_test = df_test[decision]
    maj_features_test = df_test[maj_grp]
    min_features_test = df_test[min_grp]
    X_test = df_test.drop(labels=[decision], axis = 1)
    X_test = scaler.fit_transform(X_test)
    ### cast
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)
    maj_test  = np.asarray(maj_features_test).astype(np.float32)
    min_test  = np.asarray(min_features_test).astype(np.float32)

    return X_train, y_train, X_test, y_test, maj_train, min_train, maj_test, min_test
    
    
if __name__ == '__main__':
    
    # parser initialization
    parser = argparse.ArgumentParser(description='Script preprocessing the datasets')
    parser.add_argument('--dataset', type=str, default='german_credit', help='adult_income, compas, default_credit, marketing')
    parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 9')
    parser.add_argument('--model_class', type=str, default='DNN', help='DNN, RF, AdaBoost, XgBoost')

    # get input
    args = parser.parse_args()
    dataset = args.dataset
    rseed = args.rseed
    

    X_train, y_train, X_test, y_test, maj_train, min_train, maj_test, min_test = prepare_data(dataset, rseed)
    
    print('np_train',X_train.shape, 'np_test',X_test.shape)
    print(dataset, np.unique(y_train))
    path = f'./{dataset}/{dataset}'
    np.save(f'{path}_train.npy', {'X': X_train, 'y': y_train, 'maj_train':maj_train, 'min_train':min_train})
    np.save(f'{path}_test.npy', {'X': X_test, 'y': y_test, 'maj_test':maj_test, 'min_test':min_test})
    