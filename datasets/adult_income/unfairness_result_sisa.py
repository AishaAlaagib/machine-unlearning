from __future__ import print_function

from functools import partial
from urllib.request import urlretrieve
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



# hyper-opt
from hyperopt import hp, Trials, STATUS_OK, tpe, fmin
from hyperopt import space_eval
from hyperopt.pyll import scope

from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential

# utils
import pickle
import argparse
import os
import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import load_model

from metrics import ConfusionMatrix, Metric

from collections import Counter




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
    X_train = df_train.drop(labels=[decision], axis = 1)
    X_train = scaler.fit_transform(X_train)
    ### cast
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)

    ## test set
    y_test = df_test[decision]
    X_test = df_test.drop(labels=[decision], axis = 1)
    X_test = scaler.fit_transform(X_test)
    ### cast
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)


    return X_train, y_train, X_test, y_test



def prepare_data_as_dataframe(data, rseed):

    dataset, _ = data_dict[data]
    datadir = './preprocessed/{}/'.format(dataset)    

    #filenames
    train_file      = '{}{}_trainOneHot_{}.csv'.format(datadir, dataset, rseed)
    test_file       = '{}{}_testOneHot_{}.csv'.format(datadir, dataset, rseed)

    # load dataframe
    df_train    = pd.read_csv(train_file)
    df_test     = pd.read_csv(test_file)

    return df_train, df_test


def get_metrics(dataset, model_class, rseed):

    # load data as np array
    X_train, y_train, X_test, y_test = prepare_data(dataset, rseed)
    
    # load data as dataframe
    df_train, df_test = prepare_data_as_dataframe( dataset, rseed)
    # load meta data for fairness metrics
#     _, decision = data_dict[dataset]
    decision = y_train

    min_feature, maj_feature = subgroup_dict[dataset]
    print("---------------------------->>> dataset = {}".format(dataset))
    print("---------------------------->>> model = {}".format(model_class))
    # model path
    outdir = './pretrained/{}/'.format(dataset)
    model_path = '{}{}_{}.h5'.format(outdir, model_class, rseed)

    def get_predictions(model_class, X_train, y_train, X_test, y_test):
        predictions_train, predictions_test = None, None
        acc_train, acc_test = None, None

        prediction_metrics = {}
        
        if model_class == 'DNN':
            # load model
            
            mdl = load_model(model_path)
            print('model loaded')
            # get prediction
            #---train
            predictions_train = (mdl.predict(X_train) > 0.5).astype('int32')
            predictions_train = [x[0] for x in predictions_train]
            print(Counter(predictions_train))


            #---test
            predictions_test = (mdl.predict(X_test) > 0.5).astype('int32')
            predictions_test = [x[0] for x in predictions_test]
            


            # get accuracy
            acc_train = mdl.evaluate(X_train, y_train)[1]
            acc_test = mdl.evaluate(X_test, y_test)[1]
            print(acc_train, acc_test)
            
        if model_class in ['RF', 'AdaBoost', 'XgBoost']:
            # load model
            mdl = pickle.load(open(model_path,"rb"))
            
            # get prediction
            #---train
            predictions_train = mdl.predict(X_train)
            predictions_train = [int(x) for x in predictions_train]
#             print('predictions_train',predictions_train)

            #---test
            predictions_test = mdl.predict(X_test)
            predictions_test = [int(x) for x in predictions_test]

            # get accuracy
            acc_train   = accuracy_score(y_train, mdl.predict(X_train))
            acc_test    = accuracy_score(y_test, mdl.predict(X_test))

        #----train
        prediction_metrics['predictions_train'] = predictions_train
        prediction_metrics['acc_train'] = acc_train

        #----test
        prediction_metrics['predictions_test'] = predictions_test
        prediction_metrics['acc_test'] = acc_test


        return prediction_metrics

    
    def get_fairness_metrics(df_train, df_test, prediction_metrics):
        # output object
        fairness_metrics = {}

        #----train
        df_train['predictions'] = prediction_metrics['predictions_train']
   
        cm_train = ConfusionMatrix(df_train[min_feature], df_train[maj_feature], df_train['predictions'], decision)
        cm_minority_train, cm_majority_train = cm_train.get_matrix()
        fm_train = Metric(cm_minority_train, cm_majority_train)


        #----test
        df_test['predictions'] = prediction_metrics['predictions_test']
        cm_test = ConfusionMatrix(df_test[min_feature], df_test[maj_feature], df_test['predictions'], y_test)
        cm_minority_test, cm_majority_test = cm_test.get_matrix()
        fm_test = Metric(cm_minority_test, cm_majority_test)

        fairness_metrics['train']   = fm_train
        fairness_metrics['test']    = fm_test

        return fairness_metrics

    
    def get_output(dataset, model_class, output_type, prediction_metrics, fairness_metrics):
        
        metrics = [1, 3, 4, 5]
        metrics_map  = {
            1: 'SP',
            2: 'PP',
            3: 'PE',
            4: 'EOpp',
            5: 'EOdds',
            6: 'CUAE'
        }

        res = []

        for metric in metrics:
            dd = {}
            # model class
            dd['model_class']  = model_class
            # unfairness
            dd['unfairness']   = np.round(fairness_metrics['{}'.format(output_type)].fairness_metric(metric), 3)
            # metric
            dd['metric']  = metrics_map[metric]

            res.append(dd)


        return res

    prediction_metrics = get_predictions(model_class, X_train, y_train, X_test, y_test)    
    fairness_metrics = get_fairness_metrics(df_train, df_test, prediction_metrics)

    output_train    = get_output(dataset, model_class, 'train', prediction_metrics, fairness_metrics)
    output_test     = get_output(dataset, model_class, 'test', prediction_metrics, fairness_metrics)
    

    return output_train, output_test


def process_model_class(dataset, model_class):
    test_list   = [ [], [], [], [] ]

    for rseed in range(1):
        _,  output_test = get_metrics(dataset, model_class, rseed)

        test_list[0].append(output_test[0])
        test_list[1].append(output_test[1])
        test_list[2].append(output_test[2])
        test_list[3].append(output_test[3])
    
    output_test = []

    for ll in test_list:
        dd = {
            'model_class' : ll[0]['model_class'],
            'unfairness'  : np.round(np.mean([dd['unfairness'] for dd in ll]), 3),
            'metric' : ll[0]['metric'],
        }
        output_test.append(dd)

    return  output_test


if __name__ == '__main__':
    # inputs
    datasets = ['adult_income', 'compas', 'default_credit', 'marketing']
    model_classes = ['AdaBoost','DNN', 'RF', 'XgBoost']
    


    for dataset in datasets:
        row_list = []
        for model_class in model_classes:
            output_test = process_model_class(dataset, model_class)
            
            for dd in output_test:
                print(dd)
                dd['group'] =  'Non-Members'
                print(dd)
                row_list.append(dd)
        
        save_dir = ('./results/unfairness_bbox')
        os.makedirs(save_dir, exist_ok=True)
        filename = '{}/{}.csv'.format(save_dir, dataset)
        

        df = pd.DataFrame(row_list)
        df.to_csv(filename, encoding='utf-8', index=False)

