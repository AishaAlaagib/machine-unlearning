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




dataset_dict = {
    1 : 'german_credit',
    2 : 'adult_income',
    3 : 'compas',
    4 : 'default_credit',
    5 : 'marketing'
    
}


data_map = { 
    'german_credit'     : 'German Credit',
    'adult_income'      : 'Adult Income',
    'compas'            : 'COMPAS',
    'default_credit'    : 'Default Credit',
    'marketing'         : 'Marketing'      
}

data_dict = {
    'german_credit'     : ('german_credit','target'),
    'adult_income'      : ('adult_income', 'income'),
    'compas'            : ('compas', 'low_risk'),
    'default_credit'    : ('default_credit', 'good_credit'),
    'marketing'         : ('marketing', 'subscribed')      
}  

subgroup_dict = {
    'german_credit'     : ('age_age:>=25', 'age_age:<25'),
    'adult_income'      : ('gender_Female', 'gender_Male'),
    'compas'            : ('race_African-American', 'race_Caucasian'),
    'default_credit'    : ('SEX_Female', 'SEX_Male'),
    'marketing'         : ('age_age:not30-60', 'age_age:30-60')      
}
space_DNN = {
    'units_l1'      : hp.choice('units_l1', [25, 50, 75, 100, 150, 200]),
    'dropout_l1'    : hp.uniform('dropout_l1', 0, 0.5),
    'units_l2'      : hp.choice('units_l2', [25, 50, 75, 100, 150, 200]),
    'dropout_l2'    : hp.uniform('dropout_l2', 0, 0.5),
    'nbr_layers'    : hp.choice('nbr_layers',
                    [
                        {'layers':'two',              
                        },
                        {'layers':'three',
                        'units_l3_a': hp.choice('units_l3_a', [25, 50, 75, 100, 150, 200]),
                        'dropout_l3_a': hp.uniform('dropout_l3_a', 0, 0.5),     
                        }, 
                        {'layers':'four',
                        'units_l3_b': hp.choice('units_l3_b', [25, 50, 75, 100, 150, 200]),
                        'dropout_l3_b': hp.uniform('dropout_l3_b', 0, 0.5), 
                        'units_l4': hp.choice('units_l4', [25, 50, 75, 100, 150, 200]),
                        'dropout_l4': hp.uniform('dropout_l4', 0, 0.5),     
                        },
                    ]),
    'optimizer'     : hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
    'epochs'        : 30,
    }

def prepare_data(data, rseed):
    #download the german_credit dataset
    urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', 'german.data')
    german_df = pd.read_csv('german.data', delimiter=' ',header=None)
    
    #download the doc file that have the Description of the German credit dataset.
    urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc', 'german.doc')
    f = open('german.doc')
    
    # add the columns to the dataset
    german_df.columns=['account_bal','duration','payment_status','purpose',
                   'credit_amount','savings_bond_value','employed_since',
                   'intallment_rate','sex_marital','guarantor','residence_since',
                   'most_valuable_asset','age','concurrent_credits','type_of_housing',
                   'number_of_existcr','job','number_of_dependents','telephon',
                   'foreign','target']
    
    # replace some attributes value 
    german_df= german_df.replace(['A11','A12','A13','A14', 'A171','A172','A173','A174','A121','A122','A123','A124'],
                  ['neg_bal','positive_bal','positive_bal','no_acc','unskilled','unskilled','skilled','highly_skilled',
                   'none','car','life_insurance','real_estate'])
    
    # label encoding
    le= LabelEncoder()
    le.fit(german_df.target)
    german_df.target=le.transform(german_df.target)

    # sklearn preprocessing for dealing with categorical variables
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in german_df:
        if german_df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(german_df[col].unique())) <= 2:
                # Train on the training data
                le.fit(german_df[col])
                # Transform both training and testing data
                german_df[col] = le.transform(german_df[col])

                # Keep track of how many columns were label encoded
                le_count += 1

    print('%d columns were label encoded.' % le_count)
    
    # one-hot encoding of categorical variables
    german_df = pd.get_dummies(german_df)
    print('Encoded Features shape: ', german_df.shape)
    
    
     # get the maj & min features
    german_df['age_age:<25'] = 0
    german_df['age_age:>=25'] = 0
    for i in range(german_df['age'].shape[0]):
        if  german_df['age'][i] < 25 :
            german_df['age_age:<25'][i] = '1'
            german_df['age_age:>=25'][i] = '0'
        else:
            german_df['age_age:<25'][i] = '0'
            german_df['age_age:>=25'][i] = '1'
    german_df.drop('age', axis = 1)
    
    
    # get the data and the label
    x, y = german_df.drop('target', axis=1), german_df['target']
   
    
    # split to train and test
    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=.2, random_state=42)
    # create directory to save the split data
    datadir = './split_train_test/'

    if not os.path.exists(datadir):
        os.mkdir(datadir)
    train_name = '{}train.csv'.format(datadir)
    test_name = '{}test.csv'.format(datadir)


    x_train.to_csv(train_name)
    x_test.to_csv(test_name)
    
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
   
    
    # safe the train and test file 
    # np.save(f'german_credit_train.npy', {'X': x_train, 'y': y_train})
    # np.save(f'german_credit_test.npy', {'X': x_test, 'y': y_test})
    

    # scale each feature to 0-1
#     scaler = MinMaxScaler(feature_range = (0, 1))

#     # fit on features dataset
#     scaler.fit(x)
#     x = scaler.transform(x)
#     scaler.fit(x_train)
#     scaler.fit(x_test)
#     x_train= scaler.transform(x_train)
#     x_test= scaler.transform(x_test)
    return x_train,y_train, x_test,  y_test



def prepare_data_as_dataframe( rseed):


    #filenames
    train_file      = './split_train_test/train.csv'
    test_file       = './split_train_test/test.csv'

    # load dataframe
    df_train    = pd.read_csv(train_file)
    df_test     = pd.read_csv(test_file)
#     print(df_train.head())
#     print(df_test.head())
    return df_train, df_test


def get_metrics(dataset, model_class, rseed):

    # load data as np array
    X_train, y_train, X_test, y_test = prepare_data(dataset, rseed)
    
    # load data as dataframe
    df_train, df_test = prepare_data_as_dataframe( rseed)
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
        res = {}

        # dataset
        res['Dataset']  = data_map[dataset]

        # model class
        res['Model']    = model_class

        # output type
        res['Partition']     = output_type

        # accuracy
        res['Accuracy'] = np.round(prediction_metrics['acc_{}'.format(output_type)], 3)
        
        # fairness
        res['SP']       = np.round(fairness_metrics['{}'.format(output_type)].fairness_metric(1), 3)
        res['PE']       = np.round(fairness_metrics['{}'.format(output_type)].fairness_metric(3), 3)
        res['EOpp']     = np.round(fairness_metrics['{}'.format(output_type)].fairness_metric(4), 3)
        res['EOdds']    = np.round(fairness_metrics['{}'.format(output_type)].fairness_metric(5), 3)

        return res


    prediction_metrics = get_predictions(model_class, X_train, y_train, X_test, y_test)    
    fairness_metrics = get_fairness_metrics(df_train, df_test, prediction_metrics)

    output_train    = get_output(dataset, model_class, 'train', prediction_metrics, fairness_metrics)
    output_test     = get_output(dataset, model_class, 'test', prediction_metrics, fairness_metrics)
    

    return output_train, output_test

        
if __name__ == '__main__':
    # inputs
    datasets = ['german_credit']
    model_classes = ['AdaBoost', 'DNN', 'RF', 'XgBoost']
    
    df_list = []

    for rseed in range(1):
        row_list = []
        for dataset in datasets:
            for model_class in model_classes:
                output_train, output_test = get_metrics(dataset, model_class, rseed)
                row_list.append(output_train)
                row_list.append(output_test)
        df = pd.DataFrame(row_list)
        df_list.append(df)

    
    
    average_row_list = []
    for index in range(len(df_list[0])):
        average_row = {
            'Dataset'      : df_list[0].iloc[index]['Dataset'],
            'Model'         : df_list[0].iloc[index]['Model'],
            'Partition'      : df_list[0].iloc[index]['Partition'],
            'Accuracy'     : np.round(np.mean([df_list[j].iloc[index]['Accuracy'] for j in range(1)]), 2),
            'SP'     : np.round(np.mean([df_list[j].iloc[index]['SP'] for j in range(1)]), 2),
            'PE'     : np.round(np.mean([df_list[j].iloc[index]['PE'] for j in range(1)]), 2),
            'EOpp'     : np.round(np.mean([df_list[j].iloc[index]['EOpp'] for j in range(1)]), 2),
            'EOdds'     : np.round(np.mean([df_list[j].iloc[index]['EOdds'] for j in range(1)]), 2)
        }
        average_row_list.append(average_row)
    df_average = pd.DataFrame(average_row_list) 
    
    
    save_dir = ('./results/summary')

    os.makedirs(save_dir, exist_ok=True)



    filename = '{}/summary.csv'.format(save_dir)

    

    df_average.to_csv(filename, encoding='utf-8', index=False)



    df = pd.read_csv('./results/summary/summary.csv')


    df.index = pd.MultiIndex.from_frame(df[["Dataset", "Model", "Partition"]])
    df = df.drop(["Dataset", "Model", "Partition"], axis=1)


    save_dir = ('./results/latex')

    os.makedirs(save_dir, exist_ok=True)

    filename = '{}/perfs.tex'.format(save_dir)

    df.to_latex(filename, multirow=True, index=True)


