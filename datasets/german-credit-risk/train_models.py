from __future__ import print_function

from functools import partial
from urllib.request import urlretrieve
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import load_model

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
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score

dataset_dict = {
    1 : 'german_credit',
    2 : 'adult_income',
    3 : 'compas',
    4 : 'default_credit',
    5 : 'marketing'
    
}

data_dict = {
    'german_credit'     : ('german_credit','target'),
    'adult_income'      : ('adult_income', 'income'),
    'compas'            : ('compas', 'low_risk'),
    'default_credit'    : ('default_credit', 'good_credit'),
    'marketing'         : ('marketing', 'subscribed')      
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

space_RF = {
    'max_depth'         : scope.int(hp.uniform('max_depth', 1, 11)),
    'max_features'      : hp.choice('max_features', ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    'n_estimators'      : scope.int(hp.qloguniform('n_estimators', np.log(9.5), np.log(300.5), 1)),
    'criterion'         : hp.choice('criterion', ["gini", "entropy"]),
    'min_samples_split' : hp.choice('min_samples_split', [2, 5, 10]),
    'min_samples_leaf'  : hp.choice('min_samples_leaf', [1, 2, 4]),
    'bootstrap'         : hp.choice('bootstrap', [True, False]),
}

space_AdaBoost = {
    'n_estimators'          : scope.int(hp.qloguniform('n_estimators', np.log(9.5), np.log(300.5), 1)),
    'learning_rate'         : hp.lognormal('learning_rate', np.log(0.01), np.log(10.0)),
    'algorithm'             : hp.choice('algorithm', ['SAMME', 'SAMME.R']),
}

space_XgBoost = {
    'max_depth'             : scope.int(hp.uniform('max_depth', 1, 11)),
    'n_estimators'          : scope.int(hp.qloguniform('n_estimators', np.log(9.5), np.log(300.5), 1)),
    'learning_rate'         : hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5)) - 0.0001,
    'gamma'                 : hp.loguniform('gamma', np.log(0.0001), np.log(5)) - 0.0001,
    'min_child_weight'      : scope.int(hp.loguniform('min_child_weight', np.log(1), np.log(100))),
    'subsample'             : hp.uniform('subsample', 0.5, 1),
    'colsample_bytree'      : hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel'     : hp.uniform('colsample_bylevel', 0.5, 1),
    'reg_alpha'             : hp.loguniform('reg_alpha', np.log(0.0001), np.log(1)) - 0.0001,
    'reg_lambda'            : hp.loguniform('reg_lambda', np.log(1), np.log(4))
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
    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=.1, random_state=42)
    
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


# DNN
def obj_func__DNN(params,data, rseed):

    X_train,y_train, X_test,  y_test = prepare_data(data, rseed)
    model = Sequential()

    # first layer
    model.add(Dense(params['units_l1'], input_dim = X_train.shape[1]))
    model.add(Dropout(params['dropout_l1']))
    model.add(Activation('relu'))
    
    # second layer
    model.add(Dense(params['units_l2']))
    model.add(Dropout(params['dropout_l2']))
    model.add(Activation('relu'))

    if params['nbr_layers']['layers']== 'three':
        # third layer
        model.add(Dense(params['nbr_layers']['units_l3_a']))
        model.add(Dropout(params['nbr_layers']['dropout_l3_a']))
        model.add(Activation('relu'))

    if params['nbr_layers']['layers']== 'four':
        # third layer
        model.add(Dense(params['nbr_layers']['units_l3_b']))
        model.add(Dropout(params['nbr_layers']['dropout_l3_b']))
        model.add(Activation('relu'))
        # fourth layer
        model.add(Dense(params['nbr_layers']['units_l4']))
        model.add(Dropout(params['nbr_layers']['dropout_l4']))
        model.add(Activation('relu'))

    # final layers
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # loss  function and optimizers
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=params['optimizer'])

    result = model.fit(X_train, y_train,
              epochs=params['epochs'],
              verbose=0,
              validation_split=0.1)

    #get the highest validation accuracy of the training epochs
    validation_acc = np.mean(result.history['val_accuracy']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


best = 0
# Random Forest
def obj_func__RF(params, data, rseed):

    def acc_model(params, data, rseed):
        X_train,y_train, X_test,  y_test = prepare_data(data, rseed)
        model = RandomForestClassifier(**params)

        result = cross_validate(model, X_train, y_train, return_estimator=True, n_jobs=5, verbose=2)
        
        idx_best_model = np.argmax(result['test_score'])

        score = np.mean(result['test_score'])
        best_model = result['estimator'][idx_best_model]

        return score, best_model

    global best
    acc, model = acc_model(params, data, rseed)

    if acc > best:
        best = acc

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# AdaBoost
def obj_func__AdaBoost(params, data, rseed):

    def acc_model(params, data, rseed):
        X_train,y_train, X_test,  y_test = prepare_data(data, rseed)
        model = AdaBoostClassifier(**params)

        result = cross_validate(model, X_train, y_train, return_estimator=True, n_jobs=5, verbose=2)
        
        idx_best_model = np.argmax(result['test_score'])

        score = np.mean(result['test_score'])
        best_model = result['estimator'][idx_best_model]

        return score, best_model

    global best
    acc, model = acc_model(params, data, rseed)

    if acc > best:
        best = acc

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# XgBoost
def obj_func__XgBoost(params, data, rseed):

    def acc_model(params, data, rseed):
        X_train,y_train, X_test,  y_test = prepare_data(data, rseed)
        model = xgboost.XGBClassifier(**params)

        result = cross_validate(model, X_train, y_train, return_estimator=True, n_jobs=5, verbose=2)
        
        idx_best_model = np.argmax(result['test_score'])

        score = np.mean(result['test_score'])
        best_model = result['estimator'][idx_best_model]

        return score, best_model

    global best
    acc, model = acc_model(params, data, rseed)

    if acc > best:
        best = acc

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    # parser initialization
    parser = argparse.ArgumentParser(description='Script pretraining DNN models')
    parser.add_argument('--dataset', type=str, default='german_credit', help='german_credit,adult_income, compas, default_credit, marketing')
    parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 9')
    parser.add_argument('--model_class', type=str, default='DNN', help='DNN, RF, AdaBoost, XgBoost')
    parser.add_argument('--nbr_evals', type=int, default=25, help='Number of evaluations for hyperopt')

    # get input
    args = parser.parse_args()
    dataset = args.dataset
    rseed = args.rseed
    model_class = args.model_class
    nbr_evals = args.nbr_evals
    
    # create directory to save outputs
    outdir = './pretrained/{}/'.format(dataset)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    print("---------------------------->>> dataset = {}".format(dataset))
    print("---------------------------->>> model = {}".format(model_class))
    
    # filenames of saved objects
    model_name = '{}{}_{}.h5'.format(outdir, model_class, rseed)
    stats_name = '{}{}_{}.txt'.format(outdir, model_class, rseed)

    if model_class == 'DNN':
        # Initialize an empty trials database
        trials = Trials()

        # Perform the evaluations on the search space
        obj_func__DNN = partial(obj_func__DNN, data=dataset, rseed=rseed)
        best = fmin(obj_func__DNN, space_DNN, algo=tpe.suggest, trials=trials, max_evals=nbr_evals)

        # get params of the best model
        best_params = space_eval(space_DNN, best)
        # get the best model
        best_model = trials.best_trial['result']['model']

        # accuracy of the best model
        X_train, y_train, X_test, y_test = prepare_data(dataset, rseed)

        acc_train = best_model.evaluate(X_train, y_train)[1]
        acc_test = best_model.evaluate(X_test, y_test)[1]

        # save the best model as bbox
        best_model.save(model_name)
        
        
        
        mdl = load_model(model_name)
        mdl.eval()
        # get prediction
        #---train
        predictions_train = (mdl.predict(X_train) > 0.5).astype('int32')
        predictions_train = [x[0] for x in predictions_train]
        print(Counter(predictions_train))
        
        
        
        # save best models params and perfs
        with open(stats_name,'w') as myFile:
            myFile.write('Accuracy train: {}\n'.format(acc_train))
            myFile.write('Accuracy test: {}\n'.format(acc_test))
            myFile.write('Model params: {}\n'.format(best_params))
    
    
    if model_class == 'RF':
        # Initialize an empty trials database
        trials = Trials()

        # Perform the evaluations on the search space
        obj_func__RF = partial(obj_func__RF, data=dataset, rseed=rseed)
        best = fmin(obj_func__RF, space_RF, algo=tpe.suggest, trials=trials, max_evals=nbr_evals)

        # get params of the best model
        best_params = space_eval(space_RF, best)
        print(best_params)

        # get the best model
        best_model = trials.best_trial['result']['model']

        # accuracy of the best model
        X_train, y_train, X_test, y_test = prepare_data(dataset, rseed)
        acc_train   = accuracy_score(y_train, best_model.predict(X_train))
        acc_test    = accuracy_score(y_test, best_model.predict(X_test))


        # save the best model as bbox
        pickle.dump(best_model, open(model_name,"wb"))


        # save best models params and perfs
        with open(stats_name,'w') as myFile:
            myFile.write('Accuracy train: {}\n'.format(acc_train))
            myFile.write('Accuracy test: {}\n'.format(acc_test))
            myFile.write('Model params: {}\n'.format(best_params))

    if model_class == 'AdaBoost':
        # Initialize an empty trials database
        trials = Trials()

        # Perform the evaluations on the search space
        obj_func__AdaBoost = partial(obj_func__AdaBoost, data=dataset, rseed=rseed)
        best = fmin(obj_func__AdaBoost, space_AdaBoost, algo=tpe.suggest, trials=trials, max_evals=nbr_evals)

        # get params of the best model
        best_params = space_eval(space_AdaBoost, best)
        print(best_params)

        # get the best model
        best_model = trials.best_trial['result']['model']

        # accuracy of the best model
        X_train, y_train, X_test, y_test = prepare_data(dataset, rseed)
        acc_train   = accuracy_score(y_train, best_model.predict(X_train))
        acc_test    = accuracy_score(y_test, best_model.predict(X_test))

        # save the best model as bbox
        pickle.dump(best_model, open(model_name,"wb"))

        # save best models params and perfs
        with open(stats_name,'w') as myFile:
            myFile.write('Accuracy train: {}\n'.format(acc_train))
            myFile.write('Accuracy test: {}\n'.format(acc_test))
            myFile.write('Model params: {}\n'.format(best_params))
   
    if model_class == 'XgBoost':
        # Initialize an empty trials database
        trials = Trials()

        # Perform the evaluations on the search space
        obj_func__XgBoost = partial(obj_func__XgBoost, data=dataset, rseed=rseed)
        best = fmin(obj_func__XgBoost, space_XgBoost, algo=tpe.suggest, trials=trials, max_evals=nbr_evals)

        # get params of the best model
        best_params = space_eval(space_XgBoost, best)
        print(best_params)

        # get the best model
        best_model = trials.best_trial['result']['model']

        # accuracy of the best model
        X_train, y_train, X_test, y_test = prepare_data(dataset, rseed)
        acc_train   = accuracy_score(y_train, best_model.predict(X_train))
        acc_test    = accuracy_score(y_test, best_model.predict(X_test))

        # save the best model as bbox
        pickle.dump(best_model, open(model_name,"wb"))

        # save best models params and perfs
        with open(stats_name,'w') as myFile:
            myFile.write('Accuracy train: {}\n'.format(acc_train))
            myFile.write('Accuracy test: {}\n'.format(acc_test))
            myFile.write('Model params: {}\n'.format(best_params))

