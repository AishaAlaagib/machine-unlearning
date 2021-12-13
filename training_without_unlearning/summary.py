from __future__ import print_function

from functools import partial
import torch
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
from torch.nn import Module, Linear
from torch.nn.functional import tanh
from csv import writer
import csv

data_dict = {
    'adult_income'      : ('adult_income', 'income'),
    'compas'            : ('compas', 'two_year_recid'),
    'default_credit'    : ('default_credit', 'DEFAULT_PAYEMENT'),
    'marketing'         : ('marketing', 'subscribed')    ,
    'new_adult_income' : ('new_adult_income', 'income')
}


data_map = {
    'adult_income'      : 'Adult Income',
    'compas'            : 'COMPAS',
    'default_credit'    : 'Default Credit',
    'marketing'         : 'Marketing' ,
    'new_adult_income'      : 'New Adult Income'
}



subgroup_dict = {
    'adult_income'      : ('gender_Female', 'gender_Male'),
    'compas'            : ('race_African-American', 'race_Caucasian'),
    'default_credit'    : ('SEX_Female', 'SEX_Male'),
    'marketing'         : ('age_age:30-60', 'age_age:not30-60'),
    'new_adult_income'      : ('female', 'male'),
}

class Model(Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super(Model, self).__init__()
        self.fc1 = Linear(input_shape[0], 128)
        self.fc2 = Linear(128, nb_classes)

    def forward(self, x):
#         print(x.shape)
        x = self.fc1(x)
        x = tanh(x)
        x = self.fc2(x)

        return x
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


def get_metrics(dataset, model_class, rseed, requests):

    # load data as np array
    X_train, y_train, X_test, y_test = prepare_data(dataset, rseed)
    
    # load data as dataframe
    df_train, df_test = prepare_data_as_dataframe(dataset, rseed)

    # load meta data for fairness metrics
    _, decision = data_dict[dataset]
    min_feature, maj_feature = subgroup_dict[dataset]

    # model path
    
    outdir = './pretrained/{}/'.format(dataset)
    model_path = '{}{}_{}.h5'.format(outdir, model_class, requests, rseed)
   

    def get_predictions(model_class, X_train, y_train, X_test, y_test):
        predictions_train, predictions_test = None, None
        acc_train, acc_test =  None, None

        prediction_metrics = {}
#         if model_class == 'NN':
#             # load model
#             prediction_test = np.load(os.path.join(pwd, 'outputs-adult_income.npy',allow_pickle=True)   

          
        if model_class == 'DNN':
            # load model
            mdl = load_model(model_path)

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
        if model_class == 'NN':
            device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member
            X_train = torch.from_numpy(X_train).to(device)
            X_test = torch.from_numpy(X_test).to(device)
            # load model
#             mdl = Model(X_train.shape, 2)
            model = pickle.load(open(model_path,"rb"))

            # get prediction
        
            #---train
            output = model(X_train)
            # Get the label corresponding to the highest predicted probability.
            predictions_train = output.argmax(dim=1, keepdim=True)
            predictions_train = predictions_train.cpu().numpy()
            #predictions_train = mdl.predict(X_train)
            #predictions_train = [int(x) for x in predictions_train]

            #---test
            output = model(X_test)
            # Get the label corresponding to the highest predicted probability.
            predictions_test = output.argmax(dim=1, keepdim=True)
            predictions_test = predictions_test.cpu().numpy()
            # get accuracy
            acc_train   = accuracy_score(y_train, predictions_train)
            acc_test    = accuracy_score(y_test, predictions_test)

        if model_class in ['RF', 'SVM', 'AdaBoost', 'XgBoost']:
            # load model
#             mdl = Model(X_train.shape, 2)
            mdl = pickle.load(open(model_path,"rb"))

            # get prediction
            #---train
            predictions_train = mdl.predict(X_train)
            predictions_train = [int(x) for x in predictions_train]

            #---test
            predictions_test = mdl.predict(X_test)
            predictions_test = [int(x) for x in predictions_test]


            # get accuracy
            acc_train   = accuracy_score(y_train, mdl.predict(X_train))
            acc_test    = accuracy_score(y_test, mdl.predict(X_test))

        #----train
        prediction_metrics['predictions_train'] = predictions_train
        prediction_metrics['acc_Train'] = acc_train

        #----test
        prediction_metrics['predictions_test'] = predictions_test
        prediction_metrics['acc_Test'] = acc_test

      

        return prediction_metrics

    
    def get_fairness_metrics(df_train, df_test, prediction_metrics):
        # output object
        fairness_metrics = {}

        #----train
        df_train['predictions'] = prediction_metrics['predictions_train']
        cm_train = ConfusionMatrix(df_train[min_feature], df_train[maj_feature], df_train['predictions'], df_train[decision])
        cm_minority_train, cm_majority_train = cm_train.get_matrix()
        fm_train = Metric(cm_minority_train, cm_majority_train)


        #----test
        df_test['predictions'] = prediction_metrics['predictions_test']
        cm_test = ConfusionMatrix(df_test[min_feature], df_test[maj_feature], df_test['predictions'], df_test[decision])
        cm_minority_test, cm_majority_test = cm_test.get_matrix()
        fm_test = Metric(cm_minority_test, cm_majority_test)
        

        fairness_metrics['Train']       = fm_train
        fairness_metrics['Test']        = fm_test

        return fairness_metrics

    
    def get_output(dataset, model_class, output_type, prediction_metrics, fairness_metrics, requests):
        res = {}

        # dataset
        res['Dataset']  = data_map[dataset]

        # model class
        res['Model']    = model_class

        if requests:
            res['requests'] = requests
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

    output_train    = get_output(dataset, model_class, 'Train', prediction_metrics, fairness_metrics, requests)
    output_test     = get_output(dataset, model_class, 'Test', prediction_metrics, fairness_metrics, requests)

    return output_train, output_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script pretraining DNN models')
    parser.add_argument('--requests',    type=int, default=None, help="Generate the given number of unlearning requests according to the given distribution and apply them directly to the splitfile",
)
    # get input
    args = parser.parse_args()
    requests = args.requests
    
    # inputs
    datasets = ['adult_income']#, 'compas', 'default_credit', 'marketing']
    model_classes = ['DNN']#, 'DNN']#, 'RF', 'XgBoost']
    requests=(327, 1636, 3272, 4909, 6545, 8181 ,9817, 11453 ,13089, 14726 ,16362, 17998, 19634, 21270 ,22906,24543, 26179 ,27815, 29451, 31087)
#     requests=(42, 206 ,412, 618, 824, 1030, 1236 ,1442, 1648, 1854, 2060 ,2266 ,2678, 2884,2513, 3090, 3296 ,3502, 3708, 3914) # compas

    
    save_dir = ('./results/summary')
    if not os.path.exists(save_dir):
        os.mkdir(outdir)
    filename = '{}/summary.csv'.format(save_dir)
    # adding header
    headerList = ['Dataset', 'Model', 'requests', 'Partition', 'Accuracy', 'SP','PE','EOpp', 'EOdds']
    
    with open(filename, 'w') as file:
        dw = csv.DictWriter(file, delimiter=',',fieldnames=headerList)
        dw.writeheader()
    
    for request in requests:
        df_list = []
        for rseed in range(1):
            row_list = []
            for dataset in datasets:
                for model_class in model_classes:
                    output_train, output_test = get_metrics(dataset, model_class, rseed, request)
#                     row_list.append(output_train)
                    row_list.append(output_test)
            df = pd.DataFrame(row_list)
#             print(df)
            df_list.append(df)



        average_row_list = []
        for index in range(len(df_list[0])):
            average_row = {
                'Dataset'      : df_list[0].iloc[index]['Dataset'],
                'Model'         : df_list[0].iloc[index]['Model'],
                'requests'      : df_list[0].iloc[index]['requests'],
                'Partition'      : df_list[0].iloc[index]['Partition'],
                'Accuracy'     : np.round(np.mean([df_list[j].iloc[index]['Accuracy'] for j in range(1)]), 2),
                'SP'     : np.round(np.mean([df_list[j].iloc[index]['SP'] for j in range(1)]), 2),
                'PE'     : np.round(np.mean([df_list[j].iloc[index]['PE'] for j in range(1)]), 2),
                'EOpp'     : np.round(np.mean([df_list[j].iloc[index]['EOpp'] for j in range(1)]), 2),
                'EOdds'     : np.round(np.mean([df_list[j].iloc[index]['EOdds'] for j in range(1)]), 2)
            }
            average_row_list.append(average_row)
        df_average = pd.DataFrame(average_row_list) 



    #     os.makedirs(save_dir, exist_ok=True)







        df_average.to_csv(filename, encoding='utf-8', mode='a', index=False, header=False)

#         Open our existing CSV file in append mode
#         Create a file object for this file
#         print(df_average[1:])
#         with open(filename, 'a') as f_object:

#             # Pass this file object to csv.writer()
#             # and get a writer object
#             writer_object = writer(f_object)

#             # Pass the list as an argument into
#             # the writerow()
#             writer_object.writerow(df_average[1:])

#             #Close the file object
#             f_object.close()


