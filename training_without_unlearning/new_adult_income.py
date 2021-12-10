import numpy as np
import pandas as pd
from core import save


def get_new_adult_income(save_df=False):

    # output files
    dataset = 'new_adult_income'
    decision = 'income'

    df = pd.read_csv('./raw_datasets/new_adult_income.csv')
    
    df.drop('Unnamed: 0',  axis='columns', inplace=True)
    
    # change the columns name
    column_names = ['age', 'workclass', 'education','marital-status', 'occupation', 'relationship', 'race', 'gender', 'hours-per-week', 'native-country', 'income']
    
    df.columns = column_names
    
    # split gender to male and female
    gender = df['gender']
    df['male'] = gender
    df['female'] = gender
    df = df.replace({'male': {2.0: 0, 1.0: 1}})
    df = df.replace({'female': {2.0: 1, 1.0: 0}})
    df.drop('gender',  axis='columns', inplace=True)
    
    if save_df:
        for rseed in range(1):
            save(df, dataset, decision, rseed)
    #return df