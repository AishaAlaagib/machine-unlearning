import os
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz
from sklearn.preprocessing import LabelEncoder

pwd = os.path.dirname(os.path.realpath(__file__))

# download the dataset
urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', 'german.data')
german_df = pd.read_csv('german.data',   delimiter=' ',header=None)

# download the dataset document
urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc', 'german.doc')
f = open('german.doc')
german_doc= f.read()

german_df.columns=['account_bal','duration','payment_status','purpose',
                   'credit_amount','savings_bond_value','employed_since',
                   'intallment_rate','sex_marital','guarantor','residence_since',
                   'most_valuable_asset','age','concurrent_credits','type_of_housing',
                   'number_of_existcr','job','number_of_dependents','telephon',
                   'foreign','target']
german_df= german_df.replace(['A11','A12','A13','A14', 'A171','A172','A173','A174','A121','A122','A123','A124'],
                  ['neg_bal','positive_bal','positive_bal','no_acc','unskilled','unskilled','skilled','highly_skilled',
                   'none','car','life_insurance','real_estate'])

from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
le.fit(german_df.target)
german_df.target=le.transform(german_df.target)
# german_df.target.head(5)

# using sklearn preprocessing for dealing with categorical variables
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

from sklearn.preprocessing import MinMaxScaler

# scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# fit on features dataset
scaler.fit(x_train)
scaler.fit(x_test)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)

print('np_train',x_train.shape, 'np_test',x_test.shape)
np.save(f'german_credit_train.npy', {'X': x_train, 'y': y_train})
np.save(f'german_credit_test.npy', {'X': x_test, 'y': y_test})







