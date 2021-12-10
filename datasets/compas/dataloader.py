import numpy as np
import os

pwd = os.path.dirname(os.path.realpath(__file__))

train_data = np.load(os.path.join(pwd, 'compas_train.npy'), allow_pickle=True)
test_data = np.load(os.path.join(pwd, 'compas_test.npy'), allow_pickle=True)

train_data = train_data.reshape((1,))[0]
test_data = test_data.reshape((1,))[0]

X_train = train_data['X'].astype(np.float32)
X_test  = test_data['X'].astype(np.float32)
y_train = train_data['y'].astype(np.int64)
y_test  = test_data['y'].astype(np.int64)
maj_train = train_data['maj_train'].astype(np.float32)
min_train = train_data['min_train'].astype(np.float32)

maj_test = test_data['maj_test'].astype(np.float32)
min_test = test_data['min_test'].astype(np.float32)

def load(indices, category='train'):
#     indices = indices[:15]
#     print(indices)
    if category == 'train':
        return X_train[indices], y_train[indices]#, maj_train[indices], min_train[indices]
    
    elif category == 'test':
        return X_test[indices], y_test[indices] #, maj_test[indices], min_test[indices]
    elif category == 'unf':
        return X_test[indices], y_test[indices] , maj_test[indices], min_test[indices]
