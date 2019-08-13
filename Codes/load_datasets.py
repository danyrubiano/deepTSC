import numpy as np
import os
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


"""
Load time series dataset

Parameters:
    - direc -> Directory
    - ratio -> Division rate in train and test data
    - dataset -> Dataset name
Ouputs:
    - X_train, X_val, X_test -> Time series data divided in sets of train, test and validation
    - y_train, y_val, y_test -> Time series classes divided in sets of train, test and validation
"""

def load_data1(direc,ratio,dataset):

    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir+'_TRAIN.txt',delimiter=',')
    data_test_val = np.loadtxt(datadir+'_TEST.txt',delimiter=',')
    nb_classes = len(np.unique(data_test_val))
    DATA = np.concatenate((data_train,data_test_val),axis=0)
    N = DATA.shape[0]

    ratio = (ratio*N).astype(np.int32)
    ind = np.random.permutation(N)
    X_train = DATA[ind[:ratio[0]],1:]
    X_val = DATA[ind[ratio[0]:ratio[1]],1:]
    X_test = DATA[ind[ratio[1]:],1:]
    

    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # Targets have labels 1-indexed. We subtract one for 0-indexed
    y_train = DATA[ind[:ratio[0]],0]-1
    y_val = DATA[ind[ratio[0]:ratio[1]],0]-1
    y_test = DATA[ind[ratio[1]:],0]-1

    # no es necesario cuando trabjas con una funcion de perdida cross categorial loss
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


"""
Load time series dataset 2

Parameters:
    - direc -> Directory
    - dataset -> Dataset name
    - dimensions -> Required dimension of X, depending on whether Conv1D (3) or Conv2D (4) applies
Ouputs:
    - X -> Time series data
    - y -> Time series classes
"""

def load_data(direc, dataset, dimensions):
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir+'_TRAIN.txt',delimiter=',')
    data_test_val = np.loadtxt(datadir+'_TEST.txt',delimiter=',')
    nb_classes = len(np.unique(data_test_val))
    DATA = np.concatenate((data_train,data_test_val),axis=0)

    print(print(Counter(DATA[:,0])))

    sns.countplot(DATA[:,0],label="Count")
    plt.show()

    X = DATA[:,1:]
    y = DATA[:,0]-1

    if dimensions == 3:
        X = X.reshape(X.shape + (1,))

    if dimensions == 4:
        X = X.reshape(X.shape + (1,1,))

    nb_classes = len(np.unique(y))

    y = to_categorical(y, nb_classes)

    return X, y


"""
Standardize data

Parameters:
    - X_train -> Time series data train
    - X_test -> Time series data test
Ouputs:
    - flatX_train -> Time series standardized data train
    - flatX_test -> Time series standardized data test
"""

def standardize_data(X_train, X_test):
    # remove overlap
    cut = int(X_train.shape[1] / 2)
    longX = X_train[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatX_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
    flatX_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
    
    # standardize
    s = StandardScaler()
    # fit on training data
    s.fit(longX)
    # apply to training and test data
    longX = s.transform(longX)
    flatX_train = s.transform(flatX_train)
    flatX_test = s.transform(flatX_test)
    
    # reshape
    flatX_train = flatX_train.reshape((X_train.shape))
    flatX_test = flatX_test.reshape((X_test.shape))
    
    return flatX_train, flatX_test


"""
Rescale data

Parameters:
    - X_train -> Time series data train
    - X_test -> Time series data test
Ouputs:
    - flatX_train -> Time series rescaled data train
    - flatX_test -> Time series rescaled data test
"""

def rescale_data(X_train, X_test):
    # remove overlap
    cut = int(X_train.shape[1] / 2)
    longX = X_train[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatX_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
    flatX_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
    
    # rescale
    s = MinMaxScaler(feature_range=(0, 1))
    # fit on training data
    s.fit(longX)
    # apply to training and test data
    longX = s.fit_transform(longX)
    flatX_train = s.fit_transform(flatX_train)
    flatX_test = s.fit_transform(flatX_test)
    
    # reshape
    flatX_train = flatX_train.reshape((X_train.shape))
    flatX_test = flatX_test.reshape((X_test.shape))
    
    return flatX_train, flatX_test
    

"""
Nomalize data

Parameters:
    - X_train -> Time series data train
    - X_test -> Time series data test
Ouputs:
    - flatX_train -> Time series nomalized data train
    - flatX_test -> Time series nomalized data test
"""

def normalize_data(X_train, X_test):

    # remove overlap
    cut = int(X_train.shape[1] / 2)
    longX = X_train[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatX_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
    flatX_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
    
    # normalize and fit on training data
    s = Normalizer()

    # fit on training data
    s.fit(longX)

    # apply to training and test data
    longX = s.transform(longX)
    flatX_train = s.transform(flatX_train)
    flatX_test = s.transform(flatX_test)
    
    # reshape
    flatX_train = flatX_train.reshape((X_train.shape))
    flatX_test = flatX_test.reshape((X_test.shape))
    
    return flatX_train, flatX_test


"""
Preprocess data based in mean and std

Parameters:
    - X_train -> Time series data train
    - X_test -> Time series data test
Ouputs:
    - X_train -> Time series preprocessed data train
    - X_test -> Time series preprocessed data test
"""
def preprocesing_data(X_train, X_test):

    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean)/(X_train_std)
    X_test = (X_test - X_train_mean)/(X_train_std)
    
    return X_train, X_test