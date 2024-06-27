"""This file trains the LSTM on the data in the /data folder based on the model from the model.py file"""
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import math

def DataProcessor(train_or_test):
    #Reads data and imputes missing values
    data = pd.read_csv('data/oilprice.csv')
    data = data.drop('DATE', axis=1)
    data = data.replace('.', np.NaN)
    data = data.ffill()
    data = data.astype('float')

    X_train, X_test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False)

    #Splits the sequences into feature sequence and target value
    def SeqSplit(data, context_window):
        X = []
        y = []

        for i in range(len(data)):
            lastindex = i + context_window

            if lastindex > len(data) - 1:
                break

            seq_x, seq_y = data[i:lastindex], data[lastindex]
            X.append(seq_x)
            y.append(seq_y)

        X = torch.tensor(X)
        y = torch.tensor(y)

        return X, y

    X_train, y_train = SeqSplit(X_train['DCOILBRENTEU'].tolist(), 16)
    X_test, y_test = SeqSplit(X_test['DCOILBRENTEU'].tolist(), 16)
    print(X_train.shape, y_train.shape)
    
    #Modifies dimensions to work with the LSTM model
    X_train = torch.unsqueeze(X_train, dim=2)
    X_test = torch.unsqueeze(X_test, dim=2)
    y_train = torch.unsqueeze(y_train, dim=1)
    y_train = torch.unsqueeze(y_train, dim=2)
    y_test = torch.unsqueeze(y_test, dim=1)
    y_test = torch.unsqueeze(y_test, dim=2)

    #Return train or test dataset depending on variable passed
    if train_or_test == 'train':
        return TensorDataset(X_train, y_train)
    if train_or_test == 'test':
        return TensorDataset(X_test, y_test)
    else:
        raise Exception('Neither test nor train')










