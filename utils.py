"""This file trains the LSTM on the data in the /data folder based on the model from the model.py file"""
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import math

def DataProcessor(train_or_test):
    data = pd.read_csv('data/oilprice.csv')
    data = data.drop('DATE', axis=1)
    data = data.replace('.', np.NaN)
    data = data.ffill()
    data = data.astype('float')

    # print(data.head())
    X_train, X_test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False)
    # print(X_train.head())
    # trainlen = X_train.shape[0]
    # testlen = X_test.shape[0]

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

       
        # print(X)
        # print(y)
        X = torch.tensor(X)
        y = torch.tensor(y)

        return X, y

    X_train, y_train = SeqSplit(X_train['DCOILBRENTEU'].tolist(), 16)
    X_test, y_test = SeqSplit(X_test['DCOILBRENTEU'].tolist(), 16)
    print(X_train.shape, y_train.shape)

    X_train = torch.unsqueeze(X_train, dim=2)
    X_test = torch.unsqueeze(X_test, dim=2)
    y_train = torch.unsqueeze(y_train, dim=1)
    y_train = torch.unsqueeze(y_train, dim=2)
    y_test = torch.unsqueeze(y_test, dim=1)
    y_test = torch.unsqueeze(y_test, dim=2)

    print(X_train.shape, y_train.shape)
    # batch_size = 75

    # def batcher(seq, batch_size=batch_size):
    #     num_batches = math.ceil(seq.shape[0]/batch_size)
    #     seq_batched = [seq[batch_size*y:batch_size*(y+1),:,:] for y in range(num_batches)]
    #
    #     return seq_batched

    # X_train = batcher(X_train)
    # y_train = batcher(y_train)
    #
    # X_test = batcher(X_test)
    # y_test = batcher(y_test)

    if train_or_test == 'train':
        return TensorDataset(X_train, y_train)
        # return trainlen, [(X_train[i], y_train[i]) for i in range(len(X_train))]
    if train_or_test == 'test':
        return TensorDataset(X_test, y_test)
        # return testlen, [(X_test[i], y_test[i]) for i in range(len(X_test))]
    else:
        raise Exception('Neither test nor train')










