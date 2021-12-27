import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from sklearn import decomposition
import matplotlib.pyplot as plt

class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.file_name=file_name
        self.P = window
        self.h = horizon
        self.rawdat= self.read_data()
        self.x, self.normalized, self.delta_all = self.NormalizeMult(self.rawdat)
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m,self.t= self.dat.shape
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        self.device = device



    def read_data(self):
        a = np.load(self.file_name)
        data = a[0:1600, :]
        data =data.reshape(data.shape[0],data.shape[1],1)
        return data




    #多维归一化
    def NormalizeMult(self,data):
        data = np.array(data)
        # normalize
        normalize = torch.zeros(len(data[0, :, 0]), len(data[0, 0, :]), 2)  # 14,4,2
        norm = torch.zeros(len(data[0, 0, :]), 2)  # 4,2
        delta = torch.zeros(len(data[0, 0, :]), 1)  # 4,1
        delta_all = torch.zeros(len(data[0, :, 0]), len(data[0, 0, :]), 1)  # 14,4,1
        for i in range(0, data.shape[1]):
            list = data[:, i]  # x,4
            for j in range(0, data.shape[2]):
                li = list[:, j]
                listlow, listhigh = np.percentile(li, [0, 100])
                norm[j, 0] = listlow
                norm[j, 1] = listhigh
                delta[j] = listhigh - listlow
            normalize[i] = norm
            delta_all[i] = delta
        data = torch.from_numpy(data)
        data = data.float()
        for i in range(0, data.shape[1]):
            for j in range(0, data.shape[2]):
                data[:, i, j] = (data[:, i, j] - normalize[i, j, 0]) / delta_all[i, j]
        return data, normalize, delta_all


    def FNormalizeMult(self,data):
        normalize = self.normalized[:, 0, 0]
        delta_all = self.delta_all[:, 0]
        for i in range(0, data.shape[1]):
            data[:, i]= data[:, i] * delta_all[i] + normalize[i]
        return data

    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = torch.from_numpy(self.rawdat)
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)
            self.dat=torch.from_numpy(self.dat)
        # normlized
        if (normalize == 2):
            self.dat = self.x





    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)
        #print(self.train[1].shape)


    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m,self.t))
        Y = torch.zeros((n, self.m,self.t))
        for i in range(n):
            end = idx_set[i] - self.h + 1#170-3+1=168
            start = end - self.P#168-168=0
            X[i, :, :, :] = self.dat[start:end, :, :]
            Y[i, :,:] = self.dat[idx_set[i], :,:]
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size
