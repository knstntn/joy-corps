# /usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
from sklearn.externals import joblib
import os
import time
def findnan(x):
    where_are_nan = np.isnan(x)
    x[where_are_nan] = 0
    return x

def most(y_hat):
    y_list = {}
    second = 0
    f = 0
    ff = 0
    for i in range(0, len(y_hat)):
        y_list[y_hat[i]] = y_list.get(y_hat[i], 0) + 1
    max_label = sorted(y_list.values(), reverse=True)
    for k, v in y_list.items():
        if v == max_label[0]:
            f = k
        else:
            if v > second:
                second = v
                ff = k
    return f, ff
if __name__ == '__main__':
    a, b, d = 0, 0, 0
    path = input('Input your data path, should be like : ../data/input or ...\data\input \nPlease: ')
    #path = 'C:/Users/sheng/Desktop/data/input'
    path = str(path.replace('\\', '/'))
    if '/data/input' not in path:
        print('The path is wrong, please cheack it again.')
        time.sleep(100)
        exit()
    else:
        train_data = os.listdir(path)
        train_set, test_set = train_test_split(train_data, random_state=1, train_size=0.8) # split dataset
        train_set = train_data
        path1 = 'C:/Users/sheng/Downloads/Compressed/data_3/data/input'
        test_set = os.listdir(path1)
        if os.path.isfile('lr1.model'):
            lr = joblib.load('lr1.model') # load model
        else:
            for i, train_name in enumerate(train_set):
                if i == 0:
                    X = genfromtxt('%s/%s' % (path, train_name), delimiter=',')
                    YP = genfromtxt('%s/%s' % (path.replace('input', 'labels'), train_name), delimiter=',')
                    X = findnan(X[:, 1:])
                    Y = findnan(YP[:, 0])
                else:
                    x = genfromtxt('%s/%s' % (path, train_name), delimiter=',')
                    yp = genfromtxt('%s/%s' % (path.replace('input', 'labels'), train_name), delimiter=',')
                    x = findnan(x[:, 1:])
                    y = findnan(yp[:, 0])
                    X = np.vstack((X, x))
                    Y = np.hstack((Y, y))
            lr = LogisticRegression(penalty='l2', C=10)  # train model params here
            lr.fit(X, Y.ravel()) # fit model
            joblib.dump(lr, 'lr1.model') # save model

        for j, test_name in enumerate(test_set): # read test data
            xx = genfromtxt('%s/%s' % (path1, test_name), delimiter=',')
            yy = genfromtxt('%s/%s' % (path1.replace('input', 'labels'), test_name), delimiter=',')
            XX = findnan(xx[:, 1:])
            YY = findnan(yy[:, 0])
            y_hat = lr.predict(XX)

            yyyy = [] # labels
            for i in range(0, len(y_hat)):
                f1, f2 = most(y_hat)
                yyyy.append(f1)
            if y_hat[i] == YY[i]:
                    a += 1
            if yyyy[i] == YY[i]:
                    d += 1
            b += len(YY)
            #if accuracy_score(yyyy, YY) < 0.3:
            print(j, ' ', test_name, '---Logistic accuracy : basic: %s, improved: %s ' %(accuracy_score(YY, y_hat), accuracy_score(yyyy, YY)))
        print('Total: basic: %s, improved: %s'%(a/b, d/b))
