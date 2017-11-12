import numpy as np
from numpy import genfromtxt
def most(y_hat):
    y_list = {}
    second = 0
    f = 0
    ff = 0
    rate = 0
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
                rate = second/max_label[0]
    return f, ff
def most_from_dict(d):
    y_hat = []
    max = 0
    for k, v in d.items():
        for i, j in enumerate(d[k]):
            if j > max:
                max = j
                label = i
        y_hat.append(label)
    f, ff = most(y_hat)
    return f, ff, y_hat

def score(xx, m, a_s, j):

    s = np.where([a_s > num for num in xx], 0, 1)
    rate = 2 * (0.5 - abs(np.sum(s, axis=1)/m - 0.5))
    if j < 7:
        a = 25/6
    elif j < 34:
       a = 25/27
    elif j < 70:
       a = 20/36
    else:
       a = 25/100

    return rate * a
 #    s = np.where(a_s > xx, 0, 1)
 #    rate = 2 * (0.5 - abs(np.sum(s)/m - 0.5))
 #    if j < 7:
 #        a = 25/6
 #    elif j < 34:
 #        a = 25/27
 #    elif j < 70:
 #        a = 25/36
 #    else:
 #        a = 25/100
 #    return rate * a

def delete_ele(x):
    x[np.isnan(x)] = 0
    index = np.argwhere(x == 0)
    return np.delete(x, index)

def findnan(x):
    where_are_nan = np.isnan(x)
    x[where_are_nan] = 0
    return x

def loaddata(path, train_set, p, y0_train=False):
    m = len(train_set)
    for j in range(0, m):
        train_name = train_set[j]
        x = genfromtxt('%s/%s' % (path, train_name), delimiter=',')
        yp = genfromtxt('%s/%s' % (path.replace('input-train', 'labels-train'), train_name), delimiter=',')
        x = findnan(x[:, 1:])
        y = findnan(yp[:, 0])
        y0 = findnan(yp[:, -1])
        for i in range(0, len(y)):
            y[i] = y[i] - 1
        if j == 0:
            X = x
            Y = y
            Y0 = y0
        else:
            X = np.vstack((X, x))
            Y = np.hstack((Y, y))
            Y0 = np.hstack((Y0, y0))
    for m, n in enumerate(Y0):
        if n < 0.1:
            Y0[m] = 0
        elif n < 0.2:
            Y0[m] = 1
        elif n < 0.3:
            Y0[m] = 2
        elif n < 0.4:
            Y0[m] = 3
        elif n < 0.5:
            Y0[m] = 4
        elif n < 0.6:
            Y0[m] = 5
        elif n < 0.7:
            Y0[m] = 6
        elif n < 0.8:
            Y0[m] = 7
        elif n < 0.9:
            Y0[m] = 8
        else:
            Y0[m] = 9

    if y0_train:
        return X, Y, Y0
    else:
        xx, yy, yy0 = [], [], []
        for m, n in enumerate(Y0):
            if n == p:
                yy0.append(Y0[m])
                xx.append(X[m])
                yy.append(Y[m])
            elif p > 9:
                if n > p - 5:
                    yy0.append(Y0[m])
                    xx.append(X[m])
                    yy.append(Y[m])
        return np.array(xx), np.array(yy), np.array(yy0)