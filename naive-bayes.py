import os

import numpy as np
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def findnan(x):
    where_are_nan = np.isnan(x)
    x[where_are_nan] = 0
    return x


def work(trainee, algorithm):
    path = input('Enter path: ')
    path = str(path.replace('\\', '/'))

    inputs_path = path + '/input'
    labels_path = path + '/labels'

    train_data = os.listdir(inputs_path)
    train_set, test_set = train_test_split(
        train_data,
        random_state=1,
        train_size=0.8
    )

    for index, name in enumerate(train_set):
        x = genfromtxt('%s/%s' % (inputs_path, name), delimiter=',')
        y = genfromtxt('%s/%s' % (labels_path, name), delimiter=',')

        x = findnan(x[:, 1:])
        y = findnan(y[:, 0])

        if index == 0:
            X = x
            Y = y
        else:
            X = np.vstack((X, x))
            Y = np.hstack((Y, y))

    classifier = trainee(X, Y.ravel())

    for _, name in enumerate(test_set):
        xx = genfromtxt('%s/%s' % (inputs_path, name), delimiter=',')
        xx = findnan(xx[:, 1:])
        prediction = classifier.predict(xx)

        yy = genfromtxt('%s/%s' % (labels_path, name), delimiter=',')
        yy = findnan(yy[:, 0])

        print(name,
              '---Accuracy : %s: %s ' % (accuracy_score(yy, prediction), algorithm))


if __name__ == '__main__':
    work(lambda train, target: GaussianNB().fit(train, target), 'GaussianNB')
    # work(lambda train, target: MultinomialNB().fit(train, target))
