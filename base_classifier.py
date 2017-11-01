import os

import numpy as np
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB


class BaseClassifier:
    """
    Base class for simple classifiers. 
    If you'd need more complicated behavior - you should inherit this
    and extend behavior accordingly
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        """
        Fit classifier according to X, y
        """
        return self.model.fit(X, y)

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        """
        return self.model.predict(X)


class SimpleTestPredictor:
    """
    Naive worker which loads data from files, instantiates model, trains it and generates prediction results
    """

    def predict(self, model, train_size=0.8):
        """
        Generates prediction results out of training set

        Parameters
        ----------
        model : Model for prediction

        train_size : float, int, or None, default 0.8
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
        """

        path = input('Enter path: ')
        path = str(path.replace('\\', '/'))

        inputs_path = path + '/input'
        labels_path = path + '/labels'

        train_data = os.listdir(inputs_path)
        train_set, test_set = train_test_split(
            train_data,
            random_state=1,
            train_size=train_size
        )

        for index, name in enumerate(train_set):
            x = genfromtxt('%s/%s' % (inputs_path, name), delimiter=',')
            y = genfromtxt('%s/%s' % (labels_path, name), delimiter=',')

            x = self.clean(x[:, 1:])
            y = self.clean(y[:, 0])

            if index == 0:
                X = x
                Y = y
            else:
                X = np.vstack((X, x))
                Y = np.hstack((Y, y))

        model.fit(X, Y.ravel())

        for _, name in enumerate(test_set):
            xx = genfromtxt('%s/%s' % (inputs_path, name), delimiter=',')
            xx = self.clean(xx[:, 1:])
            prediction = model.predict(xx)

            yy = genfromtxt('%s/%s' % (labels_path, name), delimiter=',')
            yy = self.clean(yy[:, 0])

            yield (name, prediction, yy)

    def clean(self, x):
        """
        Cleans given array. By default replaces NaNs as 0
        """
        where_are_nan = np.isnan(x)
        x[where_are_nan] = 0
        return x
