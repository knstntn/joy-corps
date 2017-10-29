"""
This module contains different useful utilitiy methods.
"""

from math import sqrt

import numpy as np


def mean(array, window=None):
    """
    Returns mean value for the given array (or its subarray if window is provided)
    """
    this = __import__(__name__)
    return np.mean(this.window(array, window))


def qmean(array, window=None):
    """
    Returns root mean square - RMS / rms / quadratic mean - for the given array (or its subarray if window is provided)
    """
    this = __import__(__name__)
    array = this.window(array, window)
    return sqrt(sum(n * n for n in array) / len(array))


def median(array, window=None):
    """
    Returns median value for the given array (or its subarray if window is provided)
    """
    this = __import__(__name__)
    return np.median(this.window(array, window))


def variance(array, window=None):
    """
    Returns variance for the given array (or its subarray if window is provided)
    """
    this = __import__(__name__)
    return np.var(this.window(array, window))


def sum(array, window=None):
    """
    Returns sum of array elements (or its subarray if window is provided)
    """
    this = __import__(__name__)
    return np.sum(this.window(array, window))


def max(array, window=None):
    """
    Return the largest item in an array (or its subarray if window is provided)
    """
    this = __import__(__name__)
    return max(this.window(array, window))


def min(array, window=None):
    """
    Return the smallest item in an array (or its subarray if window is provided)
    """
    this = __import__(__name__)
    return min(this.window(array, window))


def window(array, window=None):
    """
    Finds `window` in array if respective description is given

    Parameters
    ----------
    array : This could by python array or numpy array.
    window : tuple(int, int), optional - subset of the array which should be used to transform an array
        First element of tuple is index around which we should look for elements, second is an offset around indexed element

    Returns
    -------
    array

    Examples
    --------
    >>> data = range(1,5)
    >>> data
    [1, 2, 3, 4]
    >>> utils.window(data)
    [1, 2, 3, 4]
    >>> utils.window(data, (2, 1))
    [2, 3, 4]
    """
    if window is not None:
        min = window[0] - window[1]
        max = window[0] + window[1] + 1

        return array[min: max]

    return array
