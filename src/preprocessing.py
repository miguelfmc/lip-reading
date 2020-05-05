"""
CS289A INTRODUCTION TO MACHINE LEARNING
MACHINE LEARNING FOR LIP READING
Authors: Alejandro Molina & Miguel Fernandez Montes

Pre-processing tools
"""


import numpy as np


def sliding_window(tensor, window_size):
    """
    Input a D-dimensional numpy ndarray arr
    Returns a D+1-dimensional numpy ndarray

    :param arr: numpy ndarray
    :param window_size: size of sliding window
    :return: a numpy array containing the slices of the input
    """
    output_len = arr.shape[0] - window_size + 1
    indexer = np.arange(window_size)[None, :] + np.arange(output_len)[:, None]
    return arr[indexer]


video = np.random.randn(25, 120, 120)
sliced_video = sliding_window(video, 5)
print(sliced_video)