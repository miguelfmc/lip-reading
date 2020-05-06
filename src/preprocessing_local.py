"""
CS289A INTRODUCTION TO MACHINE LEARNING
MACHINE LEARNING FOR LIP READING
Authors: Alejandro Molina & Miguel Fernandez Montes

Pre-processing tools
"""


import numpy as np
import torch
import os
import cv2

# import matplotlib.pyplot as plt
#from charSet import get_charSet, init_charSet


#### GLOBAL PARAMETERS 
MAX_LENGTH = 29
WINDOW_SIZE = 5
WORD_DICTIONARY = [line.strip() for line in open('../dictionary.txt', 'r')]


def sliding_window(arr, window_size):
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


def generate_batch(batch_size):
    """
    Input  batch_size of the batch.
    Returns an array with the path to the files and the targets of each videos.
    The targets are the index of the word inside of the dictionary
    :param batch_size: int with the size of the batch
    :return: two numpy array containing the batch and the targets.
    """
    
    videos =  np.array(os.listdir('../data/'))
    random_folder = videos[np.random.randint(len(videos), size=batch_size)]
    random_number = np.random.randint(1000,size=batch_size)
    batch = ['../data/' + str(random_folder[i]) + \
             '/train/'+str(random_folder[i])+'_'+str(random_number[i]).zfill(5)+'.mp4' for i in range(batch_size)]
    t = np.array([WORD_DICTIONARY.index(l) for l in random_folder])
    return batch, t


def train_loader(num_iterations, batch_size):
    """
    Input  2 ints with the number of iterations and the batch_size
    Returns a generator with the number of iteratios for each epoc.
    :param batch_size: int with the size of the batch
    :param num_iterations: number of iterations for each epoc
    
    :return: generator with with the iterable.
    """
    for n in range(num_iterations):
        #Initializing parameters
        processed_videos = []
        batch, target = generate_batch(batch_size)
        for i in batch:
            cap = cv2.VideoCapture(i)
            tmp = np.zeros((MAX_LENGTH,120,120))
            count = 0
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret:
                    # Our operations on the frame come here
                    
                    #Here I apply the normalization of the pixels 255
                    tmp[count] = cv2.cvtColor(frame[100:220,70:190], cv2.COLOR_BGR2GRAY) /255

                    count +=1
                     #plt.imshow(gray, cmap='gray')
                    # Display the resulting frame
                else:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            #TODO: I remove the try-catch part. What happened if the video has any problem ?
            idx = [i for i in range(MAX_LENGTH - 1, -1, -1)]
            processed_videos.append(sliding_window(tmp[idx, :, :],WINDOW_SIZE))
            cap.release()
            cv2.destroyAllWindows()
        inputs = torch.from_numpy(np.array(processed_videos)).float()
        targets = torch.from_numpy(np.array(target)).long()
        yield inputs, targets







