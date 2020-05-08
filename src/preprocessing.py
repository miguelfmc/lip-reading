"""
CS289A INTRODUCTION TO MACHINE LEARNING
MACHINE LEARNING FOR LIP READING
Authors: Alejandro Molina & Miguel Fernandez Montes

Pre-processing tools
"""


import numpy as np
import torch
import os
import sys
import cv2

import matplotlib.pyplot as plt
#from charSet import get_charSet, init_charSet


#### GLOBAL PARAMETERS 
MAX_LENGTH = 29
WINDOW_SIZE = 5
WORD_DICTIONARY =  [line.strip() for line in open('/home/alemosan/lipReading/dictionary.txt', 'r')]
#Copy of the dictionary to keep track about which words are being used so far.
WORD_DICTIONARY_2 = WORD_DICTIONARY.copy()


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


def dict_loader(d,size):
    #Generation of n random numbers to load random words.
    samples = np.random.choice(len(WORD_DICTIONARY_2), size= size - len(d),replace=False)
    #I have the number inside of Word_Dictionary
    for i in samples:
        word = WORD_DICTIONARY_2[i]
        #Get the index from the Dictionary
        d[word] = WORD_DICTIONARY.index(word)
        #Update removing the word that you put into the dictionary
    
    ##Removing the element that they have already been included into the dictionary.
    for idx in sorted(samples, reverse=True):
        WORD_DICTIONARY_2.remove(WORD_DICTIONARY_2[idx])
    return d

def load_global_dict():
    global WORD_DICTIONARY_2 
    WORD_DICTIONARY_2  = WORD_DICTIONARY.copy()


def generate_batch(batch_size, n, num_iteration,epoch,dictionary):
    """
    Input  batch_size of the batch.
    Returns an array with the path to the files and the targets of each videos.
    The targets are the index of the word inside of the dictionary
    :param batch_size: int with the size of the batch
    :return: two numpy array containing the batch and the targets.
    """
    
    level = ( (n+1) / num_iteration) * 100
    
    if  epoch  == 0:
        if level > 80:
        #Total dictionary -- Words = 500
            if not(len(dictionary) == 500):
                dictionary = dict_loader(dictionary,500)
        elif level > 70:
        #Words = 400
            if not(len(dictionary) == 400):
                dictionary = dict_loader(dictionary,400)
        elif level > 60:
        #Words = 350
            if not(len(dictionary) == 350):
                dictionary = dict_loader(dictionary,350)
        elif level > 50:
        #Words = 300
            if not(len(dictionary) == 300):
                dictionary = dict_loader(dictionary,300)
        elif level > 40:
        #Words = 250    
            if not(len(dictionary) == 250):
                dictionary = dict_loader(dictionary,250) 
        elif level > 30:
        #Words = 200
            if not(len(dictionary) == 200):
                dictionary = dict_loader(dictionary,200) 
        elif level > 20:
        #Words = 150
            if not(len(dictionary) == 150):
                dictionary = dict_loader(dictionary,150)
        elif level > 15:
        #Words = 100
            if not(len(dictionary) == 100):
                dictionary = dict_loader(dictionary,100)
        elif level > 10:
        #Words = 80
            if not(len(dictionary) == 80):
                dictionary = dict_loader(dictionary,80)
        elif level > 7:
        #Words = 50
            if not(len(dictionary) == 50):
                dictionary = dict_loader(dictionary,50)
        elif level > 5:
        #Words = 40    
            if not(len(dictionary) == 40):
                dictionary = dict_loader(dictionary,40)
        elif level > 3:
        #Words = 30
            if not(len(dictionary) == 30):
                dictionary = dict_loader(dictionary,30)
        elif level > 2:
        #Words = 20
            if not(len(dictionary) == 20):
                dictionary = dict_loader(dictionary,20)         
        elif level >= 0:
        #Words = 10
            if not (len(dictionary) == 10):
                dictionary = dict_loader(dictionary,10)    

    #else when the epoch is different from the frist epoch in which we applied the curriculum learning
    else:
        load_global_dict()
        if not(len(dictionary) == 500):
            dictionary = dict_loader(dictionary,500)
    
    #folders that I can take a video
    videos = np.array(list(dictionary.keys()))
    
    random_folder = videos[np.random.choice(len(videos), size=batch_size, replace = True)]
    #TODO: change the costant of this method, it depends on the number of samples that you have in the carpet
    random_number = np.random.randint(1000,size=batch_size)
    batch = ['/home/alemosan/lipReading/data/' + str(random_folder[i]) + \
             '/train/'+str(random_folder[i])+'_'+str(random_number[i]).zfill(5)+'.mp4' for i in range(batch_size)]
    t = np.array([WORD_DICTIONARY.index(l) for l in random_folder])
    return  batch,t

def generate_batch_val(batch_size):
    """
    Input  batch_size of the batch.
    Returns an array with the path to the files and the targets of each videos.
    The targets are the index of the word inside of the dictionary
    :param batch_size: int with the size of the batch
    :return: two numpy array containing the batch and the targets.
    """
    
    videos =  np.array(os.listdir('/home/alemosan/lipReading/data/'))
    random_folder = videos[np.random.randint(len(videos), size=batch_size)]
    random_number = np.random.randint(50,size=batch_size)
    batch = ['/home/alemosan/lipReading/data/' + str(random_folder[i]) + \
             '/val/'+str(random_folder[i])+'_'+str(random_number[i]).zfill(5)+'.mp4' for i in range(batch_size)]
    t = np.array([WORD_DICTIONARY.index(l) for l in random_folder])
    return  batch,t
    
    
def train_loader(num_iterations,batch_size,epoc):
    """
    Input  2 ints with the number of iterations and the batch_size
    Returns a generator with the number of iteratios for each epoc.
    :param batch_size: int with the size of the batch
    :param num_iterations: number of iterations for each epoc
    
    :return: generator with with the iterable.
    """
    #Dictionary to storage the words that we use in the model
    dictionary = {}
    for n in range(num_iterations):
        #Initializing parameters
        processed_videos = []
        batch, target = generate_batch(batch_size, n, num_iterations, epoc,dictionary)
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
        inputs  = torch.from_numpy(np.array(processed_videos)).float()
        targets = torch.from_numpy(np.array(target)).long()
        yield (inputs, targets)

        
def val_loader(num_iterations,batch_size):
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
        batch, target = generate_batch_val(batch_size)
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
        inputs  = torch.from_numpy(np.array(processed_videos)).float()
        targets = torch.from_numpy(np.array(target)).long()
        yield (inputs, targets)







