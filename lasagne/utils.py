# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:17:18 2015

@author: richi-ubuntu
"""
import numpy as np
import lasagne
import pickle as pkl
import logging


def nextpow2(i):
    '''
    Computes the next higher exponential of 2. 
    This is often needed for FFT. E.g. 400 samples in time domain 
    --> use 512 frequency bins
    :parameters:
        - i : integer
    :returns: integer, exponential of 2
    '''
    n = 1
    while n < i: n *= 2
    return n
 

    
    
def one_hot(labels, n_classes):
    '''
    Converts an array of label integers to a one-hot matrix encoding
    :parameters:
        - labels : np.ndarray, dtype=int
            Array of integer labels, in {0, n_classes - 1}
        - n_classes : int
            Total number of classes
    :returns:
        - one_hot : np.ndarray, dtype=bool, shape=(labels.shape[0], n_classes)
            One-hot matrix of the input
    '''

    one_hot = np.zeros((labels.shape[0], n_classes)).astype(bool)
    one_hot[range(labels.shape[0]), labels] = True
    return one_hot    
    
def saveParams(layer, filename, **tags):
    params = lasagne.layers.get_all_param_values(layer, **tags)
    with open(filename, 'wb') as output:
        pkl.dump(params, output, pkl.HIGHEST_PROTOCOL)

def loadParams(filename, **tags):
    with open(filename, 'rb') as input:
        params = pkl.load(input)
    return params
    
def createLogger():
    """
    MOVE THIS INTO OTHER FILE - UTIL OR SOMETHING SIMILAR...
    """
    logger = logging.getLogger('logger')
    while logger.root.handlers:
        logger.root.removeHandler(logger.root.handlers[0])
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('testlog.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch) # add streamHandler
        logger.addHandler(fh) # and fileHandler
        
    return logger