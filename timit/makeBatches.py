# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:31:08 2015

@author: richi-ubuntu
"""
import numpy as np
import theano



def makeRandomBatchesFromNetCDF(rootgrp, batch_size):
    """
    This function generates random batches from a netCDF dataset. 
    Not only one batch is generated, but as many as possible.
    The outputs are thus of dimension num_batches x batch_size x ... 
    The elements/sequences in the batches are randomly shuffled. This function
    can be called after each epoch, generating another random constellation of batches.
    Note that enoug RAM should be available
    :parameters:
        - rootgrp : netCDF4 dataset generated with the function in frontEnd.py 
        - batch_size : int
            Mini-batch size
    :returns:
        - X_batch: np.array with dtype theano.config.floatX, num_batches x batch_size x input_seq_len x input_dim
            all batches of input data
        - y_batch: np.array with dtype theano.config.floatX, num_batches x batch_size x output_seq_len
            all batches of target data
        - X_mask: np.array with dtype theano.config.floatX, num_batches x batch_size x input_seq_len 
            all batches of input masks
        - y_mask: mp.array with dtype theano.config.floatX, num_batches x batch_size x output_seq_len 
            all batches of output data
    """
    numSeqs = len(rootgrp.dimensions['numSeqs'])
    n_batches = numSeqs//batch_size
    input_sequence_length = len(rootgrp.dimensions['maxInputLength'])
    output_sequence_length = len(rootgrp.dimensions['maxLabelLength'])
    inputPattSize = len(rootgrp.dimensions['inputPattSize'])
    
    # initialize with zeros. n_batches * batch_sz x in_seq_length x inputPatternSize --> reshape later
    X_batch = np.zeros((n_batches * batch_size, input_sequence_length, inputPattSize),
                       dtype=np.float32) 
    # n_batches * batch_sz x out_seq_length --> reshape later
    y_batch = np.zeros((n_batches * batch_size, output_sequence_length),
                       dtype=np.float32) 
              
    X_mask = np.zeros((n_batches * batch_size, input_sequence_length), dtype=np.float32)          
    y_mask = np.zeros((n_batches * batch_size, output_sequence_length), dtype=np.float32)
    
    # get as many sequences as possible: e.g. 1520 sequences with batch size 50 --> 30*50 batches, dump 20
    selected_sequences = np.sort(np.random.choice(numSeqs,n_batches*batch_size, False))
    
    seq_start_index = rootgrp.variables['seqStartIndices'][selected_sequences]
    seq_len = rootgrp.variables['seqLengths'][selected_sequences]
    for c, (si,sl) in enumerate(zip(seq_start_index,seq_len)):
        X_m = rootgrp.variables['inputs'][si:si+sl,:]
        X_batch[c, :X_m.shape[0],:] = X_m
        X_mask[c, :X_m.shape[0]] = 1.0
        
    label_start_index = rootgrp.variables['labelStartIndices'][selected_sequences]
    label_len = rootgrp.variables['labelLengths'][selected_sequences]
    for c, (si,sl) in enumerate(zip(label_start_index,label_len)):          
        y_m = rootgrp.variables['targetClasses'][si:si+sl]
        y_batch[c, :y_m.shape[0]] = y_m
        y_mask[c, :y_m.shape[0]] = 1.0
        
    # shuffle sequences, reshape and convert to theano float32
    shuffle = np.random.choice(n_batches*batch_size,n_batches*batch_size, False)
    return X_batch[shuffle].reshape([n_batches, batch_size, input_sequence_length, inputPattSize]).astype(theano.config.floatX), \
           y_batch[shuffle].reshape([n_batches, batch_size, output_sequence_length]).astype(theano.config.floatX), \
           X_mask[shuffle].reshape([n_batches, batch_size, input_sequence_length]).astype(theano.config.floatX), \
           y_mask[shuffle].reshape([n_batches, batch_size, output_sequence_length]).astype(theano.config.floatX)


def makeBatches(X, y, input_sequence_length, output_sequence_length, batch_size):
    '''
    Convert a numpy-vector(list) of matrices into batches of uniform length
    :parameters:
        - X : numpy-vector(list) of np.ndarray
            List of matrices
        - y: numpy-vector(list) of np.ndarray
            numpy-vector(list) of vectors
        - input_sequence_length : int
            Desired input sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
       - output_sequence_length : int
            Desired output sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.   
        - batch_size : int
            Mini-batch size
    :returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, input_sequence_length, n_features)
        - X_mask : np.ndarray
            Mask denoting whether to include each time step of each time series
            shape=(n_batches, batch_size, input_sequence_length)
        - y_batch : np.ndarray
            Tensor of time series batches,
            shape=(n_batches, batch_size,output_sequence_length)
            - y_batch : np.ndarray
            Mask denoting whether to include each time step of each phoneme output series,
            shape=(n_batches, batch_size,output_sequence_length)
    '''
    n_batches = len(X)//batch_size # division with ceil (no non-full batches)
    # n_batches x batch_sz x in_seq_length x feature_dim
    X_batch = np.zeros((n_batches, batch_size, input_sequence_length, X[0].shape[1]),
                       dtype=np.float32) 
    # n_batches x batch_sz x out_seq_length
    y_batch = np.zeros((n_batches, batch_size, output_sequence_length),
                       dtype=np.float32) 
              
    X_mask = np.zeros((n_batches, batch_size, input_sequence_length), dtype=np.float32)          
    y_mask = np.zeros((n_batches, batch_size, output_sequence_length), dtype=np.float32)
    
    for b in range(n_batches):
        for n in range(batch_size):
            # read sequence from raw list of sequences
            X_m = X[b*batch_size + n] # has shape in_seq_len x feat_dim
            # put in the sequence to according position in X_batch
            X_batch[b, n, :X_m.shape[0],:] = X_m
            # mask (mark) all elements of X_batch, that belong to sequence with 1,
            # sequences shorter than max-length sequence are padded with zeros
            # and marked with 0 in X_mask
            X_mask[b, n, :X_m.shape[0]] = 1.0
            
            # similar with y
            y_m = y[b*batch_size + n]
            y_batch[b, n, :y_m.shape[0]] = y_m
            y_mask[b, n, :y_m.shape[0]] = 1.0
    return X_batch.astype(theano.config.floatX), X_mask.astype(theano.config.floatX), \
           y_batch.astype(theano.config.floatX), y_mask.astype(theano.config.floatX)
    

def makeBatchesNoCTC(X, y, sequence_length, batch_size):
    """
    Not needed anymore... delete me eventually
    """
    n_batches = len(X)//batch_size # division with ceil (no non-full batches)
    # n_batches x batch_sz x in_seq_length x feature_dim
    X_batch = np.zeros((n_batches, batch_size, sequence_length, X[0].shape[1]),
                       dtype=np.float32) 
    # n_batches x batch_sz x out_seq_length
    y_batch = np.zeros((n_batches, batch_size, sequence_length),
                       dtype=np.float32) 
              
    mask = np.zeros((n_batches, batch_size, sequence_length), dtype=np.float32)          

    
    for b in range(n_batches):
        for n in range(batch_size):
            # read sequence from raw list of sequences
            X_m = X[b*batch_size + n] # has shape in_seq_len x feat_dim
            # put in the sequence to according position in X_batch
            X_batch[b, n, :X_m.shape[0],:] = X_m
            # mask (mark) all elements of X_batch, that belong to sequence with 1,
            # sequences shorter than max-length sequence are padded with zeros
            # and marked with 0 in X_mask
            mask[b, n, :X_m.shape[0]] = 1.0
            
            # similar procedure with y
            y_m = y[b*batch_size + n]
            y_batch[b, n, :y_m.shape[0]] = y_m
    return X_batch.astype(theano.config.floatX), \
           y_batch.astype(theano.config.floatX), \
           mask.astype(theano.config.floatX)