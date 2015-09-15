# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:38:53 2015

@author: richi-ubuntu
"""
"""
One single BLSTM layer AND default weight init (W: gauss, and b: 0):
Momentum 0.9, learning rate e-4 too low. with e-3 carefully choose clipping factor. 1e-3 / learning_rateseems to be too low,
does not learn. 1e-2 had infs after 1st epoch. with rescale = 5e-3 it seems to work well!
Also soft gradient clipping of each weight matrix individually seems not to work, makes sense to
clip gradients to the overall norm, otherwise scales become different.

With uniform initialization:

Next momentum: with momentum 0.95 an lr 2e-4, learning is slow (lower LR), but keeps constantly minimizing. With
momentum 0.9, learning is even way slower and seems to get stuck on a bad local minimum (20 epoch 0.97 still).
With 0.95 and 2e-4 + uniform initialization achieve 50% ER, then more and more overfitting

0.925 and 5e-5 also converged at 50% PER
3 BLSTM layers:
"""
import lasagne
import numpy as np
import theano
import theano.tensor as T
import decode
import time
from netCDF4 import Dataset
import utils
import makeBatches
import lasagneExtensions
import sys
sys.path.append('../../Lasagne-CTC')
import ctc_cost as ctc

logger = utils.createLogger()

# ***************************** load netCDF4 datasets *******************************
dataPath = '../data/CTC/'
trainDataset = Dataset(dataPath+'train.nc')
valDataset = Dataset(dataPath+'val.nc')



# *************************** definitions  for test data ****************************
# training parameters
learning_rate = lasagne.utils.floatX(1e-4)
momentum = 0.9
MIN_ITER_HIGHEST_LR, THRESH_DELTA_PER, THRESH_MIN_ITER, THRESH_STOP, FACTOR_LOWER_LR , FACTOR_LOWER_MOM = \
20,                  0.005,            6,               20,          2.0,              1.035        
BATCH_SIZE = 10
START_EPOCH = 0 #e.g. if restart training
N_EPOCHS = 100
EPOCH_SIZE = 100 


    
# clips gate activations in backward pass at each timestep, implemented in lasagne
HARD_GRAD_CLIP_LASAGNE= False 

# this is the actual grad clipping, used in sgd with momentum. The overall L2 norm is considered
# of the gradients of all weight matrices (summed)
SOFT_GRAD_CLIP_L2NORM_WEIGHT_TOTAL = 2.0e-3 / learning_rate  # 5 default (worked ~well)


GRADIENT_STEPS = -1 # defines how many timesteps are used for error backpropagation. -1 = all

# Network size parameters
N_LSTM_HIDDEN_UNITS = 250 # 3 BLSTM layers, 3 forward, 3 backward, each 250 units
INPUT_DIM = len(trainDataset.dimensions['inputPattSize']) #123
OUTPUT_DIM = len(trainDataset.dimensions['numLabels']) # 62
MAX_OUTPUT_SEQ_LEN = len(trainDataset.dimensions['maxLabelLength']) # 75
MAX_INPUT_SEQ_LEN = len(trainDataset.dimensions['maxInputLength']) # 778
#************************************************************************************

logger.info('1 layer, momentum with L2 limit {}/learning rate, {} batches, learning rate {}, continue from epoch {}' \
    .format(SOFT_GRAD_CLIP_L2NORM_WEIGHT_TOTAL, BATCH_SIZE, learning_rate, START_EPOCH))
    
logger.info('generating model...')
        
input_mask_shape = (BATCH_SIZE, MAX_INPUT_SEQ_LEN)
# masked input
l_in = lasagne.layers.InputLayer(input_mask_shape + (INPUT_DIM,))
l_mask = lasagne.layers.InputLayer(shape=input_mask_shape)

blstm0 = lasagneExtensions.BLSTMConcatLayer(incoming=l_in, mask_input=l_mask, 
    num_units=N_LSTM_HIDDEN_UNITS, gradient_steps=GRADIENT_STEPS, grad_clipping=HARD_GRAD_CLIP_LASAGNE)

l_reshape = lasagne.layers.ReshapeLayer(blstm0, (BATCH_SIZE*MAX_INPUT_SEQ_LEN, 2*N_LSTM_HIDDEN_UNITS))
l_out_lin = lasagne.layers.DenseLayer(l_reshape, num_units=OUTPUT_DIM, nonlinearity=lasagne.nonlinearities.linear)
model_lin = lasagne.layers.ReshapeLayer(l_out_lin, input_mask_shape+(OUTPUT_DIM,))

l_out_soft = lasagne.layers.NonlinearityLayer(
        l_out_lin, nonlinearity=lasagne.nonlinearities.softmax)
model_soft = lasagne.layers.ReshapeLayer(l_out_soft, input_mask_shape+(OUTPUT_DIM,))


filename = '../data/paramValues/params_1l_lr1em4_mom90_b10_epoch29.pkl'
paramValues = utils.loadParams(filename)
lasagne.layers.set_all_param_values(model_soft, paramValues)

        
output_lin = lasagne.layers.get_output(model_lin) 
output_softmax = lasagne.layers.get_output(model_soft) 

Y = T.matrix('target', dtype=theano.config.floatX) #BATCH_SIZE x MAX_OUTPUT_SEQ_LEN
Y_mask = T.matrix('target_mask', dtype=theano.config.floatX) #BATCH_SIZE x MAX_OUTPUT_SEQ_LEN

# get all parameters used for training
all_params = lasagne.layers.get_all_params(model_lin, trainable=True) 

cost_monitor = ctc.cost(y=Y, y_hat_softmax=output_softmax, 
    y_mask=Y_mask, mask=l_mask.input_var).mean(dtype=theano.config.floatX)                                
cost_train = ctc.pseudo_cost(y=Y, y_hat=output_lin, 
    y_mask=Y_mask, mask=l_mask.input_var).mean(dtype=theano.config.floatX)

grads = theano.grad(cost_train, all_params)
 

updates = lasagneExtensions.momentum_with_grad_clipping(
    grads, all_params, learning_rate=learning_rate, momentum=momentum, rescale=SOFT_GRAD_CLIP_L2NORM_WEIGHT_TOTAL) 

    
logger.info('compiling functions...')
train = theano.function([l_in.input_var, Y, l_mask.input_var, Y_mask],
                        outputs=[output_softmax, cost_monitor],
                        updates=updates)
                        
compute_cost = theano.function(
                inputs=[l_in.input_var, Y, l_mask.input_var, Y_mask],
                outputs=cost_monitor)

forward_pass = theano.function(inputs=[l_in.input_var, l_mask.input_var],
                               outputs=[output_softmax])          
                                                              
#************************* load validation data *************************************
x_val_batch, y_val_batch, x_val_mask, y_val_mask = \
    makeBatches.makeRandomBatchesFromNetCDF(valDataset, BATCH_SIZE)
# also reshape masks and target. --> flatten num_batches x batch_size    
x_val_masks = x_val_mask.reshape([x_val_mask.shape[0]*x_val_mask.shape[1],x_val_mask.shape[2]]) 
y_val_masks = y_val_mask.reshape([y_val_mask.shape[0]*y_val_mask.shape[1],y_val_mask.shape[2]]) 
tar = y_val_batch.reshape([y_val_batch.shape[0]*y_val_batch.shape[1],y_val_batch.shape[2]])     
#************************************************************************************
scoreMap = decode.getPhonemeMapForScoring()
                          
                               
#************************************************************************************
n_below_thresh_lr = 0
n_max_lr = 0
n_below_thresh_untilStop = 0
oldPER = 1000
logger.info('Training...')
for epoch in range(START_EPOCH, N_EPOCHS):  
    start_time_epoch = time.time()   
    print "make new random batches"
    x_train_batch, y_train_batch, x_train_mask, y_train_mask = \
        makeBatches.makeRandomBatchesFromNetCDF(trainDataset, BATCH_SIZE)  
        
    for counter, (x, y, x_mask, y_mask) in enumerate(zip(x_train_batch, y_train_batch, x_train_mask, y_train_mask)):
        start_time_batch = time.time()       
        _, c  = train(x, y, x_mask, y_mask)
        end_time_batch = time.time()
        print "batch " + str(counter) + " duration: " + str(end_time_batch - start_time_batch) + " cost: " + str(c)
    
    # since we have a very small val set, validate over complete val set  
#************************************************************************************      
    cost_val = np.array([compute_cost(x, y, x_mask, y_mask)[()] \
        for x, y, x_mask, y_mask
        in zip(x_val_batch, y_val_batch, x_val_mask, y_val_mask)]).mean()  
    
    # feed batches of data and mask through net, then reshape to flatten dimensions num_batches x batch_size 
    net_outputs = np.array([forward_pass(x, x_mask)[0] for x, x_mask in zip(x_val_batch, x_val_mask)]) 
    sequence_probdist = net_outputs \
        .reshape([net_outputs.shape[0]*net_outputs.shape[1],net_outputs.shape[2],net_outputs.shape[3]]) 
        
    # decode each training datum sequentially.
    # TODO: decode in batches           
    decoded = [decode.decodeSequence(decode.mapNetOutputs(sequence_probdist[i], scoreMap),
        x_val_masks[i], 39) for i in range(sequence_probdist.shape[0])]
    
    # calculate PER for each training datum sequentially. 
    # TODO: PER in batches
    PERs = [decode.calcPER(decode.mapTargets(tar[i,y_val_masks[i,:]==1], scoreMap), decoded[i]) 
        for i in range(tar.shape[0])]   
    PER = np.mean(PERs)
    #***************************************************************************************
    
    if n_max_lr > (MIN_ITER_HIGHEST_LR-THRESH_MIN_ITER): # already more than minimum num of iters with starting/highest LR
        if (oldPER - PER) < THRESH_DELTA_PER: # delta-PER too small
            if n_below_thresh_lr >= THRESH_MIN_ITER: # since too many steps --> adapt LR
                learning_rate = lasagne.utils.floatX(learning_rate / FACTOR_LOWER_LR)
                momentum = lasagne.utils.floatX(momentum / FACTOR_LOWER_MOM)
                updates = lasagneExtensions.momentum_with_grad_clipping(
                    grads, all_params, learning_rate=learning_rate, momentum = momentum, 
                    rescale=SOFT_GRAD_CLIP_L2NORM_WEIGHT_TOTAL)
                train = theano.function([l_in.input_var, Y, l_mask.input_var, Y_mask],
                    outputs=[output_softmax, cost_monitor], updates=updates)
                logger.info('new learning rate = {}'.format(learning_rate))
                n_below_thresh_lr = 0
            else: # not enough steps below thresh --> do not yet adapt LR, increment counter
               n_below_thresh_lr += 1
               n_below_thresh_untilStop += 1
        else: # delta-PER big enough --> reset counter and set oldPER
            n_below_thresh_lr = 0
            n_below_thresh_untilStop = 0
            oldPER = PER
    else: # not yet min mumber of iters with highest(=start) LR --> do not change LR!
        n_max_lr += 1
         
    print "\n"
    end_time_epoch = time.time()
    logger.info("Epoch {} took {}, cost(val) = {}, PER-mean = {}".format(
        epoch, end_time_epoch - start_time_epoch, cost_val, PER))
        
        
    if n_below_thresh_untilStop > THRESH_STOP:
        logger.info('Finished Training')
        filename = '../data/paramValues/params_1l_lr1em4_mom90_b10_epoch' + str(epoch) + '.pkl' 
        utils.saveParams(model_lin, filename)
        break    
#************************************ save param values **************************************
    if ((epoch+1)%10 == 0):
        filename = '../data/paramValues/params_1l_lr1em4_mom90_b10_epoch' + str(epoch) + '.pkl' 
        utils.saveParams(model_lin, filename)
        print 'save'                          
                               
                               
                               
                               