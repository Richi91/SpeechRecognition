# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:57:49 2015

@author: richi-ubuntu
"""
from __future__ import print_function
import theano
import theano.tensor as T
import lasagne
import numpy as np

from collections import OrderedDict




def BLSTMConcatLayer(*args, **kwargs):
    """
    This function generates a BLSTM by concatenating a forward and a backward LSTM
    at axis 2, which should be the axis for the hidden dimension (batch_size x seq_len x hidden_dim)
    :parameters: See LSTMLayer for inputs, this layer receives the same inputs as a LSTM-Layer
    :returns: lasagne.layers.ConcatLayer of 2 LSTM layers 
    """
    kwargs.pop('backwards', None)
    
    
    ingate=lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
        W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))
    forgetgate=lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
        W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Constant(1.5))        
    outgate=lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
        W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))
    cell=lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
        W_cell=None, b=lasagne.init.Uniform(0.1), nonlinearity=lasagne.nonlinearities.tanh)
    
    l_fwd = lasagne.layers.LSTMLayer(
        *args, backwards=False, ingate=ingate, forgetgate=forgetgate, cell=cell, outgate=outgate, **kwargs)

        
    ingate=lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
        W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))
    forgetgate=lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
        W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Constant(1.5))        
    outgate=lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
        W_cell=lasagne.init.Uniform(0.1), b=lasagne.init.Uniform(0.1))
    cell=lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Uniform(0.1), W_hid=lasagne.init.Uniform(0.1), \
        W_cell=None, b=lasagne.init.Uniform(0.1), nonlinearity=lasagne.nonlinearities.tanh)
    
    l_bwd = lasagne.layers.LSTMLayer(
        *args, backwards=True,  ingate=ingate, forgetgate=forgetgate, cell=cell, outgate=outgate, **kwargs)
        
    return lasagne.layers.ConcatLayer((l_fwd,l_bwd),axis=2)


def categorical_crossentropy_batch(coding_dist, true_dist, mask):
    """
    Apply categorical crossentropy and zero out entropies, where mask = 0.
    Compare with theano.tensor.nnet.categorical_crossentropy.
    The first 2 inputs for this function are the same but with an additional dimension (first) for batch_size
    The last parameter is the mask that will be applied to calculate cross_entropy only for valid timesteps
    
    :parameters:
        - coding_dist: model output distribution, dimensions = batch_size x output_seq_length x output_dim
        - true_dist: target output sequence, dimensions = batch_size x output_seq_length
        - mask: mask for marking valid timesteps, dimensions = batch_size x output_seq_length
    :returns:
        - cross_entropy: mean of cross_entropys for each timestep, multiplied with batch_size for better scaling
    """
    shapes = coding_dist.shape
    cross_entropy = \
        (theano.tensor.nnet.categorical_crossentropy( \
            coding_dist.clip(1e-10,1-1e-10). \
                reshape([shapes[0]*shapes[1], shapes[2]]), true_dist.reshape([shapes[0]*shapes[1]])) \
        ) \
        * mask.reshape([shapes[0]*shapes[1]])
    return cross_entropy.mean()*shapes[0]
    
    
def momentum_with_grad_clipping(loss_or_grads, params, learning_rate, momentum=0.9, rescale=5.0):

    updates = sgd_with_grad_clipping(loss_or_grads, params, learning_rate, lasagne.utils.floatX(rescale))
    return lasagne.updates.apply_momentum(updates, momentum=lasagne.utils.floatX(momentum))


def sgd_with_grad_clipping(loss_or_grads, params, learning_rate, rescale):

    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    grad_norm = T.sqrt(grad_norm)
    scaling_num = rescale
    scaling_den = T.maximum(rescale, grad_norm)
    for param, grad in zip(params, grads):
        grad = T.switch(not_finite, 0.1 * param, grad * (scaling_num / scaling_den))
        updates[param] = param - learning_rate * grad
    return updates   



def rmsprop_with_grad_clipping(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6, rescale=5.0):

    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    grad_norm = T.sqrt(grad_norm)
    scaling_num = rescale
    scaling_den = T.maximum(rescale, grad_norm)
    
    
    for param, grad in zip(params, grads):
        grad = T.switch(not_finite, 0.1 * param, grad * (scaling_num / scaling_den))
        
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates


def adadelta_with_grad_clipping(loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6, rescale=5.0):

    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    grad_norm = T.sqrt(grad_norm)
    scaling_num = rescale
    scaling_den = T.maximum(rescale, grad_norm)
    
    for param, grad in zip(params, grads):
        grad = T.switch(not_finite, 0.1 * param, grad * (scaling_num / scaling_den))
        
        value = param.get_value(borrow=True)
        # accu: accumulate gradient magnitudes
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates



def genModel(batch_size, max_input_seq_len, input_dim, output_dim, gradient_steps, grad_clip, lstm_hidden_units):
    """
    Creates a deep BLSTM model with 3 layers of BLSTM units, where BLSTM units consist of 2 LSTM units,
    one calculating outputs for the forward sequence and one for backward sequence (reversed). The
    forward and backward LSTMs are merged by concatenating (alternative is sum). 
    The output of the 3 BLSTM Layers is fed into a fully connected layer. 
    The "post-output" layer is a Softmax.
    This function outputs both the model for the linear and the softmax output
    """
#************************************* Input Layer ********************************************
    l_in = lasagne.layers.InputLayer(shape=(batch_size, max_input_seq_len, input_dim))
    l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_input_seq_len), 
                                       input_var=theano.tensor.matrix('input_mask', dtype=theano.config.floatX))

    blstm0 = BLSTMConcatLayer(incoming=l_in, mask_input=l_mask, 
        num_units=lstm_hidden_units[0], gradient_steps=gradient_steps, grad_clipping=grad_clip)
    blstm1 = BLSTMConcatLayer(incoming=blstm0, mask_input=l_mask,
        num_units=lstm_hidden_units[1], gradient_steps=gradient_steps, grad_clipping=grad_clip)
    blstm2 = BLSTMConcatLayer(incoming=blstm1, mask_input=l_mask, 
        num_units=lstm_hidden_units[2], gradient_steps=gradient_steps, grad_clipping=grad_clip)
        
# Need to reshape hidden LSTM layers --> Combine batch size and sequence length for the output layer 
# Otherwise, DenseLayer would treat sequence length as feature dimension        

    l_reshape2 = lasagne.layers.ReshapeLayer(
        blstm2, (batch_size*max_input_seq_len, lstm_hidden_units[2]*2))
    l_out_lin = lasagne.layers.DenseLayer(
        incoming=l_reshape2, num_units=output_dim, nonlinearity=lasagne.nonlinearities.linear)
    
#************************************ linear output ******************************************
    l_out_lin_shp = lasagne.layers.ReshapeLayer(
        l_out_lin, (batch_size, max_input_seq_len, output_dim))
    
#************************************ Softmax output *****************************************
    l_out_softmax = lasagne.layers.NonlinearityLayer(
        l_out_lin, nonlinearity=lasagne.nonlinearities.softmax)
    l_out_softmax_shp = lasagne.layers.ReshapeLayer(
        l_out_softmax, (batch_size, max_input_seq_len, output_dim))   
        
    return l_out_lin_shp, l_out_softmax_shp, l_in, l_mask