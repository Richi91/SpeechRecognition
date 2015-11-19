# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 23:33:50 2015

@author: Richard Kurle
"""

import numpy as np
import theano
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension, TrainingExtension
from blocks.bricks.base import application
from blocks.bricks.cost import Cost
from fuel.transformers import Transformer
from blocks.monitoring.aggregation import MonitoredQuantity
import logging
import conf
from TimitFeatureExtraction import getPhonemeMapForScoring


def ShuffleDim(data):
    """
    Parameters
    ----------
    This function is used together with the function Mapping from fuel.transformers.
    The first 2 dimensions get swapped, the rest stay like they are.
    """
    dims = [len(d.shape) for d in data]
    dimshuffle = [[1,0] for i in range(len(dims))]
    for i in range(len(dims)):
        for k in range(2, dims[i]): # if one of the inputs from the data tuple has more than 2 dimensions...
            dimshuffle[i].append(k) # ...we have to append the indices for these dimensions (needed for transpose)
    return tuple([d.transpose(dimshuffle[i]) for i,d in enumerate(data)])


class AddInputNoise(Transformer):
    """Applies white noise to input.
    Applies noise only to input with 3 dimensions.
    E.g. input-sequence of shape Time x Batch x Features
    And not Masks, targets, target_masks, which have 2 dimensions

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    std: std dev of gaussian noise

    """
    def __init__(self, data_stream, std=0.0, **kwargs):
        super(AddInputNoise, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)
        self.std = std

    @property
    def sources(self):
        return self.data_stream.sources

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        return  self.addNoise(next(self.child_epoch_iterator))

    def addNoise(self, data):
        return tuple([d + np.random.normal(0,self.std, d.size).reshape(d.shape).astype('float32')
                      if len(d.shape)==3 else d for d in data])


class PhonemeErrorRate(SimpleExtension, MonitoringExtension):
    """ Monitors Phoneme Error Rate in case of phoneme recognition with CTC.

    Parameters
    ----------
    data_stream: fuel datastream, containing 'inputs', 'input_masks',
    'target_masks' and 'targets'. These Variable names have been given
    when creating the dataset and are used for the corresponding theano variables

    getOutputFunction: theano function, which gives needs inputs 'inputs' and 'input_masks'
    and returns the output (y_hat) of the model.
 
    """
    def __init__(self, data_stream, getOutputFunction, **kwargs):
        self.getOutput = getOutputFunction
        self.data_stream = data_stream
        super(PhonemeErrorRate, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        """calculate PER and monitor it's mean, min and max """
        per = []
        for batch in self.data_stream.get_epoch_iterator(as_dict=True):
            inputs=batch[conf.input_theano]
            input_masks = batch[conf.input_mask_theano]
            targets = batch[conf.target_theano]
            target_masks = batch[conf.target_mask_theano]

            outputs = self.getOutput(inputs, input_masks)
            if conf.mapTo39Phonemes_Decoding:
                scoreMap = getPhonemeMapForScoring()
                outputs = self.mapOutputsForScoring(outputs, scoreMap)
                targets = self.mapTargetsForScoring(targets, scoreMap)
            decoded = self.decodeSequences(outputs, input_masks)

            targets = self.maskTargets(targets,target_masks)

            for t,d in zip(targets,decoded):
                per.append(np.min([self.per(t,d),1]))
        per=np.asarray(per)

        accuracies_dict={'meanPER':np.mean(per), # mean accuracy over epoch
                         'minPER':np.min(per),
                         'maxPER':np.max(per),
                         }
        self.add_records(self.main_loop.log, accuracies_dict.items())


    def mapOutputsForScoring(self, batch, scoreMap):
        """
        This function maps 61 phonemes + blank to 39 phonemes + blank
        Adds up network outputs that are mapped to the same output for scoring
        Not sure if it is better to map-sum the softmax outputs, the linear outputs
        or to map after decoding.

        Here: sum the softmax outputs (if the provided output function is the softmax output)
        """
        mappedBlankSymbol=np.max(scoreMap.values())
        mappedBatch=np.zeros((batch.shape[0],batch.shape[1], mappedBlankSymbol+1))
        for key in scoreMap.keys():
            if scoreMap[key] is not None:
                mappedBatch[:,:,scoreMap[key]] += batch[:,:,key]
        # blank symbol is not in the scoreMap, but it's the last element
        mappedBatch[:,:,-1] = batch[:,:,-1]
        return mappedBatch

    def mapTargetsForScoring(self, batch, scoreMap):
        mappedBlankSymbol=np.max(scoreMap.values())
        # 'q' == None --> make it a blank, blanks will be removed anyway during decoding
        return np.array([scoreMap[s] if scoreMap[s] is not None else mappedBlankSymbol
                for s in batch.flatten() ]).reshape(batch.shape)


    def maskTargets(self, targets,target_masks):
	"""
        This function masks the targets with their mask and puts the resulting
        masked targets in a list, where each element has a different length
        """
        masked=[]
        for iter_batch in range(targets.shape[1]):
            masked.append(targets[:,iter_batch][target_masks[:,iter_batch]==1])
        return np.array(masked)

    def decodeSequences(self, outputs, input_masks):
        """
        This function decodes each timestep + batch by outputting the label with the highest probability,
        given an output distribution.
        mask outputs with input_mask, take maximum output, remove blanks, remove repeated outputs
        :parameters:
            - outputs: numpy 3D array of output distribution timestep x batch x output_dim
	    - input_masks: mask of input sequence, which has dimensions timestep x batch
        :returns:
            - decoded: list of length batch of decoded sequences of different lengths (num timesteps)
        """
        decoded = []
        _blank = outputs.shape[-1]-1 #blank is highest label
        for iter_batch in range(outputs.shape[1]):
            reduced = np.array(outputs[:,iter_batch,:][input_masks[:,iter_batch]==1]).argmax(axis=1)
            reduced = reduced[reduced!=_blank]
            reduced = np.array([seq for index, seq in enumerate(reduced) \
                    if (reduced[index] != reduced[index-1] or index == 0)])
            decoded.append(reduced)
        return np.array(decoded)


    def levenshtein(self, y, y_hat):
        """levenshtein distance between two sequences.
        the minimum number of symbol edits (i.e. insertions,
        deletions or substitutions) required to change one
        word into the other.
        """
        plen, tlen = len(y_hat), len(y)

        dist = [[0 for i in range(tlen+1)] for x in range(plen+1)]
        for i in xrange(plen+1):
            dist[i][0] = i
        for j in xrange(tlen+1):
            dist[0][j] = j

        for i in xrange(plen):
            for j in xrange(tlen):
                if y_hat[i] != y[j]:
                    cost = 1
                else:
                    cost = 0
                dist[i+1][j+1] = min(
                    dist[i][j+1] + 1, #  deletion
                    dist[i+1][j] + 1, #  insertion
                    dist[i][j] + cost #  substitution
                    )
        return dist[-1][-1]

    def per(self, y, y_hat):
        """ phoneme error rate = Levenstein distance divided by target length """
        return self.levenshtein(y, y_hat) / float(len(y))



class PhonemeErrorRateFramewise(SimpleExtension, MonitoringExtension):
    """ Monitors Phoneme Error Rate in case of framewise phoneme recognition.

    Parameters
    ----------
    data_stream: fuel datastream, containing 'inputs', 'input_masks',
    'target_masks' and 'targets'. These Variable names have been given
    when creating the dataset and are used for the corresponding theano variables

    getOutputFunction: theano function, which gives needs inputs 'inputs' and 'input_masks'
    and returns the output (y_hat) of the model.   
    """

    def __init__(self, data_stream, getOutputFunction, **kwargs):
        self.getOutput = getOutputFunction
        self.data_stream = data_stream

        super(PhonemeErrorRateFramewise, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        per = []
        for batch in self.data_stream.get_epoch_iterator(as_dict=True):
            inputs=batch[conf.input_theano]
            masks = batch[conf.input_mask_theano]
            targets = batch[conf.target_theano]
            outputs = self.getOutput(inputs, masks)
            if conf.mapTo39Phonemes_Decoding:
                scoreMap = getPhonemeMapForScoring()
                outputs = self.mapOutputsForScoring(outputs, scoreMap)
                targets = self.mapTargetsForScoring(targets, scoreMap)
            decoded = self.decodeSequences(outputs)

            per.append(np.min([self.per(targets,decoded,masks),1]))
        per=np.asarray(per)

        accuracies_dict={'meanPER':np.mean(per), # mean accuracy over epoch
                         'varPER':np.var(per),
                         'minPER':np.min(per),
                         'maxPER':np.max(per),
                         #'decoded':d,
                         #'taget':t,
                         #'orig outputs':outputs,
                         }
        self.add_records(self.main_loop.log, accuracies_dict.items())


    def mapOutputsForScoring(self, batch, scoreMap):
        """
        This function maps 61 phonemes + blank to 39 phonemes + blank
        Adds up network outputs that are mapped to the same output for scoring
        Not sure if it is better to map-sum the softmax outputs, the linear outputs
        or to map after decoding.

        Here: sum the softmax outputs (if the provided output function is the softmax output)
        """
        mappedBlankSymbol=np.max(scoreMap.values())
        mappedBatch=np.zeros((batch.shape[0],batch.shape[1], mappedBlankSymbol+1))
        for key in scoreMap.keys():
            if scoreMap[key] is not None:
                mappedBatch[:,:,scoreMap[key]] += batch[:,:,key]
        return mappedBatch


    def mapTargetsForScoring(self, batch, scoreMap):
        return np.array([scoreMap[s] # 'q' == None --> Leave it None, take care during calc of PER
                for s in batch.flatten() ]).reshape(batch.shape)

    def decodeSequences(self, outputs):
        """
        This function decodes each timestep + batch by outputting the label with the highest probability,
        given an output distribution.
        No mask used here
        :parameters:
            - sequence_probdist: numpy 3D array of output distribution time x batch x output_dim
        :returns:
            - decoded: list of length batch of decoded sequences of different length (time)
        """
        return outputs.argmax(axis=2)

    def per(self, y, y_hat, mask):
        """
        This function calculates the phoneme-error-rate, when not using CTC, but having a network output
        for every input. just compares target output (tar) and actual output (out)
        :parameters:
            - tar: target output: time x batch
            - out: network output (decoded): time x batch
            - mask: time x batch
        :returns:
            - phoneme error rate
        """
        if conf.mapTo39Phonemes_Decoding:
            return np.float(np.sum(y[mask==1] != y_hat[mask==1]))/np.sum(np.isfinite(y))
        else:
            return np.mean(y[mask==1] != y_hat[mask==1])



class categorical_crossentropy_batch(Cost):
    """ 3-Dimensional Crossentropy, by flattening batch_size * timesteps

    Apply categorical crossentropy and zero out entropies, where mask = 0.
    Compare with theano.tensor.nnet.categorical_crossentropy.
    The first 2 inputs to this function are the same but with an additional dimension for timesteps.
    The parameter 'mask' that will be applied to calculate cross_entropy only for valid timesteps

    :parameters:
        - coding_dist: model output distribution, dimensions = output_seq_length x batch_size x output_dim
        - true_dist: target output sequence, dimensions = output_seq_length x batch_size
        - mask: mask for masking valid timesteps, dimensions = output_seq_length x batch_size
    :returns:
        - cross_entropy: mean of cross_entropys over timesteps, sum over batches
    """
    @application(outputs=["cost"])
    def apply(self, coding_dist, true_dist, mask):
        shapes = coding_dist.shape
        cost = theano.tensor.nnet.categorical_crossentropy( \
                coding_dist.clip(1e-10,1-1e-10).reshape([shapes[0]*shapes[1], shapes[2]]),
                true_dist.reshape([shapes[0]*shapes[1]]).astype('int64'))
        if mask is not None:
            cost *= mask.reshape([shapes[0]*shapes[1]])

        return cost.sum()/shapes[1]


def createLogger(filename):
    """

    """
    logger = logging.getLogger('logger')
    while logger.root.handlers:
        logger.root.removeHandler(logger.root.handlers[0])
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename)
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