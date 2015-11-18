# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:38:49 2015

@author: Richard Kurle
"""
import os
"""
Description, default value and dtype of the configs, that must be specified.
--------
- task='CTC': string, either 'CTC' or 'framewise', determines,
    whether model shall be trained to predict labels for each frame
    or CTC-Algorithm shall be used to train on the actual phoneme sequence.
    Targets for the dataset are extracted accordingly to this configuration.
- mapTo39Phonemes_Training=True: bool, train only on 39 phonemes instead of 61?
- mapTo39Phonemes_Decoding=False: bool, map original 61 phonemes to 39
    during decoding, for scoring?

- winlen=0.0256: float, window length for frames in [s]
- winstep=0.010: float, step size between succeeding windows in [s]
- nfilt=40: number, int of filterbanks
- lowfreq=133.3333: float, lowest frequency of filterbanks
- highfreq=6855.4976: float, highest frequency of filterbanks
- preemph=0.97: float, preemphasis on speech signal
- winSzForDelta=2: int, "N" in formula for calculating deltas
- samplerate=16000: int, samplerate of Timit dataset
- n_speaker_val=50: int, number of speakers in valid set (with all sentences)
- featureType='logFB': string, either 'logFB' or 'mfcc'
- numcep=13: int, number of cepstral coefficients
- ceplifter=22: int,
- normalisation='trainData': string, either 'trainData' or 'individual',
    the former normalises on the train data's mean and variance,
    the latter normalizes each feature individually

  Theano var names specified here, because they are used at several places, e.g.
  creation of the dataset, reading out from batch, creating datastream,
  names must be conform
- input_theano='input': string, naming the theano input variable
- input_mask_theano='input_mask': string, naming the theano input_mask variable
- target_theano='target': string, naming the theano target variable
- target_mask_theano='target_mask': string, naming the theano target_mask variable

- timitRoot=os.getenv("HOME") + '/data/TIMIT/timit': string, Specifies root directory 
    of Timit Database, containing the folders train and test.
- path_to_dataset_CTC=os.getenv("HOME")+'/data/TimitFeat/CTC/Timit.hdf5':
    string, path to save (preprocessing) or load (training) features for CTC 
- path_to_dataset_framewise=os.getenv("HOME")+'/data/TimitFeat/Framewise/Timit.hdf5'
    string, path to save (preprocessing) or load (training) features for framewise 
- path_to_model=os.getenv("HOME")+'/data/TimitModelParams/blocks/model.pkl':
    string, path to save/load the trained blocks model (MainLoop)
    
- max_epochs=100: int, number of epochs until stop training
- batch_size=50: int, number of sequences trained in parallel
- step_rule='AdaDelta': string, either 'Momentum' or 'AdaDelta',
    use SGD with either AdaDelta or basic Momentum <-- need lr and momentum
- learning_rate=0.1: float, only needed in case of Momentum
- momentum=0.9: floa [0...1], only needed in case of Momentum
- step_clipping=1: float, clip gradients to have maximum norm
    
    
- transition='GRU': string, either 'RNN', 'GRU' or 'LSTM',
    which is the simple RNN, a gated recurrent unit or long short term memory
- dims_transition=[250,250,200]: list of int: dimensions of the transition(recurrent)
- num_classes=40 # either 40 or 61 (CTC or Framewise)

- std_input_noise=0.2: float, input noise std-dev, applied to input features
- weight_noise=0.05: float, weight noise std-dev, applied to weights every batch
- epochs_early_stopping=5, int, number of epochs until stop if no improvement
"""
# ************************* GENERAL ********************************
task='CTC'
mapTo39Phonemes_Training=False
mapTo39Phonemes_Decoding=True


# ***************** FEATURE EXTRACTION PARAMETERS ******************
winlen=0.0256 
winstep=0.010 
nfilt=40 
lowfreq=133.3333
highfreq=6855.4976
preemph=0.97
winSzForDelta=2
samplerate=16000
n_speaker_val=50
featureType='mfcc'
numcep=13
ceplifter=22
normalisation='trainData'


# ******************* THEANO VARIABLE NAMES ***********************
input_theano='input'
input_mask_theano='input_mask'
target_theano='target'
target_mask_theano='target_mask'


#*************************** PATHS *********************************
path_to_timitRoot=os.getenv("HOME")+'/data/TIMIT/timit'
path_to_dataset_CTC=os.getenv("HOME") + '/data/TimitFeat/CTC/Timit.hdf5'
path_to_dataset_framewise=os.getenv("HOME") + '/data/TimitFeat/Framewise/Timit.hdf5'
path_to_model=os.getenv("HOME")+'/data/TimitModelParams/blocks/model.pkl'
#**************************** TRAIN PARAMETERS *********************
max_epochs=100
batch_size=32
step_rule='AdaDelta' 
learning_rate=0.1 
momentum=0.9 
step_clipping=1.


#**************************** MODEL PARAMETERS *********************
transition='LSTM'
dims_transition  = [300,250,200]


#**************************** REGULARIZATION ***********************
std_input_noise=0.1
weight_noise=0.075
epochs_early_stopping=5


#*******************************************************************
assert not (mapTo39Phonemes_Training and mapTo39Phonemes_Decoding) , \
    'cant map 61 to 39 phonemes both before training and during decoding'