# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:10:24 2015

@author: Richard Kurle
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:27:04 2015

@author: richi-ubuntu
"""

import features
import numpy as np
import glob
import os
import soundfile as sf
import csv
import random
from features import sigproc
from scipy.fftpack import dct


def hamming(n):
    """
    Generate a hamming window of n points as a numpy array.
    """
    return 0.54 - 0.46 * np.cos(2 * np.pi / n * (np.arange(n) + 0.5))
    
    
def delta(feat, deltawin):
    '''
    Computes the derivatives of a feature-matrix wrt. 1st dimension.
    Borders of size 'deltawin' at the beginning and end (of 1st dim)
    are padded with zeros before computing the derivatives.
    --> Output dim = Input dim
    :parameters:
        - feat : np.ndarray, dtype=float
            Array of feature vectors, 1st dim: time, 2nd dim: feature-dimension
        - deltawin : int
            number of neighbour features, used for computing the deltas
    :returns:
        - deltafeat: np.ndarray, dtype=float
            deltas of the input features
    '''
    # repeat the values at the borders is default behaviour of HTK --> adapt it here aswell instead of zero padding
    pad_bot = np.tile(feat[0,:],(deltawin,1))
    pad_top = np.tile(feat[-1,:],(deltawin,1))
    padfeat = np.concatenate((pad_bot,feat,pad_top),axis=0)

    norm = 2.0 * sum([it**2 for it in range(1, deltawin+1)])
    deltafeat = sum([it*(np.roll(padfeat,-it,axis=0)-np.roll(padfeat,it,axis=0)) for it in range(1, deltawin+1)])

    deltafeat /= norm;
    return deltafeat[deltawin:padfeat.shape[0]-deltawin,:]



def logFilterbankFeatures(signal,samplerate=16000,winlen=0.0255,winstep=0.01,
          nfilt=40,nfft=512,lowfreq=133.3333,highfreq=6855.4976,preemph=0.97,
          winSzForDelta=2):
    '''
    Computes log filterbank energies on a mel scale + total energy using 
    with the code taken from features.fbank, which does not accept
    window function as a param. 
    function from package 'python_speech_features', see
    http://python-speech-features.readthedocs.org/en/latest/ or
    https://github.com/jameslyons/python_speech_features

    Therefore it calculates the FFT of the signal and sums the the weighted
    bins, distributed on a mel scale. Weighting is done with tri-angular filters.
    For these filter energies + total energy, deltas are calculated.
    
    :parameters:
        - signal : np.ndarray, dtype=float
            input vector of the speech signal
        - samplerate : int
        - winlen: float
            length of analysis window in seconds
        - winstep: float
            step size between successive windows in seconds
        - nfilt: int
             number of filter energies to compute (total energy not included).
             e.g. 40 --> Output dim = (40+1)*3
        - nfft: int
            FFT size
        - lowfreq: int
            lower end on mel frequency scale, on which filter banks are distributed
        - highfreq: int
            upper end on mel frequency scale, on which filter banks are distributed
        - preemph: float
            pre-emphasis coefficient
        - deltafeat: np.ndarray, dtype=float
            deltas of the input features
        - winSzForDelta: int
            window size for computing deltas. E.g. 2 --> t-2, t-1, t+1 and t+2 are
            for calculating the deltas
    :returns:
        - features: numpy.array: float
            feature-matrix. 1st dimension: time steps of 'winstep',
            2nd dim: feature dimension: (nfilt + 1)*3,
            +1 for energy, *3 because of deltas

    '''
    # Part of the following code is copied from function features.fbank
    # Unfortunately, one can't specify the window function in features.fbank
    # Hamming window is used here
    
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate,winfunc=hamming)
    pspec = sigproc.powspec(frames,nfft)
    energy = np.sum(pspec,1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log  
    fb = features.get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log
    
    # Use log feature bank and log energy
    feat = np.column_stack((np.log(energy),np.log(feat)))
    # calculate delta and acceleration
    deltaFeat = delta(feat, winSzForDelta)
    accFeat = delta(deltaFeat, winSzForDelta)
    # stack features + delta + acceleration
    return np.concatenate((feat,deltaFeat,accFeat),axis=1)


def mfccFeatures(signal,samplerate=16000,winlen=0.0256,winstep=0.01, nfilt=40, 
                 nfft=512,lowfreq=133.3333,highfreq=6855.4976,preemph=0.97, 
                 winSzForDelta=2, numcep=13, ceplifter=22, appendEnergy=True):
    '''
    Computes MFCC features with the code taken from features.fbank and 
    features.mfcc, which do not accept window function as a param. 
    function from package 'python_speech_features', see
    http://python-speech-features.readthedocs.org/en/latest/ or
    https://github.com/jameslyons/python_speech_features
    
    :parameters:
        - signal : np.ndarray, dtype=float
            input vector of the speech signal
        - samplerate : int
        - winlen: float
            length of analysis window in seconds
        - winstep: float
            step size between successive windows in seconds
        - nfilt: int
             number of filter energies to compute (total energy not included).
             e.g. 40 --> Output dim = (40+1)*3
        - nfft: int
            FFT size
        - lowfreq: int
            lower end on mel frequency scale, on which filter banks are distributed
        - highfreq: int
            upper end on mel frequency scale, on which filter banks are distributed
        - preemph: float
            pre-emphasis coefficient
        - deltafeat: np.ndarray, dtype=float
            deltas of the input features
        - ceplifter: int
        - numcep: number of cepstral coefficients
        - appendEnergy: bool
        - winSzForDelta: int
            window size for computing deltas. E.g. 2 --> t-2, t-1, t+1 and t+2 are
            for calculating the deltas

    :returns:
        - features: numpy.array: float
            feature-matrix. 1st dimension: time steps of 'winstep',
            2nd dim: feature dimension: (nfilt + 1)*3,
            +1 for energy, *3 because of deltas

    '''
    # Part of the following code is copied from function features.fbank
    # Unfortunately, one can't specify the window function in features.fbank
    # Hamming window is used here
    from features import fbank
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate,winfunc=hamming)
    pspec = sigproc.powspec(frames,nfft)
    energy = np.sum(pspec,1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log  
    fb = features.get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log

    energy=np.log(energy)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = features.lifter(feat,ceplifter)
    if appendEnergy:
        feat = np.column_stack((energy,feat))
    # calculate delta and acceleration
    deltaFeat = delta(feat, winSzForDelta)
    accFeat = delta(deltaFeat, winSzForDelta)
    # stack features + delta + acceleration
    return np.concatenate((feat,deltaFeat,accFeat),axis=1)
    

def getAllFeatures(featureType, wavFileList, samplerate=16000,winlen=0.0256,winstep=0.01, 
                  nfilt=40, nfft=512,lowfreq=133.3333,highfreq=6855.4976,preemph=0.97,
                  winSzForDelta=2, numcep=13, ceplifter=22, appendEnergy=True):
    '''
    Computes all features of a given numpy vector of file paths to .wav files. Reads the
    wav files specified in 'wavFileList' the package 'PySoundFile'.
    PySoundFile is able to read the format of the files from TIMIT database.
    See: http://pysoundfile.readthedocs.org/en/0.7.0/ and
    https://github.com/bastibe/PySoundFile

    For other parameters see function getFeatures, once signal is read from path,
    signal and other parameters are forwarded to 'getFeatures'
    :parameters:
        - featureType: either 'mfcc' or 'logFB'
        - wavFileList: list of file paths
        - samplerate
        - winlen
        - winstep
        - nfilt
        - nfft
        - lowfreq
        - highfreq
        - preemph
        - winSzForDelta
    :returns:
        - featureList: numpy vector of np.arrays
        list of same length as input wavFileList, dimensions of every element
        of the list specified by signal duration and winstep (1st dim), and
        number of filters (2nd dim)
    '''
        
    featureList = []
    for f in wavFileList:
        signal, _ = sf.read(f)
        # equalize rms --> same power in all speech signals. Note that later features will be normalised
        # to have zero mean and unit variance, but that is w.r.t all signals. Before, make sure that signals
        # have same energy.
        rms = np.sqrt(np.mean(np.square(signal)))
        signal=signal/rms
        if featureType == 'mfcc':
            featureList.append(mfccFeatures(
                signal=signal,samplerate=samplerate,winlen=winlen,
                winstep=winstep, nfilt=nfilt,nfft=nfft,lowfreq=lowfreq,
                highfreq=highfreq,preemph=preemph, winSzForDelta=winSzForDelta, 
                numcep=numcep, ceplifter=ceplifter, appendEnergy=appendEnergy))
        elif featureType == 'logFB':
            featureList.append(logFilterbankFeatures(
                signal=signal,samplerate=samplerate,winlen=winlen,winstep=winstep,
                nfilt=nfilt,nfft=nfft,lowfreq=lowfreq,highfreq=highfreq,preemph=preemph,
                winSzForDelta=winSzForDelta))
        else:
            raise ValueError
    return np.array(featureList)


def getTargets(phnFilesList):
    '''
    This function is used to read the phoneme sequences from TIMIT .phn files
    :parameters:
        - phnFilesList: list of filepaths to .phn files
    :returns:
        - labelSequenceList: list of phonemes
            each element of labelSequenceList is again a list of strings (phoneme)
    '''
    labelSequenceList = []
    for f in phnFilesList:
        with open(f, 'r') as csvfile:
            tmpPhoneSeq = []
            reader = csv.reader(csvfile, delimiter = ' ')
            for _,_,phone in reader:
                tmpPhoneSeq.append(phone)
            labelSequenceList.append(tmpPhoneSeq)
    return labelSequenceList


def getTargetsFramewise(phnFilesList, listSequenceLengths, winstep, fs, winLen):
    '''
    This function is used to read the phoneme sequences from TIMIT .phn files for all timesteps
    :parameters:
        - phnFilesList: list of filepaths to .phn files
        - winstep: delta t (window step size) in seconds
        - fs: sampling frequency of raw data
        - lengths: list of frame-lengths
        - winLen: length of analysis window of training data
    :returns:
        - labelSequenceList: list of phonemes
            each element of labelSequenceList is again a list of strings (phoneme)
    '''
    labelSequenceList = []
    dSample = winstep*fs
    for index_seqLen in range(len(listSequenceLengths)):
        f = phnFilesList[index_seqLen]

        # create temporary list for phones and stops for file f.
        tmp_phone = []
        tmp_stop = []
        tmp_start = []
        with open(f,'r') as csvfile:
            tmpPhoneSeq = []
            reader = csv.reader(csvfile, delimiter = ' ')
            for start,stop,phone in reader:
                tmp_phone.append(phone)
                tmp_stop.append(int(stop))
                tmp_start.append(int(start))

        # use temporary lists to determine which phonemes to read for each frame
        currentSample = winLen/2.0*fs
        for length in range(listSequenceLengths[index_seqLen]):

            # find in tmp_stop the index of the (nearest) stop sample, that is greater than currentSample
            for i in range(len(tmp_stop)):
                if currentSample < tmp_stop[i] and currentSample > tmp_start[i]:
                    break # stop the search, we found the index --> i
            phone = tmp_phone[i]
            tmpPhoneSeq.append(phone)
            currentSample += dSample
        labelSequenceList.append(tmpPhoneSeq)
    return labelSequenceList


def getPhonemeDictionary():
    '''
    Use hard coded phoneme dict for timit, found here:
    http://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database
    Use hard coded, so that enumeration is the same --> makes sure h# is last entry and
    and mapping to fewer classes for scoring can be done analogous.
    '''
    phonemes= ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao',
               'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr','ax-h',
               'jh', 'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's',
               'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng',
               'em', 'nx', 'en', 'eng', 'l', 'r', 'w', 'y', 'hh', 'hv',
               'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau','epi',
               'h#']
    phonemeValues = range(0,61)
    phonemeDic = dict(zip(phonemes, phonemeValues))
    return phonemeDic



def getPhonemeMapForScoring():
    '''
    This function maps the 61 phones + blank from timit to the 39 phonemes that are commonly used.
    glottal stops, silence phones, etc. get each mapped to one class
    'q' is not used for scoring at all. it is assigned to None.
    Other functions 
    '''

    phonemes39 = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah',
                  'oy', 'ow', 'uh', 'uw', 'er',
                  'jh', 'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's',
                  'sh', 'z', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng',
                  'l', 'r', 'w', 'y', 'hh', 'sil']

    # construct dictionary with the 39 phonemes +blank and with with the 61 phonemes +blank for mapping
    dic39 = dict(zip(phonemes39, range(0, 39)))
    dic39[None] = None #needed for 'q'
    dic61 = getPhonemeDictionary()
    phonemes61 = dic61.keys()

    strMap = dict(zip(phonemes61, phonemes61))
    strMap['ao'] = 'aa'
    strMap['ax'] = 'ah'
    strMap['ax-h'] = 'ah'
    strMap['axr'] = 'er'
    strMap['hv'] = 'hh'
    strMap['ix'] = 'ih'
    strMap['el'] = 'l'
    strMap['em'] = 'm'
    strMap['en'] = 'n'
    strMap['nx'] = 'n'
    strMap['eng'] = 'ng'
    strMap['zh'] = 'sh'
    strMap['ux'] = 'uw'

    strMap['pcl'] = 'sil'
    strMap['tcl'] = 'sil'
    strMap['kcl'] = 'sil'
    strMap['bcl'] = 'sil'
    strMap['dcl'] = 'sil'
    strMap['gcl'] = 'sil'
    strMap['h#'] = 'sil'
    strMap['pau'] = 'sil'
    strMap['epi'] = 'sil'

    strMap['q'] = None  # this one shall not be scored!--> delete it

    # Now we have a dict for 61 phonemes and a dict for the 39 phonemes (+blank)
    # map integers from 61-phn dict to integers from 39-phn dict
    intMap = {}
    for str61 in strMap:
        str39 = strMap[str61]
        int61 = dic61[str61]
        int39 = dic39[str39]
        intMap[int61] = int39
    return intMap



def phonemeToInt(labelSequenceList, mapToSubset):
    '''
    Converts the list of phoneme Sequences (list of list of strings)
    into its equivalent values in the phonemeDictionary
    list of 1-D numpy arrays (values corresponding to phonemes)
    :parameters:
        - labelSequenceList: list of phoneme sequences
        - mapToSubset: bool, map 61 original phonemes to 39 for scoring?
    :returns:
        - y: numpy vector of sequences(numpy vector) in the form needed for training
            each element in vector is a 1d numpy array of integers
    '''
    phonemeDic = getPhonemeDictionary()
    scoreMap = getPhonemeMapForScoring()
    y = []
    for sequence in labelSequenceList:
        # convert from phoneme strings to sequence vector in form of label numbers
        if mapToSubset:
            convertedSequence = [scoreMap[phonemeDic[s]] for s in sequence]
        else:
            convertedSequence = [phonemeDic[s] for s in sequence]
        convertedSequence = np.array([x for x in convertedSequence if x is not None]).astype(np.float32)
        y.append(convertedSequence)
    return np.array(y)


def getPaths(timitRootDir,numSpeakersVal):
    '''
    Gets the paths for all training, validation and test files
    of the TIMIT database. Training and Test is already split
    in TIMIT, validation is created by taking randomly selected
    speakers from training set.
    All sentences from speakers, selected for the validation set
    are added to validation and deleted from training set.
    :parameters:
        - timitRootDir: path-string
            root directory to TIMIT database, that contains
            the test and train folder
        - numSpeakersVal: int
            the number of speakers which are used for the validation
            set. E.g. 50 speakers, each speaker has 5 sentences
            --> 250 sentences used for validation, where not any
            sentence of the speakers are known to training set
    :returns:
        - wav_train: list of filepaths to .wav files for training
        - wav_val: list of filepaths to .wav files for validation
        - wav_test: list of filepaths to .wav files for testing
        - phn_train: list of filepaths to .phn files for training
        - phn_val: list of filepaths to .phn files for validation
        - phn_test: list of filepaths to .phn files for testing
    '''
    trainPaths = glob.glob(timitRootDir+ '/train/'+'*/*')
    valPaths = random.sample(trainPaths, numSpeakersVal)
    for valPath in valPaths:
        trainPaths.remove(valPath)

    wav_train = glob.glob(timitRootDir+'/train/'+'*/*/*.wav')
    phn_train = []

    wav_test = glob.glob(timitRootDir+'/test/'+'*/*/*.wav')
    phn_test = []

    wav_val = []
    phn_val = []
    for f in wav_train:
        if f.rsplit('/',1)[0] in valPaths:
            wav_val.append(f)
            wav_train.remove(f)

    # remove .wav and append .phn for phoneme files
    for f in wav_train:
        phn_train.append(os.path.splitext(f)[0]+'.phn')
    for f in wav_val:
        phn_val.append(os.path.splitext(f)[0]+'.phn')
    for f in wav_test:
        phn_test.append(os.path.splitext(f)[0]+'.phn')

    return wav_train, wav_val, wav_test, phn_train, phn_val, phn_test



def normaliseFeatures(x_train, x_val, x_test, normalisation):
    """
    :prarams:
        - x_train: numpy vector (list) of training set
        - x_val: numpy vector (list) of validation set
        - x_test: numpy vector (list) of test set
        - normalisation: either 'trainData' or 'individual'
            Normalize over all training data or each feature individually?
    """  
    if normalisation=='trainData':
        return normaliseFeaturesOverTrainData(x_train, x_val, x_test)
    elif normalisation=='individual':
        return normaliseFeaturesIndividual(x_train, x_val, x_test)
    else:
        raise ValueError


def normaliseFeaturesOverTrainData(x_train, x_val, x_test):
    '''
    Normalises the training, validation and test set.
    Normalisation is done by subtracting the mean-vector and divding
    by std-deviation-vector of all training data.
    This is not the typical use-case
    Given numpy vector of training samples with dimension sequenceLength x feature_dim,
    mean and variance is calculated for every feature independently.
    :params:
        - x_train: numpy vector of training set
            used for calculating the mean and std-deviation
        - x_val: numpy vector of validation set
        - x_test: numpy vector of test set
    :returns:
        - x_train: numpy vector of normalised training set
            has zero mean, unit variance over training data, for each dim
        - x_val: numpy vector of normalised validation set
        - x_test: numpy vector of normalised test set
        - mean_vector: mean-vector
            calculated from training set. Has dimension equal to feature-dim
        - std_vector: std-deviation-vector
            calculated form training set. Has dimension equal to feature-dim
    '''
    stacked = np.vstack(x_train)
    mean_vector = np.mean(stacked,axis=0)
    std_vector = np.sqrt(np.var(stacked,axis=0))
    # normalize to zero mean and variance 1 and convert to float32 for GPU.
    # convert after normalization to ensure no precision is wasted.
    for it in range(len(x_train)):
        x_train[it] = (np.divide(np.subtract(x_train[it],mean_vector),std_vector)).astype(np.float32)
    for it in range(len(x_val)):
        x_val[it] = np.divide(np.subtract(x_val[it],mean_vector),std_vector).astype(np.float32)
    for it in range(len(x_test)):
        x_test[it] = np.divide(np.subtract(x_test[it],mean_vector),std_vector).astype(np.float32)

    return x_train, x_val, x_test
    

def normaliseFeaturesIndividual(x_train, x_val, x_test):
    '''
    Normalises the training, validation and test set.
    Normalisation is done by subtracting the mean-vector and divding
    by std-deviation-vector of data. This is done per utterance
    Given numpy vector of training samples with dimension sequenceLength x feature_dim,
    mean and variance is calculated for every feature independently.
    :params:
        - x_train: numpy vector of training set
            used for calculating the mean and std-deviation
        - x_val: numpy vector of validation set
        - x_test: numpy vector of test set
    :returns:
        - x_train: numpy vector of normalised training set
            has zero mean, unit variance over training data, for each dim
        - x_val: numpy vector of normalised validation set
        - x_test: numpy vector of normalised test set
        - mean_vector: mean-vector
            calculated from training set. Has dimension equal to feature-dim
        - std_vector: std-deviation-vector
            calculated form training set. Has dimension equal to feature-dim
    '''
    # normalize to zero mean and variance 1 and convert to float32 for GPU.
    # convert after normalization to ensure no precision is wasted.
    
    for it in range(len(x_train)):
        mean_vector = np.mean(x_train[it],axis=0)
        std_vector = np.sqrt(np.var(x_train[it],axis=0))
        x_train[it] = (np.divide(np.subtract(x_train[it],mean_vector),std_vector)).astype(np.float32)
    for it in range(len(x_val)):
        mean_vector = np.mean(x_val[it],axis=0)
        std_vector = np.sqrt(np.var(x_val[it],axis=0))
        x_val[it] = np.divide(np.subtract(x_val[it],mean_vector),std_vector).astype(np.float32)
    for it in range(len(x_test)):
        mean_vector = np.mean(x_test[it],axis=0)
        std_vector = np.sqrt(np.var(x_test[it],axis=0))
        x_test[it] = np.divide(np.subtract(x_test[it],mean_vector),std_vector).astype(np.float32)

    return x_train, x_val, x_test


def getLongestSequence(train_set, val_set, test_set):
    '''
    Return the longest sequence from train set and val set.
    :parameters:
        - train_set: numpy vector of numpy.ndarray
        - val_set: numpy vector of numpy.ndarray
    :returns:
        - integer value for length of longest sequence
    '''
    return max(max([X.shape[0] for X in train_set]),
             max([X.shape[0] for X in val_set]),
             max([X.shape[0] for X in test_set]))


def padAndReshape(X, y, X_padLength, y_padLength):
    """
    input is list or 1-dim np array, containing numpy 2-d arrays with dimensions
    sequence-length x feature-dim. sequence-length can be different for
    each individual entry of X. This function shall 1. pad all entries with
    zeros to have the same sequence-length and reshape the input X
    to have dimension len(X) x MAX-sequence-length x feature-dim
    """
    X_padResh = np.zeros((len(X), X_padLength, X[0].shape[1]),dtype=np.float32)
    X_padResh_mask = np.zeros((len(X), X_padLength),dtype=np.float32)
    y_padResh = np.zeros((len(y), y_padLength),dtype=np.float32)
    y_padResh_mask = np.zeros((len(y), y_padLength),dtype=np.float32)

    for i, x_i in enumerate(X):
        X_padResh[i,0:x_i.shape[0],:] = x_i
        X_padResh_mask[i,0:x_i.shape[0]] = 1.0
    for i, y_i in enumerate(y):
        y_padResh[i,0:y_i.shape[0]] = y[i]
        y_padResh_mask[i,0:y_i.shape[0]] = 1.0
    return X_padResh, y_padResh, X_padResh_mask, y_padResh_mask
    
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