# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:27:04 2015

@author: richi-ubuntu
"""

from features import fbank
import numpy as np
import glob
import os
import soundfile as sf
import csv
import random
import netCDF4
import utils
from netCDF4 import Dataset




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
    # init deltafeat with zeros, 2xdeltawin more timesteps, because on both ends needed for calculating differences
    deltafeat = np.zeros([feat.shape[0]+2*deltawin,feat.shape[1]])
    # repeat the values at the borders is default behaviour of HTK --> adapt it here aswell instead of zero padding
    pad_bot = np.tile(feat[0,:],(deltawin,1))
    pad_top = np.tile(feat[-1,:],(deltawin,1))
    padfeat = np.concatenate((pad_bot,feat,pad_top),axis=0)
    
    norm = 2.0 * sum([it**2 for it in range(1, deltawin+1)])
    deltafeat = sum([it*(np.roll(padfeat,-2,axis=0)-np.roll(padfeat,2,axis=0)) for it in range(1, deltawin+1)])
    
    deltafeat /= norm;
    return deltafeat[deltawin:padfeat.shape[0]-deltawin,:]
 
   
def getFeatures(signal, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq, preemph, winSzForDelta):
    '''
    Computes filterbank energies on a mel scale + total energy using 'fbank'
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
    # calculate fbank energies and total energy
    feat,energy = fbank(signal, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq,preemph)
    feat = np.column_stack((energy,feat))
    # calculate delta and acceleration
    deltaFeat = delta(feat, winSzForDelta)
    accFeat = delta(deltaFeat, winSzForDelta)
    # stack features + delta + acceleration
    return np.concatenate((feat,deltaFeat,accFeat),axis=1)


def getAllFeatures(wavFileList, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq, preemph, winSzForDelta):
    '''
    Computes all features of a given numpy vector of file paths to .wav files. Reads the
    wav files specified in 'wavFileList' the package 'PySoundFile'. 
    PySoundFile is able to read the format of the files from TIMIT database.
    See: http://pysoundfile.readthedocs.org/en/0.7.0/ and
    https://github.com/bastibe/PySoundFile
    
    For other parameters see function getFeatures, once signal is read from path,
    signal and other parameters are forwarded to 'getFeatures'
    :parameters:
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
        featureList.append(getFeatures( \
            signal/rms, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq,preemph,winSzForDelta)) 
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


def getTargetsEachTimestep(phnFilesList, listSequenceLengths, winstep, fs, winLen):
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
    Use hard coded phoneme dict for timit found on: 
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



        
    
def convertPhoneSequences(labelSequenceList,phonemeDic):
    '''
    Converts the list of phoneme Sequences (list of list of strings) 
    into its equivalent values in the phonemeDictionary
    list of 1-D numpy arrays (values corresponding to phonemes)
    :parameters:
        - labelSequenceList: list of phoneme sequences
        - phonemeDic: dictionary of phonemes and corresponding values
    :returns:
        - y: numpy vector of sequences(numpy vector) in the form needed for training
            each element in vector is a 1d numpy array of integers 
    '''
    y = []
    #n_classes = len(phonemeDic)
    for sequence in labelSequenceList:
        # convert from phoneme strings to sequence vector in form of label numbers
        convertedSequence = np.array([phonemeDic[h] for h in sequence]).astype(np.float32)
        #y.append(one_hot(convertedSequence,n_classes))
        y.append(convertedSequence)
    return np.array(y)
    
        
def getTrainValTestPaths(timitRootDir,numSpeakersVal):
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


def normaliseFeaturesOverAllData(x_train, x_val, x_test):
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
    
    return x_train, x_val, x_test, mean_vector, std_vector


def normaliseFeatures(x_train, x_val, x_test):
    '''
    Normalises the training, validation and test set.
    Normalisation is done by subtracting the mean-vector and divding
    by std-deviation-vector of training data. This is done per utterance
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
    
    return x_train, x_val, x_test, mean_vector, std_vector
    
    
def getLongestSequence(train_set, val_set):
    '''
    Return the longest sequence from train set and val set.
    :parameters:
        - train_set: numpy vector of numpy.ndarray
        - val_set: numpy vector of numpy.ndarray
    :returns:
        - integer value for length of longest sequence
    '''
    return max(max([X.shape[0] for X in train_set]),
             max([X.shape[0] for X in val_set]))



def CreateNetCDF(useCTC):
    """
    TODO: add some description
    """
    rootdir = os.getenv("HOME") + '/data/TIMIT/timit' # root directory of timit
    dataPath = os.getenv("HOME") + '/data/TimitFeat' # store Path
    if useCTC:
        dataPath+= '/CTC'
    else:
        dataPath+= '/noCTC'
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    
    winlen, winstep, nfilt, lowfreq, highfreq, preemph, winSzForDelta, samplerate = \
    0.025,  0.01,   40,     200,     8000,     0.97,    2,             16000           
    nfft = utils.nextpow2(samplerate*winlen) 
    n_speaker_val = 50 
    
    wav_train, wav_val, wav_test, phn_train, phn_val, phn_test = getTrainValTestPaths(rootdir,n_speaker_val)
    
    X_train = getAllFeatures(wav_train, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq,preemph,winSzForDelta)
    X_val = getAllFeatures(wav_val, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq,preemph,winSzForDelta)
    X_test = getAllFeatures(wav_test, samplerate, winlen, winstep, nfilt,nfft, lowfreq, highfreq,preemph,winSzForDelta)
    
    X_train, X_val, X_test, mean_vector, std_vect = normaliseFeatures(X_train, X_val, X_test)
    
    if useCTC:
        labelSequence_train = getTargets(phn_train)
        labelSequence_val = getTargets(phn_val)
        labelSequence_test = getTargets(phn_test)
    else:
        labelSequence_train = getTargetsEachTimestep(phn_train, [len(s) for s in X_train], winstep, samplerate, winlen)
        labelSequence_val = getTargetsEachTimestep(phn_val, [len(s) for s in X_val], winstep, samplerate, winlen)
        labelSequence_test = getTargetsEachTimestep(phn_test, [len(s) for s in X_test], winstep, samplerate, winlen)
    
    phonemeDic = getPhonemeDictionary()
    
    y_train = convertPhoneSequences(labelSequence_train, phonemeDic)
    y_val = convertPhoneSequences(labelSequence_val, phonemeDic)
    y_test = convertPhoneSequences(labelSequence_test, phonemeDic)
    
    numSeqs_train = len(X_train)
    numSeqs_val = len(X_val)
    numSeqs_test = len(X_test)
    numTimesteps_train = sum(len(s) for s in X_train)
    numTimesteps_val = sum(len(s) for s in X_val)
    numTimesteps_test = sum(len(s) for s in X_test)
    inputPattSize = 123
    maxSeqTagLength = 100 # This will be the name of the files. path from timit to file
    maxLabelLength = getLongestSequence(y_train,y_val)
    maxInputLength = getLongestSequence(X_train,X_val)
    if useCTC:
        numLabels = 62
    else:
    #    numLabels = 61 #uncommented
        numLabels = 62

    train = Dataset(dataPath+'/train.nc', 'w', format='NETCDF4')
    try:
        # create dimension
        train.createDimension('numSeqs', numSeqs_train)
        train.createDimension('numTimesteps', numTimesteps_train)
        train.createDimension('inputPattSize', inputPattSize)
        train.createDimension('numLabels', numLabels)
        train.createDimension('maxLabelLength', maxLabelLength)
        train.createDimension('maxSeqTagLength', maxSeqTagLength)
        train.createDimension('maxInputLength', maxInputLength)
        # create variables   
        seqTags = train.createVariable('seqTags','S1',('numSeqs', 'maxSeqTagLength',))
        seqLengths = train.createVariable('seqLengths','i4',('numSeqs',))
        inputs = train.createVariable('inputs','f4',('numTimesteps','inputPattSize',))
        targetClasses = train.createVariable('targetClasses','i4',('numTimesteps',))
        
        seqStartIndices = train.createVariable('seqStartIndices','i4',('numSeqs',))
        labelLengths = train.createVariable('labelLengths','i4',('numSeqs',))
        labelStartIndices = train.createVariable('labelStartIndices','i4',('numSeqs',))
        
        # write into variables    
        index = 0
        for s in wav_train:
            tmp_str = s.split('train',1)[1]
            tmp_str = np.array(tmp_str.split(","))
            tmp_str = netCDF4.stringtochar(tmp_str)
            seqTags[index,0:tmp_str.shape[1]] = tmp_str
            index += 1
           
        seqLengths[:] = [len(s) for s in X_train]
        labelLengths[:] = [len(s) for s in y_train]
            
        index = 0
        for counter, s in enumerate(X_train):
            inputs[index:index+len(s),:] = s
            seqStartIndices[counter] = index
            index += len(s)
            
            
        index = 0     
        for counter, s in enumerate(y_train):
            targetClasses[index:index+len(s)] = s
            labelStartIndices[counter] = index
            index += len(s) 
    
    except:
        print '\n\nfail train\n\n'
        pass
    train.close()
    
    
    
    
    val = Dataset(dataPath+'/val.nc', 'w', format='NETCDF4')
    try:
        # create dimension
        val.createDimension('numSeqs', numSeqs_val)
        val.createDimension('numTimesteps', numTimesteps_val)
        val.createDimension('inputPattSize', inputPattSize)
        val.createDimension('numLabels', numLabels)
        val.createDimension('maxLabelLength', maxLabelLength)
        val.createDimension('maxSeqTagLength', maxSeqTagLength)
        val.createDimension('maxInputLength', maxInputLength)
        # create variables   
        seqTags = val.createVariable('seqTags','S1',('numSeqs', 'maxSeqTagLength',))
        seqLengths = val.createVariable('seqLengths','i4',('numSeqs',))
        seqStartIndices = train.createVariable('seqStartIndices','i4',('numSeqs',))
        labelLengths = train.createVariable('labelLengths','i4',('numSeqs',))
        labelStartIndices = train.createVariable('labelStartIndices','i4',('numSeqs',))
        inputs = val.createVariable('inputs','f4',('numTimesteps','inputPattSize',))
        targetClasses = val.createVariable('targetClasses','i4',('numTimesteps',))
        
        # write into variables    
        index = 0
        for s in wav_val:
            tmp_str = s.split('train',1)[1]
            tmp_str = np.array(tmp_str.split(","))
            tmp_str = netCDF4.stringtochar(tmp_str)
            seqTags[index,0:tmp_str.shape[1]] = tmp_str
            index += 1
                 
        seqLengths[:] = [len(s) for s in X_val]
        labelLengths[:] = [len(s) for s in y_val]
        
        index = 0
        for counter,s in enumerate(X_val):
            inputs[index:index+len(s),:] = s
            seqStartIndices[counter] = index
            index += len(s)
            
        index = 0 
        # targets for every timestep --> no CTC
        for counter, s in enumerate(y_val):
            targetClasses[index:index+len(s)] = s
            labelStartIndices[counter] = index
            index += len(s) 
    
    except:
        print '\n\nfail val\n\n'
        pass
    val.close()
    
    
    
    test = Dataset(dataPath+'/test.nc', 'w', format='NETCDF4')
    try:
        # create dimension
        test.createDimension('numSeqs', numSeqs_test)
        test.createDimension('numTimesteps', numTimesteps_test)
        test.createDimension('inputPattSize', inputPattSize)
        test.createDimension('numLabels', numLabels)
        test.createDimension('maxLabelLength', maxLabelLength)
        test.createDimension('maxSeqTagLength', maxSeqTagLength)
        test.createDimension('maxInputLength', maxInputLength)
        # create variables   
        seqTags = test.createVariable('seqTags','S1',('numSeqs', 'maxSeqTagLength',))
        seqLengths = test.createVariable('seqLengths','i4',('numSeqs',))
        seqStartIndices = train.createVariable('seqStartIndices','i4',('numSeqs',))
        labelLengths = train.createVariable('labelLengths','i4',('numSeqs',))
        labelStartIndices = train.createVariable('labelStartIndices','i4',('numSeqs',))
        inputs = test.createVariable('inputs','f4',('numTimesteps','inputPattSize',))
        targetClasses = test.createVariable('targetClasses','i4',('numTimesteps',))
        
        # write into variables    
        index = 0
        for s in wav_test:
            tmp_str = s.split('test',1)[1]
            tmp_str = np.array(tmp_str.split(","))
            tmp_str = netCDF4.stringtochar(tmp_str)
            seqTags[index,0:tmp_str.shape[1]] = tmp_str
            index += 1
                 
        seqLengths[:] = [len(s) for s in X_test]
        labelLengths[:] = [len(s) for s in y_test]
            
        index = 0
        for counter,s in enumerate(X_test):
            inputs[index:index+len(s),:] = s
            seqStartIndices[counter] = index
            index += len(s)
                    
        index = 0       
        # targets for every timestep --> no CTC
        for counter, s in enumerate(y_test):
            targetClasses[index:index+len(s)] = s
            labelStartIndices[counter] = index
            index += len(s) 
    
    except:
        print '\n\nfail test\n\n'
        pass
    test.close()  




if __name__ == '__main__':
    # adapt useCTC to decide if CTC will be used for cost. DIfference is training target sequence length
    useCTC = True
    
    print "using CTC: "+str(useCTC)
    CreateNetCDF(useCTC)
