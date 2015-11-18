"""
Using paths and parameters, specified in conf.py, this script
computes features from the Timit dataset and stores them in an hdf5 file
@author: Richard Kurle
"""
from TimitFeatureExtraction import *
import conf
import h5py
from fuel.datasets.hdf5 import H5PYDataset

nfft = nextpow2(conf.samplerate*conf.winlen)
# file paths to all .wav and .phn files, split into train val test
wav_train, wav_val, wav_test, phn_train, phn_val, phn_test = getPaths(
    conf.path_to_timitRoot, conf.n_speaker_val)

# Calculate features in train, val and test set
# TODO: Maybe make dictionoray and drop arguments with **kwargs?
X_train = getAllFeatures(conf.featureType, wav_train, conf.samplerate, conf.winlen, 
                         conf.winstep, conf.nfilt, nfft, conf.lowfreq, 
                         conf.highfreq, conf.preemph,
                         conf.winSzForDelta, conf.numcep, conf.ceplifter, True)
X_val = getAllFeatures(conf.featureType, wav_val, conf.samplerate, conf.winlen, 
                       conf.winstep, conf.nfilt, nfft, conf.lowfreq, 
                       conf.highfreq, conf.preemph,
                       conf.winSzForDelta, conf.numcep, conf.ceplifter, True)
X_test = getAllFeatures(conf.featureType, wav_test, conf.samplerate, conf.winlen, 
                       conf.winstep, conf.nfilt, nfft, conf.lowfreq, 
                       conf.highfreq, conf.preemph,
                       conf.winSzForDelta, conf.numcep, conf.ceplifter, True)
                       
X_train, X_val, X_test = normaliseFeatures(X_train, X_val, X_test, conf.normalisation)

if conf.task=='CTC':
    labelSequence_train = getTargets(phn_train)
    labelSequence_val = getTargets(phn_val)
    labelSequence_test = getTargets(phn_test)

elif conf.task=='framewise':
    labelSequence_train = getTargetsFramewise(
        phn_train, [len(s) for s in X_train], conf.winstep, conf.samplerate, conf.winlen)
    labelSequence_val = getTargetsFramewise(
        phn_val, [len(s) for s in X_val], conf.winstep, conf.samplerate, conf.winlen)
    labelSequence_test = getTargetsFramewise(
        phn_test, [len(s) for s in X_test], conf.winstep, conf.samplerate, conf.winlen)
else:
    raise ValueError, conf.task

y_train = phonemeToInt(labelSequence_train, mapToSubset=conf.mapTo39Phonemes_Training)
y_val = phonemeToInt(labelSequence_val, mapToSubset=conf.mapTo39Phonemes_Training)
y_test = phonemeToInt(labelSequence_test, mapToSubset=conf.mapTo39Phonemes_Training)

maxOutputLength = getLongestSequence(y_train, y_val, y_test)
maxInputLength = getLongestSequence(X_train, X_val, X_test)
assert maxOutputLength == maxInputLength or conf.task=='CTC', \
    'max input sequ length != max output sequ length, using framewise'

X_train, y_train, X_train_mask, y_train_mask = padAndReshape(
    X_train, y_train, maxInputLength, maxOutputLength)
X_val, y_val, X_val_mask, y_val_mask = padAndReshape(
    X_val, y_val, maxInputLength, maxOutputLength)
X_test, y_test, X_test_mask, y_test_mask = padAndReshape(
    X_test, y_test, maxInputLength, maxOutputLength)

numSeqs_train = len(X_train)
numSeqs_val = len(X_val)
numSeqs_test = len(X_test)
numSeqs = numSeqs_train + numSeqs_val + numSeqs_test

X = np.vstack((X_train, X_val, X_test))
X_mask = np.vstack((X_train_mask, X_val_mask, X_test_mask))
y = np.vstack((y_train, y_val, y_test))
y_mask = np.vstack((y_train_mask,y_val_mask,y_test_mask))

# model featues dim
inputPattSize = X.shape[-1]

if conf.task=='framewise':
    path_to_dataset = conf.path_to_dataset_framewise 
elif conf.task=='CTC':
    path_to_dataset = conf.path_to_dataset_CTC
else:
    raise ValueError    
    
# create path if it doesnt exist
if not os.path.exists(os.path.split(path_to_dataset)[0]):
    os.makedirs(os.path.split(path_to_dataset)[0])
    
with h5py.File(path_to_dataset, mode="w") as f:
# specify dataset dims 
    f_X = f.create_dataset( \
        conf.input_theano, (numSeqs, maxInputLength, inputPattSize), dtype='float32')
    f_X.dims[0].label = 'sequence'
    f_X.dims[1].label = 'timestep'
    f_X.dims[2].label = 'feature'

    f_y = f.create_dataset( \
        conf.target_theano, (numSeqs, maxOutputLength), dtype='int64')
    f_y.dims[0].label = 'sequence'
    f_y.dims[1].label = 'timestep'

    f_X_mask = f.create_dataset( \
        conf.input_mask_theano, (numSeqs, maxInputLength), dtype='float32')
    f_X_mask.dims[0].label = 'sequence'
    f_X_mask.dims[1].label = 'timestep'
    
    f_y_mask = f.create_dataset(\
        conf.target_mask_theano, (numSeqs, maxOutputLength), dtype='float32')
    f_y_mask.dims[0].label = 'sequence'
    f_y_mask.dims[1].label = 'timestep'

# load in actual values
    f_X[...] = X
    f_y[...] = y
    f_X_mask[...] = X_mask
    f_y_mask[...] = y_mask

    split_dict = {
        'train': {conf.input_theano: (0, numSeqs_train),
                  conf.input_mask_theano: (0, numSeqs_train),
                  conf.target_theano: (0, numSeqs_train),
                  conf.target_mask_theano: (0, numSeqs_train)},

        'val': {conf.input_theano: (numSeqs_train, numSeqs_train + numSeqs_val),
                conf.input_mask_theano: (numSeqs_train, numSeqs_train + numSeqs_val),
                conf.target_theano: (numSeqs_train, numSeqs_train + numSeqs_val),
                conf.target_mask_theano: (numSeqs_train, numSeqs_train+numSeqs_val)},

        'test': {conf.input_theano: (numSeqs_train + numSeqs_val, numSeqs_train + numSeqs_val + numSeqs_test),
                 conf.input_mask_theano: (numSeqs_train + numSeqs_val, numSeqs_train + numSeqs_val + numSeqs_test),
                 conf.target_theano: (numSeqs_train + numSeqs_val, numSeqs_train + numSeqs_val + numSeqs_test),
                 conf.target_mask_theano: (numSeqs_train+numSeqs_val,numSeqs_train+numSeqs_val+numSeqs_test)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
print('done')