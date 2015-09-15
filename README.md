# Attempt to implement deep BLSTM with Lasagne

The network is applied to phoneme recognition on the TIMIT dataset. 

Still under development... achieve phone error rate of not better than 40% with 1 layer BLSTM and CTC

requirements: 
	- Theano + Lasagne
	- netCDF + netCDF4-Python to read and store data (same is used in CURRENNT library)
	- PySoundFile to read timit's depcrecated .wav-like format, See: http://pysoundfile.readthedocs.org/en/0.7.0/ and https://github.com/bastibe/PySoundFile
	- python_speech_features for preprocessing (FFT-based filterbank), see http://python-speech-features.readthedocs.org/en/latest/ and https://github.com/jameslyons/python_speech_features
	- Lasagne-CTC

TODO:
1) beamsearch decoding of networks output sequences
