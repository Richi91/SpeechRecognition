# BiRNN in Blocks trained with CTC on TIMIT

Trainable with CTC on phoneme transcription or framewise. 

Implementation both in Blocks and Lasagne. Note that the Lasagne implementation is at the moment a bit messed up and will be deleted or updated soon.
Use the blocks implementation instead.

requirements: 
	- Theano: http://deeplearning.net/software/theano/install.html
	-  Blocks: http://blocks.readthedocs.org/en/latest/setup.html
	- Blocks extras: https://github.com/mila-udem/blocks-extras
	- Fuel: http://fuel.readthedocs.org/en/latest/setup.html
	- PySoundFile to read timit's depcrecated .wav-like format, See: 
http://pysoundfile.readthedocs.org/en/0.7.0/ and https://github.com/bastibe/PySoundFile
	- python_speech_features for preprocessing (FFT-based filterbank), see 
	http://python-speech-features.readthedocs.org/en/latest/ + https://github.com/jameslyons/python_speech_features


#Notes: 
- Decoding: simple argmax, no expensive beamsearch
- Mapping from original 61 to reduced 39 Phonemes can be done before training or during decoding.
	Mapping during decoding is apparently done wrong atm. I fold the 61 output distributions after softmax to 39, by adding up. This however produces accuracies


----------

**3 layer BiRNN with [300,250,200] hidden units, batch size 40, AdaDelta, mapping to 39 classes before training:**

 GRU on MFCC features: 19.5% PER
- GRU on Log-FB features: 20.5% PER
- LSTM on MFCC features: 19.5% PER
- LSTM on Log-FB features: ?


#Credits
CTC Implementation: ctc_cost.py is copied from Philemon Brakel's repository: 
	https://github.com/pbrakel/CTC-Connectionist-Temporal-Classification
