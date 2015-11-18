from __future__ import division
from __future__ import print_function
import scipy.io
import os
import cochlea
from thorns import waves as wv
import time
import featureExtraction
import soundfile as sf

"""
Instead of using command line optional parameters, define the parameters as constants.
Note that species is always set to human, random seed is chosen automatically (depends on current time)
and CF cannot be a single value (no use for speech recognition). All other parameters can be specified below.
"""
#********************************* Parameters **************************************************
# Total number of nerve fibers to be modeled
FIBERS_TOTAL = 310
# ratios of each fiber type
HSR_RATIO, MSR_RATIO, LSR_RATIO = 0.61, 0.16, 0.23
# Number of high-, medium and low-spontaneous rate fibers per frequency
HSR, MSR, LSR = \
int(round(HSR_RATIO*FIBERS_TOTAL)), int(round(MSR_RATIO*FIBERS_TOTAL)), int(round(LSR_RATIO*FIBERS_TOTAL))
# Center frequency (CF) of a single frequency channel
CF = 10
# Min, Max and number of CF in a the frequency map (calculated with a Greenwood function)

CF_MIN, CF_MAX, CF_NUM = 200, 6000, 100
# Sound level applied to the sound before entering the inner ear
dB_SPL = 60
#***********************************************************************************************

#def convert_sound_to_mat(
#        sound_file,
#        anf_num,
#        cf,
#        species,
#        seed,
#        dbspl
#):
#
#    print("Processing " + sound_file)
#
#    fs = 100e3
#
#
#    ### Read the sound file + resample + scale
#    f = audiolab.Sndfile(sound_file, 'r')
#    sound_raw = f.read_frames(f.nframes)
#    sound_raw = wv.resample(sound_raw, f.samplerate, fs)
#    if dbspl is not None:
#        sound = wv.set_dbspl(sound_raw, dbspl)
#    else:
#        sound = sound_raw
#
#    ### Run the inner ear model
#    anf_trains = cochlea.run_zilany2014(
#        sound=sound,
#        fs=fs,
#        anf_num=anf_num,
#        cf=cf,
#        species=species,
#        seed=seed
#    )
#
#    ### Save spikes to matlab file
#    trains = anf_trains.to_records()
#    mat_fname = os.path.splitext(sound_file)[0]
#    mdict = {'trains': trains}
#
#    scipy.io.savemat(
#        mat_fname,
#        mdict,
#        do_compression=True
#    )


# Have trouble installing audiolab + timit's wav have different header, it's very probable,
# that audiolab cannot read it. Use soundfile instead
def convert_sound_to_mat_timit(sound_file, anf_num, cf, species, seed, dbspl):
    print("Processing " + sound_file)
    fs = 100e3
    ### Read the sound file + resample + scale
    sound_raw, sound_fs = sf.read(sound_file)
    sound_raw = sound_raw[:,0] # sf give dimensions N x 1 (1 channel), but need just dim N
    sound_raw = wv.resample(sound_raw, sound_fs, fs)
    if dbspl is not None:
        sound = wv.set_dbspl(sound_raw, dbspl)
    else:
        sound = sound_raw
        
    ### Run the inner ear model
    anf_trains = cochlea.run_zilany2014(
        sound=sound,
        fs=fs,
        anf_num=anf_num,
        cf=cf,
        species=species,
        seed=seed
    )
    
    ### Save spikes to matlab file
    trains = anf_trains.to_records()
    mat_fname = os.path.splitext(sound_file)[0]
    mdict = {'trains': trains}

    scipy.io.savemat(
        mat_fname,
        mdict,
        do_compression=True
    )


def convert_sound_to_mat_unpack(args):
    """Unpack the args (dict) and pass to convert_sound_to_mat()"""
    convert_sound_to_mat_timit(**args)



def main():
    start_time = time.time()
    species = 'human' # for this script, test only human...
    seed = int(round(time.time()) % 42) # seed = random number between 0 and "42"
    #copied TIMIT, dunno what zilany does with the files... no idea where it saves outputs etc!
    timit_root_dir = '../../TIMIT_Zilany/timit' 
    n_speaker_val = 50 # number of speakers in validation set
    anf_num = HSR,MSR,LSR
    cf = float(CF_MIN),float(CF_MAX),float(CF_NUM)
    dbspl = dB_SPL
    
    wav_train, wav_val, wav_test, phn_train, phn_val, phn_test = \
        featureExtraction.getTrainValTestPaths(timit_root_dir, n_speaker_val)
    sound_files = [wav_train[0]]


    # Prepare a list of dict arguments.  Each dict contains
    # parameters for the processing function (convert_sound_to_mat)
    space = [
        {
            'anf_num': anf_num,
            'cf': cf,
            'species': species,
            'seed': seed,
            'dbspl': dbspl,
            'sound_file': sound_file
        }
        for sound_file in sound_files
    ]

    ### Apply the function to each parameter dict
    map(
        convert_sound_to_mat_unpack,
        space
    )
    stop_time = time.time()
    print("time = " + str(stop_time-start_time))





if __name__ == "__main__":
    main()