# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:32:50 2015

@author: richi-ubuntu
"""
import numpy as np
import featureExtraction


def decodeSequence(sequence_probdist, mask, blanksymbol):
    """
    This function decodes the output sequence from the network, which has the same length
    as the input sequence. This one takes just the most probable label and removes blanks and same phonemes
    """
    # just for testing, brute-force take output with highest prob and then eleminate repeated labels+blanks
    mostProbSeq = sequence_probdist[mask==1].argmax(axis=1)
    reduced = mostProbSeq[mostProbSeq!=blanksymbol]
    reduced = np.array([seq for index, seq in enumerate(reduced) \
        if (reduced[index] != reduced[index-1] or index == 0)])
    return reduced
    

def decodeSequenceNoCTC(sequence_probdist, mask):
    """
    This function decodes each timestep by outputting the label with the highest probability,
    given an output distribution
    :parameters:
        - sequence_probdist: numpy array of output distribution num_seq x output_dim
        - mask: mask for marking which elements in sequence_probdist are valid
    """
    return np.array(sequence_probdist[mask==1].argmax(axis=1))
    

#    
#def beamsearch(sequence):
#    T = len(sequence)
#    #Initalise: B = {∅}; Pr(∅) = 1
#    B = [[]] #list of lists --> list of sequences! take indices of sequences to access Pr(...)
#    Pr = [1]
#    for t in range (1, T):
#        A = B
#        B = [[]]
#        for y_index, y in enumerate(A):
#            Pr(y_index) += Pyˆ∈pref(y)∩A Pr(yˆ) Pr(y|yˆ, t)
#    
#        while B contains less than W elements moreprobable than the most probable in A:
#            y∗ = most probable in A
#            Remove y∗from A
#            Pr(y∗) = Pr(y∗) Pr(∅|y, t)
#            Add y∗to B
#            for k ∈ Y:
#                Pr(y∗ + k) = Pr(y∗) Pr(k|y∗, t)
#                Add y∗ + k to A
#        Remove all but the W most probable from B
#        
#    return y with highest log Pr(y)/|y| in B
        
def beamsearch(cost, extra, initial, B, E):
	"""A breadth-first beam search.
	B = max number of options to keep,
	E = max cost difference between best and worst threads in beam.
	initial = [ starting positions ]
	extra = arbitrary information for cost function.
	cost = fn(state, extra) -> (total_cost, [next states], output_if_goal)
 
     THIS FUNCTION IS HERE JUST FOR COMPARISON; WILL NEED OWN IMPLEMENTATION OF BEAMSEARCH
	"""

	o = []
	B = max(B, len(initial))
	hlist = [ (0.0, tmp) for tmp in initial ]
	while len(hlist)>0:
		# print "Len(hlist)=", len(hlist), "len(o)=", len(o)
		hlist.sort()
		if len(hlist) > B:
			hlist = hlist[:B]
		# print "E=", hlist[0][0], " to ", hlist[0][0]+E
		hlist = filter(lambda q, e0=hlist[0][0], e=E: q[0]-e0<=e, hlist)
		# print "		after: Len(hlist)=", len(hlist)
		nlist = []
		while len(hlist) > 0:
			c, point = hlist.pop(0)
			newcost, nextsteps, is_goal = cost(point, extra)
			if is_goal:
				o.append((newcost, is_goal))
			for t in nextsteps:
				nlist.append((newcost, t))
		hlist = nlist
	o.sort()
	return o   




def calcPER(tar, out):
    """
    This function calculates the Phoneme Error Rate (PER) of the decoded networks output
    sequence (out) and a target sequence (tar) with Levenshtein distance and dynamic programming.
    This is the same algorithm as commonly used for calculating the word error rate (WER)
        :parameters:
        - tar: target output
        - out: network output (decoded)
    :returns:
        - phoneme error rate
    """
    # initialize dynammic programming matrix
    D = np.zeros((len(tar)+1)*(len(out)+1), dtype=np.uint16)
    D = D.reshape((len(tar)+1, len(out)+1))
    # fill border entries, horizontals with timesteps of decoded networks output
    # and vertical with timesteps of target sequence.
    for t in range(len(tar)+1):
        for o in range(len(out)+1):
            if t == 0:
                D[0][o] = o
            elif o == 0:
                D[t][0] = t
                
    # compute the distance by calculating each entry successively. 
    # 
    for t in range(1, len(tar)+1):
        for o in range(1, len(out)+1):
            if tar[t-1] == out[o-1]:
                D[t][o] = D[t-1][o-1]
            else:
                # part-distances are 1 for all 3 possible paths (diag,hor,vert). 
                # Each elem of distance matrix D represents the accumulated part-distances
                # to reach this location in the matrix. Thus the distance at location (t,o)
                # can be calculated from the already calculated distance one of the possible 
                # previous locations(t-1,o), (t-1,o-1) or (t,o-1) plus the distance to the
                # desired new location (t,o). Since we are interested only in the shortes path,
                # take the shortes (min)
                substitution = D[t-1][o-1] + 1 # diag path
                insertion    = D[t][o-1] + 1 # hor path
                deletion     = D[t-1][o] + 1 # vert path
                D[t][o] = min(substitution, insertion, deletion)
    # best distance is bottom right entry of Distance-Matrix D.
    return float(D[len(tar)][len(out)])/len(tar)
    
def calcPERNoCTC(tar,out):
    """
    This function calculates the phoneme-error-rate, when not using CTC, but having a network output
    for every input. just compares target output (tar) and actual output (out)
    :parameters:
        - tar: target output
        - out: network output (decoded)
    :returns:
        - phoneme error rate
    """
    return (tar!=out).mean()       
    
    
    
    
def getPhonemeMapForScoring():
    '''
    This function maps the 61 phones from timit to the 39 phonemes that are commonly used.
    glottal stops, silence phones, etc. get each mapped to one class (or deleted (q))
    '''
    
    phonemes39 = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah',
               'oy', 'ow', 'uh', 'uw', 'er',
               'jh', 'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's',
               'sh', 'z', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng',
               'l', 'r', 'w', 'y', 'hh', 'sil']
               
    # construct dictionary with the 39 phonemes and with with the 61 phonemes for mapping
    dic39 = dict(zip(phonemes39, range(0,39)))           
    dic61 = featureExtraction.getPhonemeDictionary()
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
    del(strMap['q'])



    # Now we have a dict for 61 phonemes and a dict for the 39 phonemes (+blank)
    # map integers from 61-phn dict to integers from 39-phn dict (+ add blank)
    intMap = {}
    for str61 in strMap:
        str39 = strMap[str61]
        int61 = dic61[str61]
        int39 = dic39[str39]        
        intMap[int61] = int39    
    return intMap
    
    
def mapNetOutputs(netOutputs, scoreMap):
    '''
    This function maps 61 phonemes + blank to 39 phonemes + blank
    Adds up network outputs that are mapped to the same output for scoring
    '''    
    mappedOutputs = np.zeros((netOutputs.shape[0],40))
    for key in scoreMap.keys():
        mappedOutputs[:, scoreMap[key]] += netOutputs[:,key]
    # blank again last element
    mappedOutputs[:,-1] = netOutputs[:,-1]
    return mappedOutputs
    
def mapTargets(target, scoreMap):
    '''
    This function maps 61 phonemes + blank to 39 phonemes + blank 
    '''
    # make sure 'q' (=57) is not in the sequence, it is not included for scoring...
    # TODO: maybe handle that differently, don't like explicit 57 here
    target = target[target != 57.0]
    mappedSeq = [scoreMap[t] for t in target]
    return np.array([mappedSeq[i] for i in range(len(mappedSeq)) if mappedSeq[i-1]!=mappedSeq[i] or i == 0])
    