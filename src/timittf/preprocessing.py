# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:44:03 2016

@author: Timo van Niedek
"""

import numpy as _np
import pdb

def normalize_mfcc(mfcc):
    """Normalize mfcc data using the following formula:
    
    normalized = (mfcc - mean)/standard deviation
    
    Args:
        mfcc (numpy.ndarray):
            An ndarray containing mfcc data.
            Its shape is [sentence_length, coefficients]
    
    Returns:
        numpy.ndarray:
            An ndarray containing normalized mfcc data with the same shape as
            the input.
    """
    
    means = _np.mean(mfcc, 0)
    stds = _np.std(mfcc, 0)
    return (mfcc - means)/stds
    
def to_one_hot(labels, num_classes=48):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels.shape[0]
    index_offset = _np.arange(num_labels) * num_classes
    labels_one_hot = _np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
    
def add_context(sentence, num_frames=1):
    """
    Convert the mfcc data for a sentence to a format that contains the context
    of each frame. For the first and last frames, where context is missing,
    the context consists of copies of that frame.
    
    Args:
        sentence (numpy.ndarray): 
            An ndarray containing mfcc data. 
            Its shape is [sentence_length, coefficients]
        num_frames (int): 
            The number of frames after adding context. Must be odd.
    
    Returns:
        numpy.ndarray:
            An ndarray containing mfcc data with context.
            Its shape is [sentence_length, num_frames, coefficients]
    
    """
    
    assert num_frames%2 == 1, "num_frames must be odd: %i"%num_frames    
    
    if num_frames == 1:
        return sentence
    
    context_sent = []
    
    for i in xrange(0, len(sentence)):
        context_sent.append([context for context in _enumerate_context(i, sentence, (num_frames-1)/2)])
        
    return _np.array(context_sent)



def to_sequences(sentence, seq_length, batch_size):
    # We use the batch size in terms of number of sequences
    # Batch size = 6, seq_length = 20 means we get 6*20 = 120 frames in one batch
    
    frames_per_batch = batch_size * seq_length
    
    target_length = frames_per_batch - (sentence.shape[0] % frames_per_batch) + sentence.shape[0]
    sentence = pad_last(sentence, target_length)
    
    sentence = [sentence[seq_length*i:seq_length*(i+1)] for i in xrange(target_length/seq_length)]
    return _np.array(sentence)
    
def pad_to_sequence_length(sentence, seq_length):
    """
    Pad a sentence with its last frames such that the length of the sequence
    becomes a multiple of seq_length.
    
    This method can be used for padding labels and data.
    
    Args:
        sentence (numpy.ndarray):
            An ndarray containing the data that should be padded.
        seq_length (int):
            The length of one sequence.
    
    Returns:
        numpy.ndarray:
            The same sentence as the input sentence, padded to a multpile of 
            seq_lengths
    """
    
    padding_length = seq_length - (sentence.shape[0] % seq_length)
    target_length = sentence.shape[0] + padding_length
    return pad_last(sentence, target_length)
        
def pad_last(sentence, target_length):
    """
    Pad the mfcc of a sentence to a given target length using the last frame.
    This last frame is considered to be silence in the mfcc files, therefore
    acting as a neutral element.
    
    Args:
        sentence (numpy.ndarray):
            An ndarray containing mfcc data.
            Its shape is [sentence_length, coefficients]
        target_length (int):
            The total length after padding. If this is smaller than the 
            sentence length, an error will be raised.
            
    Returns:
        numpy.ndarray:
            An ndarray containing mfcc data padded to target_length.
    """
    
    assert len(sentence) <= target_length, "Can't pad sentence of length %i to length %i"%(len(sentence), target_length)
        
    last = sentence[-1]
    pad = [last for i in xrange(0, target_length - len(sentence))]
    
    return _np.append(sentence, pad, 0)
    
def _enumerate_context(i, sentence, num_frames):
    r = xrange(i-num_frames, i+num_frames+1)
    r = [x if x>=0 else 0 for x in r]
    r = [x if x<len(sentence) else len(sentence)-1 for x in r]
    return sentence[r]