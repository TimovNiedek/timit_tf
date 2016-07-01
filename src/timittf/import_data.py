# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:12:28 2016

@author: Timo van Niedek

Module for importing pickle files and creating a DataSet.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import numpy as np
import pdb
from timittf import preprocessing as pp

NUM_CLASSES = 48

def load_dataset(data_pickle, 
                 labels_pickle, 
                 keep_sentences=False, 
                 context_frames=1, 
                 to_one_hot=True, 
                 seq_length=1,
                 batch_size=6):
    
    with open(data_pickle, 'rb') as data_dump:
        data_sentences = cPickle.load(data_dump)
    with open(labels_pickle, 'rb') as labels_dump:
        labels_sentences = cPickle.load(labels_dump)
    
    # Normalize the mfccs
    print("Normalizing")
    data = [pp.normalize_mfcc(s) for s in data_sentences]
    
    # Possibly add context
    if context_frames > 1:
        print("Adding context")
        data = [pp.add_context(s, context_frames) for s in data]
    
    # Possibly flatten the sentences
    if not keep_sentences:
        data = np.vstack(data)
        if to_one_hot:
            labels = pp.to_one_hot(np.hstack(labels_sentences), NUM_CLASSES)
        else:
            labels = np.hstack(labels_sentences)
        if seq_length > 1:
            data = pp.pad_to_sequence_length(data, seq_length)
            labels = pp.pad_to_sequence_length(labels, seq_length)
    else:
        if to_one_hot:
            labels = [pp.to_one_hot(labels_scalar, NUM_CLASSES) for labels_scalar in labels_sentences]
        if seq_length > 1:
            data = [pp.pad_to_sequence_length(s, seq_length) for s in data]
            labels = [pp.pad_to_sequence_length(s, seq_length) for s in labels]

    print("Preprocessing done")
    return DataSet(np.array(data), np.array(labels), batch_size)    

class DataSet(object):
    def __init__(self, data, labels, batch_size=6):
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = data.shape[0]
        # required for next_sequence_batch
        self._batch_size = batch_size
        # List for the indexes of the sentences in the current batch
        self._indexes_in_epoch = range(batch_size)
        # Array for the frame indexes in the sentences for the current batch
        # Values are always a multiple of seqence_size
        self._index_in_sentences = np.zeros([batch_size], dtype=int)
        
    @property
    def data(self):
        return self._data
        
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
        
    @property
    def batch_size(self):
        return self._batch_size
    
    def reset_epoch(self, batch_size):
        self._batch_size = batch_size
        self._indexes_in_epoch = range(batch_size)
        self._index_in_sentences = np.zeros([self._batch_size], dtype=int)
        # Shuffle the data
        perm = np.arange(len(self._data))
        np.random.shuffle(perm)
        self._data = self._data[perm]
        self._labels = self._labels[perm]
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(len(self._data))
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
        
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]
        
    def next_sequence_batch(self, seq_length):
        # Boolean array. If _end_of_sentence[i] = False, the state should be
        # preserved in an RNN structure. Otherwise the state for sentence i
        # should be reset
        end_of_sentence = np.zeros([self._batch_size], dtype=bool)        
        starts = np.copy(self._index_in_sentences)
        for i in range(self._batch_size):
            self._index_in_sentences[i] += seq_length
            # If the end of this sentence is reached ...
            if (self._index_in_sentences[i] > self.data[self._indexes_in_epoch[i]].shape[0]):
                # ..set the end_of_sentence flag
                end_of_sentence[i] = True
                # ..fetch a new sentence (the sentence after the furthest sentence in this epoch)
                new_index = np.max(self._indexes_in_epoch) + 1
                # Check if there are no more sentences in this epoch
                if (new_index >= self._num_examples):
                    print("End of epoch")
                    self._epochs_completed += 1
                    # Reset epoch
                    self.reset_epoch(self._batch_size)                  
                    # Return Nones to signal the training loop
                    return None, None, None
                # Otherwise, reset the index in sentence and set the new index in epoch
                self._indexes_in_epoch[i] = new_index
                self._index_in_sentences[i] = seq_length
                starts[i] = 0
        
        # Get for each sentence in the batch the sequence from 
        # index_in_sentence to starts + seq_length
        batch_data = [
            self._data[self._indexes_in_epoch[i]][starts[i]:starts[i]+seq_length] 
            for i in range(self._batch_size)
        ]
        # Same for the labels
        batch_labels = [
            self._labels[self._indexes_in_epoch[i]][starts[i]:starts[i]+seq_length] 
            for i in range(self._batch_size)
        ]
        return np.array(batch_data), np.array(batch_labels), np.array(end_of_sentence)
        
    """
    def next_sequence_batch(self, batch_size, sequence_length):
        cur_sentence = self._index_in_epoch
        start = self._index_in_sentence
        
        eos = False # End of sentence
        
        self._index_in_sentence += batch_size        
        
        if self._index_in_sentence > self._data[cur_sentence].shape[0]:
            # Finished sentence
            self._index_in_epoch += 1
            start = 0
            self._index_in_sentence = batch_size
            cur_sentence = self._index_in_epoch
            eos = True
            
        if self._index_in_epoch >= self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            perm = np.arange(len(self._data))
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            cur_sentence = 0
            start = 0
            self._index_in_epoch = batch_size
            
        end = self._index_in_sentence
        
        return self._data[cur_sentence][start:end], self._labels[cur_sentence][start:end], eos
        
        """
    
    """
    def next_sequence_batch(self, batch_size, sequence_length):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch+sequence_length > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(len(self._data))
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
        
        end = self._index_in_epoch
        batch_data = [self._data[(start+i):(start+i+sequence_length)] for i in xrange(batch_size)]
        batch_labels = [self._labels[(start+i):(start+i+sequence_length)] for i in xrange(batch_size)]
        return batch_data, batch_labels"""
        
    def set_as_sequences(self, sequence_length):
        num_sequences = len(self._data) - sequence_length
        sequence_data = [self._data[i:(i+sequence_length)] for i in xrange(0, num_sequences)]
        sequence_labels = [self._labels[i:(i+sequence_length)] for i in xrange(0, num_sequences)]
        return sequence_data, sequence_labels