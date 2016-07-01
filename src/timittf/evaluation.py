# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:24:12 2016

@author: Timo van Niedek
"""

import tensorflow as _tf
import numpy as _np
import sklearn as _sk
import os as _os

### Constants
# Mapping from training phones to testing phones
TEST_FOLDINGS = {
    3: 0,       # ao -> aa
    5: 2,       # ax -> ah
    9: 37,      # cl -> sil
    14: 27,     # el -> l
    15: 29,     # en -> n
    16: 37,     # epi -> sil
    23: 22,     # ix -> ih
    43: 37,     # vcl -> sil
    47: 35      # zh -> sh
}

# Ordered list of testing phones; used for confusion matrix
TEST_NAMES = ['aa','ae','ah','aw','ay','b','ch','d','dh','dx','eh','er','ey',
              'f','g','hh','ih','iy','jh','k','l','m','n','ng','ow','oy','p',
              'r','s','sh','sil','t','th','uh','uw','v','w','y','z']
              
save_loc = "/scratch-mirror/tvniedek"
              
def fold_labels(labels):
    """
    Fold the labels from the training phones to the testing phones.
    
    Args:
        labels (numpy.array):
            Integer array of labels in [0, 48]
    
    Returns:
        numpy.array:
            Integer array of labels in [0, 48] folded according to TEST_FOLDINGS.
    """
    
    return _np.array([TEST_FOLDINGS[l] if l in TEST_FOLDINGS else l for l in labels])
    
def frames_to_phones(outputs, true_labels, alignments):
    """
    Convert frame-level outputs and true labels to phone-level 
    outputs and true labels for one sentence. The phone-level outputs are
    computed using the mean value for each class over one phone (as given by
    alignments) and then taking the maximum
    of those means.
    Also compute the phone-level softmax cross entropy.
    
    Args:
        outputs (numpy.ndarray):
            Array containing output values.
            The shape is [sentence_length, num_classes].
        true_labels (numpy.ndarray):
            Array containing true classes per frame.
            The shape is [sentence_length]
        alignments (numpy.array):
            Array containing (start, end, phone) tuples denoting the alignments
            for this sentence.

    Returns:
        list:
            Phone-level outputs computed as described above.
            The shape is [num_phones]
        list:
            Phone-level true labels.
            The shape is [num_phones]
        tf.Tensor
            float Tensor with softmax cross entropy loss.
    """    
    
    phones_pred = []
    output_means = []
    phones_true = []
    
    for (s, e, p) in alignments:
        # Phones with start == end don't have any data corresponding to them
        # they are in the test times pickle due to integer rounding
        if s == e:
            continue
        
        y = true_labels[s:e]
        # Check if the true labels are identical (we are only evaluating 1 phone)
        assert len(_np.unique(_np.argmax(y))) == 1
        
        y_ = outputs[s:e]
        means = _np.mean(y_, 0)
        phone_pred = _np.argmax(means)
        if phone_pred in TEST_FOLDINGS:
            phone_pred = TEST_FOLDINGS[phone_pred]
            
        phones_pred.append(phone_pred)
        output_means.append(means)
        true_label = _np.argmax(y[0])
        if true_label in TEST_FOLDINGS:
            true_label = TEST_FOLDINGS[true_label]
        phones_true.append(true_label)
        
    logits = _np.array(output_means)
    labels = _np.array(phones_true)
    cross_entropy = _tf.reduce_mean(_tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
    return phones_pred, phones_true, cross_entropy
    
def save_confusion_matrix(phones_true, phones_pred, date):
    print "Confusion matrix:"
    
    conf_mat = _sk.metrics.confusion_matrix(phones_true, phones_pred, [l for l in xrange(48) if l not in TEST_FOLDINGS])
    print conf_mat
    
    if not _os.path.isdir('%s/summaries/%s'%(save_loc, date)):
        try:
            _os.makedirs('%s/summaries/%s'%(save_loc, date))
        except OSError:
            if not _os.path.isdir('%s/summaries/%s'%(save_loc, date)):
                raise    
    
    conf_mat = _np.asarray(conf_mat)
    saved = _np.zeros((conf_mat.shape[0]+1,conf_mat.shape[0]+1),dtype='S8')
    saved[1:,1:]=conf_mat
    saved[0,1:]=TEST_NAMES
    saved[1:,0]=TEST_NAMES
    _np.savetxt(('%s/summaries/%s/conf_mat.csv'%(save_loc,date)), saved, delimiter=',',header=date, fmt='%s')
    print "Confusion matrix saved in %s/summaries/%s/conf_mat.csv"%(save_loc,date)