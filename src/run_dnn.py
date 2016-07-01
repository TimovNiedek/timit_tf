# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:20:11 2016

@author: Timo van Niedek
"""

import cPickle
import tensorflow as tf
import numpy as np
import argparse
import time
import pdb
from timittf import import_data
from timittf import layers
from timittf import evaluation
import os

# Required; otherwise python won't find timittf modules
import sys
sys.path.append("/timittf")

### Parameters (overidden by argparse, default values listed here)
num_epochs = 15
batch_size = 128
context_frames = 11
optimizer_name = 'Adam'
num_hidden_layers = 3
size_hidden = 1024
use_dropout = True

### Internal variables
num_features = 13*context_frames
num_classes = 48
start_date = time.strftime("%d-%m-%y/%H.%M.%S")
save_loc = "/scratch-mirror/tvniedek"

### Placeholders
# Probability of keeping a node at dropout (0.5 during training, 1 during testing)
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# Input features
# x is defined after parsing command line args
x = None #tf.placeholder(tf.float32, [None, num_features], name='input')
# Output classes (true values)
y_ = tf.placeholder(tf.float32, [None, num_classes], name='true_labels')

### Ops
summary_op = None
train_writer = None
sess = None
saver = None

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
        
def DNN(x):
    layer1 = layers.fully_connected_layer(x, num_features, size_hidden, 'layer1')
    if use_dropout:
        layer1 = layers.dropout_layer(layer1, keep_prob, 'dropout1')
    output = layer1
    for i in range(num_hidden_layers-1):
        output = layers.fully_connected_layer(output, size_hidden, size_hidden, 'hidden%i'%(i+1))
        if use_dropout:
            output = layers.dropout_layer(output, keep_prob, 'dropout%i'%(i+1))
    
    #layer2 = fully_connected_layer(layer1, 1024, 1024, 'layer2')
    #layer3 = fully_connected_layer(layer2, 1024, 1024, 'layer3')
    #layer4 = fully_connected_layer(layer3, 1024, 1024, 'layer4')
    #layer5 = fully_connected_layer(layer4, 1024, 1024, 'layer5')
    
    # Dropout layer
    #dropout = tf.nn.dropout(output, keep_prob, name='dropout')
    
    # Output layer
    y = layers.output_layer(output, size_hidden, num_classes, 'output')
    
    return y
    
def train(train_set, y, cost, optimizer):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # This accuracy is only used for training
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        with tf.name_scope('summaries'):

            tf.scalar_summary('accuracy/accuracy', accuracy)

    global summary_op
    global train_writer
    
    i = 0 
    
    for epoch in range(num_epochs):
        print "epoch %d:" % epoch
        for batch in range(int(train_set.data.shape[0] / batch_size)):
            i += 1
            batch_data, batch_labels = train_set.next_batch(batch_size)
            
            if context_frames > 1:
                reshaped_data = np.reshape(batch_data, [batch_size, 13*context_frames])
            else:
                reshaped_data = batch_data
                        
            if batch%100 == 0:
                acc_summ, train_accuracy = sess.run([summary_op, accuracy],feed_dict={x:reshaped_data, y_: batch_labels, keep_prob:1.0})  
                
                train_writer.add_summary(acc_summ, i)            
                print("epoch %d, batch %d, training accuracy %g"%(epoch, batch, train_accuracy))
            c_e, _ = sess.run([cost, optimizer], feed_dict={x:reshaped_data, y_:batch_labels, keep_prob:0.8})
            if batch%100 == 0:
                print("cross entropy: %g"%c_e)
        # After every epoch, compute the frame error rate of the total training set
        if context_frames > 1:
            data = np.reshape(train_set.data, [train_set.data.shape[0], 13*context_frames])
        else:
            data= train_set.data
        acc = sess.run(accuracy, feed_dict={x:data, y_:train_set.labels, keep_prob:1.0})
        print("epoch %d finished, accuracy: %g" % (epoch, acc))
        save_path = saver.save(sess, "%s/models/%s/epoch%d_model.ckpt"%(save_loc,start_date, epoch))
        print("Model for epoch %d saved in file: %s"%(epoch, save_path))
    
def evaluate(output, true_labels, frames_per_sentence, test_times_pickle):
    """
    Evaluate the results of training a model.    
    
    This function converts the results of training a model from a frame-based 
    structure to a phone-based structure. For each output value, the mean is
    computed over one phone, after which the maximum output value is taken as 
    the prediction for that phone.
    
    Additionally, the cross entropy is calculated and returned for each phone.
    
    Args:
        output (numpy.ndarray):
            The output values of the DNN classification. 
            The shape is [num_frames, num_classes]
        true_labels (numpy.array):
            The true labels with shape [num_frames]
        frames_per_sentence (numpy.array):
            An array containing the number of frames in the i-th sentence.
            For example, ``[10, 20, ...]`` means that frame 0-10 correspond to 
            the first sentence, 10-20 to the second sentence, and so on.
            The sum of all elements in this array should be equal to num_frames.
        test_times_pickle (file):
            The .pickle file containing at which frame each phone starts
            and ends for every sentence.
            
    Returns:
        np.array:
            The predicted phones according to the model.
        np.ndarray:
            The cross entropy values for each phone.
        np.array:
            The true labels of each phone.
            
    """
    with open(test_times_pickle, 'rb') as times_dump:
        times = cPickle.load(times_dump)
        
    phones_pred = []
    output_means = []
    phones_true = []
    
    offset=0
    for i, time_sentence in enumerate(times):        
        for j, (s, e, p) in enumerate(time_sentence):
            y = true_labels[offset+s:offset+e]
            
            # Phones with start == end don't have any data corresponding to them
            # they are in the test times pickle due to integer rounding
            if s == e:
                continue
            
            # Check if the true labels are identical (we are only evaluating 1 phone)
            assert len(np.unique(np.argmax(y))) == 1
            
            y_ = output[offset+s:offset+e]
            means = np.mean(y_, 0)
            prediction = np.argmax(means)
            if prediction in TEST_FOLDINGS:
                prediction = TEST_FOLDINGS[prediction]
                
            phones_pred.append(prediction)
            output_means.append(means)
            
            true_label = np.argmax(y[0])
            
            if true_label in TEST_FOLDINGS:
                true_label = TEST_FOLDINGS[true_label]
            phones_true.append(true_label)  
        offset += frames_per_sentence[i]
    
    logits = np.array(output_means)
    labels = np.array(phones_true)
    
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
    
    return np.array(phones_pred), sess.run(cross_entropy), np.array(phones_true)
            
def train_dnn(data_folder, model_file): 
    # Output of dnn using input x
    y = DNN(x)
    
    print "Loading training pickles..."  
    train_set = import_data.load_dataset(data_folder + '/train_data.pickle', 
                                         data_folder + '/train_labels.pickle',
                                         context_frames=context_frames)      
        
    # Create the dir for the model
    if not os.path.isdir('%s/models/%s'%(save_loc,start_date)):
        try:
            os.makedirs('%s/models/%s'%(save_loc,start_date))
        except OSError:
            if not os.path.isdir('%s/models/%s'%(save_loc,start_date)):
                raise
    
    # Create the session
    global sess
    sess = tf.InteractiveSession()    
    global summary_op
    global train_writer
    global saver
    saver = tf.train.Saver()
        
    # Op for merging all summaries
    summary_op = tf.merge_all_summaries()
    # Summary Writer
    train_writer = tf.train.SummaryWriter('%ssummaries/%s'%(save_loc, start_date), sess.graph)
        
    # Cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # Optimizer
    # For gradient descend, learning rate = 0.002 (see Hinton et al.)
    # For AdamOptimizer, learning rate = 0.0001 (better than default (exp 1.2))
    if (optimizer_name == 'Adam'):
        # Hacky solution for always making sure that the beta2_power var
        # is always initialized
        temp = set(tf.all_variables())
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
    else:
        optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(cost)
    
    if model_file:
        saver.restore(sess, model_file)
        print "Model restored"
    else:
        # Initialization
        init_op = tf.initialize_all_variables()
        sess.run(init_op)    
    
    print("Training network. Date: %s" % start_date)
    train(train_set, y, cost, optimizer)
    
    save_path = saver.save(sess, "%s/models/%s/model.ckpt"%(save_loc, start_date))
    print("Model saved in file: %s" % save_path)
    print("Summaries written to summaries/%s" % start_date)
    
    evaluate_dnn(data_folder, y)

def evaluate_model_from_file(data_folder, model_file):
    y = DNN(x)
    global sess
    sess = tf.InteractiveSession()
    global saver
    saver = tf.train.Saver()
    saver.restore(sess, model_file)
    print("Model restored")
    evaluate_dnn(data_folder, y)
    
def evaluate_dnn(data_folder, y):
    print "Loading testing pickles..."
    
    # We load the test set as sentences to get the frames per sentence from it.
    test_set_sentences = import_data.load_dataset(data_folder + '/test_data.pickle', 
                                                  data_folder + '/test_labels.pickle',
                                                  keep_sentences=True,
                                                  context_frames=context_frames)
    
    fps = [s.shape[0] for s in test_set_sentences.data]    
    
    # The flat test set is actually used for predictions.
    test_set_flat = import_data.DataSet(np.vstack(test_set_sentences.data), np.vstack(test_set_sentences.labels))
    
    if context_frames > 1:
        reshaped_data = np.reshape(test_set_flat.data, [test_set_flat.data.shape[0], 13*context_frames])
    else:
        reshaped_data = test_set_flat.data

    outputs = sess.run(y, feed_dict={x:reshaped_data, y_:test_set_flat.labels, keep_prob:1.0})
    
    phones_pred, cross_entropy, phones_true = evaluate(outputs, test_set_flat.labels, fps, data_folder + '/test_times.pickle')
    frame_acc = np.mean(np.equal(np.argmax(outputs, 1), np.argmax(test_set_flat.labels, 1)))
    accuracy = np.mean(np.equal(phones_pred, phones_true))
    
    print("Test frame accuracy: %f"%frame_acc)
    print("Test phone accuracy: %f"%accuracy)
    print("Average phone-level cross entropy: %f"%cross_entropy)
    evaluation.save_confusion_matrix(phones_true, phones_pred, start_date)
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Train and evaluate the Neural Network model.')
    ap.add_argument('data_folder', type=str, help='Folder containing train_data.pickle, train_labels.pickle, test_data.pickle and test_labels.pickle.')
    ap.add_argument('-b', '--batch_size', type=int, help='Size of each minibatch')
    ap.add_argument('-c', '--context_frames', type=int, dest='context', help='Number of frames used as context (1 for no context). Must be odd.')
    ap.add_argument('-d', '--dropout', action='store_false', help="Don't apply dropout regularization")
    ap.add_argument('-e', '--epochs', type=int, help='Number of epochs for training')
    ap.add_argument('-m', '--model', type=str, help='.ckpt file of the model')
    ap.add_argument('-n', '--num_hidden_layers', type=int, help='Number of fully connected hidden layers')
    ap.add_argument('-o', '--optimizer', type=str, help='Optimizer. Either "Adam" or "GradDesc"')
    ap.add_argument('-s', '--size_hidden', type=int, help='Number of neurons in each hidden layer')
    ap.add_argument('-t', '--test', action='store_true', help='Evaluate a given model.')
    ap.set_defaults(context=11, 
                    epochs=15,
                    batch_size=128,
                    num_hidden_layers=3,
                    size_hidden=1024,
                    optimizer='GradDesc',
                    test=False,
                    dropout=True)
    args = ap.parse_args()
    
    assert args.context%2 == 1, "context_frames must be odd: %i"%args.context
    context_frames = args.context
    num_features = 13*args.context
    
    assert (args.optimizer == 'Adam' or args.optimizer == 'GradDesc'), 'Optimizer must be either "Adam" or "GradDesc": %s'%args.optimizer
    optimizer_name = args.optimizer
    
    num_epochs = args.epochs
    batch_size = args.batch_size
    num_hidden_layers = args.num_hidden_layers
    size_hidden = args.size_hidden
    use_dropout = args.dropout
    
    x = tf.placeholder(tf.float32, [None, num_features], name='input')   
    
    if args.test:
        assert args.model, "Model file is required for evaluation."
        evaluate_model_from_file(args.data_folder, args.model)
    else:
        train_dnn(args.data_folder, args.model)
