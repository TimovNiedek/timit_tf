# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:44:11 2016

@author: Timo van Niedek
"""

import argparse
import numpy as np
import os
from timittf import import_data
from timittf import layers
from timittf import evaluation
import tensorflow as tf
import time
import pdb
import cPickle

# Required; otherwise python won't find timittf modules
import sys
sys.path.append("/timittf")

### Parameters (overidden by argparse, default values listed here)
num_epochs = 15
train_batch_size = 6
num_hidden = 650
num_lstm_layers = 2
num_steps = 20
use_dropout = True
optimizer_name='Adam'

### Internal variables
num_features = 13
num_classes = 48
start_date = time.strftime("%d-%m-%y/%H.%M.%S")
save_loc = "/scratch-mirror/tvniedek"

x = tf.placeholder(tf.float32, [None, num_steps, num_features], name='input')

# Times when we want to have an early stop (length = batch_size)
batch_size = tf.placeholder(tf.int32, name='batch_size')

# Output classes (true values).
y_ = tf.placeholder(tf.float32, [None, num_steps, num_classes], name='y_')
true_labels = tf.reshape(y_, [-1, num_classes])

# Initial state for the LSTM
initial_state = tf.placeholder(tf.float32, [None, None], name='initial_state')

# Probability of keeping nodes at dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

### Ops
summary_op = None
train_writer = None
sess = None
# Op for saving and restoring the model
saver = None
    
def RNN(x):
    """
    Create the model for the RNN and return the output.
    Based on:
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py
    
    Args:
        x (Tensor):
            A tensor of shape [batch_size, num_steps, num_features] containing
            input features.
            
    Returns:
        y (Tensor):
            A tensor of shape [num_frames, num_classes] containing output activations.
            num_frames equals the sum of frames of each sentence in the batch.
    """
    
    print "Creating model"    
        
    # Permute the batch_size and num_steps dimensions
    x = tf.transpose(x, [1, 0, 2])
    
    # Reshape for input to multiLSTM layer
    x = tf.reshape(x, [-1, num_features])
    
    layer1 = layers.fully_connected_layer(x, num_features, num_hidden, 'layer1')
    
    # We can't use layers.dropout_layer because dropout is applied
    # within the Multi-LSTM layer. Instead, we pass None for keep_prob if
    # we don't want to apply dropout, and keep_prob if we do.
    if use_dropout:
        k_p = keep_prob
    else:
        k_p = None
    
    # Multi-LSTM layer
    rnn_out, rnn_state = layers.multiLSTM_layer(layer1, 
                                                num_steps,
                                                num_hidden,
                                                num_lstm_layers,
                                                initial_state,
                                                k_p)
    
    # Output layer
    y = layers.output_layer(rnn_out, num_hidden, num_classes, 'output')
    print "Model creation done"    
    
    return y, rnn_state

def train(train_set, y, rnn_state, cost, optimizer):        
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(true_labels, 1))
        #correct_prediction = tf.Print(cp, [tf.argmax(y, 1), tf.argmax(true_labels, 1)], message="y, y_")
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        with tf.name_scope('summaries'):
            tf.scalar_summary('accuracy', accuracy)

    global summary_op
    global train_writer
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print "epoch %d:" % epoch
        
        last_state = np.zeros([train_batch_size, num_hidden*2*num_lstm_layers])
        epoch_done = False
        batch = 0
        while not epoch_done:
            global_step += 1
            batch += 1
            batch_data, batch_labels, eos = train_set.next_sequence_batch(num_steps)
            
            if batch_data is None:
                # Epoch is finished
                epoch_done = True
                break
            
            # Reset the LSTM State for the sequences that ended, 
            # otherwise use the previous state
            start_state = [
                np.zeros([num_hidden*2*num_lstm_layers]) if eos[i]
                else last_state[i]
                for i in range(train_batch_size)
            ]
            
            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                batch_size: train_batch_size,
                initial_state: start_state,
                keep_prob: 1.0
            }                        
            
            if batch%100 == 0:
                acc_summ, train_accuracy = sess.run([summary_op, accuracy],feed_dict=feed_dict)  
                train_writer.add_summary(acc_summ, global_step)            
                print("epoch %d, batch %d, training accuracy %g"%(epoch, batch, train_accuracy))
            
            feed_dict[keep_prob] = 0.8
            c_e, _, last_state = sess.run([cost, optimizer,rnn_state], feed_dict=feed_dict)
            if batch%100 == 0:
                print("cross entropy: %g"%c_e)
            
        
        # After every epoch, we use the entire data set as one batch
        # and evaluate the performance up to the length of the shortest
        # sentence. This is a very rough estimate of the accuracy,
        # because the ends of many sentences are not taken into account
        num_examples = train_set.data.shape[0]
        last_state = np.zeros([num_examples, num_hidden*2*num_lstm_layers])
        train_set.reset_epoch(num_examples)
        
        end_of_eval = False

        accuracies = []

        while not end_of_eval:
            batch_data, batch_labels, eos = train_set.next_sequence_batch(num_steps)
            if batch_data is None:
                end_of_eval = True
                break
            # Reset the LSTM State for the sequences that ended, 
            # otherwise use the previous state
            start_state = [
                np.zeros([num_hidden*2*num_lstm_layers]) if eos[i]
                else last_state[i]
                for i in range(num_examples)
            ]
            
            feed_dict = {
                x: batch_data,
                y_: batch_labels,
                batch_size: num_examples,
                initial_state: start_state,
                keep_prob: 1.0
            }
            
            acc, last_state = sess.run([accuracy,rnn_state], feed_dict=feed_dict)
            accuracies.append(acc)
            
        acc_mean = np.mean(np.array(accuracies))
        print("epoch %d finished, accuracy: %g" % (epoch, acc_mean))
        save_path = saver.save(sess, "%s/models/%s/epoch%d_model.ckpt"%(save_loc, start_date, epoch))
        print("Model for epoch %d saved in file: %s"%(epoch, save_path))
        train_set.reset_epoch(train_batch_size)

def train_rnn(data_folder, model_file):
    y, rnn_state = RNN(x)
    
    print "Loading training pickles.."    

    # We want to keep the sentences in order to train per sentence
    # Sentences are padded to num_steps
    train_set = import_data.load_dataset(data_folder + '/train_data.pickle', 
                                         data_folder + '/train_labels.pickle',
                                         keep_sentences=True,
                                         context_frames=1,
                                         seq_length=num_steps,
                                         batch_size=train_batch_size)
    
    
    print "Loading done"
    
    global sess
    global summary_op
    global train_writer
    global saver
    saver = tf.train.Saver()

    # Create the dir for the model
    if not os.path.isdir('%s/models/%s'%(save_loc,start_date)):
        try:
            os.makedirs('%s/models/%s'%(save_loc,start_date))
        except OSError:
            if not os.path.isdir('%s/models/%s'%(save_loc,start_date)):
                raise
    
    sess = tf.InteractiveSession()
    summary_op = tf.merge_all_summaries()        
    train_writer = tf.train.SummaryWriter('%s/summaries/%s'%(save_loc, start_date), sess.graph)
        
    # Cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, true_labels))
    # Optimizer
    # For gradient descend, learning rate = 0.002 (see Hinton et al.)
    # For AdamOptimizer, learning rate = 0.001 (default)
    if (optimizer_name == 'Adam'):
        # Hacky solution for always making sure that the beta2_power var
        # is always initialized
        temp = set(tf.all_variables())
        optimizer = tf.train.AdamOptimizer().minimize(cost)
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
    train(train_set, y, rnn_state, cost, optimizer)
    
    save_path = saver.save(sess, "%s/models/%s/model.ckpt"%(save_loc,start_date))
    print("Model saved in file: %s" % save_path)
    print("Summaries written to %s/summaries/%s" % (save_loc, start_date))
    
    evaluate_rnn(data_folder, y, rnn_state)
    
def evaluate_rnn_model_from_file(data_folder, model_file):
    y, rnn_state = RNN(x)
    # Op for saving and restoring the model
    saver = tf.train.Saver()
    global sess
    sess = tf.InteractiveSession()
    saver.restore(sess, model_file)
    print "Model restored"
    
    evaluate_rnn(data_folder, y, rnn_state)

def evaluate_sentence(test_set, y, rnn_state, times, i):
    end_of_sentence = False
    
    prediction = []
    true_labels = []    
    
    last_state = np.zeros([1, num_hidden*2*num_lstm_layers])
    
    while not end_of_sentence:
        data, labels, eos = test_set.next_sequence_batch(num_steps)
        if data is None:
            # Entire test set is done
            return None, None, None, None
        
        #if i == 6:
        #    pdb.set_trace()
        
        end_of_sentence = eos[0]
        if end_of_sentence:
            # If the end of a sentence is reached, next_sequence_batch will
            # return the first subsequence of the next sentence and update
            # the index in that sentence. To make sure the first subsequence is
            # still returned for the next sentence, we set the index to 0.
            test_set._index_in_sentences[0] = 0
            break            
            
        # Initial state is the zero state defined above for the first subsequence
        # last_state is updated for each subsequence
        start_state = last_state
            
        feed_dict = {
            x: data,
            y_: labels,
            batch_size: 1,
            initial_state: start_state,
            keep_prob: 1.0
        }                        
        
        pred, last_state = sess.run([y, rnn_state], feed_dict=feed_dict)
        prediction.append(pred)
        true_labels.append(labels[0])
    
    prediction = np.reshape(prediction, [-1, num_classes])
    true_labels = np.reshape(true_labels, [-1, num_classes])
    # Compute the frame accuracy
    correct_prediction = np.equal(evaluation.fold_labels(np.argmax(prediction, 1)), 
                                  evaluation.fold_labels(np.argmax(true_labels, 1)))
    frame_acc = np.mean(np.asarray(correct_prediction, float))
    
    phones_pred, phones_true, cross_entropy = evaluation.frames_to_phones(prediction, true_labels, times)
    return phones_pred, phones_true, frame_acc, sess.run(cross_entropy)
        
def evaluate_rnn(data_folder, y, rnn_state):
    # For evaluation, we run the same loop as in training, 
    # without optimization. The batch size remains the same, because a higher
    # batch size would lead to less sentences being evaluated
    test_set = import_data.load_dataset(data_folder + '/test_data.pickle', 
                                         data_folder + '/test_labels.pickle',
                                         keep_sentences=True,
                                         context_frames=1,
                                         seq_length=num_steps,
                                         batch_size=1)
                                         
    with open(data_folder + '/test_times.pickle', 'rb') as times_dump:
        times = cPickle.load(times_dump)

    # Create arrays that will contain evaluation metrics per sentence    
    phones_pred = []
    phones_true = []
    frame_accuracies = []
    cross_entropies = []
    
    for i in range(len(times)):
        print i
        pred, true_l, frame_acc, cross_entropy = evaluate_sentence(test_set, y, rnn_state, times[i],i)    
        if frame_acc is None:
            break
        frame_accuracies.append(frame_acc)
        phones_pred.append(pred)
        phones_true.append(true_l)
        cross_entropies.append(cross_entropy)
    
    phones_pred = np.hstack(phones_pred)
    phones_true = np.hstack(phones_true)
    
    phone_accuracy = np.mean(np.equal(phones_pred, phones_true))
    
    # Get the total accuraries
    frame_accuracy = np.mean(frame_accuracies)
    cross_entropy = np.mean(cross_entropies)
    
    print("Test frame accuracy: %f"%frame_accuracy)
    print("Test phone accuracy: %f"%phone_accuracy)
    print("Average phone-level cross entropy: %f"%cross_entropy)
    evaluation.save_confusion_matrix(phones_true, phones_pred, start_date)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Train and evaluate the LSTM model.')
    ap.add_argument('data_folder', type=str, help='Folder containing train_data.pickle, train_labels.pickle, test_data.pickle and test_labels.pickle.')
    ap.add_argument('-e', '--epochs', type=int, help='Number of epochs for training')
    ap.add_argument('-b', '--batch_size', type=int, help='Size of each minibatch')
    ap.add_argument('-d', '--dropout', action='store_false', help="Don't apply dropout regularization")
    ap.add_argument('-l', '--seq_length', type=int, help='Number of frames per subsequence.')
    ap.add_argument('-m', '--model', type=str, help='.ckpt file of the model. If -t option is used, evaluate this model. Otherwise, train it.')
    ap.add_argument('-n', '--num_hidden_layers', type=int, help='Number of hidden LSTM layers')
    ap.add_argument('-o', '--optimizer', type=str, help='Optimizer. Either "Adam" or "GradDesc"')
    ap.add_argument('-s', '--size_hidden', type=int, help='Number of neurons in each hidden LSTM layer')
    ap.add_argument('-t', '--test', action='store_true', help='Evaluate a given model.')
    
    ap.set_defaults(epochs=15,
                    batch_size=6, 
                    test=False,
                    size_hidden=250,
                    num_hidden_layers=3,
                    seq_length=20,
                    dropout=True,
                    optimizer='Adam')
    args = ap.parse_args()

    assert (args.optimizer == 'Adam' or args.optimizer == 'GradDesc'), 'Optimizer must be either "Adam" or "GradDesc": %s'%args.optimizer
    optimizer_name = args.optimizer

    num_epochs = args.epochs
    train_batch_size = args.batch_size    
    num_hidden = args.size_hidden
    num_lstm_layers = args.num_hidden_layers
    num_steps = args.seq_length
    use_dropout = args.dropout
    
    if args.test:
        assert args.model, "Model file is required for evaluation."
        evaluate_rnn_model_from_file(args.data_folder, args.model)
    else:
        train_rnn(args.data_folder, args.model)