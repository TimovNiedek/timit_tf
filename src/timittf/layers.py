# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:23:56 2016

@author: Timo van Niedek
"""

import tensorflow as _tf
from tensorflow.models.rnn import rnn_cell as _rnn_cell
from tensorflow.models.rnn import rnn as _rnn

""" Layers """
def fully_connected_layer(input_tensor, input_dim, output_dim, layer_name):
    """
    Create ops for a fully connected feedforward layer. The units use
    ReLU activation.
    Also adds some summary ops for this layer.
    
    Args:
        input_tensor (tf.Tensor):
            The input Tensor for this layer.
            The shape is [batch_size, input_dim].
        input_dim (int):
            Dimension of the input Tensor.
        output_dim (int):
            Dimension of the output Tensor. Corresponds to the number of units
            in this layer.
        layer_name (str):
            Name of the layer (used for name scope and summaries).
            
    Returns:
        tf.Tensor:
            The output Tensor for this layer.
            Corresponds to: relu(input_tensor * weights + biases).
            The shape is [batch_size, output_dim].
    
    Example:
        Create a fully connected layer from 143 input features to 1024 hidden
        units with:
        
        ``layer1 = fully_connected_layer(x, 143, 1024, 'layer1')``
    """
    
    with _tf.name_scope(layer_name):
        with _tf.name_scope("weights"):
            weights = _weight_variable([input_dim, output_dim], name='W_'+layer_name)
            _variable_summaries(weights, layer_name + '/weights')
        with _tf.name_scope("biases"):
            biases = _bias_variable([output_dim], name='b_'+layer_name)
            _variable_summaries(biases, layer_name + '/biases')
        with _tf.name_scope('Wx_plus_b'):
            activations = _tf.matmul(input_tensor, weights) + biases
            _tf.histogram_summary(layer_name + '/activations', activations)
        relu = _tf.nn.relu(activations, 'relu')
        _tf.histogram_summary(layer_name + '/activations_relu', relu)
        return relu
        
def output_layer(input_tensor, input_dim, output_dim, layer_name):
    """
    Create ops for a fully connected feedforward output layer. The units use
    no activation function.
    Also adds some summary ops for this layer.
    
    Args:
        input_tensor (tf.Tensor):
            The input Tensor for this layer.
            The shape is [batch_size, input_dim].
        input_dim (int):
            Dimension of the input Tensor.
        output_dim (int):
            Dimension of the output Tensor. Corresponds to the number of units
            in this layer.
        layer_name (str):
            Name of the layer (used for name scope and summaries).
            
    Returns:
        tf.Tensor:
            The output Tensor for this layer.
            Corresponds to: input_tensor * weights + biases.
            The shape is [batch_size, output_dim].
    """
    with _tf.name_scope(layer_name):
        with _tf.name_scope("weights"):
            weights = _weight_variable([input_dim, output_dim], name='W_'+layer_name)
            _variable_summaries(weights, layer_name + '/weights')
        with _tf.name_scope("biases"):
            biases = _bias_variable([output_dim], name='b_'+layer_name)
            _variable_summaries(biases, layer_name + '/biases')
        with _tf.name_scope('Wx_plus_b'):
            activations = _tf.matmul(input_tensor, weights) + biases
            _tf.histogram_summary(layer_name + '/activations', activations)
        return activations
        
def dropout_layer(input_tensor, keep_prob, layer_name):
    """
    Create an op for applying dropout. Technically, this is not a layer since
    it has no weights or biases and is not trainable.
    
    Args:
        input_tensor (tf.Tensor):
            The input tensor that dropout should be applied to.
        keep_prob (tf.Tensor or float):
            The probability for nodes to remain in the network when applying
            dropout. Its value should be a single float value between 0 and 1.
            Pass 1 to apply no dropout (i.e. when training).
        layer_name (str):
            Name of the layer (used for name scope).
            
    Returns:
        tf.Tensor:
            A tensor of same shape as input_tensor with dropout applied.
            See also the documentation for tf.nn.dropout for explanation of
            how TensorFlow applies dropout.
    """
    
    return _tf.nn.dropout(input_tensor, keep_prob, name=layer_name)
    
def multiLSTM_layer(input_tensor, 
                    subsequence_length,
                    num_units, 
                    num_layers, 
                    initial_state,
                    keep_prob=None):
    """
    Create ops for multiple LSTM layers. Returns the output Tensor for these
    layers.
    
    Args:
        input_tensor (tf.Tensor):
            The input tensor for the LSTM layer. 
            The shape is [batch_size*subsequence_length, num_units]
        subsequence_length (int):
            The length of each subsequence. The LSTM layers will be unrolled for
            this number of time steps.
        num_units (int):
            The number of units in each LSTM layer.
        num_layers (int):
            The number of LSTM layers in the Multi-LSTM layers.
        intial_state (tf.Tensor):
            The initial state of the LSTM layers. To train one sequence with
            multiple subsequences, set the initial state for the first sub-
            sequence to the zero state (np.zeros([batch_size, num_units*2*num_layers]))
            and use the returned state as initial state for the next subsequence.
            The shape is [batch_size, num_units*2*num_layers].
        keep_prob (tf.Tensor, float or None):
            The probability for nodes to remain in the network when applying
            dropout. Its value should be a single float value between 0 and 1.
            Pass 1 to apply no dropout (i.e. when training) 
            and None when dropout should not be included in the graph.
            
    Returns:
        tf.Tensor:
            Output Tensor of the Multi-LSTM layer.
            The shape is [batch_size * subsequence_length, num_units]
        tf.Tensor:
            Tensor representing the final state of the Multi-LSTM layer.
            The shape is [batch_size, num_units*2*num_layers]
    """
    # LSTM requires list input 
    #(of shape subsequence_length * [batch_size, num_units])
    input_tensor = _tf.split(0, subsequence_length, input_tensor)
    
    # LSTM cell
    lstm_cell = _rnn_cell.BasicLSTMCell(num_units)
    
    if keep_prob is not None:
        # Dropout
        dropout_cell = _rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob) 
    
        # Create multiple LSTM layers with dropout
        cell = _rnn_cell.MultiRNNCell([dropout_cell] * num_layers)
    else:
        # Create multiple LSTM layers without dropout
        cell = _rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
    
    # Create the RNN for the specified cell
    # rnn_out is a TensorList of length (subsequence_length) 
    # with Tensors of shape [batch_size, num_units]
    # rnn_state is a Tensor of shape [batch_size, num_units*num_layers*2]
    # containing the final state of the RNN.
    rnn_out, rnn_state = _rnn.rnn(cell, input_tensor, initial_state=initial_state)
    
    # Pack the outputs in one Tensor
    rnn_out = _tf.pack(rnn_out)     # [subsequence_length, batch_size, num_units]
        
    # Transpose subsequence_length and batch_size in order to match the shape of input
    rnn_out = _tf.transpose(rnn_out, [1, 0, 2])      # [batch_size, subsequence_length, num_units]
    
    # Reshape into [batch_size * subsequence_length, num_units]
    rnn_out = _tf.reshape(rnn_out, [-1, num_units])
    
    return rnn_out, rnn_state

""" Internal methods """
""" Weight and bias initialization """
def _weight_variable(shape, name=None):
    initial = _tf.truncated_normal(shape, stddev=0.1)
    return _tf.Variable(initial, name=name)

def _bias_variable(shape, name=None):
    initial = _tf.constant(0.1, shape=shape)
    return _tf.Variable(initial, name=name)

""" Summaries """
def _variable_summaries(var, name):
    with _tf.name_scope("summaries"):
        mean = _tf.reduce_mean(var)
        _tf.scalar_summary('mean/' + name, mean)
        with _tf.name_scope('stddev'):
            stddev = _tf.sqrt(_tf.reduce_sum(_tf.square(var - mean)))
        _tf.scalar_summary('sttdev/' + name, stddev)
        _tf.scalar_summary('max/' + name, _tf.reduce_max(var))
        _tf.scalar_summary('min/' + name, _tf.reduce_min(var))
        _tf.histogram_summary(name, var)