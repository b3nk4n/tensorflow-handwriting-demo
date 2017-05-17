import numpy as np
import tensorflow as tf


def neural_net(x, layers, keep_prob, weight_decay):
    y = x
    embedding_layer = None
    for i, layer in enumerate(layers[:-1]):
        y = tf.contrib.layers.fully_connected(x, layer,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        embedding_layer = y
        y = tf.nn.relu(y)
    y = tf.nn.dropout(y, keep_prob=keep_prob)
    return tf.contrib.layers.fully_connected(y , layers[-1],
                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                                             weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)), embedding_layer

def conv_net(x, convs, fullys, keep_prob, weight_decay):
    x_dims = x.get_shape().as_list()[-1]
    x_dims_sqrt = int(np.sqrt(x_dims))
    y = tf.reshape(x, [-1, x_dims_sqrt, x_dims_sqrt, 1])
    for conv in convs:
        y = tf.contrib.layers.conv2d(y, conv[0], kernel_size=[conv[1], conv[1]], stride=[1, 1], padding='SAME',
                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                        activation_fn=tf.nn.relu)
        y = tf.contrib.layers.max_pool2d(y, kernel_size=[2, 2], padding='SAME')
    
    y = tf.reshape(y, [-1, np.prod(y.get_shape().as_list()[1:])])
    return neural_net(y, fullys, keep_prob, weight_decay)