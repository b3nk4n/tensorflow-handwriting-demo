import numpy as np
import tensorflow as tf


def neural_net(x, keep_prob, weight_decay):
    y = tf.contrib.layers.fully_connected(x, 32,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                            activation_fn=tf.nn.relu)
    y = tf.contrib.layers.fully_connected(y, 32,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                            activation_fn=tf.nn.relu)
    y = tf.nn.dropout(y, keep_prob=keep_prob)
    return tf.contrib.layers.fully_connected(y , 26,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

def conv_net(x, keep_prob):
    y = tf.reshape(x, [-1, 32, 32, 1])

    y = tf.contrib.layers.conv2d(y, 8, kernel_size=[5, 5], stride=[1, 1], padding='SAME',
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                    activation_fn=tf.nn.relu)
    y = tf.contrib.layers.max_pool2d(y, kernel_size=[2, 2], padding='SAME')
    y = tf.contrib.layers.conv2d(y, 8, kernel_size=[3, 3], stride=[1, 1], padding='SAME',
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                    activation_fn=tf.nn.relu)
    y = tf.contrib.layers.max_pool2d(y, kernel_size=[2, 2], padding='SAME')
    
    y = tf.reshape(y, [-1, np.prod(y.get_shape().as_list()[1:])])

    y = tf.contrib.layers.fully_connected(y, 32,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                            activation_fn=tf.nn.relu)
    
    y = tf.nn.dropout(y, keep_prob=keep_prob)
    return tf.contrib.layers.fully_connected(y, 26,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))