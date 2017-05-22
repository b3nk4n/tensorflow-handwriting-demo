""" Restores a trained model and predicts handwritings. """
from __future__ import absolute_import, division, print_function

import sys
import argparse

import numpy as np
import tensorflow as tf

# Required import to reload the graph using tf.contrib.image.rotate():
# See: https://stackoverflow.com/questions/43287850
import tensorflow.contrib.image

import utils.ui

FLAGS = None


def main(_):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('checkpoint/model.ckpt.meta')
        new_saver.restore(sess, 'checkpoint/model.ckpt')
        x_ph = tf.get_collection('x_ph')[0]
        dropout_ph = tf.get_collection('dropout_ph')[0]

        model_y = tf.get_collection('model_y')[0]

        data_shape = x_ph.get_shape().as_list()[1:]

        while True:
            dialog = utils.ui.CanvasDialog("Read Handwriting...", data_shape[0], data_shape[1],
                                           scale=5, num_letters=FLAGS.num_letters)
            data = dialog.show()
            writing = np.asarray(data)
            writing = writing.reshape([-1] + data_shape)

            if writing.shape[0] == 0:
                break

            prediction = sess.run(model_y, feed_dict={x_ph: writing, dropout_ph: 1.0})
            for i in range(prediction.shape[0]):
                index_pos = np.argmax(prediction[i])
                if data_shape[0] == 28:  # MNIST
                    print(index_pos, end='')
                else:  # Handwriting
                    print(chr(ord('A') + index_pos), end='')
            print()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--num_letters', type=int, default=6,
                        help='The number of letters for the handwriting.')
    PARSER.add_argument('--dataset', type=str, default='mnist',
                        help='The dataset to use.')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
