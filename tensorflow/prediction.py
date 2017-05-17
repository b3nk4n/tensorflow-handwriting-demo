""" Restores a trained model and predicts handwritings. """
from __future__ import absolute_import, division, print_function

import sys
import argparse

import numpy as np
import tensorflow as tf

import utils.ui
import datasets

FLAGS = None


def main(_):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('checkpoint/model.ckpt.meta')
        new_saver.restore(sess, 'checkpoint/model.ckpt')
        x_ph = tf.get_collection('x_ph')[0]
        y_ph = tf.get_collection('y_ph')[0]
        dropout_ph = tf.get_collection('dropout_ph')[0]
        loss_ = tf.get_collection('loss')[0]
        accuracy_ = tf.get_collection('accuracy')[0]

        model_y = tf.get_collection('model_y')[0]

        sess.run(tf.local_variables_initializer())  # FIXME https://github.com/tensorflow/tensorflow/issues/9747

        if FLAGS.input_type == 'handwriting':
            while True:
                dialog = utils.ui.CanvasDialog("Read Handwriting...", 28, 28,
                                            scale=5, num_letters=FLAGS.num_letters)
                data = dialog.show()
                writing = np.asarray(data)

                if writing.shape[0] == 0:
                    break

                prediction = sess.run(model_y, feed_dict={x_ph: writing, dropout_ph: 1.0})
                for i in range(prediction.shape[0]):
                    alpha_pos = np.argmax(prediction[i])
                    print(chr(ord('A') + alpha_pos), end='')
                print()

        elif FLAGS.input_type == 'valid':
            dataset = datasets.HandwritingDataset()
            valid_x, valid_y = dataset.valid()
            prediction, loss, accuracy = sess.run([model_y, loss_, accuracy_], feed_dict={x_ph: valid_x, y_ph: valid_y, dropout_ph: 1.0})
            for i in range(prediction.shape[0]):
                alpha_pos = np.argmax(prediction[i])
                print("{} -> {}".format(chr(ord('A') + valid_y[i]), chr(ord('A') + alpha_pos)))
            print('Loss: {:.5f} Accuracy: {:3f}'.format(loss, accuracy))

        else:
            raise "Unknown input type"


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--num_letters', type=int, default=6,
                        help='The number of letters for the handwriting.')
    PARSER.add_argument('--input_type', type=str, default='handwriting',
                        help='The number of letters for the handwriting.')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
