import sys
import argparse
import json
import urllib
import collections
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def main(_):
    """Executed only if run as a script."""

    print('Fetiching data...')
    url = 'http://localhost:3000/api/handwriting'
    response = urllib.urlopen(url)
    handwriting_list = json.loads(response.read())
    n_data = len(handwriting_list)
    print(n_data)

    print('Prepricessing data...')
    handwritings = np.zeros((n_data, 1024), dtype=np.float32)
    labels = np.zeros((n_data, 1), dtype=np.float32)
    for i, handwriting in enumerate(handwriting_list):
        handwritings[i, :] = np.asarray(handwriting['img'], dtype=np.float32)
        labels[i, 0] = ord(handwriting['label']) - ord('A')
    dataset = {
        'data': handwritings,
        'labels': labels
    }
    print(dataset['data'].shape, dataset['labels'].shape)
    
    # show data distribution of alphabet as histogram 
    plt.hist(dataset['labels'].reshape(-1), bins=26)
    plt.show()

    # TODO split data into different sets

    print('Training...')
    x_ph = tf.placeholder(tf.float32, shape=[None, 1024])
    y_ph = tf.placeholder(tf.int32, shape=[None, 1])

    def neural_net(x):
        y = tf.contrib.layers.fully_connected(x, 32,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation_fn=tf.nn.relu)
        y = tf.contrib.layers.fully_connected(y, 32,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation_fn=tf.nn.relu)
        return tf.contrib.layers.fully_connected(y , 26,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope('model'):
        model_y = neural_net(x_ph)
    
    with tf.name_scope('loss'):
        y_one_hot = tf.one_hot(indices=y_ph, depth=26, on_value=1.0, off_value=0.0, axis=-1)
        y_one_hot = tf.reshape(y_one_hot, [-1, 26])
        loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_y, labels=y_one_hot))

    train_ = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_)

    with tf.name_scope('metrics'):
        model_out = tf.sigmoid(model_y)
        _, accuracy_ = tf.metrics.accuracy(labels=tf.argmax(y_ph, axis=1), predictions=tf.argmax(model_out, axis=1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        f, ax = plt.subplots(2, 1)
        train_losses = {'step': [], 'value': []}
        valid_losses = {'step': [], 'value': []}
        valid_accuracy= {'step': [], 'value': []}

        step = 1
        for epoch in range(FLAGS.train_epochs):
            print('Starting epoch {}...'.format(epoch + 1))
            num_batches = int(n_data / FLAGS.batch_size)
            # shuffle data
            perm = np.random.permutation(n_data)
            dataset['data'] = dataset['data'][perm]
            dataset['labels'] = dataset['labels'][perm]

            loss_sum = 0.0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * FLAGS.batch_size
                end_idx = start_idx + FLAGS.batch_size
                batch_x =  dataset['data'][start_idx:end_idx]
                batch_y =  dataset['labels'][start_idx:end_idx]

                _, loss = sess.run([train_, loss_], feed_dict={x_ph: batch_x, y_ph: batch_y})
                loss_sum += loss

                if step % 5 == 0:
                    loss_avg = loss_sum / num_batches
                    train_losses['step'].append(step)
                    train_losses['value'].append(loss_avg)
                    print('Step {:3d} with train-loss: {:.5f}'.format(step, loss_avg))
                    loss_sum = 0.0

                step += 1

            loss, accuracy = sess.run([loss_, accuracy_], feed_dict={x_ph: dataset['data'], y_ph: dataset['labels']})
            valid_losses['step'].append(step)
            valid_losses['value'].append(loss)
            valid_accuracy['step'].append(step)
            valid_accuracy['value'].append(accuracy)
            print('Step {:3d} with valid-loss: {:.5f}'.format(step, loss))

        ax[0].plot(train_losses['step'], train_losses['value'], label='Train loss')
        ax[0].plot(valid_losses['step'], valid_losses['value'], label='Valid loss')
        ax[0].legend(loc='upper right')
        ax[1].plot(valid_accuracy['step'], valid_accuracy['value'], label='Valid accuracy')
        ax[1].legend(loc='lower right')
        plt.show()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--batch_size', type=int, default=10,
                        help='The batch size.')
    PARSER.add_argument('--learning_rate', type=float, default=0.001,
                        help='The initial learning rate.')
    PARSER.add_argument('--train_epochs', type=int, default=10,
                        help='The number of training epochs.')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)

