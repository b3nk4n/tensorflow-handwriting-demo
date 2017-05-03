import os
import sys
import time
import argparse
import json
import urllib
import collections
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import utils.tensor
import models


def main(_):
    """Executed only if run as a script."""

    print('\nFetiching data...')
    url = 'http://localhost:3000/api/handwriting'
    response = urllib.urlopen(url)
    handwriting_list = json.loads(response.read())
    n_data = len(handwriting_list)
    print(n_data)

    print('\nPrepricessing data...')
    handwritings = np.zeros((n_data, 1024), dtype=np.float32)
    labels = np.zeros((n_data, 1), dtype=np.float32)
    for i, handwriting in enumerate(handwriting_list):
        handwritings[i, :] = np.asarray(handwriting['img'], dtype=np.float32)
        labels[i, 0] = ord(handwriting['label']) - ord('A')
    dataset = {
        'data': handwritings,
        'labels': labels
    }
    
    # split data into different sets
    split_idx = int(n_data * FLAGS.train_split)
    trainset = {
        'size': split_idx,
        'data': dataset['data'][:split_idx],
        'labels': dataset['labels'][:split_idx]
    }
    validset = {
        'size': n_data - split_idx,
        'data': dataset['data'][split_idx:],
        'labels': dataset['labels'][split_idx:]
    }

    print('Training set: {:5d} samples, Vaidation set: {:5d} samples'.format(trainset['size'], validset['size']))

    # show data distribution of alphabet as histogram 
    plt.hist(dataset['labels'].reshape(-1), bins=range(26))
    plt.show()

    x_ph = tf.placeholder(tf.float32, shape=[None, 1024])
    y_ph = tf.placeholder(tf.int32, shape=[None, 1])
    dropout_ph = tf.placeholder(tf.float32)

    with tf.name_scope('model'):
        if FLAGS.model == 'neural_net':
            model_y = models.neural_net(x_ph, [32, 32, 26],
                                        dropout_ph, FLAGS.weight_decay)
        elif FLAGS.model == 'conv_net':
            model_y = models.conv_net(x_ph, [(8, 5), (8, 3)], [32, 26],
                                      dropout_ph, FLAGS.weight_decay)
        else:
            raise 'Unknown network model type.'
    
    with tf.name_scope('loss'):
        y_one_hot = tf.one_hot(indices=y_ph, depth=26, on_value=1.0, off_value=0.0, axis=-1)
        y_one_hot = tf.reshape(y_one_hot, [-1, 26])
        loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_y, labels=y_one_hot))
        regularization_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(regularization_list) > 0:
            loss_ += tf.add_n(regularization_list) 

    train_ = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_)

    with tf.name_scope('metrics'):
        model_out = tf.sigmoid(model_y)
        _, accuracy_ = tf.metrics.accuracy(labels=tf.argmax(y_ph, axis=1), predictions=tf.argmax(model_out, axis=1))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print('\nModel with {} trainable parameters.'.format(utils.tensor.get_num_trainable_params()))
        time.sleep(3)
        print('\nTraining...')

        f, ax = plt.subplots(2, 1)
        train_losses = {'step': [], 'value': []}
        valid_losses = {'step': [], 'value': []}
        valid_accuracy= {'step': [], 'value': []}

        step = 1
        loss_sum = 0.0
        loss_n = 0
        for epoch in range(FLAGS.train_epochs):
            print('\nStarting epoch {}...'.format(epoch + 1))
            sess.run(tf.local_variables_initializer())
            num_batches = int(trainset['size'] / FLAGS.batch_size)
            # shuffle data
            perm = np.random.permutation(trainset['size'])
            trainset['data'] = trainset['data'][perm]
            trainset['labels'] = trainset['labels'][perm]

            for batch_idx in range(num_batches):
                start_idx = batch_idx * FLAGS.batch_size
                end_idx = start_idx + FLAGS.batch_size
                batch_x =  trainset['data'][start_idx:end_idx]
                batch_y =  trainset['labels'][start_idx:end_idx]

                _, loss = sess.run([train_, loss_], feed_dict={x_ph: batch_x,
                                                               y_ph: batch_y,
                                                               dropout_ph: FLAGS.dropout})
                loss_sum += loss
                loss_n += 1

                if step % 5 == 0:
                    loss_avg = loss_sum / loss_n
                    train_losses['step'].append(step)
                    train_losses['value'].append(loss_avg)
                    print('Step {:3d} with loss: {:.5f}'.format(step, loss_avg))
                    loss_sum = 0.0
                    loss_n = 0

                step += 1

            loss, accuracy = sess.run([loss_, accuracy_], feed_dict={x_ph: validset['data'],
                                                                     y_ph: validset['labels'],
                                                                     dropout_ph: 1.0})
            valid_losses['step'].append(step)
            valid_losses['value'].append(loss)
            valid_accuracy['step'].append(step)
            valid_accuracy['value'].append(accuracy)
            print('VALIDATION > Step {:3d} with loss: {:.5f}, accuracy: {:.4f}'.format(step, loss, accuracy))

        if FLAGS.save_checkpoint:
            checkpoint_dir = "checkpoint"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # save checkpoint
            save_path = saver.save(sess, os.path.join(checkpoint_dir, "model.ckpt"))
            print("Model saved in file: {}".format(save_path))

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
    PARSER.add_argument('--learning_rate', type=float, default=0.0005,
                        help='The initial learning rate.')
    PARSER.add_argument('--train_epochs', type=int, default=10,
                        help='The number of training epochs.')
    PARSER.add_argument('--train_split', type=float, default=0.8,
                        help='The data ratio for training.')
    PARSER.add_argument('--dropout', type=float, default=0.5,
                        help='The keep probability of the dropout layer.')
    PARSER.add_argument('--weight_decay', type=float, default=5e-4,
                        help='The lambda koefficient for weight decay regularization.')
    PARSER.add_argument('--model', type=str, default='neural_net',
                        help='The network model no use.')
    PARSER.add_argument('--save_checkpoint', type=bool, default=False,
                        help='Whether we save a checkpoint or not.')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)

