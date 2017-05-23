""" Trains a model on handwriting data. """
from __future__ import absolute_import, division, print_function

import os
import sys
import time
import math
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2

import utils.ui
import utils.tensor
import utils.embedding
import models
import datasets

FLAGS = None
# disable TensorFlow C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(_):
    """Executed only if run as a script."""
    if FLAGS.dataset == 'mnist':
        dataset = datasets.MnistDataset()
    elif FLAGS.dataset == 'hw-local':
        dataset = datasets.HandwritingDataset('http://localhost:64303')
    elif FLAGS.dataset == 'hw-production':
        dataset = datasets.HandwritingDataset('http://bsautermeister.de/handwriting-service')
    else:
        raise Exception('Unknown dataset.')

    dataset.show_info()

    if FLAGS.dataset_check:
        exit()

    with tf.name_scope('placeholders'):
        x_ph = tf.placeholder(tf.float32, shape=[None] + list(dataset.data_shape))
        y_ph = tf.placeholder(tf.int32, shape=[None, 1])
        dropout_ph = tf.placeholder(tf.float32)
        augment_ph = tf.placeholder_with_default(tf.constant(False, tf.bool), shape=[])
        tf.add_to_collection('x_ph', x_ph)
        tf.add_to_collection('y_ph', y_ph)
        tf.add_to_collection('dropout_ph', dropout_ph)
        tf.add_to_collection('augment_ph', augment_ph)

    with tf.name_scope('data_augmentation'):

        def augment_data(input_data, angle, shift):
            num_images_ = tf.shape(input_data)[0]
            # random rotate
            processed_data = tf.contrib.image.rotate(input_data,
                                                     tf.random_uniform([num_images_],
                                                                       maxval=math.pi / 180 * angle,
                                                                       minval=math.pi / 180 * -angle))
            # random shift
            base_row = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], shape=[1, 8], dtype=tf.float32)
            base_ = tf.tile(base_row, [num_images_, 1])
            mask_row = tf.constant([0, 0, 1, 0, 0, 1, 0, 0], shape=[1, 8], dtype=tf.float32)
            mask_ = tf.tile(mask_row, [num_images_, 1])
            random_shift_ = tf.random_uniform([num_images_, 8], minval=-shift, maxval=shift, dtype=tf.float32)
            transforms_ = base_ + random_shift_ * mask_

            processed_data = tf.contrib.image.transform(images=processed_data,
                                                        transforms=transforms_)
            return processed_data

        preprocessed = tf.cond(augment_ph, lambda: augment_data(x_ph, angle=5.0, shift=2.49), lambda: x_ph)

    with tf.name_scope('model'):
        model_y = emb_layer = None
        if FLAGS.model == 'neural_net':
            model_y, emb_layer = models.neural_net(preprocessed, [128, 128, dataset.num_classes],
                                                   dropout_ph, FLAGS.weight_decay)
        elif FLAGS.model == 'conv_net':
            model_y, emb_layer = models.conv_net(preprocessed, [(16, 5), (32, 3)], [128, dataset.num_classes],
                                                 dropout_ph, FLAGS.weight_decay)
        else:
            raise Exception('Unknown network model type.')

    tf.add_to_collection('model_y', tf.nn.softmax(model_y))

    with tf.name_scope('loss'):
        y_one_hot = tf.one_hot(indices=y_ph, depth=dataset.num_classes, on_value=1.0, off_value=0.0, axis=-1)
        y_one_hot = tf.reshape(y_one_hot, [-1, dataset.num_classes])
        loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_y, labels=y_one_hot))
        regularization_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        tf.summary.scalar('xe-loss', loss_)
        if len(regularization_list) > 0:
            loss_ += tf.add_n(regularization_list) 

    train_ = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_)

    with tf.name_scope('metrics'):
        model_out = tf.nn.softmax(model_y)
        reshaped_y_ph = tf.reshape(y_ph, [-1])
        _, accuracy_ = tf.metrics.accuracy(labels=reshaped_y_ph, predictions=tf.argmax(model_out, axis=1))
        tf.summary.scalar('accuracy', accuracy_)

    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)

    summary_ = tf.summary.merge_all()

    with tf.Session() as sess:
        # delete old summaries
        summary_dir = 'summary'
        if tf.gfile.IsDirectory(summary_dir):
            tf.gfile.DeleteRecursively(summary_dir)

        train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'training'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'validation'))

        sess.run(tf.global_variables_initializer())

        print('\nModel with {} trainable parameters.'.format(utils.tensor.get_num_trainable_params()))
        time.sleep(3)
        print('\nTraining...')

        f, ax = plt.subplots(2, 1)
        train_losses = {'step': [], 'value': []}
        valid_losses = {'step': [], 'value': []}
        valid_accuracy = {'step': [], 'value': []}

        step = 1
        loss_sum = 0.0
        loss_n = 0

        for epoch in range(FLAGS.train_epochs):
            print('\nStarting epoch {} / {}...'.format(epoch + 1, FLAGS.train_epochs))
            sess.run(tf.local_variables_initializer())

            num_batches = int(dataset.train_size / FLAGS.batch_size)
            for b in range(num_batches):
                batch_x, batch_y = dataset.train_batch(FLAGS.batch_size)

                _, loss, summary = sess.run([train_, loss_, summary_],
                                            feed_dict={x_ph: batch_x,
                                                       y_ph: batch_y,
                                                       dropout_ph: FLAGS.dropout,
                                                       augment_ph: FLAGS.augmentation})

                loss_sum += loss
                loss_n += 1

                if step % 10 == 0:
                    loss_avg = loss_sum / loss_n
                    train_losses['step'].append(step)
                    train_losses['value'].append(loss_avg)
                    print('Step {:3d} with loss: {:.5f}'.format(step, loss_avg))
                    loss_sum = 0.0
                    loss_n = 0
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                step += 1

            valid_x, valid_y = dataset.valid()
            loss, accuracy, summary = sess.run([loss_, accuracy_, summary_], 
                                               feed_dict={x_ph: valid_x,
                                                          y_ph: valid_y,
                                                          dropout_ph: 1.0})
            valid_losses['step'].append(step)
            valid_losses['value'].append(loss)
            valid_accuracy['step'].append(step)
            valid_accuracy['value'].append(accuracy)
            print('VALIDATION > Step {:3d} with loss: {:.5f}, accuracy: {:.4f}'.format(step, loss, accuracy))
            valid_writer.add_summary(summary, step)
            valid_writer.flush()

        if FLAGS.save_checkpoint:
            checkpoint_dir = 'checkpoint'
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # save checkpoint
            print('Saving checkpoint...')
            save_path = saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))
            print('Model saved in file: {}'.format(save_path))

        if FLAGS.save_embedding:
            log_dir = 'summary/validation'
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)

            valid_x, valid_y = dataset.valid()

            print('Saving embedding...')
            embvis = utils.embedding.EmbeddingVisualizer(sess, valid_x, valid_y, x_ph, emb_layer)
            embvis.write(log_dir, alphabetical=FLAGS.dataset != 'mnist')

        print('Showing plot...')
        ax[0].plot(train_losses['step'], train_losses['value'], label='Train loss')
        ax[0].plot(valid_losses['step'], valid_losses['value'], label='Valid loss')
        ax[0].legend(loc='upper right')
        ax[1].plot(valid_accuracy['step'], valid_accuracy['value'], label='Valid accuracy')
        ax[1].legend(loc='lower right')
        plt.show()
        print('DONE')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--batch_size', type=int, default=64,  # large batch size (>>100) gives much better results
                        help='The batch size.')
    PARSER.add_argument('--learning_rate', type=float, default=0.001,
                        help='The initial learning rate.')
    PARSER.add_argument('--train_epochs', type=int, default=5,
                        help='The number of training epochs.')
    PARSER.add_argument('--dropout', type=float, default=0.5,
                        help='The keep probability of the dropout layer.')
    PARSER.add_argument('--weight_decay', type=float, default=0.001,
                        help='The lambda koefficient for weight decay regularization.')
    PARSER.add_argument('--model', type=str, default='neural_net',
                        help='The network model no use.')
    PARSER.add_argument('--save_checkpoint', type=bool, default=True,
                        help='Whether we save a checkpoint or not.')
    PARSER.add_argument('--save_embedding', type=bool, default=True,
                        help='Whether we save the embedding.')
    PARSER.add_argument('--dataset', type=str, default='mnist',
                        help='The dataset to use.')
    PARSER.add_argument('--augmentation', type=bool, default=False,
                        help='Whether data augmentation (rotate/shift) is used or not.')
    PARSER.add_argument('--dataset_check', type=bool, default=False,
                        help='Whether the dataset should be checked only.')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
