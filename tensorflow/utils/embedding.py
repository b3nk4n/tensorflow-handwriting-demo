""" Embedding visualization utility module. """
from __future__ import absolute_import, division, print_function

import os
import scipy.misc
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import utils.graphics


class EmbeddingVisualizer(object):
    """ Mangages the visualization of an embedding tensor. """
    def __init__(self, session, input_data, labels, input_placeholder, fetch_tensor):
        self.sess = session
        self.input_data = input_data
        self.labels = labels
        self.input_ph = input_placeholder
        self.fetch_tensor = fetch_tensor
        self.num_examples = input_data.shape[0]  # N
        self.emb_dim = fetch_tensor.shape[1]  # p

    def write(self, log_dir, alphabetical=False):
        """ Write the embadding matrix to the given log path. """
        self._create_metadata(log_dir, alphabetical)
        self._create_sprite(log_dir)
        self._write_embedding_matrix(log_dir)

    def _create_embedding(self):
        """ We now create a N x p matrix holding the embeddings, for the N images. """
        emb = np.zeros((self.num_examples, self.emb_dim), dtype=np.float32)
        for i in range(self.num_examples):  # Of course you could do mini-batches
            emb[i] = self.sess.run(self.fetch_tensor,
                                   feed_dict={self.input_ph: self.input_data[i:i+1, ...]})
        return emb

    def _write_embedding_matrix(self, log_dir):
        """ Write the embedding matrix to disc. """
        # The embedding variable, which needs to be stored
        # Note this must a Variable not a Tensor!
        emb = self._create_embedding()
        embedding_var = tf.Variable(emb, name='EmbeddingVisualization')
        self.sess.run(embedding_var.initializer)
        summary_writer = tf.summary.FileWriter(log_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # Comment out if you don't have metadata
        embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')

        # Comment out if you don't want sprites
        embedding.sprite.image_path = os.path.join(log_dir, 'sprite.png')
        input_shape = self.input_data.shape
        embedding.sprite.single_image_dim.extend([input_shape[1],
                                                  input_shape[2]])

        projector.visualize_embeddings(summary_writer, config)
        saver = tf.train.Saver([embedding_var])
        saver.save(self.sess, os.path.join(log_dir, 'model2.ckpt'))

    def _create_metadata(self, log_dir, alphabetical):
        """ Crates the meta data file """
        def map_alpha(number):
            return chr(65 + number)

        def map_num(number):
            return number

        if alphabetical:
            mapping = map_alpha
        else:
            mapping = map_num

        metadata_file = open(os.path.join(log_dir, 'metadata.tsv'), 'w')
        metadata_file.write('Name\tClass\n')
        for i in range(self.num_examples):
            metadata_file.write('%06d\t%s\n' % (i, mapping(int(self.labels[i, 0]))))
        metadata_file.close()

    def _create_sprite(self, log_dir):
        input_shape = self.input_data.shape
        images = self.input_data.reshape(-1, input_shape[1], input_shape[2]).astype(np.float32)
        sprite = utils.graphics.images_to_sprite(images)
        scipy.misc.imsave(os.path.join(log_dir, 'sprite.png'), sprite)
