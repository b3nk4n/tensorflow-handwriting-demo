from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty 

import json
import urllib
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class Dataset:
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    def show_info(self):
        print()
        print("# Dataset:        {}".format(self.name))
        print("# Training Set:   {} examples".format(self.train_size))
        print("# Validation Set: {} examples".format(self.valid_size))
        print("# Classes:        {}".format(self.num_classes))
        print("# Data Shape:     {}".format(self.data_shape))
        print()

    @abstractmethod
    def train_batch(self, batch_size):
        pass
    
    @abstractmethod
    def valid(self):
        pass
    
    @abstractproperty
    def data_shape(self):
        pass
        
    @abstractproperty
    def num_classes(self):
        pass
    
    @abstractproperty
    def train_size(self):
        pass
    
    @abstractproperty
    def valid_size(self):
        pass
   

class MnistDataset(Dataset):
    
    def __init__(self):
        self.mnist = input_data.read_data_sets('tmp/mnist', one_hot=False)
        super(MnistDataset, self).__init__("MNIST Dataset")
    
    def train_batch(self, batch_size):
        batch_x, batch_y = self.mnist.train.next_batch(batch_size)
        batch_y = batch_y.reshape(-1, 1)
        return batch_x, batch_y
    
    def valid(self):
        data_x, data_y = self.mnist.validation.next_batch(self.valid_size)
        print(data_x.shape)
        data_y = data_y.reshape(-1, 1)
        return data_x, data_y
    
    @property
    def data_shape(self):
        return (28, 28)
    
    @property
    def num_classes(self):
        return 10
    
    @property
    def train_size(self):
        return self.mnist.train.images.shape[0]
    
    @property 
    def valid_size(self):
        return self.mnist.validation.images.shape[0]


class HandwritingDataset(Dataset):
    
    def __init__(self):
        self.batch_idx = 0
        self._download_data('http://localhost:3000/api/handwriting')
        super(HandwritingDataset, self).__init__("Handwriting Dataset")
        
    def _download_data(self, url):
        print('\nFetiching data...')
        response = urllib.urlopen(url)
        handwriting_list = json.loads(response.read())
        n_data = len(handwriting_list)
        
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
        TRAIN_SPLIT = 0.8
        split_idx = int(n_data * TRAIN_SPLIT)
        self.trainset = {
            'size': split_idx,
            'data': dataset['data'][:split_idx],
            'labels': dataset['labels'][:split_idx]
        }
        self.validset = {
            'size': n_data - split_idx,
            'data': dataset['data'][split_idx:],
            'labels': dataset['labels'][split_idx:]
        }
        
    def train_batch(self, batch_size):
        if self.batch_idx + batch_size > self.train_size:
            # shuffle data
            perm = np.random.permutation(trainset['size'])
            trainset['data'] = trainset['data'][perm]
            trainset['labels'] = trainset['labels'][perm]
            self.batch_idx = 0
        
        start_idx = self.batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_x =  self.trainset['data'][start_idx:end_idx]
        batch_y =  self.trainset['labels'][start_idx:end_idx]
        return batch_x, batch_y
        
    def valid(self):
        return self.validset['data'], self.validset['labels']
    
    @property
    def data_shape(self):
        return (32, 32)
    
    @property
    def num_classes(self):
        return 26
    
    @property
    def train_size(self):
        return self.trainset['size']
    
    @property
    def valid_size(self):
        return self.validset['size']
