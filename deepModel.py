"""
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from PIL import Image
from six.moves import urllib
import tensorflow as tf
import numpy as np

# image data constants information
class DeepModel(object):
    """

    """
    def __init__(self, particle_size, model_input_size, num_class):
        self.particle_size = particle_size
        self.batch_size = model_input_size[0]
        self.num_col = model_input_size[1]
        self.num_row = model_input_size[2]
        self.num_channel = model_input_size[3]
        self.num_class = num_class

    def init_learning_rate(self, learning_rate = 0.01, learning_rate_decay_factor = 0.95, decay_steps = 400, staircase = True):
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.decay_steps = decay_steps
        self.staircase = staircase
        # define a global step variable
        self.global_step = tf.Variable(0,trainable = False)
        
    def init_momentum(self, momentum = 0.9):
        self.momentum = momentum

    """ create variable with weight decay    
    """
    # why not using tf.Variable()...
    # if the initializer is not None, then it has the same effect as tf.Variable()
    def __variable_with_weight_decay(self, name, shape, stddev, wd):
        var = tf.get_variable(name, shape,
                        initializer = tf.truncated_normal_initializer(stddev=stddev, seed = 1234))
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses',weight_decay)
        return var

    def __inference(self, data, train=True):
        """ build cnn model, 
        input : data
        return : predictions
        """
        conv1 = tf.nn.conv2d(data, self.kernel1, strides=[1, 1, 1, 1], padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.biases1))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = tf.nn.conv2d(pool1, self.kernel2, strides=[1, 1, 1, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.biases2))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3 = tf.nn.conv2d(pool2, self.kernel3, strides=[1, 1, 1, 1], padding='VALID')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.biases3))
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv4 = tf.nn.conv2d(pool3, self.kernel4, strides=[1, 1, 1, 1], padding='VALID')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, self.biases4))
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        hidden = tf.reshape(pool4, [self.batch_size, -1])
        #print(hidden.get_shape())
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=6543)

        fc1 = tf.nn.relu(tf.matmul(hidden, self.weights_fc1) + self.biases_fc1)
        sotfmax = tf.add(tf.matmul(fc1, self.weights_fc2), self.biases_fc2)
        return (sotfmax)

    def __loss(self, logits):
        """compute loss with prediction and label, also will acount for L2 loss
        input : prediction, label
        output : loss
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits, self.train_label_node, name = 'cross_entropy_all')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        all_loss = tf.add_n(tf.get_collection('losses'), name='all_loss')
        return all_loss

    def __preprocess_particle(self, batch_data):
        # scale the image to the model input size
        #batch_data = tf.image.resize_images(batch_data, self.num_col, self.num_row)
        # get the scale tensor shape
        batch_data_shape = batch_data.get_shape().as_list()
        # uppack the tensor into sub-tensor
        batch_data_list = tf.unpack(batch_data)
        for i in xrange(batch_data_shape[0]):
            # Pass image tensor object to a PIL image
            image = Image.fromarray(batch_data_list[i].eval())
            # Use PIL or other library of the sort to rotate
            random_degree = random.randint(0, 359)
            rotated = Image.Image.rotate(image, random_degree)
            # Convert rotated image back to tensor
            rotated_tensor = tf.convert_to_tensor(np.array(rotated))
            #slice_image = tf.slice(batch_data, [i, 0, 0, 0], [1, -1, -1, -1])
            #slice_image_reshape = tf.reshape(slice_image, [batch_data_shape[1], batch_data_shape[2], batch_data_shape[3]])
            #distorted_image = tf.image.random_flip_up_down(batch_data_list[i], seed = 1234)
            #distorted_image = tf.image.random_flip_left_right(distorted_image, seed = 1234)
            #distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
            #distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
            # Subtract off the mean and divide by the variance of the pixels.
            distorted_image = tf.image.per_image_whitening(rotated_tensor)
            batch_data_list[i] = distorted_image
        # pack the list of tensor into one tensor
        batch_data = tf.pack(batch_data_list)
        return batch_data

    def init_model_graph_train(self):
        self.kernel1 = self.__variable_with_weight_decay('weights1', shape=[9, 9, 1, 8], stddev=0.05, wd = 0.0)
        self.biases1 = tf.get_variable('biases1', [8], initializer=tf.constant_initializer(0.0))

        self.kernel2 = self.__variable_with_weight_decay('weights2', shape=[5, 5, 8, 16], stddev=0.05, wd = 0.0)
        self.biases2 = tf.get_variable('biases2', [16], initializer=tf.constant_initializer(0.0))

        self.kernel3 = self.__variable_with_weight_decay('weights3', shape=[3, 3, 16, 32], stddev=0.05, wd = 0.0)
        self.biases3 = tf.get_variable('biases3', [32], initializer=tf.constant_initializer(0.0))

        self.kernel4 = self.__variable_with_weight_decay('weights4', shape=[2, 2, 32, 64], stddev=0.05, wd = 0.0)
        self.biases4 = tf.get_variable('biases4', [64], initializer=tf.constant_initializer(0.0))

        dim = 64*2*2
        self.weights_fc1 = self.__variable_with_weight_decay('weightsf1', shape=[dim, 128], stddev=0.05, wd=0.0005)
        self.biases_fc1 = tf.get_variable('biasesf1', [128], initializer=tf.constant_initializer(0.0))

        self.weights_fc2 = self.__variable_with_weight_decay('weightsf2', shape=[128, self.num_class], stddev=0.05, wd=0.0005)
        self.biases_fc2 = tf.get_variable('biasesf2', [self.num_class], initializer=tf.constant_initializer(0.0))


	# define the holder for training procedure
        self.train_data_node = tf.placeholder( tf.float32,
                                               shape=(self.batch_size, self.num_col, self.num_row, self.num_channel))
        self.train_label_node = tf.placeholder(tf.int64, shape=(self.batch_size,))
        self.eval_data_node = tf.placeholder( tf.float32,
                                              shape=(self.batch_size, self.num_col, self.num_col, self.num_channel))
        # preprocess to the train data
        #train_data_node_process = self.__preprocess_particle(self.train_data_node)
        #eval_data_node_process = self.__preprocess_particle(self.eval_data_node)

	# define the training procedure
        # the value is not processed by softmax function.
	logits = self.__inference(self.train_data_node, train=True)
	# define the loss computation process and prediction computation process 
        self.train_prediction_operation = tf.nn.softmax(logits)
        self.loss_operation = self.__loss(logits)
        # define the learning rate decay during training
        self.learningRate_operation = tf.train.exponential_decay(self.learning_rate,
                                        self.global_step,
                                        self.decay_steps,
                                        self.learning_rate_decay_factor, staircase=self.staircase)
        # define the Optimizer
        self.optimizer_operation = tf.train.MomentumOptimizer(self.learningRate_operation, self.momentum).minimize(self.loss_operation, 
                                                      global_step = self.global_step)
            
        # define the evaluation procedure
        evaluation_logits = self.__inference(self.eval_data_node, train=False)
        self.evaluation_prediction_operation = tf.nn.softmax(evaluation_logits)
    
    def init_model_graph_evaluate(self):
        self.kernel1 = self.__variable_with_weight_decay('weights1', shape=[9, 9, 1, 8], stddev=0.05, wd = 0.0)
        self.biases1 = tf.get_variable('biases1', [8], initializer=tf.constant_initializer(0.0))

        self.kernel2 = self.__variable_with_weight_decay('weights2', shape=[5, 5, 8, 16], stddev=0.05, wd = 0.0)
        self.biases2 = tf.get_variable('biases2', [16], initializer=tf.constant_initializer(0.0))

        self.kernel3 = self.__variable_with_weight_decay('weights3', shape=[3, 3, 16, 32], stddev=0.05, wd = 0.0)
        self.biases3 = tf.get_variable('biases3', [32], initializer=tf.constant_initializer(0.0))

        self.kernel4 = self.__variable_with_weight_decay('weights4', shape=[2, 2, 32, 64], stddev=0.05, wd = 0.0)
        self.biases4 = tf.get_variable('biases4', [64], initializer=tf.constant_initializer(0.0))

        dim = 64*2*2
        self.weights_fc1 = self.__variable_with_weight_decay('weightsf1', shape=[dim, 128], stddev=0.05, wd=0.0005)
        self.biases_fc1 = tf.get_variable('biasesf1', [128], initializer=tf.constant_initializer(0.0))

        self.weights_fc2 = self.__variable_with_weight_decay('weightsf2', shape=[128, self.num_class], stddev=0.05, wd=0.0005)
        self.biases_fc2 = tf.get_variable('biasesf2', [self.num_class], initializer=tf.constant_initializer(0.0))


        self.eval_data_node = tf.placeholder( tf.float32,
                                              shape=(self.batch_size, self.num_col, self.num_col, self.num_channel))
        # define the evaluation procedure
        evaluation_logits = self.__inference(self.eval_data_node, train=False)
        self.evaluation_prediction_operation = tf.nn.softmax(evaluation_logits)
 
    def evaluation(self, data, sess):
        size = data.shape[0]
        predictions = np.ndarray(shape=(size, self.num_class), dtype=np.float32)
        for begin in xrange(0, size, self.batch_size):
            end = begin + self.batch_size
            if end <= size:
                batch_data = data[begin:end, ...]
                predictions[begin:end, :] = sess.run(
                    self.evaluation_prediction_operation,
                    feed_dict={self.eval_data_node: batch_data})
            else:
                batch_data = data[-self.batch_size:, ...]
                batch_predictions = sess.run(
                    self.evaluation_prediction_operation,
                    feed_dict={self.eval_data_node: batch_data})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def train_batch(self, batch_data, batch_label, sess):
         # do the computation
         feed_dict = {self.train_data_node: batch_data, self.train_label_node: batch_label}
         _, loss_value, learning_rate, prediction = sess.run(
                       [self.optimizer_operation, self.loss_operation, self.learningRate_operation, self.train_prediction_operation],
                       feed_dict=feed_dict)
         return loss_value, learning_rate, prediction

