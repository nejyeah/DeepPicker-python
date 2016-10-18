from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from optparse import OptionParser

from deepModel import DeepModel
from dataLoader import DataLoader
import display

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def error_rate(prediction, label):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(prediction, 1) == label) / prediction.shape[0])

def train():
    parser = OptionParser()
    parser.add_option("--train_inputDir", dest="train_inputDir", help="Input directory", metavar="DIRECTORY")
    parser.add_option("--train_inputFile", dest="train_inputFile", help="Input file", metavar="FILE")
    parser.add_option("--train_type", dest="train_type", help="Training type, 1|2|3|4.", metavar="VALUE", default=2)
    parser.add_option("--particle_number", dest="train_number", help="Number of positive samples to train.", metavar="VALUE", default=-1)
    parser.add_option("--mrc_number", dest="mrc_number", help="Number of mrc files to be trained.", metavar="VALUE", default=-1)
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol", help="The symbol of the coordinate file, like '_manualPick'", metavar="STRING")
    parser.add_option("--particle_size", dest="particle_size", help="the size of the particle.", metavar="VALUE", default=-1)
    parser.add_option("--validation_ratio", dest="validation_ratio", help="the ratio.", metavar="VALUE", default=0.1)
    parser.add_option("--model_retrain", action="store_true", dest="model_retrain", help="train the model using the pre-trained model as parameters initialization .", default=False)
    parser.add_option("--model_load_file", dest="model_load_file", help="pre-trained model", metavar="FILE")
    parser.add_option("--model_save_dir", dest="model_save_dir", help="save the model to this directory", metavar="DIRECTORY", default="../trained_model")
    parser.add_option("--model_save_file", dest="model_save_file", help="save the model to file", metavar="FILE")
    (opt, args) = parser.parse_args()
 
    # set the tensoflow seed
    tf.set_random_seed(1234)
    # set the numpy seed
    np.random.seed(1234)

    # define the input size of the model
    model_input_size = [100, 64, 64, 1]
    num_class = 2                   # the number of the class
    batch_size = model_input_size[0]

    # define input parameters
    train_type = int(opt.train_type)
    train_inputDir = opt.train_inputDir
    train_inputFile = opt.train_inputFile
    train_number = float(opt.train_number) 
    mrc_number = int(opt.mrc_number)
    coordinate_symbol = opt.coordinate_symbol
    debug_dir = '../train_output'   # output dir
    particle_size = int(opt.particle_size)
    validation_ratio = float(opt.validation_ratio)   

    # define the save model
    model_retrain = opt.model_retrain
    model_load_file = opt.model_load_file
    model_save_dir = opt.model_save_dir
    model_save_file = os.path.join(model_save_dir, opt.model_save_file)

    if not os.access(model_save_dir, os.F_OK):
        os.mkdir(model_save_dir)
    if not os.access(debug_dir, os.F_OK):
        os.mkdir(debug_dir)

    # define the learning rate decay parameters
    # more information about this, refer to function tf.train.exponential_decay()
    learning_rate = 0.01
    learning_rate_decay_factor = 0.95
    # the value will be changed base on the train_size and batch size
    learning_rate_decay_steps = 400
    learning_rate_staircase = True
    # momentum
    momentum = 0.9

    # load training dataset
    dataLoader = DataLoader()
    if train_type == 1:
        # load train data from mrc file dir
        train_number = int(train_number)
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_mrcFileDir(train_inputDir, particle_size, model_input_size, validation_ratio, coordinate_symbol, mrc_number, train_number)
    elif train_type == 2:
        # load train data from numpy data struct
        train_number = int(train_number)
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_ExtractedDataFile(train_inputDir, train_inputFile, model_input_size, validation_ratio, train_number)
    elif train_type == 3:
        # load train data from prepicked results
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_PrePickedResults(train_inputDir, train_inputFile, particle_size, model_input_size, validation_ratio, train_number)
    elif train_type == 4:
        # load train data from relion .star file 
        train_number = int(train_number)
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_RelionStarFile(train_inputFile, particle_size, model_input_size, validation_ratio, train_number)
    else:
        print("ERROR: invalid value of train_type:", train_type)    

    # display.show_particle(train_data, os.path.join(debug_dir, 'positive.png'))
    # test whether train_data exist
    try: 
        train_data
    except NameError:
        print("ERROR: in function load.loadInputTrainData.")
        return None
    else:
        print("Load training data successfully!")
    # shuffle the training data
    train_data, train_label = shuffle_in_unison_inplace(train_data, train_label)
    eval_data, eval_label = shuffle_in_unison_inplace(eval_data, eval_label)

    train_size = train_data.shape[0]
    eval_size = eval_data.shape[0]    
    # initalize the decay_steps based on train_size and batch size.
    # change the learning rate each 2 epochs
    learning_rate_decay_steps = 10*(train_size // batch_size)
    # initialize the parameters of deepModel
    deepModel = DeepModel(particle_size, model_input_size, num_class)
    deepModel.init_learning_rate(learning_rate = learning_rate, learning_rate_decay_factor = learning_rate_decay_factor,
                                  decay_steps = learning_rate_decay_steps, staircase = learning_rate_staircase)
    deepModel.init_momentum(momentum = momentum)
    # initialize the model
    # define the computation procedure of optimizer, loss, lr, prediction, eval_prediction 
    deepModel.init_model_graph_train()
    saver = tf.train.Saver(tf.all_variables())
    
    start_time = time.time()
    init = tf.initialize_all_variables()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        # initialize all the parameters
        sess.run(init)
        max_epochs = 200   # the max number of epoch to train the model
        best_eval_error_rate = 100
        toleration_patience = 10 
        toleration_patience_flag  = 0
        eval_frequency = train_size // batch_size   # the frequency to evaluate the evaluation dataset
        for step in xrange(int(max_epochs * train_size) // batch_size):
            # get the batch training data
            offset =  (step * batch_size) % (train_size - batch_size)
            batch_data = train_data[offset:(offset+batch_size), ...]
            batch_label = train_label[offset:(offset+batch_size)]
            # online augmentation
            #batch_data = DataLoader.preprocess_particle_online(batch_data)
            loss_value, lr, train_prediction = deepModel.train_batch(batch_data, batch_label,sess)

            # do the computation
            if step % eval_frequency == 0:
                stop_time = time.time() - start_time
                start_time = time.time()
                eval_prediction = deepModel.evaluation(eval_data, sess)
                eval_error_rate = error_rate(eval_prediction, eval_label)
                print('epoch: %.2f , %.2f ms' % (step * batch_size /train_size, 1000 * stop_time / eval_frequency)) 
                print('train loss: %.6f,\t learning rate: %.6f' % (loss_value, lr)) 
                print('train error: %.6f%%,\t valid error: %.6f%%' % (error_rate(train_prediction, batch_label), eval_error_rate))         
                if eval_error_rate < best_eval_error_rate:
                    best_eval_error_rate = eval_error_rate
                    toleration_patience = 10
                else:
                    toleration_patience = toleration_patience - 1
            if toleration_patience == 0:
                saver.save(sess, model_save_file)
                break


def main(argv=None):
    train()

if __name__ == '__main__':
    #tf.app.run()
    main()
