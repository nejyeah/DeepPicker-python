from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import re
import pickle
import numpy as np
from optparse import OptionParser
import tensorflow as tf

from deepModel import DeepModel
from autoPicker import AutoPicker

def pick_particle():
    # define the options
    parser = OptionParser()
    parser.add_option("--inputDir", dest="inputDir", help="Input directory", metavar="DIRECTORY")
    parser.add_option("--pre_trained_model", dest="pre_trained_model", help="Input the pre-trained model", metavar="FILE")
    parser.add_option("--mrc_number", dest="mrc_number", help="Number of mrc files to be picked.", metavar="VALUE", default=-1)
    parser.add_option("--particle_size", dest="particle_size", help="the size of the particle.", metavar="VALUE", default=-1)
    parser.add_option("--outputDir", dest="outputDir", help="Output directory, the coordinates file will be saved here.", metavar="DIRECTORY")
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol", help="The symbol of the saveed coordinate file, like '_cnnPick'", metavar="STRING")
    parser.add_option("--threshold", dest="threshold", help="Pick the particles, the prediction value is larger than the threshold..", metavar="VALUE", default=0.5)
    (opt, args) = parser.parse_args()

    # set the random seed for numpy and tensorflow
    tf.set_random_seed(1234)
    np.random.seed(1234)
    
    # define the input size of the model
    model_input_size = [1000, 64, 64, 1]
    num_class = 2                   # the number of the class
    batch_size = model_input_size[0]

    particle_size = int(opt.particle_size)

    pre_trained_model = opt.pre_trained_model
    input_dir = opt.inputDir
    output_dir = opt.outputDir
    threshold = float(opt.threshold)
    coordinate_symbol = opt.coordinate_symbol
    mrc_number = int(opt.mrc_number)

    if not os.path.isfile(pre_trained_model):
        print("ERROR:%s is not a valid file."%(pre_trained_model))
    
    if not os.path.isdir(input_dir):
        print("ERROR:%s is not a valid dir."%(input_dir))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # initialize the model 
    deepModel = DeepModel(particle_size, model_input_size, num_class)
    deepModel.init_model_graph_evaluate()

    # load mrc files 
    mrc_file_all = []
    files = os.listdir(input_dir)
    for f in files:
        if re.search('\.mrc$', f): 
            filename = os.path.join(input_dir, f)
            mrc_file_all.append(filename)
    
    mrc_file_all.sort()
    if mrc_number<=0:
        mrc_number = len(mrc_file_all)

    if mrc_number > len(mrc_file_all):
        mrc_number = len(mrc_file_all)
    
    with tf.Session() as sess:
        # reload the pre-trained model
        saver = tf.train.Saver()
        saver.restore(sess, pre_trained_model)
        
        # do the autopick
        autopicker = AutoPicker(sess, model_input_size, deepModel, particle_size)    
        time1 = time.time()
        candidate_particle_all = []
        for i in range(mrc_number):
            # elements in list 'coordinate' are small list, [x_coordinate, y_coordinate, prediction_value, micrograph_name]
            coordinate = autopicker.pick(mrc_file_all[i])
            candidate_particle_all.append(coordinate)   
            AutoPicker.write_coordinate(coordinate, mrc_file_all[i], coordinate_symbol, threshold, output_dir)
        time_cost = time.time() - time1
        print("time cost: %.1f s"%time_cost)

        # write the pick all results(threshold=0) to file
        output_file = os.path.join(output_dir, 'autopick_results.pickle')
        AutoPicker.write_pick_results(candidate_particle_all, output_file)

def main(argv=None):
    pick_particle()
if __name__ == '__main__':
    #tf.app.run()
    main()
