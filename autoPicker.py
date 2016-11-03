from __future__ import absolute_import  
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import time
import math
import pickle

from six.moves import urllib
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter

from deepModel import DeepModel
from dataLoader import DataLoader
import display
# image data constants information
class AutoPicker(object):
    """

    """
    def __init__(self, sess, model_input_size, deepModel, particle_size):
        """Initialize the Autopicker.

        Args:
            sess: an instance of tensorflow session.
            model_input_size: a list of length 4, it is the input size of a placeholder of tensorflow.
            deepModel: an instance of class deepModel
            particle_size: the particle size of the molecular
        
        """
        self.sess = sess
        self.model_input_size = model_input_size
        self.deepModel = deepModel
        self.particle_size = particle_size
        self.SEED = 6543

    def peak_detection(self, image_2D, local_window_size):
        """Do the local peak dection to get the best coordinate of molecular center.

        This function does a local peak dection to the score map to get the best coordinates.

        Args:
            image_2d: numpy.array, it is a 2d array, the dim is 2, the value of it was a prediction score given by the CNN model.
            local_window_size: this is the distance threshold between two particles. The peak detection is done in the local window.

        Returns:
            return list_coordinate_clean
            list_coordinate_clean: a list, the length of this list stands for the number of picked particles.
                                   Each element in the list is also a list, the length is 3.
                                   The first one is x-axis, the second one is y-axis, the third one is the predicted score.
        """
        col = image_2D.shape[0] 
        row = image_2D.shape[1]
        # filter the array in local, the values are replaced by local max value. 
        data_max = filters.maximum_filter(image_2D, local_window_size)
        # compare the filter array to the original one, the same value in the same location is the local maximum.
        # maxima is a bool 2D array, true stands for the local maximum
        maxima = (image_2D == data_max)
        data_min = filters.minimum_filter(image_2D, local_window_size)
        diff = ((data_max - data_min) > 0)
        maxima[diff == 0] = 0 

        labeled, num_objects = ndimage.label(maxima)
        # get the coordinate of the local maximum
        # the shape of the array_y_x is (number, 2)
        array_y_x = np.array(ndimage.center_of_mass(image_2D, labeled, range(1, num_objects+1)))
        array_y_x = array_y_x.astype(int)
        list_y_x = array_y_x.tolist()
        #print("number of local maximum:%d"%len(list_y_x))
        for i in range(len(list_y_x)):
            # add the prediction score to the list
            list_y_x[i].append(image_2D[ array_y_x[i][0] ][array_y_x[i][1] ]) 
            # add a symbol to the list, and it is used to remove crowded candidate
            list_y_x[i].append(0)
       
        # remove close candidate
        for i in range(len(list_y_x)-1):
            if list_y_x[i][3] == 1:
                continue
            
            for j in range(i+1, len(list_y_x)):
                if list_y_x[i][3] == 1:
                    break
                if list_y_x[j][3] == 1:
                    continue
                d_y = list_y_x[i][0] - list_y_x[j][0]
                d_x = list_y_x[i][1] - list_y_x[j][1]
                d_distance = math.sqrt(d_y**2 + d_x**2)
                if d_distance < local_window_size/2:
                    if list_y_x[i][2] >= list_y_x[j][2]:
                        list_y_x[j][3] = 1
                    else:
                        list_y_x[i][3] = 1  
                
        list_coordinate_clean = []
        for i in range(len(list_y_x)):
            if list_y_x[i][3] == 0:
                # remove the symbol element
                list_x_y = []
                list_x_y.append(list_y_x[i][1])
                list_x_y.append(list_y_x[i][0])
                list_x_y.append(list_y_x[i][2])
                list_coordinate_clean.append(list_x_y)

        return list_coordinate_clean


    def pick(self, mrc_filename):
        """Do the picking job through tensorflow.

        This function read the micrograph data information based on the given filename of micrograph.
        Then do the auto picking based on pre-trained CNN model.

        Args:
            mrc_filename: string, it is the filename of the target micrograph.

        Returns:
            return list_coordinate
            list_coordinate: a list, the length of this list stands for the number of picked particles.
                                   Each element in the list is also a list, the length is 4, the first one is y-axis, 
                                   the second one is x-axis, the third one is the predicted score, the fourth is the micrograph filename.
        """
        # read the micrograph image data
        print(mrc_filename)
        header, body = DataLoader.readMrcFile(mrc_filename)
        num_col = header[0]
        num_row = header[1]
        body_2d = np.array(body, dtype = np.float32).reshape(num_row, num_col)
        
        # do process to micrograph
        body_2d, bin_size = DataLoader.preprocess_micrograph(body_2d)
        
        # Edge detection to get the ice noise mask
        # a binary matrix, 1 stands for the ice noise site
        # mask = edge_detection_ice(body_2d)

        step_size = 4
        candidate_patches = None
        candidate_patches_exist = False
        num_total_patch = 0
        patch_size = int(self.particle_size/bin_size)
        # the size to do peak detection 
        local_window_size = int(0.6*patch_size/step_size)

        #print("image_col:", body_2d.shape[0])
        #print("particle_size:", patch_size)
        #print("step_size:", step_size)
        map_col = int((body_2d.shape[0]-patch_size)/step_size)
        map_row = int((body_2d.shape[1]-patch_size)/step_size)
         
        #prediction = np.zeros((map_col, map_row), dtype = float)
        time1 = time.time()
        particle_candidate_all = []
        map_index_col = 0
        for col in range(0, body_2d.shape[0]-patch_size+1, step_size):
            for row in range(0, body_2d.shape[1]-patch_size+1, step_size):
                # extract the particle patch
                patch = np.copy(body_2d[col:(col+patch_size), row:(row+patch_size)])
                # do preprocess to the particle
                patch = DataLoader.preprocess_particle(patch, self.model_input_size)
                particle_candidate_all.append(patch)
                num_total_patch = num_total_patch + 1
            map_index_col = map_index_col + 1

        map_index_row = map_index_col-map_col+map_row
        #print("map_col:",map_col)
        #print("map_row:",map_row)
        #print(len(particle_candidate_all))
        #print("map_index_col:",map_index_col)
        #print("map_index_row:",map_index_row)
        #print("col*row:",map_index_col*map_index_row)
        # reshape it to fit the input format of the model
        particle_candidate_all = np.array(particle_candidate_all).reshape(num_total_patch, self.model_input_size[1], self.model_input_size[2], 1)
        # predict
        predictions = self.deepModel.evaluation(particle_candidate_all, self.sess)
        predictions = predictions[:, 1:2]
        predictions = predictions.reshape(map_index_col, map_index_row)

        time_cost = time.time() - time1
        print("time cost: %d s"%time_cost)
        #display.save_image(prediction, "prediction.png")
        # get the prediction value to be a positive sample, it is a value between 0~1
        # the following code not tested
        # do a connected component analysis
        # prediction = detete_large_component(prediction)

        # do a local peak detection to get the best coordinate
        # list_coordinate is a 2D list of shape (number_particle, 3)
        # element in list_coordinate is [x_coordinate, y_coordinate, prediction_value]
        list_coordinate = self.peak_detection(predictions, local_window_size)
        # add the mrc filename to the list of each coordinate
        for i in range(len(list_coordinate)):
            list_coordinate[i].append(mrc_filename)
            # transform the coordinates to the original size 
            list_coordinate[i][0] = (list_coordinate[i][0]*step_size+patch_size/2)*bin_size
            list_coordinate[i][1] = (list_coordinate[i][1]*step_size+patch_size/2)*bin_size
            
        return list_coordinate
  
    @staticmethod
    def write_coordinate(coordinate, mrc_filename, coordinate_symbol, threshold, output_dir):
        """Write the picking results in the Relion '.star' format.

        This function selects the particles based on the given threshold and saves these particles in Relion '.star' file. 

        Args:
            coordinate: a list, all the coordinates in it are come from the same micrograph. 
                        The length of the list stands for the number of the particles.
                        And each element in the list is a small list of length of 3 at least.
                        The first element in the small list is the coordinate x-aixs. 
                        The second element in the small list is the coordinate y-aixs. 
                        The third element in the small list is the prediction score. 
                        The fourth element in the small list is the micrograph name. 
            mrc_filename: string, the corresponding micrograph file.
            coordinate_symbol: the symbol is used in the output star file name, like '_manualPick', '_cnnPick'. 
            threshold: particles over the threshold are stored, a default value is 0.5.
            output_dir: the directory to store the coordinate file.
        """
        mrc_basename = os.path.basename(mrc_filename)
        print(mrc_basename)
        coordinate_name = os.path.join(output_dir, mrc_basename[:-4]+coordinate_symbol+".star")
        print(coordinate_name)
        f = open(coordinate_name, 'w')
        f.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
        for i in range(len(coordinate)):
            if coordinate[i][2] > threshold:
                f.write(str(coordinate[i][0])+' '+str(coordinate[i][1])+'\n') 

        f.close() 
    
    @staticmethod
    def write_pick_results(coordinate, output_file):
        """Write the picking results in a file of binary format.

        This function writes the coordinates of all micrographs into a binary file. 

        Args:
            coordinate: a list, the length of it stands for the number of picked micrograph file.
                        Each element is a list too, which contains all coordinates from the same micrograph. 
                        The length of the list stands for the number of the particles.
                        And each element in the list is a small list of length of 4.
                        The first element in the small list is the coordinate x-aixs. 
                        The second element in the small list is the coordinate y-aixs. 
                        The third element in the small list is the prediction score. 
                        The fourth element in the small list is the micrograh name. 
            output_file: string, the output file.
        """
        with open(output_file, 'wb') as f:
            pickle.dump(coordinate, f)
    
    @staticmethod
    def analysis_pick_results(pick_results_file, reference_coordinate_dir, reference_coordinate_symbol, particle_size, minimum_distance_rate):
        """Load the picking results from a file of binary format and compare it with the reference coordinate.

        This function analysis the picking results with reference coordinate and calculate the recall, precision and the deviation from the center.

        Args:
            pick_results_file: string, the file name of the pre-picked results.
            reference_mrc_dir: string, the directory of the mrc file dir.
            reference_coordinate_symbol: the symbol of the coordinate, like '_manualpick'
            particle_size: int, the size of particle
            minimum_distance_rate: float, the default is 0.2, a picked coordinate is considered to be a true positive only when the distance between the picked coordinate and the reference coordinate is less than minimum_distance_rate mutiplicate particle_size.
        """
        with open(pick_results_file, 'rb') as f:
            coordinate = pickle.load(f)
            """
            coordinate: a list, the length of it stands for the number of picked micrograph file.
                        Each element is a list too, which contains all coordinates from the same micrograph. 
                        The length of the list stands for the number of the particles.
                        And each element in the list is a small list of length of 4.
                        The first element in the small list is the coordinate x-aixs. 
                        The second element in the small list is the coordinate y-aixs. 
                        The third element in the small list is the prediction score. 
                        The fourth element in the small list is the micrograh name. 
            """
        tp = 0
        total_pick = 0
        total_reference = 0
        coordinate_total = []
        for i in range(len(coordinate)):
            mrc_filename = os.path.basename(coordinate[i][0][3])
            #print(mrc_filename)
            reference_coordinate_file = mrc_filename.replace('.mrc', reference_coordinate_symbol+'.star')
            reference_coordinate_file = os.path.join(reference_coordinate_dir, reference_coordinate_file)
            #print(reference_coordinate_file)
            if os.path.isfile(reference_coordinate_file):
                reference_coordinate = DataLoader.read_coordinate_from_star(reference_coordinate_file)
                """
                reference_coordinate: a list, the length of it stands for the number of picked particles.
                            And each element in the list is a small list of length of 2.
                            The first element in the small list is the coordinate x-aixs. 
                            The second element in the small list is the coordinate y-aixs. 
                """    
                tp_sigle, average_distance = AutoPicker.calculate_tp(coordinate[i], reference_coordinate, particle_size*minimum_distance_rate)
                #print("tp:",tp_sigle)
                #print("average_distance:",average_distance)
                # calculate the number of true positive, when the threshold is set to 0.5
                tp_sigle = 0
                total_reference = total_reference + len(reference_coordinate)
                for j in range(len(coordinate[i])):
                    coordinate_total.append(coordinate[i][j])
                    if coordinate[i][j][2]>0.5:
                        total_pick = total_pick + 1
                        if coordinate[i][j][4] == 1:
                            tp = tp + 1
                            tp_sigle = tp_sigle + 1
                print(tp_sigle/len(reference_coordinate))
            else:
                print("Can not find the reference coordinate:"+reference_coordinate_file)
        precision = tp/total_pick
        recall = tp/total_reference
        print("(threshold 0.5)precision:%f recall:%f"%(precision, recall))
        # sort the coordinate based on prediction score in a descending order.
        coordinate_total = sorted(coordinate_total, key = itemgetter(2), reverse = True) 
        total_tp = []
        total_recall = []
        total_precision = []
        total_probability = []
        total_average_distance = []
        total_distance = 0
        tp_tem = 0
        for i in range(len(coordinate_total)):
            if coordinate_total[i][4] == 1:
                tp_tem = tp_tem + 1
                total_distance = total_distance + coordinate_total[i][5]
            precision = tp_tem/(i+1)
            recall = tp_tem/total_reference
            total_tp.append(tp_tem)
            total_recall.append(recall)
            total_precision.append(precision)
            total_probability.append(coordinate_total[i][2])
            if tp_tem==0:
                average_distance = 0
            else:
                average_distance = total_distance/tp_tem
            total_average_distance.append(average_distance)
        # write the list results in file
        directory_pick = os.path.dirname(pick_results_file)
        total_results_file = os.path.join(directory_pick, 'results.txt')
        f = open(total_results_file, 'w')
        # write total_tp
        f.write(','.join(map(str, total_tp))+'\n')
        f.write(','.join(map(str, total_recall))+'\n')
        f.write(','.join(map(str, total_precision))+'\n')
        f.write(','.join(map(str, total_probability))+'\n')
        f.write(','.join(map(str, total_average_distance))+'\n')
        f.write('#total autopick number:%d\n'%(len(coordinate_total))) 
        f.write('#total manual pick number:%d\n'%(total_reference))
        f.write('#the first row is number of true positive\n')
        f.write('#the second row is recall\n')
        f.write('#the third row is precision\n')
        f.write('#the fourth row is probability\n')
        f.write('#the fiveth row is distance\n')    
        
        # show the recall and precision
        times_of_manual = len(coordinate_total)//total_reference + 1
        for i in range(times_of_manual):
            print('autopick_total sort, take the head number of total_manualpick * ratio %d'%(i+1))
            f.write('#autopick_total sort, take the head number of total_manualpick * ratio %d \n'%(i+1))
            if i==times_of_manual-1:
                print('precision:%f \trecall:%f'%(total_precision[-1], total_recall[-1]))
                f.write('precision:%f \trecall:%f \n'%(total_precision[-1], total_recall[-1]))
            else:
                print('precision:%f \trecall:%f'%(total_precision[(i+1)*total_reference-1], total_recall[(i+1)*total_reference-1]))
                f.write('precision:%f \trecall:%f \n'%(total_precision[(i+1)*total_reference-1], total_recall[(i+1)*total_reference-1]))
        f.close()

      
    @staticmethod
    def calculate_tp(coordinate_pick, coordinate_reference, threshold):
        if len(coordinate_pick)<1 or len(coordinate_reference)<1:
            print("Invalid coordinate parameters in function calculate_tp()!")
        
        # add a symbol to index whether the coordinate is matched with a reference coordinate
        for i in range(len(coordinate_pick)):
            coordinate_pick[i].append(0)

        tp = 0
        average_distance = 0

        for i in range(len(coordinate_reference)):
            coordinate_reference[i].append(0)
            coor_x = coordinate_reference[i][0]
            coor_y = coordinate_reference[i][1]
            neighbour = []
            for k in range(len(coordinate_pick)):
                if coordinate_pick[k][4]==0:
                    coor_mx = coordinate_pick[k][0]
                    coor_my = coordinate_pick[k][1]
                    abs_x = math.fabs(coor_mx-coor_x)
                    abs_y = math.fabs(coor_my-coor_y)
                    length = math.sqrt(math.pow(abs_x, 2)+math.pow(abs_y, 2)) 
                    if length < threshold: 
                        same_n = [] 
                        same_n.append(k)
                        same_n.append(length)
                        neighbour.append(same_n)
            if len(neighbour)>=1: 
                if len(neighbour)>1:
                    neighbour = sorted(neighbour, key = itemgetter(1))
                index = neighbour[0][0]
                # change the symbol to 1, means it matchs with a reference coordinate
                coordinate_pick[index][4] = 1
                # add the distance to the list
                coordinate_pick[index].append(neighbour[0][1])
                coordinate_pick[index].append(coor_x)
                coordinate_pick[index].append(coor_y)
                tp = tp + 1 
                average_distance = average_distance+neighbour[0][1]
                coordinate_reference[i][2] = 1
        average_distance = average_distance/tp
        return tp, average_distance
