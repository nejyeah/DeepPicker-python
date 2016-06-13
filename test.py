import numpy as np
import scipy.misc
import time
import tensorflow as tf
#from dataLoader import DataLoader

def random_test1():
    image_2d = np.random.random_sample((1950,1950,1))
    step = 4
    num_total = 0
    col = (1950-60)/4+1
    number = col*col
    particles = np.zeros((number, 60, 60, 1))
    
    for i in range(0, 1950-60, 4):
        for j in range(0, 1950-60, 4):
            particles[num_total] = image_2d[i:i+60, j:j+60, 0:1]
            num_total = num_total + 1
   
    batch = 1000 
    particle_node = tf.constant(particles[])
    particle_resize = tf.image.resize_images(particle_node, 100, 100)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        particles = sess.run(particle_resize) 
    #max_value = particle.max()
    #min_value = particle.min()
    #particle = (particle-min_value)/(max_value-min_value)
    #particle = scipy.misc.imresize(particle, (60,60))
    #mean_value = particle.mean()
    #std_value = particle.std()
    #particle = (particle-mean_value)/std_value
    #patch = DataLoader.preprocess_particle(patch, [1, 60, 60, 1])
    print(num_total)
    
def random_test():
    time_start = time.time()
    random_test1()
    time_cost = time.time() - time_start
    print("time_cost:",time_cost)

np.random.seed(1234)
random_test()
