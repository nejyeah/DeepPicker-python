from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import scipy.misc

def plot_circle_in_micrograph(micrograph_2d, coordinate, particle_size, filename, color = 'white'):
    """plot the particle circle in micrograph image 

    Based on the coordinate of particle, plot circles of the particles in the micrograph.
    And save the ploted image in filename.
 
    Args:
        micrograph_2d: numpy.array,it is a 2D numpy array.
        coordinate: list, it is a 2D list, the shape is (num_particle, 2).
        particle_size: int, the value of the particle size
        filename: the filename of the image to be save.
        color: define the color of the circle

    Raises:
        pass
    """
    micrograph_2d = micrograph_2d.reshape(micrograph_2d.shape[0], micrograph_2d.shape[1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')
    plt.gray()
    plt.imshow(micrograph_2d)
    radius = particle_size/2
    i = 0 
    while True: 
        if i >= len(coordinate):
            break
        coordinate_x = coordinate[i][0]
        coordinate_y = coordinate[i][1]
        cir1 = Circle(xy = (coordinate_x, coordinate_y), radius = radius, alpha = 0.5, color = color, fill = False)
        ax.add_patch(cir1)
        # extract the particles
        i = i + 1
    plt.savefig(filename)

def save_image(image_2d, filename):
    scipy.misc.imsave(filename, image_2d)

def show_particle(numpy_array, filename):
    numpy_array_small = numpy_array[:100, ...]
    numpy_array_small = numpy_array_small.reshape(numpy_array_small.shape[0], numpy_array_small.shape[1], numpy_array_small.shape[2])
    plt.figure(1)
    index = 1
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, index)
            plt.gray()
            plt.imshow(numpy_array_small[index-1])
            plt.axis('off')
            index = index + 1
    plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.99, bottom=0.01, left=0.01, right=0.99)
    plt.savefig(filename) 


