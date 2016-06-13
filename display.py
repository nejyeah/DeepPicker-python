from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import scipy.misc

def plot_circle_in_micrograph(micrograph_2d, coordinate, particle_size, filename, color = 'white'):
    """plot the particle circle in micrograph image 

    Based on the coordinate of particle, plot circles of the particles in the micrograph.
    And save the ploted image in filename.
 
    Args:
        micrograph_2d: numpy.array,it is a 2D numpy array.
        coordinate: list, it is a 2D list, the shape is (2, num_particle).
                    The coordinate[0] is a list of the x coordinate,            
                    and the coordinate[1] is a list of the y coordinate.
        particle_size: int, the value of the particle size
        filename: the filename of the image to be save.
        color: define the color of the circle

    Raises:
        pass
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gray()
    plt.imshow(micrograph_2d)
    radius = particle_size/2
    i = 0 
    while True: 
        if i >= len(coordinate[0]):
            break
        coordinate_x = coordinate[0][i]
        coordinate_y = coordinate[1][i]
        cir1 = Circle(xy = (coordinate_x,coordinate_y), radius = radius, alpha = 0.5, color = color, fill = False)
        ax.add_patch(cir1)
        # extract the particles
        i = i + 1
    plt.savefig(filename)

def save_image(image_2d, filename):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.gray()
    #plt.imshow(image_2d)
    #plt.savefig(filename)
    scipy.misc.imsave(filename, image_2d)


