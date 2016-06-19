from dataLoader import DataLoader

import os
from optparse import OptionParser

def extractData():
    parser = OptionParser()
    parser.add_option("--inputDir", dest="inputDir", help="Input directory", metavar="DIRECTORY")
    parser.add_option("--mrc_number", dest="mrc_number", help="Number of mrc files to be trained.", metavar="VALUE", default=-1)
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol", help="The symbol of the coordinate file, like '_manualPick'", metavar="STRING")
    parser.add_option("--particle_size", dest="particle_size", help="the size of the particle.", metavar="VALUE", default=-1)
    parser.add_option("--save_dir", dest="save_dir", help="save the training samples to this directory", metavar="DIRECTORY", default="../trained_model")
    parser.add_option("--save_file", dest="save_file", help="save the training samples to file", metavar="FILE")
    (opt, args) = parser.parse_args()

    inputDir = opt.inputDir
    particle_size = int(opt.particle_size)
    coordinate_symbol = opt.coordinate_symbol
    mrc_number = int(opt.mrc_number)
    output_dir = opt.save_dir
    output_filename = opt.save_file
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if particle_size == -1:
        print("particle size should be a positive value!")
        return 

    output_filename = os.path.join(output_dir, output_filename)
    DataLoader.extractData(inputDir, particle_size, coordinate_symbol, mrc_number, output_filename)

def main(argv=None):
    extractData()

if __name__ == '__main__':
    main()
