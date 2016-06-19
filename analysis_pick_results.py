from autoPicker import AutoPicker
from optparse import OptionParser

def analysis_results():
    parser = OptionParser()
    parser.add_option("--inputFile", dest="inputFile", help="Input picking results file, like '/PATH/autopick_results.list'", metavar="FILE")
    parser.add_option("--inputDir", dest="inputDir", help="Reference coordinate directory", metavar="DIRECTORY")
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol", help="The symbol of the coordinate file, like '_manualPick'", metavar="STRING")
    parser.add_option("--particle_size", dest="particle_size", help="the size of the particle.", metavar="VALUE", default=-1)
    parser.add_option("--minimum_distance_rate", dest="minimum_distance_rate", help="Use the value particle_size*minimum_distance_rate as the distance threshold for estimate the number of true positive samples, the default value is 0.2", metavar="VALUE", default=0.2)
    (opt, args) = parser.parse_args()

    pick_results_file = opt.inputFile
    reference_mrc_file_dir = opt.inputDir
    reference_coordinate_symbol = opt.coordinate_symbol
    particle_size = int(opt.particle_size)
    minimum_distance_rate = float(opt.minimum_distance_rate)
    AutoPicker.analysis_pick_results(pick_results_file, reference_mrc_file_dir, reference_coordinate_symbol, particle_size, minimum_distance_rate)

def main(argv=None):
    analysis_results()

if __name__ == '__main__':
    main()
