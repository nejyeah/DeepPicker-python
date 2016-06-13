from dataLoader import DataLoader
import os

trainInputDir = "/media/bioserver1/Data/paper_test/ss/train/lowpass"
particle_size = 320
coordinate_symbol = "_manual_checked"
mrc_number = -1
output_dir = "../extract_data"
output_filename = "ss_lowpass10A.pickle"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

output_filename = os.path.join(output_dir, output_filename)
DataLoader.extractData(trainInputDir, particle_size, coordinate_symbol, mrc_number, output_filename)

