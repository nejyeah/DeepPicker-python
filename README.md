# DeepPicker
------- By Wang Feng 2016/06/14-------
More detail about 'DeepPicker', please refer to the paper [DeepPicker](https://arxiv.org/abs/1605.01838). 
This is the python version, the code is based on the TensorFlow. And it supports all linux platform.


## 1. Install TensorFlow 
Please refer to the offical website of [Tensorflow](https://www.tensorflow.org/) to install it. A GPU version is required. There are 5 different ways to install tensorflow, and "Virtualenv install" is recommended for not impacting any existing Python programs on your machine.

## 2. Training Model
The main function for trainig a model is `train.py`. There are 4 types to train a model.
Type 1: Load the training data from the relion `.star` file. The `.star` should contains all the training particles. It can be a `classification2D.star` or `classification3D.star` file. The program will extract all the particles in the `.star` file as the positive training samples to train a CNN model. Model detail refer to section 2.1

Type 2: Load the training data from numpy data struct. This is the only way to train a cross-molecule model. More detail refer to section 2.2.

Type3: Load the training data from pre-picked results. More detail refer to section 2.3.

Type4: Load the training data from micrograph directory directly. More detail refer to section 2.4.

### 2.1 Train Type 1

Options for training:
    
    --train_type, 1, specify the training type
    --train_inputFile, string, specify the input `.star` file, like '/${YOUR_PATH}/classification2D.star'
    --particle_size, int, the size of the particle
    --particle_number, int, the default value is -1, so all the particles in the `demo.star` file will be used as training samples. It was set to 10000 in our paper. 
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.

run the script `train.py`:
    
    python train.py --train_type 1 --train_inputFile '../demo/demo.star' --particle_size 180 --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'Model_demo'

After finished, the trained model will be saved in **'../trained_model/Model_demo'**.

### 2.2 Train Type2
Before training a model based on different molecules, you should extract the positive samples and negative samples from different molecule into a binary file through script `extractData.py`.
### 2.2.1 extract particles into numpy binary file
Options for extracting the positive and negative samples into a file:

    --inputDir, string, specify the directory of micrograph files
    --particle_size, int, the size of the particle
    --mrc_number, int, the default value is -1, so all particles in the micrographs with coordinate file will be extracted.
    --coordinate_symbol, string, the symbol of the coordinate file, like '_manualpick'. The coordinate files should be in the same directory as micrographs.
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.

run the script `extractData.py`:
  
    python extractData.py --inputDir '/media/bioserver1/Data/paper_test/trpv1/train/lowpass' --mrc_number 100 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'trpv1.pickle'

After finished, the particles of molecule trpv1 are stored in file **'../extracted_data/trpv1.pickle'**.
Run this script to extract all the training samples into file.

### 2.2.2 training
Options for cross-molecule training:

    --train_type, 2, specify the training type
    --train_inputDir, string, specify the input directory, like '../extracted_data'
    --train_inputFile, string, specify the input file, like 'trpv1.pickle;gammas.pickle;ss.pickle', the separator must be ';'. So the particles from different molecules will be trained together.
    --particle_number, int, the default value is -1, so all the particles in the `demo.star` file will be used as training samples. It was set to 10000 in our paper. 
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.
    
run the script `train.py`:
    
    python train.py --train_type 2 --train_inputDir '../extracted_data' --train_inputFile 'trpv1.pickle;gammas.pickle;ss.pickle' --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'Model_demo'

After finished, the trained model will be saved in **'../trained_model/Model_demo'**.  The model was trained by 3 different molecules, each contributes 3333 positive training samples.  

## 2.3 Train Type 3

## 2.4 Train Type 4
Options for training directly from micrographs directory:

    --train_type, 4, specify the training type
    --train_inputDir, string, specify the directory of micrograph files, like '/media/bioserver1/Data/paper_test/trpv1/train/lowpass'
    --particle_size, int, the size of the particle
    --mrc_number, int, the default value is -1, so all particles in the micrographs with coordinate file will be extracted.
    --particle_number, int, the default value is -1, so all the particles extracted will be used as training samples.
    --coordinate_symbol, string, the symbol of the coordinate file, like '_manualpick'. The coordinate files should be in the same directory as micrographs.
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.

run the script `train.py`:
    
    python train.py --train_type 4 --train_inputDir '/media/bioserver1/Data/paper_test/trpv1/train/lowpass' --particle_size 180 --mrc_number 100 --particle_number 10000 --coordinate_symbol '_manual_cheecked' --model_save_dir '../trained_model' --model_save_file 'Model_demo'

After finished, the trained model will be saved in **'../trained_model/Model_demo'**.

## 3. Picking
Picking the particles based on pre-trained model.
Options:

    --inputDir, string, specify the directory of micrograph files 
    --particle_size, int, the size of the particle
    --mrc_number, int, the default value is -1, so all particles in the micrographs with coordinate file will be picked.
    --pre_trained_model, string, specify the pre-trained model.
    --outputDir, string, specify the directory of output coordinate files
    --coordinate_symbol, string, the symbol of the saved coordinate file, like '_cnnPick'. 
    --threshold, float, specify the threshold to pick particle, the default is 0.5.
 
 run the script `train.py`:
    
    python autopick.py --inputDir '/media/bioserver1/Data/paper_test/trpv1/test/lowpass' --pre_trained_model '../trained_model/Model_demo' --particle_size 180 --mrc_number 100 --outputDir '../autopick-trpv1-by-demo' --coordinate_symbol '_cnnPick' --threshold 0.5

After finished, the picked coordinate file will be saved in **'../autopick-trpv1-by-demo'**. The format of the coordinate file is Relion '.star'.
