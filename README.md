# DeepPicker

------- By Wang Feng 2016/06/14-------

More details about 'DeepPicker', please refer to the paper [DeepPicker](https://arxiv.org/abs/1605.01838). 
This is the python version based on TensorFlow. 
It only supports Ubuntu 12.0+, centOS 7.0+, and RHEL 7.0+.

## 1. Install TensorFlow 
Please refer to the website of [Tensorflow](https://www.tensorflow.org/) to install it. CUDA-Toolkit 7.5 is required to install the GPU version. There are 5 different ways to install tensorflow, and "Virtualenv install" is recommended for not impacting any existing Python programs on your machine.

## 2. Install other python packages
    
    # install package matplotlib and scipy
    # ubuntu system
    > sudo apt-get install python-matplotlib
    > sudo apt-get install python-scipy
    
## 3. Training model
The main script for trainig a model is `train.py`. There are 4 ways to train a CNN model.

Type 1: it is for single-molecule and trained with a semi-automated manner. Load the training data from micrograph directory directly.

Type 2: it is for multi-molecules and trained with a cross-molecule manner . Load the training data from numpy data struct files. This is the only way to train a cross-molecule model.

Type 3: Load the training data from the relion `.star` file. The `.star` should contains all the training positive samples. It can be a `classification2D.star` or `classification3D.star` file. The program will extract all the particles in the `.star` file as the positive training samples to train a CNN model. It mainly aims for optimizing the picking results after optimize the sample through relion classification operation. 
 
Type 4: Load the training data from pre-picked results. It is used for iteration training. This way is very tricky.

All the following demo running command can be found in the `Makefile`.

### 3.1 Train Type 1
Options for training directly from micrographs directory:

    --train_type, 1, specify the training type
    --train_inputDir, string, specify the directory of micrograph files, like '/media/bioserver1/Data/paper_test/trpv1/train/'
    --particle_size, int, the size of the particle
    --mrc_number, int, the default value is -1, so all the micrographs with coordinate file will be used for training.
    --particle_number, int, the default value is -1, so all the particles extracted will be used as training samples.
    --coordinate_symbol, string, the symbol of the coordinate file, like '_manualpick'. The coordinate files should be in the same directory as micrographs.
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.

run the script `train.py`:

    python train.py --train_type 1 --train_inputDir '/media/bioserver1/Data/paper_test/trpv1/train' --particle_size 180 --mrc_number 100 --particle_number 10000 --coordinate_symbol '_manual_checked' --model_save_dir '../trained_model' --model_save_file 'model_demo_type1'

After finished, the trained model will be saved in **'../trained_model/model_demo_type1'**.

### 3.2 Train Type2
Before training a model based on different molecules, you should extract the positive samples and negative samples from different molecule into binary files through script `extractData.py`.
#### 3.2.1 extract particles into numpy binary file
Options for extracting the positive and negative samples into a file:

    --inputDir, string, specify the directory of micrograph files
    --particle_size, int, the size of the particle
    --mrc_number, int, the default value is -1, so all particles in the micrographs with coordinate file will be extracted.
    --coordinate_symbol, string, the symbol of the coordinate file, like '_manualpick'. The coordinate files should be in the same directory as micrographs.
    --save_dir, string, specify the diretory to save the extracted samples.
    --save_file, string, specify the file to save the extracted samples, e.g., 'gammas.pickle'

run the script `extractData.py`:

    # extract the samples of spliceosome
    python extractData.py --inputDir '/media/bioserver1/Data/paper_test/ss/train' --particle_size 320 --mrc_number 300 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'ss.pickle'
    
    # extract the samples of gamma-secretase
    python extractData.py --inputDir '/media/bioserver1/Data/paper_test/gammas/train/' --particle_size 180 --mrc_number 100 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'gammas.pickle'

After finished, the particles of molecule "spliceosome" and molecule "gamma-secretase" are stored in **'../extracted_data/ss.pickle'** and **'../extracted_data/gammas.pickle'** respectively.

#### 3.2.2 training
Options for cross-molecule training:

    --train_type, 2, specify the training type
    --train_inputDir, string, specify the input directory, like '../extracted_data'
    --train_inputFile, string, specify the input file, like 'ss.pickle;gammas.pickle', the separator must be ';'. So the particles from different molecules will be trained together.
    --particle_number, int, the default value is -1, so all the particles in the data file will be used for training. If it is set to 10000, and there are two kind of molecules, then each one contributes only 5,000 positive samples.  
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.
    
run the script `train.py`:

    python train.py --train_type 2 --train_inputDir '../extracted_data' --train_inputFile 'ss.pickle;gammas.pickle' --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type2_ss_gammas'

After finished, the trained model will be saved in **'../trained_model/model_demo_type2_ss_gammas'**.  The model was trained by 2 different molecules, each contributes 5,000 positive training samples.  

### 3.3 Train Type 3
Options for training:

    --train_type, 3, specify the training type
    --train_inputFile, string, specify the input `.star` file, like '/${YOUR_PATH}/classification2D.star'
    --particle_size, int, the size of the particle
    --particle_number, int, the default value is -1, so all the particles in the `classification2D.star` file will be used as training samples. It was set to 10000 in our paper. 
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.

run the script `train.py`:
    
    python train.py --train_type 3 --train_inputFile '/media/bioserver1/Data/paper_test/trpv1/train/trpv1_manualpick_less.star' --particle_size 180 --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type3'

After finished, the trained model will be saved in **'../trained_model/model_demo_type3'**.

### 3.4 Train Type 4
Before we do the iteration training, we need the pick the particles based on previous trained model. Suppose we have done the picking step in Section 3. Then we can training a new model based on the picked results.
Options for training the model based on pre-picked results:

    --train_type, 4, specify the training type
    --train_inputDir, string, specify the input directory of the micrograph files
    --train_inputFile, string, specify the input file of the pre-picked results, like '/PICK_PATH/autopick_results.pickle'
    --particle_number, value, if the value is ranging (0,1), then it means the prediction threshold. If the value is ranging (1,100), then it means the proportion of top sorted ranking particles. If the value is larger than 100, then it means the number of top sorted ranking particles.

run the script `train.py`:
 
    python train.py --train_type 4 --train_inputDir '/media/bioserver1/Data/paper_test/trpv1/test/' --train_inputFile '../autopick-trpv1-by-demo-ss-gammas/autopick_results.pickle' --particle_size 180 --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type4_trpv1_iter1_by_ss_gammas'

After finished, the trained model will be saved in **'../trained_model/model_demo_type4_trpv1_iter1_by_ss_gammas'**

## 4. Picking
Picking the particles based on previous trained model.
Options:

    --inputDir, string, specify the directory of micrograph files 
    --particle_size, int, the size of the particle
    --mrc_number, int, the default value is -1, so the micrographs in the directory will be picked.
    --pre_trained_model, string, specify the pre-trained model.
    --outputDir, string, specify the directory of output coordinate files
    --coordinate_symbol, string, the symbol of the saved coordinate file, like '_cnnPick'. 
    --threshold, float, specify the threshold to pick particle, the default is 0.5.
 
 run the script `train.py`:
    
    python autoPick.py --inputDir '/media/bioserver1/Data/paper_test/trpv1/test/' --pre_trained_model '../trained_model/model_demo_type2_ss_gammas' --particle_size 180 --mrc_number 20 --outputDir '../autopick-trpv1-by-demo-ss-gammas' --coordinate_symbol '_cnnPick' --threshold 0.5


After finished, the picked coordinate file will be saved in **'../autopick-trpv1-by-demo-ss-gammas'**. The format of the coordinate file is Relion '.star'.

Besides, a binary file called **'../autopick-trpv1-by-demo-ss-gammas/autopick_results.pickle'** is produced. It contains all the particles information no matter the threshold. It will be used to do an iteration training or to estimate the precision and recall compared to reference results like manually picking.

## 5. Comparing the picking results with reference
Estimate the precision and recall based on the reference results (like manually picking by experts). The script `analysis_pick_results.py` 

Options:
 
    --inputFile, string, specify the file of picking results, like '/PICK_PATH/autopick_results.pickle'  
    --inputDir, string, specify the directory of the reference coordinate files 
    --particle_size, int, the size of the particle
    --coordinate_symbol, string, the symbol of the reference coordinate file, like '_manualPick'. 
    --minimum_distance_rate, float, Use the value particle_size*minimum_distance_rate as the distance threshold for estimate the number of true positive samples, the default value is 0.2.
    
run the script `analysis_pick_results.py`:

    python analysis_pick_results.py --inputFile '../autopick-trpv1-by-demo-ss-gammas/autopick_results.pickle' --inputDir '/media/bioserver1/Data/paper_test/trpv1/test' --particle_size 180 --coordinate_symbol '_refine_frealign' --minimum_distance_rate 0.2

After finished, a result file `../autopick-trpv1-by-demo-ss-gammas/results.txt` will be produced. It records the precision and recall value as well as the deviation of the center.

## 6. Recommended procedure
Step 1, do the picking job(see Section 4) based on the pre-trained demo model './trained_model/model_demo_type3', or you can train your own model(see Section 3.2).

    python autoPick.py --inputDir 'Your_mrc_file_DIR' --pre_trained_model './trained_model/model_demo_type3' --particle_size Your_particle_size --mrc_number 100 --outputDir '../autopick-results-by-demo-type3' --coordinate_symbol '_cnnPick' --threshold 0.5

Step 2, do the iteration training(see Section 3.4):

    python train.py --train_type 4 --train_inputDir 'Your_mrc_file_DIR' --train_inputFile '../autopick-results-by-demo-type3/autopick_results.pickle' --particle_size Your_particle_size --particle_number 10000 --model_save_dir './trained_model' --model_save_file 'model_demo_type3_iter1_by_type3'
    
Step 3, do the picking job(see Section 4):

    python autoPick.py --inputDir 'Your_mrc_file_DIR' --pre_trained_model './trained_model/model_demo_type3_iter1_by_type3' --particle_size Your_particle_size --mrc_number 100 --outputDir '../autopick-results-by-demo-type3-iter1' --coordinate_symbol '_cnnPick' --threshold 0.5
    
Step 4, do the 2D classification job using Relion based on the picked coordinate file in '../autopick-results-by-demo-type3-iter1'.
Save the good 2D average results in a '.star' file, like 'classification2D_demo.star'.

Step 5, do the training job based on the 'classification2D_demo.star'(see Section 3.3)

    python train.py --train_type 3 --train_inputFile '/Your_DIR/classification2D_demo.star' --particle_size Your_particle_size --particle_number -1 --model_save_dir './trained_model' --model_save_file 'model_demo_type3_2D'
    
Step 6, do the final picking job.

    python autoPick.py --inputDir 'Your_mrc_file_DIR' --pre_trained_model './trained_model/model_demo_type3_2D' --particle_size Your_particle_size --mrc_number -1 --outputDir '../autopick-results-by-demo-type3-2D' --coordinate_symbol '_cnnPick' --threshold 0.5

So the final picked coordinate files are produced in '../autopick-results-by-demo-type3-2D'.
