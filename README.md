# DeepPicker

For more details about 'DeepPicker', please refer to the paper [DeepPicker](https://arxiv.org/abs/1605.01838). 
This is the python version based on [TensorFlow](https://www.tensorflow.org/). 
So far it only supports Ubuntu 12.0+, centOS 7.0+, and RHEL 7.0+.

## 1. Install TensorFlow 
For more details about [Tensorflow](https://www.tensorflow.org/), please refer to the [website](https://www.tensorflow.org/). Cuda toolkit 7.5 and cuDNN v4 are required to install the GPU version of Tensorflow. There are 5 different ways to install tensorflow, and "Virtualenv install" is recommended for not impacting any existing Python program on your machine.

### 1.1 Install Cuda Toolkit 7.5
download and install Cuda Toolkit 7.5

https://developer.nvidia.com/cuda-downloads

### 1.2 Install cudnn v4
Download and install cuDNN v4

https://developer.nvidia.com/cudnn

Uncompress and copy the cuDNN files into the toolkit directory. Assuming the toolkit is installed in /usr/local/cuda, run the following commands (edited to reflect the cuDNN version you downloaded):

    tar xvzf cudnn-7.0-linux-x64-v4.0-prod.tgz
    sudo cp cudnn-7.0-linux-x64-v4/cudnn.h /usr/local/cuda/include
    sudo cp cudnn-7.0-linux-x64-v4/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn

### 1.3 Virtualenv installation of Tensorflow
[Virtualenv](https://pypi.python.org/pypi/virtualenv) is a tool to keep the dependencies required by different Python projects in separate places. The Virtualenv installation of TensorFlow will not override pre-existing version of the Python packages needed by TensorFlow.

With [Virtualenv](https://pypi.python.org/pypi/virtualenv) the installation is as follows:

 - Install pip and Virtualenv.
 - Create a Virtualenv environment.
 - Activate the Virtualenv environment and install TensorFlow in it.
 - After the install you will activate the Virtualenv environment each time you want to use TensorFlow.
 
Install pip abd Virtualenv:

    # Ubuntu/Linux 64-bit
    $ sudo apt-get install python-pip python-dev python-virtualenv

Create a Virtualenv environment in the directory ~/tensorflow:

    $ virtualenv --system-site-packages ~/tensorflow

Activate the environment:

    $ source ~/tensorflow/bin/activate  # If using bash
$ source ~/tensorflow/bin/activate.csh  # If using csh
    (tensorflow)$  # Your prompt should change

Now, install TensorFlow just as you would for a regular Pip installation:

    # Ubuntu/Linux 64-bit, GPU enabled, Python 2.7 
    # Requires CUDA toolkit 7.5 and CuDNN v4. 
    (tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl

With the Virtualenv environment activated, you can now test your installation.

    $ python
    ...
    >>> import tensorflow as tf
    >>> hello = tf.constant('Hello, TensorFlow!')
    >>> sess = tf.Session()
    >>> print(sess.run(hello))
    Hello, TensorFlow!
    >>> a = tf.constant(10)
    >>> b = tf.constant(32)
    >>> print(sess.run(a + b))
    42
    >>>

When you are done using TensorFlow, deactivate the environment.

    (tensorflow)$ deactivate
$  # Your prompt should change back

To use TensorFlow later you will have to activate the Virtualenv environment again:

    $ source ~/tensorflow/bin/activate  # If using bash.
$ source ~/tensorflow/bin/activate.csh  # If using csh.
    (tensorflow)$  # Your prompt should change.
    
    # Run Python programs that use TensorFlow.
    ...
    # When you are done using TensorFlow, deactivate the environment.
    (tensorflow)$ deactivate

### 1.4 Install other python packages
    
    # install package matplotlib and scipy
    # ubuntu system
    > sudo apt-get install python-matplotlib
    > sudo apt-get install python-scipy

## 2. Recommended procedure
### 2.1 fully automated particle picking
This is the way we used in our paper to do the fully automated particle picking. There are three steps.

Step 1, before doing the automatic picking job, a pre-trained model is needed. Here we have offered a demo model in './trained_model/model_demo_type3'. It was trained in a cross-molecule manner (see Section 3.2 in our paper) with three types of molecules, including TRPV1, gammas-secretase and spliceosome. The number of positive samples for training is 30,000. You can either do your automatic particle picking job based on this model or train your own model based on more types of molecules and more training samples (see Section 3.2). After you get a pre-trained model, do the picking job. 

    python autoPick.py --inputDir 'Your_mrc_file_DIR' --pre_trained_model './trained_model/model_demo_type3' --particle_size Your_particle_size --mrc_number 100 --outputDir '../autopick-results-by-demo-type3' --coordinate_symbol '_cnnPick' --threshold 0.5

Step 2, do the iterative training (see Section 3.3):

    python train.py --train_type 3 --train_inputDir 'Your_mrc_file_DIR' --train_inputFile '../autopick-results-by-demo-type3/autopick_results.pickle' --particle_size Your_particle_size --particle_number 10000 --model_save_dir './trained_model' --model_save_file 'model_demo_type3_iter1_by_type3'
    
Step 3, do the final picking job (see Section 3.2):
    
    python autoPick.py --inputDir 'Your_mrc_file_DIR' --pre_trained_model './trained_model/model_demo_type3_iter1_by_type3' --particle_size Your_particle_size --mrc_number -1 --outputDir '../autopick-results-by-demo-type3-iter1' --coordinate_symbol '_cnnPick' --threshold 0.5

So the final picked coordinate files are produced in '../autopick-results-by-demo-type3-iter1'.

### 2.2 cooperate with Relion 2D classification 
This is a practical way to do the particle picking cooperating with Relion 2D classification.

Step 1, before doing the automatic picking job, a pre-trained model is needed. Here we have offered a demo model in './trained_model/model_demo_type3'. It was trained in a cross-molecule manner with three types of molecules, including TRPV1, gammas-secretase and spliceosome. And the number of positive samples for training is 30,000. You can either do your automatic picking job based on this model or train your own model based on more kinds of molecules and more training samples. After you get a pre-trained model, do the automatic particle picking job.

    python autoPick.py --inputDir 'Your_mrc_file_DIR' --pre_trained_model './trained_model/model_demo_type3' --particle_size Your_particle_size --mrc_number 100 --outputDir '../autopick-results-by-demo-type3' --coordinate_symbol '_cnnPick' --threshold 0.4

Step 2, do the 2D classification in Relion based on the picked coordinate files in '../autopick-results-by-demo-type3'. 
Select those good average results to store in a '.star' file, like 'classification2D_demo.star'.

Step 3, do the training job based on the 'classification2D_demo.star' (see Section 3.4)

    python train.py --train_type 4 --train_inputFile '/Your_DIR/classification2D_demo.star' --particle_size Your_particle_size --particle_number -1 --model_save_dir './trained_model' --model_save_file 'model_demo_type3_2D'
    
Step 4, do the final picking job.

    python autoPick.py --inputDir 'Your_mrc_file_DIR' --pre_trained_model './trained_model/model_demo_type3_2D' --particle_size Your_particle_size --mrc_number -1 --outputDir '../autopick-results-by-demo-type3-2D' --coordinate_symbol '_cnnPick' --threshold 0.5

So the final picked coordinate files are produced in '../autopick-results-by-demo-type3-2D'.

## 3. Training the model
The main script for training a model is `train.py`. There are 4 ways to train a CNN model.

Type 1: It aims to train a CNN model based on a single type of molecule. The script loads the training data from micrograph directory directly.

Type 2: It aims to train a CNN model based on multiple types of molecules. It coorperates with script `extractData.py` to train a cross-molecule model (see Section 3.2).

Type 3: It aims to do the iterative training. It is a complement to the fully automatic particle picking which is based on a cross-molecule manner. Here we take the pre-picked particles as training samples to train a new model and then pick the particles based on the new model to mimic the semi-automated manner.

Type 4: It aims to improve the picking results coorperating with Relion 2D classification. It is a complement to the fully automatic particle picking. When the fully automatic particle picking finished, do the Relion 2D classification job to the picked particles and save the good class averaging results in a `.star` file. The program will extract all the particles in the `.star` file as the positive samples to train a CNN model. 
 
All the following commands can be found in the `Makefile`. 

### 3.1 Train Type 1
Options for training model in single-molecule manner are:

    --train_type, 1, specify the training type
    --train_inputDir, string, specify the directory of micrograph files, like '/media/bioserver1/Data/paper_test/trpv1/train/'
    --particle_size, int, the size of the particle
    --mrc_number, int, the default value is -1, so all the micrographs with coordinate files will be used for training.
    --particle_number, int, the default value is -1, so all the extracted particles will be used as training samples.
    --coordinate_symbol, string, the symbol of the coordinate file, like '_manualpick'. The coordinate files should be in the same directory as micrographs.
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.

run the script `train.py`:

    python train.py --train_type 1 --train_inputDir '/media/bioserver1/Data/paper_test/trpv1/train' --particle_size 180 --mrc_number 100 --particle_number 10000 --coordinate_symbol '_manual_checked' --model_save_dir '../trained_model' --model_save_file 'model_demo_type1'

When finished, the trained model will be saved in file **'../trained_model/model_demo_type1'**.

### 3.2 Train Type2
Before training a model in multi-molecule manner, the positive samples and negative samples from different molecules should be extracted through script `extractData.py` at first.
#### 3.2.1 extract particles into numpy binary file
Options for extracting the positive and negative samples into a binary file are:

    --inputDir, string, specify the directory of micrograph files
    --particle_size, int, the size of the particle
    --mrc_number, int, the default value is -1, so all the micrographs with coordinate files will be extracted.
    --coordinate_symbol, string, the symbol of the coordinate file, like '_manualpick'. The coordinate files should be in the same directory as the micrographs.
    --save_dir, string, specify the diretory to save the extracted samples.
    --save_file, string, specify the file to save the extracted samples, e.g., 'trpv1.pickle'

run the script `extractData.py`:
    
    # extract the samples of molecule A
    python extractData.py --inputDir '/media/bioserver1/Data/paper_test/molecule_A/train' --particle_size 320 --mrc_number 300 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'molecule_A.pickle'
    
    # extract the samples of molecule B
    python extractData.py --inputDir '/media/bioserver1/Data/paper_test/molecule_B/train/' --particle_size 180 --mrc_number 100 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'molecule_B.pickle'

When finished, the particles of molecule A and molecule B are stored in **'../extracted_data/molecule_A.pickle'** and **'../extracted_data/molecule_B.pickle'** respectively.

#### 3.2.2 training
Options for training model in multi-molecule manner are:

    --train_type, 2, specify the training type
    --train_inputDir, string, specify the input directory, like '../extracted_data'
    --train_inputFile, string, specify the input file, like 'molecule_A.pickle;molecule_B.pickle', the separator must be ';'.
    --particle_number, int, the default value is -1, so all the particles in the data file will be used for training. If it is set to 10000, and there are two kinds of molecules, then each one contributes only 5,000 positive samples.  
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.
    
run the script `train.py`:

    python train.py --train_type 2 --train_inputDir '../extracted_data' --train_inputFile 'molecule_A.pickle;molecule_B.pickle' --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type2_molecule_A_B'

When finished, the trained model will be saved in file **'../trained_model/model_demo_type2_molecule_A_B'**.  The model was trained by two kinds of molecules, each contributes 5,000 positive training samples.  

### 3.3 Train Type 3
Before we do the iterative training, we need to pick the particles based on pre-trained model. Suppose we have finished the picking step in Section 4. Then we can train a new model based on the picked results.
Options for training model based on pre-picked results are:

    --train_type, 3, specify the training type
    --train_inputDir, string, specify the input directory of the micrograph files
    --train_inputFile, string, specify the input file of the pre-picked results, like '/PICK_PATH/autopick_results.pickle'
    --particle_number, value, if the value is ranging (0,1), then it means the prediction threshold. If the value is ranging (1,100), then it means the proportion of the top sorted ranking particles. If the value is larger than 100, then it means the number of top sorted ranking particles.

run the script `train.py`:
 
    python train.py --train_type 3 --train_inputDir '/media/bioserver1/Data/paper_test/trpv1/test/' --train_inputFile '../autopick-trpv1-by-demo-molecule-A-B/autopick_results.pickle' --particle_size 180 --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type3_trpv1_iter1_by_molecule_A_B'

When finished, the trained model will be saved in file **'../trained_model/model_demo_type3_trpv1_iter1_by_molecule_A_B'**

### 3.4 Train Type 4
Options for training model based on Relion 2D classification results are:

    --train_type, 4, specify the training type
    --train_inputFile, string, specify the input `.star` file, like '/${YOUR_PATH}/classification2D.star'
    --particle_size, int, the size of the particle
    --particle_number, int, the default value is -1, so all the particles in the `classification2D.star` file will be used as training samples.
    --model_save_dir, string, specify the diretory to save the model.
    --model_save_file, string, specify the file to save the model.

run the script `train.py`:
    
    python train.py --train_type 4 --train_inputFile '/media/bioserver1/Data/paper_test/trpv1/train/trpv1_manualpick_less.star' --particle_size 180 --particle_number -1 --model_save_dir '../trained_model' --model_save_file 'model_demo_type4'

When finished, the trained model will be saved in **'../trained_model/model_demo_type4'**.

## 4. Picking
Options for picking particles based on pre-trained model are:

    --inputDir, string, specify the directory of micrograph files 
    --particle_size, int, the size of the particle
    --mrc_number, int, the default value is -1, so the micrographs in the directory will be picked.
    --pre_trained_model, string, specify the pre-trained model.
    --outputDir, string, specify the directory of output coordinate files
    --coordinate_symbol, string, the symbol of the saved coordinate file, like '_cnnPick'. 
    --threshold, float, specify the threshold to pick particle, the default is 0.5.
 
 run the script `train.py`:
    
    python autoPick.py --inputDir '/media/bioserver1/Data/paper_test/trpv1/test/' --pre_trained_model '../trained_model/model_demo_type2_molecule_A_B' --particle_size 180 --mrc_number 20 --outputDir '../autopick-trpv1-by-demo-molecule-A-B' --coordinate_symbol '_cnnPick' --threshold 0.5


When finished, the picked coordinate file will be saved in **'../autopick-trpv1-by-demo-molecule-A-B'**. The format of the coordinate file is Relion '.star'.

Besides, a binary file called **'../autopick-trpv1-by-demo-molecule-A-B/autopick_results.pickle'** is produced. It contains all the particles information. It will be used to do an iterative training or to estimate the precision and recall compared to the reference (e.g., those particles manually picked by experts).

## 5. Comparing the picking results with reference
The script `analysis_pick_results.py` is used to estimate the precision and recall based on the reference results (e.g., those particles manually picked by experts).

Options:
 
    --inputFile, string, specify the file of the picking results, like '/PICK_PATH/autopick_results.pickle'  
    --inputDir, string, specify the directory of the reference coordinate files 
    --particle_size, int, the size of the particle
    --coordinate_symbol, string, the symbol of the reference coordinate file, like '_manualPick'. 
    --minimum_distance_rate, float, take the value particle_size*minimum_distance_rate as the distance threshold for estimate the number of true positive samples, the default value is 0.2.
    
run the script `analysis_pick_results.py`:

    python analysis_pick_results.py --inputFile '../autopick-trpv1-by-demo-molecule-A-B/autopick_results.pickle' --inputDir '/media/bioserver1/Data/paper_test/trpv1/test' --particle_size 180 --coordinate_symbol '_refine_frealign' --minimum_distance_rate 0.2

When finished, a result file `../autopick-trpv1-by-demo-molecule-A-B/results.txt` will be produced. It records the precision and recall values as well as the deviations of the centers from the reference particles.

If you have any questions, please contact us at "*251138964@qq.com*".

