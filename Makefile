# 


trainType1:
	python train.py --train_inputFile -particle_size 180 --particle_number  

extractData:
	python extractData.py --inputDir '/media/bioserver1/Data/paper_test/snare/train/lowpass' --mrc_number 100 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'demo.pickle'	

trainType2:
	python train.py --train_type 2 --train_inputDir '../extract_data' --train_inputFile 'gammas_lowpass10A.pickle;ss_lowpass10A.pickle' --particle_size 180 --particle_number 1000 --model_save_dir '../trained_model' --model_save_file 'Model_demo'

pick:

