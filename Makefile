# demo excution to the scripts as the `README.md` suggested.
trainType1:
	python train.py --train_type 1 --train_inputDir '/media/bioserver1/Data/paper_test/trpv1/train/lowpass' --particle_size 180 --mrc_number 100 --particle_number 10000 --coordinate_symbol '_manual_checked' --model_save_dir '../trained_model' --model_save_file 'model_demo_type1'

extractData:
	python extractData.py --inputDir '/media/bioserver1/Data/paper_test/ss/train/lowpass' --particle_size 180 --mrc_number 100 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'ss_lowpass.pickle'
	python extractData.py --inputDir '/media/bioserver1/Data/paper_test/gammas/train/lowpass' --particle_size 180 --mrc_number 100 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'gammas_lowpass.pickle'

trainType2:
	python train.py --train_type 2 --train_inputDir '../extracted_data' --train_inputFile 'ss_lowpass.pickle;gammas_lowpass.pickle' --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type2_ss_gammas'

trainType3:
	python train.py --train_type 3 --train_inputFile '/media/bioserver1/Data/paper_test/trpv1/train/trpv1_manualpick_less.star' --particle_size 180 --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type1'
        
pick:
	python autoPick.py --inputDir '/media/bioserver1/Data/paper_test/trpv1/test/lowpass' --pre_trained_model '../trained_model/model_demo_type2_ss_gammas' --particle_size 180 --mrc_number 20 --outputDir '../autopick-trpv1-by-demo-ss-gammas' --coordinate_symbol '_cnnPick' --threshold 0.5

trainType4:
	python train.py --train_type 4 --train_inputDir '/media/bioserver1/Data/paper_test/trpv1/test/lowpass' --train_inputFile '../autopick-trpv1-by-demo-ss-gammas/autopick_results.pickle' --particle_size 180 --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type4_trpv1_iter1_by_ss_gammas'

analysis:
	python analysis_pick_results.py --inputFile '../autopick-trpv1-by-demo-ss-gammas/autopick_results.pickle' --inputDir '/media/bioserver1/Data/paper_test/trpv1/test/lowpass' --particle_size 180 --coordinate_symbol '_refine_frealign' --minimum_distance_rate 0.2
