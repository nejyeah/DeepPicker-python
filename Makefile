# demo excution to the scripts as the `README.md` suggested.
trainType1:
	python train.py --train_type 1 --train_inputDir '/media/bioserver1/Data/paper_test/dataset/trpv1/train/original' --particle_size 180 --mrc_number 100 --particle_number 10000 --coordinate_symbol '_manual_checked' --model_save_dir '../trained_model' --model_save_file 'model_demo_type1'

extractData:
	python extractData.py --inputDir '/media/bioserver1/Data/paper_test/dataset/ss/train/original' --particle_size 320 --mrc_number 100 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'ss_original.pickle'
	python extractData.py --inputDir '/media/bioserver1/Data/paper_test/dataset/gammas/train/original' --particle_size 180 --mrc_number 100 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'gammas_original.pickle'
	python extractData.py --inputDir '/media/bioserver1/Data/paper_test/dataset/trpv1/train/original' --particle_size 180 --mrc_number 100 --coordinate_symbol '_manual_checked' --save_dir '../extracted_data' --save_file 'trpv1_original.pickle'

trainType2:
	python train.py --train_type 2 --train_inputDir '../extracted_data' --train_inputFile 'ss_original.pickle;gammas_original.pickle;trpv1_original.pickle' --particle_number 30000 --model_save_dir './trained_model' --model_save_file 'model_demo_type3'

trainType3:
	python train.py --train_type 3 --train_inputFile '/media/bioserver1/Data/paper_test/dataset/trpv1/train/trpv1_manualpick_less.star' --particle_size 180 --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type1'
        
pick:
	python autoPick.py --inputDir '/media/bioserver1/Data/paper_test/dataset/trpv1/test/original' --pre_trained_model '../trained_model/model_demo_type2_ss_gammas' --particle_size 180 --mrc_number 20 --outputDir '../autopick-trpv1-by-demo-ss-gammas' --coordinate_symbol '_cnnPick' --threshold 0.5

trainType4:
	python train.py --train_type 4 --train_inputDir '/media/bioserver1/Data/paper_test/dataset/trpv1/test/original' --train_inputFile '../autopick-trpv1-by-demo-ss-gammas/autopick_results.pickle' --particle_size 180 --particle_number 10000 --model_save_dir '../trained_model' --model_save_file 'model_demo_type4_trpv1_iter1_by_ss_gammas'

analysis:
	python analysis_pick_results.py --inputFile '../autopick-trpv1-by-demo-ss-gammas/autopick_results.pickle' --inputDir '/media/bioserver1/Data/paper_test/dataset/trpv1/test/original' --particle_size 180 --coordinate_symbol '_refine_frealign' --minimum_distance_rate 0.2

recommended_step1:
	python autoPick.py --inputDir 'Your_mrc_file_DIR' --pre_trained_model './trained_model/model_demo_type3' --particle_size Your_particle_size --mrc_number 100 --outputDir '../autopick-results-by-demo-type3' --coordinate_symbol '_cnnPick' --threshold 0.5
	python autoPick.py --inputDir '/media/bioserver1/Data/paper_test/dataset/trpv1/test/original' --pre_trained_model '../trained_model/model_demo_type2_ss_gammas' --particle_size 180 --mrc_number 20 --outputDir '../autopick-trpv1-by-demo-ss-gammas' --coordinate_symbol '_cnnPick' --threshold 0.5

recommended_step2:

testTrain:
	python train.py --train_type 1 --train_inputDir '../data/Micrographs' --particle_size 60 --mrc_number 100 --particle_number 10000 --coordinate_symbol '_manual' --model_save_dir '../trained_model' --model_save_file 'model_test'

