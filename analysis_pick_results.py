from autoPicker import AutoPicker

pick_results_file = '../autopick-trpv1-by-demo/autopick_results.list'
reference_mrc_file_dir = '/media/bioserver1/Data/paper_test/trpv1/test/lowpass'
reference_coordinate_symbol = '_refine_frealign'
particle_size = 180
minimum_distance_rate = 0.2

AutoPicker.analysis_pick_results(pick_results_file, reference_mrc_file_dir, reference_coordinate_symbol, particle_size, minimum_distance_rate)


