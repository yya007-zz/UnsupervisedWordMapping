from cyclegan-text.CycleGAN import *

if __name__ == '__main__':

	exp_01={
		"description": "The word translation cyclegan.", 
		"pool_size": 50,
		"base_lr":0.0002,
		"max_step": 200,
		"network_version": "tensorflow",
		"train_dataset_name": "./data/en_it_train_vec.npy",
		"test_dataset_name": "./data/en_it_test_vec.npy",
		"_LAMBDA_A": 10,
		"_LAMBDA_B": 10

		"to_train" : 1
		"log_dir" : './output/cyclegan/exp_01'
		"checkpoint_dir" : './output/cyclegan/exp_01/#timestamp#'
		"skip" : False;
	}

    
    run_cyclegan(exp_01)