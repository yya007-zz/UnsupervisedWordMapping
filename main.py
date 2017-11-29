# Copyright (c) 2017-present, Yuang Yao
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


# from __future__ import print_function
# import time

# from CycleGAN import *

# def run_cyclegan(config):
#     """
#     :param 
#     to_train: Specify whether it is training or testing. 
#     0: testing
#     1: training; 
#     2: resuming from latest checkpoint; 
#     :param log_dir: The root dir to save checkpoints and the prediction. The actual dir
#     is the root dir appended by the folder with the name timestamp.
#     :param config_filename: The configuration file.
#     :param checkpoint_dir: The directory that saves the latest checkpoint. It
#     only takes effect when to_train == 2.
#     :param skip: A boolean indicating whether to add skip connection between
#     input and output.
#     """

#     to_train=config['to_train']
#     log_dir=config['log_dir']
#     checkpoint_dir=config['checkpoint_dir']
#     skip=config['skip']

#     if not os.path.isdir(log_dir):
#         os.makedirs(log_dir)

#     lambda_a = float(config['_LAMBDA_A']) if '_LAMBDA_A' in config else 10.0
#     lambda_b = float(config['_LAMBDA_B']) if '_LAMBDA_B' in config else 10.0
#     pool_size = int(config['pool_size']) if 'pool_size' in config else 50

#     to_restore = (to_train == 2)
#     base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
#     max_step = int(config['max_step']) if 'max_step' in config else 200
#     network_version = str(config['network_version'])
#     train_dataset_name = str(config['train_dataset_name'])
#     test_dataset_name = str(config['test_dataset_name'])

#     do_test=True
#     if to_train > 0:
#         do_train=True
#     elif not to_restore:
#         raise 

#     cyclegan_model = CycleGAN(do_train, do_test, pool_size, lambda_a, lambda_b, log_dir,
#                               to_restore, base_lr, max_step, network_version,
#                               train_dataset_name, test_dataset_name , checkpoint_dir, skip)

    
#     cyclegan_model.run()

# if __name__ == '__main__':

# 	print('start to run')
# 	exp_01={
# 		"description": "The word translation cyclegan.", 
# 		"pool_size": 50,
# 		"base_lr":0.0002,
# 		"max_step": 200,
# 		"network_version": "tensorflow",
# 		"train_dataset_name": "./data/en_it_train_vec.npy",
# 		"test_dataset_name": "./data/en_it_test_vec.npy",
# 		"_LAMBDA_A": 10,
# 		"_LAMBDA_B": 10,

# 		"to_train" : 1,
# 		"log_dir" : './output/cyclegan/exp_01',
# 		# "checkpoint_dir" : './output/cyclegan/exp_01/#timestamp#',
# 		"checkpoint_dir" : './output/cyclegan/exp_01',
# 		"skip" : False
# 	}
# 	run_cyclegan(exp_01)