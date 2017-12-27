# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
from copy import deepcopy

from src.utils import bool_flag, initialize_exp
from src.models import build_model, build_model_cycle
from src.trainer import Trainer
from src.trainer_Cycle import  Trainer_Cycle
from src.evaluation import Evaluator
from src.evaluation import Evaluator_Cycle

VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=bool_flag, default=True, help="Export embeddings after training")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='it', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size")
# mapping if beta is zero, there is no orthogonalization
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
#Cycle consistency
parser.add_argument("--lambda_a", type=int, default=10, help="Cycle consistency loss feedback coefficient from src to src")
parser.add_argument("--lambda_b", type=int, default=10, help="Cycle consistency loss feedback coefficient from tgt to tgt")
parser.add_argument("--cc_method", type=str, default='default', help="The method to calculate cycle consistency")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
# training refinement
parser.add_argument("--refinement", type=bool_flag, default=False, help="Use iterative Procrustes refinement")
parser.add_argument("--n_iters", type=int, default=5, help="Number of iterations")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
# quick test
parser.add_argument("--quick_test", type=bool_flag, default=False, help="USE quick test")
parser.add_argument("--use_dico_train", type=bool_flag, default=False, help="USE dico train")


# parse parameters
params = parser.parse_args()

params1 = params
params2 = deepcopy(params1)
params2.src_emb = deepcopy(params1.tgt_emb) 
params2.tgt_emb = deepcopy(params1.src_emb)
params2.src_lang = deepcopy(params1.tgt_lang)
params2.tgt_lang = deepcopy(params1.src_lang)

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)

# build model / trainer / evaluator
logger = initialize_exp(params)

src_emb, tgt_emb, mapping1, mapping2, discriminator1, discriminator2= build_model_cycle(params, True, True)
trainer = Trainer_Cycle(src_emb, tgt_emb, mapping1, mapping2, discriminator1, discriminator2, params)

evaluator1 = Evaluator_Cycle(trainer, params1, True)
evaluator2 = Evaluator_Cycle(trainer, params2, False)

expname=params.exp_path.split('/')[-1]
figPath='./fig/'+expname

plot_info=OrderedDict({
    'expname': expname,

    "iter_train":[],
    'DIS_A_COSTS':[],
    'DIS_B_COSTS':[],
    'GAN_A_COSTS': [],
    'GAN_B_COSTS':[],
    'CYC_A_COSTS':[],
    'CYC_B_COSTS':[],

    "epoch_train":[],
    "precision_at_1-nn_t_train":[],
    "precision_at_5-nn_t_train":[],
    "precision_at_10-nn_t_train":[],
    "precision_at_1-csls_knn_10_t_train":[],
    "precision_at_5-csls_knn_10_t_train":[],
    "precision_at_10-csls_knn_10_t_train":[],

    "precision_at_1-nn_f_train":[],
    "precision_at_5-nn_f_train":[],
    "precision_at_10-nn_f_train":[],
    "precision_at_1-csls_knn_10_f_train":[],
    "precision_at_5-csls_knn_10_f_train":[],
    "precision_at_10-csls_knn_10_f_train":[],

    "precision_at_1-nn_t_train_best":[],
    "precision_at_5-nn_t_train_best":[],
    "precision_at_10-nn_t_train_best":[],
    "precision_at_1-csls_knn_10_t_train_best":[],
    "precision_at_5-csls_knn_10_t_train_best":[],
    "precision_at_10-csls_knn_10_t_train_best":[],

    "precision_at_1-nn_f_train_best":[],
    "precision_at_5-nn_f_train_best":[],
    "precision_at_10-nn_f_train_best":[],
    "precision_at_1-csls_knn_10_f_train_best":[],
    "precision_at_5-csls_knn_10_f_train_best":[],
    "precision_at_10-csls_knn_10_f_train_best":[],

    "iter_refine":[],
    "precision_at_1-nn_t_refine":[],
    "precision_at_5-nn_t_refine":[],
    "precision_at_10-nn_t_refine":[],
    "precision_at_1-csls_knn_10_t_refine":[],
    "precision_at_5-csls_knn_10_t_refine":[],
    "precision_at_10-csls_knn_10_t_refine":[],

    "precision_at_1-nn_f_refine":[],
    "precision_at_5-nn_f_refine":[],
    "precision_at_10-nn_f_refine":[],
    "precision_at_1-csls_knn_10_f_refine":[],
    "precision_at_5-csls_knn_10_f_refine":[],
    "precision_at_10-csls_knn_10_f_refine":[],

    "precision_at_1-nn_t_refine_best":[],
    "precision_at_5-nn_t_refine_best":[],
    "precision_at_10-nn_t_refine_best":[],
    "precision_at_1-csls_knn_10_t_refine_best":[],
    "precision_at_5-csls_knn_10_t_refine_best":[],
    "precision_at_10-csls_knn_10_t_refine_best":[],

    "precision_at_1-nn_f_refine_best":[],
    "precision_at_5-nn_f_refine_best":[],
    "precision_at_10-nn_f_refine_best":[],
    "precision_at_1-csls_knn_10_f_refine_best":[],
    "precision_at_5-csls_knn_10_f_refine_best":[],
    "precision_at_10-csls_knn_10_f_refine_best":[],

    })

def update_plot_info(to_log, postfix):
    for key in to_log:
        if key+postfix in plot_info:
            plot_info[key+postfix].append(to_log[key])

if params.quick_test:
    logger.info('\n\n----> THIS IS DEBUGGING MODE <----\n\n')
else:
    logger.info('\n\n----> THIS IS NOT DEBUGGING MODE <----\n\n')

if params.use_dico_train:
    # load a training dictionary. if a dictionary path is not provided, use a default
    # one ("default") or create one based on identical character strings ("identical_char")
    trainer.load_training_dico(params.dico_train)

"""
Learning loop for Adversarial Training
"""
if params.adversarial:
    logger.info('\n\n----> ADVERSARIAL TRAINING <----\n\n')

    # training loop
    for n_epoch in range(params.n_epochs):

        logger.info('Starting adversarial training epoch %i...' % n_epoch)
        tic = time.time()
        n_words_proc = 0
        stats = {'DIS_A_COSTS':[],'DIS_B_COSTS':[],'GAN_A_COSTS': [],'GAN_B_COSTS':[],'CYC_A_COSTS':[],'CYC_B_COSTS':[]}

        for n_iter in range(0, params.epoch_size, params.batch_size):

            # discriminator training
            for _ in range(params.dis_steps):
                trainer.dis_step(stats,False)
                trainer.dis_step(stats,True)
                
            # mapping training (discriminator fooling)
            trainer.mapping_step(stats,False)
            trainer.mapping_step(stats,True)
            
            n_words_proc += 2*params.batch_size

            # log stats
            if n_iter % (params.epoch_size/params.batch_size/20*params.batch_size) == 0:
                stats_log=[""]
                for cost in stats:
                    if len(stats[cost]) > 0:
                        stats_log.extend(['%s: %.4f' % (cost, np.mean(stats[cost]))])
                        plot_info[cost].append(np.mean(stats[cost]))

                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
                
                plot_info['iter_train'].append(n_iter+params.epoch_size*n_epoch)
                #clear
                for cost in stats:
                    del stats[cost][:]

                # reset
                tic = time.time()
                n_words_proc = 0

        # embeddings / discriminator evaluation
        to_log1 = OrderedDict({'n_epoch': n_epoch})
        
        logger.info('Normal Direction:')
        evaluator1.word_translation(to_log1)
        evaluator1.dist_mean_cosine(to_log1)
        if not params.quick_test:
            evaluator1.all_eval(to_log1)
            evaluator1.eval_dis(to_log1)

        to_log2 = OrderedDict({'n_epoch': n_epoch})
        logger.info('Reverse Direction:')
        evaluator2.word_translation(to_log2)
        evaluator2.dist_mean_cosine(to_log2)
        if not params.quick_test:
            evaluator2.all_eval(to_log2)
            evaluator2.eval_dis(to_log2)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log1))
        logger.info("__log__:%s" % json.dumps(to_log2))
        trainer.save_best(to_log1, VALIDATION_METRIC)
        logger.info('End of epoch %i.\n\n' % n_epoch)
        
        plot_info['epoch_train'].append(n_epoch)
        update_plot_info(to_log1, "_t_train")
        update_plot_info(to_log2, "_f_train")

        # update the learning rate (stop if too small)
        trainer.update_lr(to_log1, VALIDATION_METRIC)
        if trainer.map_optimizer(True).param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break
        if trainer.map_optimizer(False).param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break

    logger.info('\n\n----> BEST TRAINING MODEL <----\n\n')
    trainer.reload_best()
    to_log1 = OrderedDict({'final_t': 0})
    logger.info('Normal Direction:')
    evaluator1.word_translation(to_log1)
    evaluator1.dist_mean_cosine(to_log1)
    if not params.quick_test:
        evaluator1.all_eval(to_log1)
        evaluator1.eval_dis(to_log1)

    to_log2 = OrderedDict({'final_f': 0})
    logger.info('Reverse Direction:')
    evaluator2.word_translation(to_log2)
    evaluator2.dist_mean_cosine(to_log2)
    if not params.quick_test:
        evaluator2.all_eval(to_log2)
        evaluator2.eval_dis(to_log2)

    logger.info("__log__:%s" % json.dumps(to_log1))
    logger.info("__log__:%s" % json.dumps(to_log2))

    update_plot_info(to_log1, "_t_train_best")
    update_plot_info(to_log2, "_f_train_best")

    if params.quick_test:
        address=os.path.join(params.exp_path, 'plot_info.test')
        with open(address, 'w') as outfile:  
            json.dump(plot_info, outfile)

        #test
        with open(address) as json_file:  
            data = json.load(json_file)
            print data


"""
Learning loop for Procrustes Iterative Refinement
"""
if params.refinement:
    # Get the best mapping according to VALIDATION_METRIC
    logger.info('\n\n----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
    trainer.reload_best()

    # training loop
    for n_iter in range(params.n_iters):

        logger.info('Starting refinement iteration %i...' % n_iter)

        # build a dictionary from aligned embeddings
        trainer.build_dictionary(True)
        # apply the Procrustes solution
        trainer.procrustes(True)
        # embeddings evaluation
        logger.info('Normal Direction:')
        to_log1 = OrderedDict({'n_iter_no': n_iter})
        evaluator1.word_translation(to_log1)
        evaluator1.dist_mean_cosine(to_log1)
        if not params.quick_test:
            evaluator1.all_eval(to_log1)
            evaluator1.eval_dis(to_log1)
        
        # build a dictionary from aligned embeddings
        trainer.build_dictionary(False)
        # apply the Procrustes solution
        trainer.procrustes(False)
        logger.info('Reverse Direction:')
        # embeddings evaluation
        to_log2 = OrderedDict({'n_iter_re': n_iter})
        evaluator2.word_translation(to_log2)
        evaluator2.dist_mean_cosine(to_log2)
        if not params.quick_test:
            evaluator2.all_eval(to_log2)
            evaluator2.eval_dis(to_log2)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log1))
        logger.info("__log__:%s" % json.dumps(to_log2))
        trainer.save_best(to_log1, VALIDATION_METRIC)
        logger.info('End of refinement iteration %i.\n\n' % n_iter)

        plot_info['iter_refine'].append(n_iter)
        update_plot_info(to_log1, "_t_refine")
        update_plot_info(to_log2, "_f_refine")

    logger.info('\n\n----> BEST PROCRUSTES REFINEMENT MODEL <----\n\n')

    #show best
    trainer.reload_best()
    to_log1 = OrderedDict({'final_t': 0})
    logger.info('Normal Direction:')
    evaluator1.word_translation(to_log1)
    evaluator1.dist_mean_cosine(to_log1)
    if not params.quick_test:
        evaluator1.all_eval(to_log1)
        evaluator1.eval_dis(to_log1)

    to_log2 = OrderedDict({'final_f': 0})
    logger.info('Reverse Direction:')
    evaluator2.word_translation(to_log2)
    evaluator2.dist_mean_cosine(to_log2)
    if not params.quick_test:
        evaluator2.all_eval(to_log2)
        evaluator2.eval_dis(to_log2)

    logger.info("__log__:%s" % json.dumps(to_log1))
    logger.info("__log__:%s" % json.dumps(to_log2))

    update_plot_info(to_log1, "_t_refine_best")
    update_plot_info(to_log2, "_f_refine_best")

# export embeddings to a text format
if params.export:
    trainer.reload_best()
    # trainer.export()

    address=os.path.join(params.exp_path, 'plot_info.test')
    with open(address, 'w') as outfile:  
        json.dump(plot_info, outfile)

    if params.quick_test:
        #test
        with open(address) as json_file:  
            data = json.load(json_file)
            print data
    

