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
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
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

parser.add_argument("--lambda_a", type=int, default=10, help="Lambda a")
parser.add_argument("--lambda_b", type=int, default=10, help="Lambda ")

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

"""
Learning loop for Adversarial Training
"""
if params.adversarial:
    logger.info('----> ADVERSARIAL TRAINING <----\n\n')

    # training loop
    for n_epoch in range(params.n_epochs):

        logger.info('Starting adversarial training epoch %i...' % n_epoch)
        tic = time.time()
        n_words_proc = 0
        stats1 = {'DIS_COSTS': []}
        stats2 = {'DIS_COSTS': []}

        for n_iter in range(0, params.epoch_size, params.batch_size):

            # discriminator training
            for _ in range(params.dis_steps):
                trainer.dis_step(stats1,True)
                trainer.dis_step(stats2,False)

            # mapping training (discriminator fooling)
            trainer.mapping_step(stats1,True)
            n_words_proc += trainer.mapping_step(stats2,False)

            # log stats
            if n_iter % 500 == 0:
                logger.info('Normal Direction:')
                stats = stats1
                stats_str = [('DIS_COSTS', 'Discriminator loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
                for k, _ in stats_str:
                    del stats1[k][:]

                logger.info('Reverse Direction:')
                stats = stats2
                stats_str = [('DIS_COSTS', 'Discriminator loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
                for k, _ in stats_str:
                    del stats2[k][:]

                # reset
                tic = time.time()
                n_words_proc = 0
                

        # embeddings / discriminator evaluation
        to_log = OrderedDict({'n_epoch': n_epoch})
        
        logger.info('Normal Direction:')
        evaluator1.all_eval(to_log)
        evaluator1 .eval_dis(to_log)
        logger.info('Reverse Direction:')
        evaluator2.all_eval(to_log)
        evaluator2.eval_dis(to_log)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of epoch %i.\n\n' % n_epoch)

        # update the learning rate (stop if too small)
        trainer.update_lr(to_log, VALIDATION_METRIC)
        if trainer.map_optimizer(True).param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break
        if trainer.map_optimizer(False).param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break


"""
Learning loop for Procrustes Iterative Refinement
"""
if params.refinement:
    # Get the best mapping according to VALIDATION_METRIC
    logger.info('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
    trainer.reload_best()

    # training loop
    for n_iter in range(params.n_iters):

        logger.info('Starting refinement iteration %i...' % n_iter)

        # build a dictionary from aligned embeddings
        trainer.build_dictionary(True)

        # apply the Procrustes solution
        trainer.procrustes(True)

        # embeddings evaluation
        to_log = OrderedDict({'n_iter': n_iter})
        
        logger.info('Normal Direction:')
        evaluator1.all_eval(to_log)
        evaluator1.eval_dis(to_log)


         # build a dictionary from aligned embeddings
        trainer.build_dictionary(False)

        # apply the Procrustes solution
        trainer.procrustes(False)

        # embeddings evaluation
        to_log = OrderedDict({'n_iter': n_iter})

        logger.info('Reverse Direction:')
        evaluator2.all_eval(to_log)
        evaluator2.eval_dis(to_log)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of refinement iteration %i.\n\n' % n_iter)


# export embeddings to a text format
if params.export:
    trainer.reload_best()
    trainer.export()
