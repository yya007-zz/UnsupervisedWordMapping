#! /bin/bash
git pull
python unsupervised.py --n_epochs 20 --map_beta 0 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
python unsupervised.py --n_epochs 20 --map_beta 0 --lambda_a 0 --lambda_b 0 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
