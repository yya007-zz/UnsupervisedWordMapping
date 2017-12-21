#! /bin/bash
git pull
python unsupervised.py --lambda_a 0 --lambda_b 0 --n_epochs 40 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
