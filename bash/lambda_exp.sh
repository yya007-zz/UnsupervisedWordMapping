#! /bin/bash
git pull
python unsupervised.py --lambda_a 1 --lambda_b 1 --n_epochs 20 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
python unsupervised.py --lambda_a 5 --lambda_b 5 --n_epochs 20 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
python unsupervised.py --lambda_a 10 --lambda_b 10 --n_epochs 20 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
python unsupervised.py --lambda_a 20 --lambda_b 20 --n_epochs 20 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
python unsupervised.py --lambda_a 100 --lambda_b 100 --n_epochs 20 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True


