#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/yaoy4/UnsupervisedWordMapping/log/run3.log
module load Anaconda2
source activate FCN
cd /scratch/yaoy4/UnsupervisedWordMapping/
python unsupervised.py --n_epochs 10 --epoch_size 500000 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
python unsupervised.py --n_epochs 10 --epoch_size 2000000 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
python unsupervised.py --n_epochs 10 --epoch_size 4000000 --src_lang en --tgt_lang it --src_emb data/pretrained/en.vec --tgt_emb data/pretrained/it.vec --refinement True
