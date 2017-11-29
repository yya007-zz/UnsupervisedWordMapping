#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/yaoy4/UnsupervisedWordMapping/log/run.log
module load Anaconda2
source activate FCN
cd /scratch/yaoy4/UnsupervisedWordMapping/
python  /scratch/yaoy4/UnsupervisedWordMapping/main.py exp exp1
