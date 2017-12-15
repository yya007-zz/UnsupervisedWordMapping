# Unsupervised Word Mapping
Facebook recently open-sourced word vectors in [89 languages](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).

## Installing Required Package

### Requirements

For Mac OS and Linux User.
Since it uses C++11 features, it requires a compiler with good C++11 support.
These include :

* (gcc-4.6.3 or newer) or (clang-3.3 or newer)

Compilation is carried out using a Makefile, so you will need to have a working **make**.
For the word-similarity evaluation script you will need:

* python 2.6 or newer
* numpy & scipy
* tensorflow 1.1 or newer
Install PyTorch and dependencies from http://pytorch.org
Install Torch vision from the source.
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
Install python libraries visdom and dominate.
pip install visdom
pip install dominate
```

### (Optional)Multi-threaded GIZA++

```
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
cmake .
make
make install
export BINDIR=~/workspace/bin/training-tools
cp bin/* $BINDIR/mgizapp
cp scripts/merge_alignment.py $BINDIR
```


## Usage
```
git clone https://github.com/yya007/UnsupervisedWordMapping.git
cd UnsupervisedWordMapping
```
### Dataset Preparation
./get_data.sh
python prepare_data.py
```

## Reference
* https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
* https://github.com/leehomyc/cyclegan-1
* https://github.com/artetxem/vecmap
* https://github.com/Babylonpartners/fastText_multilingual
* https://github.com/moses-smt/mgiza.git
* https://github.com/facebookresearch/MUSE
* https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py

