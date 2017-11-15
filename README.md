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

### Building fastText

In order to build `fastText`, use the following:

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```
This will produce object files for all the classes as well as the main binary `fasttext`.
If you do not plan on using the default system-wide compiler, update the two macros defined at the beginning of the Makefile (CC and INCLUDES).

### Multi-threaded GIZA++

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

### Dataset Preparation
```
git clone https://github.com/yya007/UnsupervisedWordMapping.git
cd UnsupervisedWordMapping
./get_data.sh
python prepare_data.py
```

## Reference
* https://github.com/leehomyc/cyclegan-1
* https://github.com/artetxem/vecmap
* https://github.com/Babylonpartners/fastText_multilingual
* https://github.com/moses-smt/mgiza.git


