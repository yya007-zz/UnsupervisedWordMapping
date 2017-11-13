#!/bin/bash
#
# Copyright (C) 2017  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA="$ROOT/data"
mkdir -p "$DATA/analogies"
mkdir -p "$DATA/similarity"
mkdir -p "$DATA/dictionaries"
mkdir -p "$DATA/embeddings/original"
mkdir -p "$DATA/embeddings/unit"
mkdir -p "$DATA/embeddings/unit-center"

# Download English-Italian data from Dinu et al. (2015)
wget -q http://clic.cimec.unitn.it/~georgiana.dinu/down/resources/transmat.zip -O "$DATA/transmat.zip"
unzip -p "$DATA/transmat.zip" data/EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt > "$DATA/embeddings/original/en.emb.txt"
unzip -p "$DATA/transmat.zip" data/IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt > "$DATA/embeddings/original/it.emb.txt"
unzip -p "$DATA/transmat.zip" data/OPUS_en_it_europarl_train_5K.txt > "$DATA/dictionaries/en-it.train.txt"
unzip -p "$DATA/transmat.zip" data/OPUS_en_it_europarl_test.txt > "$DATA/dictionaries/en-it.test.txt"
rm -f "$DATA/transmat.zip"

# Download word analogy data from Mikolov et al. (2013)
wget -q https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip -O "$DATA/word2vec.zip"
unzip -p "$DATA/word2vec.zip" word2vec/trunk/questions-words.txt > "$DATA/analogies/questions-words.txt"
rm -f "$DATA/word2vec.zip"

# Download crosslingual word similarity data from Camacho-Collados et al. (2015)
wget -q http://lcl.uniroma1.it/similarity-datasets/datasets/rg65_EN-DE.txt -O "$DATA/similarity/en-de.rg65.txt"
wget -q http://lcl.uniroma1.it/similarity-datasets/datasets/MWS353_Cross-lingual_datasets.zip -O "$DATA/mws353.zip"
unzip -p "$DATA/mws353.zip" MWS353_Cross-lingual_datasets/cross_en_de_MWS353.txt > "$DATA/similarity/en-de.mws353.txt"
unzip -p "$DATA/mws353.zip" MWS353_Cross-lingual_datasets/cross_en_it_MWS353.txt > "$DATA/similarity/en-it.mws353.txt"
rm -f "$DATA/mws353.zip"

# Download remaining data from our own release
wget -q http://ixa2.si.ehu.es/eneko/tmp/vecmap/de.emb.txt.gz -O "$DATA/embeddings/original/de.emb.txt.gz"
wget -q http://ixa2.si.ehu.es/eneko/tmp/vecmap/fi.emb.txt.gz -O "$DATA/embeddings/original/fi.emb.txt.gz"
wget -q http://ixa2.si.ehu.es/eneko/tmp/vecmap/dictionaries.tar.gz -O "$DATA/dictionaries.tar.gz"
gunzip "$DATA/embeddings/original/de.emb.txt.gz"
gunzip "$DATA/embeddings/original/fi.emb.txt.gz"
tar -xzf "$DATA/dictionaries.tar.gz" -C "$DATA/dictionaries"
rm -f "$DATA/dictionaries.tar.gz"

# Normalize word embeddings
python3 "$ROOT/normalize_embeddings.py" unit -i "$DATA/embeddings/original/en.emb.txt" -o "$DATA/embeddings/unit/en.emb.txt"
python3 "$ROOT/normalize_embeddings.py" unit -i "$DATA/embeddings/original/it.emb.txt" -o "$DATA/embeddings/unit/it.emb.txt"
python3 "$ROOT/normalize_embeddings.py" unit -i "$DATA/embeddings/original/de.emb.txt" -o "$DATA/embeddings/unit/de.emb.txt"
python3 "$ROOT/normalize_embeddings.py" unit -i "$DATA/embeddings/original/fi.emb.txt" -o "$DATA/embeddings/unit/fi.emb.txt"
python3 "$ROOT/normalize_embeddings.py" unit center -i "$DATA/embeddings/original/en.emb.txt" -o "$DATA/embeddings/unit-center/en.emb.txt"
python3 "$ROOT/normalize_embeddings.py" unit center -i "$DATA/embeddings/original/it.emb.txt" -o "$DATA/embeddings/unit-center/it.emb.txt"
python3 "$ROOT/normalize_embeddings.py" unit center -i "$DATA/embeddings/original/de.emb.txt" -o "$DATA/embeddings/unit-center/de.emb.txt"
python3 "$ROOT/normalize_embeddings.py" unit center -i "$DATA/embeddings/original/fi.emb.txt" -o "$DATA/embeddings/unit-center/fi.emb.txt"
