#!/bin/bash
mkdir -p data
wget -O data/MSD.zip http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip
unzip -o data/MSD.zip -d data
rm -f data/MSD.zip
mv data/train_triplets.txt data/MSD.txt
