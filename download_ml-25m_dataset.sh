#!/bin/bash
mkdir -p data
wget -O data/ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip -o data/ml-25m.zip -d data
rm -f data/ml-25m.zip