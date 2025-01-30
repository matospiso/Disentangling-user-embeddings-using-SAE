#!/bin/bash
mkdir -p data
wget -O data/ML-25M.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip -o data/ML-25M.zip -d data
rm -f data/ML-25M.zip
mv data/ml-25m data/ML-25M