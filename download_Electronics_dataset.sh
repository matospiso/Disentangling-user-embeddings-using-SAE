#!/bin/bash
mkdir -p data
wget -O data/Electronics.json.gz https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz
gzip -d data/Electronics.json.gz