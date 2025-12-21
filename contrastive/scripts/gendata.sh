#!/bin/bash

python src/gendata.py --output_dir data/train --num_samples 1000
python src/gendata.py --output_dir data/valid --num_samples 100