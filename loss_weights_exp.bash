#!/bin/bash

# Iterate over the range of values for parameter -a
for a in $(seq 0 0.1 1); do
    echo "Running command with value: $a"
    python3 train.py -n 4 -b conv -f depth normal albedo relative_normal roughness -a $a -t sample_tag
done
