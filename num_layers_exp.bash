#!/bin/bash

# Define the values for -n parameter
n_values=(1 2 4 6 8)

# Loop through each value of -n
for n in "${n_values[@]}"
do
    # Run the command with the current -n value
    python3 train.py -n $n -b conv -f depth normal albedo relative_normal roughness -a 0.5 -t sample_tag
done