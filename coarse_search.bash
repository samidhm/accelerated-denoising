#!/bin/bash

# Define the range for -n and -a parameters
n_values=(1 2 4)
a_values=(0 0.25 0.5 0.75 1)

# Loop through each combination of -n and -a values
for n in "${n_values[@]}"; do
    for a in "${a_values[@]}"; do
        # Run your Python script with the current -n and -a values
        python3 train.py -n $n -b conv -f depth normal albedo relative_normal roughness -a $a -t sample_tag
    done
done
