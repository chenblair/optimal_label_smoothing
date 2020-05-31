#!/bin/bash

for p in 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0
do
    for noise_rate in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        python3 main.py --result_dir results/grid_search/$1/$2 --method $2 --dataset $1 --noise_rate $noise_rate --smoothing $p --lr 0.01 --batch_size 128 --noise_type symmetric --n_epoch 100 $ARGS || break 2
    done
done
