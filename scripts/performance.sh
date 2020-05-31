#!/bin/bash

for p in 0.11
do
    for noise_rate in 0.5
    do
        python3 main.py --result_dir results/grid_search/$1/$2 --method $2 --dataset $1 --noise_rate $noise_rate --smoothing $p --lr 0.01 --batch_size 128 --noise_type symmetric --n_epoch 2000 --scale_lr 1.0 --seed 2 $ARGS || break 2
    done
done
