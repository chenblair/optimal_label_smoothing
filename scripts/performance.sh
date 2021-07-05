#!/bin/bash

# for p in 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00
# do
#     for noise_rate in 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05 0.00
#     do
#         ((i=i%52)); ((i++==0)) && wait
#         echo running $p $noise_rate $(date)
#         python3 main.py --result_dir results/grid_search/$1/$2 --gpu $((i%2)) --method $2 --dataset $1 --noise_rate $noise_rate --smoothing $p --lr 0.1 --batch_size 128 --noise_type symmetric --n_epoch 800 --scale_lr 1.0 --seed 2 $ARGS > /dev/null || break 2 &
#     done
# done

for p in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for noise_rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
    do
        ((i=i%52)); ((i++==0)) && wait
        echo running $p $noise_rate $(date)
        python3 main.py --result_dir results/grid_search/$1/$2 --gpu $((i%2)) --method $2 --dataset $1 --noise_rate $noise_rate --smoothing $p --lr 0.1 --batch_size 128 --noise_type symmetric --n_epoch 400 --scale_lr 0.72 --seed 2 $ARGS > /dev/null || break 2 &
    done
done

# for p in 1.00
# do
#     for noise_rate in 0.00
#     do
#         python3 main.py --result_dir results/grid_search/$1/$2 --gpu $((i%2)) --method $2 --dataset $1 --noise_rate $noise_rate --smoothing $p --lr 0.1 --batch_size 128 --noise_type symmetric --n_epoch 400 --scale_lr 1.0 --seed 1 $ARGS || break 2
#     done
# done