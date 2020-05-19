#!/bin/bash

# case on method
case $2 in
    nll)
        ARGS=""
        ;;
    smoothing)
        ARGS="--optimal_smoothing"
        ;;
    agnostic_smoothing)
        ARGS="--agnostic_smoothing"
        ;;
    coteaching)
        ARGS=""
        ;;
esac

set -x

for seed in 1 2 3 4 5
do
    for noise_rate in 0.2 0.5 0.8
    do
        python3 main.py --result_dir results/performance/$1/$2 --method $2 --dataset $1 --noise_rate $noise_rate --lr 0.01 --batch_size 128 --noise_type symmetric --n_epoch 100 --gpu 1 --seed $seed $ARGS || break 2
    done

done

set +x
# python3 main.py --result_dir results/performance/$1/known/nll --dataset $1 --noise_rate $noise_rate --lr 0.00001 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type nll || break 1
# python3 main.py --result_dir results/performance/$1/known/smoothing --dataset $1 --noise_rate $noise_rate --lr 0.00001 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type nll --optimal_smoothing || break 1