# Why, When, and How to Apply Label Smoothing

This code runs most of the experiments detailed in the paper.

To run the grid search shown in the paper, run

`./scripts/grid_search.sh [mnist/cifar10] [smoothing/fsmoothing]`

Smoothing will run the experiments with label smoothing as shown in the paper, and fsmoothing will run experiments with F-matrix forward correction.
