# Learning Not to Learn in the Presence of Noisy Labels

This code runs most of the experiments detailed in the paper.

The following commands run experiments with the same hyperparameters that were used in the paper (varying the noise rate):
### MNIST:
nll: 
- `python3 main.py --dataset mnist --noise_rate 0.2 --lr 0.001 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type nll`
Gamblers + Early Stopping:
- `python3 main.py --dataset mnist --noise_rate 0.2 --lr 0.001 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type gmblers --eps 9.9 --early_stopping`
Gamblers + Autoscheduling: 
- `python3 main.py --dataset mnist --noise_rate 0.2 --lr 0.001 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type euc`


### CIFAR10:
nll: 
- `python3 main.py --dataset mnist --noise_rate 0.2 --lr 0.001 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type nll`
Gamblers + Early Stopping:
- `python3 main.py --dataset cifar10 --noise_rate 0.2 --lr 0.001 --batch_size 128 --noise_type symmetric --n_epoch 100 --start_gamblers 10 --lambda_type gmblers --eps 9.9 --early_stopping`
Gamblers + Autoscheduling: 
- `python3 main.py --dataset cifar10 --noise_rate 0.2 --lr 0.001 --batch_size 128 --noise_type symmetric --n_epoch 100 --start_gamblers 10 --lambda_type euc`

### IMDB:
nll: 
- `python3 main.py --dataset imdb --noise_rate 0.2 --lr 0.001 --batch_size 32 --noise_type symmetric --n_epoch 100 --start_gamblers 10 --lambda_type nll`
Gamblers + Early Stopping:
- `python3 main.py --dataset imdb --noise_rate 0.2 --lr 0.001 --batch_size 32 --noise_type symmetric --n_epoch 100 --start_gamblers 10 --lambda_type gmblers --eps 1.95 --early_stopping`
Gamblers + Autoscheduling:
- `python3 main.py --dataset imdb --noise_rate 0.2 --lr 0.001 --batch_size 32 --noise_type symmetric --n_epoch 100 --start_gamblers 10 --lambda_type euc`

By manipulating the options, the full range of experiments involving Gambler's Loss in the paper are runnable.
# optimal_label_smoothing
