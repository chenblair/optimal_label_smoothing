import os
import sys
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path

parser = argparse.ArgumentParser(
        description='Performance Runner')

parser.add_argument(
        '--method', type=str, help='[nll, smoothing]', default="nll")

parser.add_argument(
        '--y', type=str, help='[train_loss, test_acc, test_loss]', default="test_acc")

parser.add_argument(
        '--dataset', type=str, help='[mnist, cifar10]', default="mnist")

parser.add_argument(
        '--task', type=str, help='[stats, curve]', default="stats")

args = parser.parse_args()
# results/grid_search/mnist/smoothing/mnist_smoothing_symmetric_0.11_0.50_2.json
data_dir = "results/grid_search/{}/{}".format(args.dataset, args.method)
plot_dir = "analysis/plots/performance"

def get_path(a, p, seed):
    file_name = "{}_smoothing_symmetric_{:.2f}_{:.2f}_{}.json".format(args.dataset, p, 1 - a, seed)
    return "{}/{}".format(data_dir, file_name)

def get_data(a, p, seed):
    path = get_path(a, p, seed)
    data = json.load(open(path, 'r'))
    return data

save_file = ''
epoch = 99

if (args.task == 'stats'):
    print("method: {}".format(args.method))
    for clean_rate in [0.2, 0.5, 0.8]:
        data = [get_data(clean_rate, seed)[args.y][epoch - 1] for seed in range(1, 6)]
        print(data)
        # print(data)
        print("clean rate: {}, {}: {:.2f}+-{:.2f}".format(clean_rate, args.y, np.mean(data), np.std(data)))
elif (args.task == 'curve'):
    save_file = '{}/{}_{}_{}.png'.format(plot_dir, args.task, args.y, args.method)
    clean_rate = 0.5
    
    data = get_data(clean_rate, 0.11, 2)[args.y]
    plt.plot(np.arange(0, len(data), 1), data)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("{}".format(args.y), fontsize=12)
    plt.savefig('{}/{}_{}_{}.png'.format(plot_dir, args.task, args.y, args.method))
print(save_file)