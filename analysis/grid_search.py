import os
import sys
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_data(p, a):
    file_name = "mnist_symmetric_{:.2f}_{:.2f}_1000.0_1.json".format(p, 1 - a)
    data = json.load(open("{}/{}".format(data_dir, file_name), 'r'))
    return data

data_dir = "../results/grid_search"
plot_dir = "plots"
p_grid = [0.05 * i for i in range(3, 21)]

parser = argparse.ArgumentParser(
        description='Grid Search Runner')

parser.add_argument(
        '--graph', type=str, help='[grid]', default="grid")

parser.add_argument(
        '--y', type=str, help='[train_loss, test_acc, test_loss]', default="test_acc")

args = parser.parse_args()

if (args.graph == "grid"):
    epoch = 100

    for a in [0.05 * i for i in range(3, 21)]:
        accs = []
        for p in p_grid:
            data = get_data(p, a)
            accs.append(data[args.y][epoch - 1])
        plt.plot(p_grid, accs, label="a={:.2f}".format(a))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("p", fontsize=12)
    plt.ylabel("{}".format(args.y), fontsize=12)
    plt.savefig('{}/grid_search/{}_{}.png'.format(plot_dir, args.graph, args.y), bbox_inches='tight')
    plt.clf()

if (args.graph == "fixed_a"):
    a = 0.2
    for p in p_grid:
        data = get_data(p, a)[args.y]
        plt.plot(np.arange(0, len(data), 1), data, label="p={:.2f}".format(p))
    plt.legend()
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("{}".format(args.y), fontsize=12)
    plt.savefig('{}/grid_search/{}_{}_{}.png'.format(plot_dir, args.graph, args.y, a))