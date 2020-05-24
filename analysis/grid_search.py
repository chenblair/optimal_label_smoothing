import os
import sys
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path

parser = argparse.ArgumentParser(
        description='Grid Search Runner')

parser.add_argument(
        '--graph', type=str, help='[grid]', default="grid")

parser.add_argument(
        '--y', type=str, help='[train_loss, test_acc, test_loss]', default="test_acc")

parser.add_argument(
        '--dataset', type=str, help='[mnist, cifar10]', default="mnist")

args = parser.parse_args()

def get_path(p, a):
    file_name = "{}_symmetric_{:.2f}_{:.2f}_1000.0_1.json".format(args.dataset, p, 1 - a)
    return "{}/{}".format(data_dir, file_name)

def get_data(p, a):
    path = get_path(p, a)
    data = json.load(open(path, 'r'))
    return data

data_dir = "results/grid_search_lr/{}".format(args.dataset)
plot_dir = "analysis/plots/grid_search_lr"
# p_grid = [0.05 * i for i in range(3, 21)]
# a_grid = [0.05 * i for i in range(3, 21)]
p_grid = [0.01 * i for i in range(11, 101)]
a_grid = [0.01 * i for i in range(11, 101)]

if (args.graph == "grid"):
    epoch = 100

    for a in a_grid:
        accs = []
        for p in p_grid:
            data = get_data(p, a)
            accs.append(data[args.y][epoch - 1])
        plt.plot(p_grid, accs, label="a={:.2f}".format(a))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("p", fontsize=12)
    plt.ylabel("{}".format(args.y), fontsize=12)
    plt.savefig('{}/{}_{}.png'.format(plot_dir, args.graph, args.y), bbox_inches='tight')
    plt.clf()

if (args.graph == "fixed_a"):
    a = 0.2
    for p in p_grid:
        data = get_data(p, a)[args.y]
        plt.plot(np.arange(0, len(data), 1), data, label="p={:.2f}".format(p))
    plt.legend()
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("{}".format(args.y), fontsize=12)
    plt.savefig('{}/{}_{}_{}.png'.format(plot_dir, args.graph, args.y, a))

if (args.graph == "heatmap"):
    epoch = 100
    data = []
    for a in reversed(a_grid):
        data.append([get_data(p, a)[args.y][epoch - 1] for p in p_grid])
    plt.imshow(data, cmap='viridis', extent=[0.15, 1.0, 0.15, 1.0], vmin=0, vmax=100)
    plt.xlabel("p", fontsize=12)
    plt.ylabel("a", fontsize=12)
    plt.colorbar()
    plt.savefig('{}/{}_{}.png'.format(plot_dir, args.graph, args.y))

if (args.graph == "progress"):
    data = []
    for a in reversed(a_grid):
        data.append([int(path.exists(get_path(p, a))) for p in p_grid])
    completed = sum([sum(d) for d in data])
    total = sum([len(d) for d in data])
    print("{} / {} Experiments Completed".format(completed, total)) 
    print("{:.2f} %".format(100. * completed / total))  
    plt.imshow(data, cmap='Blues', extent=[0.15, 1.0, 0.15, 1.0])
    plt.xlabel("p", fontsize=12)
    plt.ylabel("a", fontsize=12)
    plt.colorbar()
    plt.savefig('{}/{}_{}.png'.format(plot_dir, args.graph, args.y))
            