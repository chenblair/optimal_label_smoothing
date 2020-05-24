import os
import sys
import json
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path
from sklearn import linear_model

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
    file_name = "{}_smoothing_symmetric_{:.2f}_{:.2f}_1000.0_1.json".format(args.dataset, p, 1 - a)
    return "{}/{}".format(data_dir, file_name)

def get_data(p, a):
    path = get_path(p, a)
    data = json.load(open(path, 'r'))
    return data

data_dir = "results/grid_search/{}/smoothing".format(args.dataset)
plot_dir = "analysis/plots/grid_search/smoothing"
# p_grid = [0.05 * i for i in range(3, 21)]
# a_grid = [0.05 * i for i in range(3, 21)]
p_grid = np.array([0.01 * i for i in range(11, 101)])
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
    epoch = 60
    data = []
    scatter_data = [[], []]
    for a in reversed(a_grid):
        data.append([get_data(p, a)[args.y][epoch - 1] for p in p_grid])
        scatter_data[0].append(a - 0.005)
        scatter_data[1].append(p_grid[np.argmax(data[-1])] - 0.005)
    data = np.array(data)[::-1,::-1].T
    plt.scatter(scatter_data[0], scatter_data[1], marker="+", color="red")
    plt.imshow(data, cmap='viridis', extent=[0.1, 1.0, 0.1, 1.0], vmin=0, vmax=100)
    
    plt.xlim(0.1, 1.0)
    plt.ylim(0.1, 1.0)
    plt.xlabel("a", fontsize=12)
    plt.ylabel("p", fontsize=12)
    plt.colorbar()
    plt.savefig('{}/smoothing_{}_{}.png'.format(plot_dir, args.graph, args.y))
    print('{}/smoothing_{}_{}.png'.format(plot_dir, args.graph, args.y))

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

if (args.graph == 'curve'):
    clean_rate = 0.5
    p = 0.8
    
    data = get_data(p, clean_rate)[args.y]
    plt.plot(np.arange(0, len(data), 1), data)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("{}".format(args.y), fontsize=12)
    plt.savefig('{}/{}_{}_{}_{}.png'.format(plot_dir, args.graph, args.y, p, clean_rate))

if (args.graph == 'relaxation'):
    clean_rates = [0.25, 0.4, 0.65]
    colormap = {0.25: "blue", 0.4: "orange", 0.65: "green"}
    
    data = {}
    largest = 0
    all_x = []
    all_y = []
    for a in clean_rates:
        data[a] = [np.argmax(np.array(get_data(p, a)['test_acc'])) for p in p_grid]
        all_x += list(np.log(p_grid - 0.1))
        all_y += list(np.log(data[a]))
        largest = max(largest, max(data[a]))
    for a in clean_rates:
        # data[a] = 10 * np.array(data[a]) / largest
        plt.scatter(p_grid - 0.1, data[a], marker="+", color=colormap[a], label=r'a={}'.format(a), s=0.001)
    
    regr = linear_model.LinearRegression()
    all_x = [[x] for x in all_x]
    regr.fit(all_x, all_y)
    print('Coefficients: {}, {}'.format(regr.coef_, regr.intercept_))
    regr.coef_ = np.array([-0.5])
    plt.plot(p_grid - 0.1, np.exp(regr.predict(np.log(p_grid - 0.1).reshape(-1, 1))), label=r'$(p - 0.1)^{-0.5}$', color="red")
    
    plt.legend()
    plt.xlabel("p", fontsize=12)
    plt.xscale("log")
    plt.ylabel("relaxation time", fontsize=12)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(plot_dir, args.graph))
    print('{}/{}.png'.format(plot_dir, args.graph))
    