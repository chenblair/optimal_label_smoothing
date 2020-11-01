import os
import sys
import ujson as json
import colorcet as cc
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path
from sklearn import linear_model
from tqdm import tqdm
import matplotlib.cm as cm
from operator import itemgetter

parser = argparse.ArgumentParser(
        description='Grid Search Runner')

parser.add_argument(
        '--graph', type=str, help='[grid]', default="grid")

parser.add_argument(
        '--y', type=str, help='[train_loss, test_acc, test_loss]', default="test_loss")

parser.add_argument(
        '--dataset', type=str, help='[mnist, cifar10]', default="mnist")

parser.add_argument(
        '--a', type=float, default=0.3)

args = parser.parse_args()

def get_path(p, a):
    file_name = "{}_smoothing_symmetric_{:.2f}_{:.2f}_1.json".format(args.dataset[:7], p, 1 - a)
    # file_name = "{}_symmetric_{:.2f}_{:.2f}_1000.0_1.json".format(args.dataset[:5], p, 1 - a)
    return "{}/{}".format(data_dir, file_name)

def get_data(p, a):
    path = get_path(p, a)
    # print(path)
    data = json.load(open(path, 'r'))
    return data

data_dir = "results/grid_search/{}".format(args.dataset)
if (args.dataset == 'mnist10000'):
    args.dataset = 'mnist'
plot_dir = "analysis/plots/aistats"
# p_grid = np.sort(np.array([0.05 * i for i in range(3, 21)] + [0.11 + 0.05 * i for i in range(18)] + [0.13 + 0.05 * i for i in range(18)]))[::2]
#p_grid = np.array([0.01 * i for i in range(11, 101)])
#a_grid = np.array([0.01 * i for i in range(11, 101)])
p_grid = np.array([0.05 * i for i in range(3, 21)])
# p_grid = np.array([0.1 * i for i in range(2, 11)])
#a_grid = np.array([0.1 * i for i in range(2, 11)])
# a_grid = [0.05]

if (args.graph == "grid"):
    epoch = 100
    M = 10
    a = args.a
    accs = []
    epsilon = 1e-10
    for p in p_grid:
        data = get_data(p, a)
        accs.append(data[args.y][epoch - 1])
    plt.plot(p_grid, accs, label='empirical')
    plt.plot(p_grid, -a*a*np.log(p_grid) - 2*a*(1-a)*np.log((1-p_grid)/(M-1)+epsilon) - (1-a)*(1-a)*np.log(p_grid)/(M-1) - (1-a)*(1-a)*(M-2)*np.log((1-p_grid)/(M-1) + epsilon)/(M-1), label='beta')
    # plt.plot(p_grid, (-a * np.log(p_grid)) - ((1 - a) * np.log((1 - p_grid + epsilon)/(M-1))), label='alpha')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("p", fontsize=12)
    plt.ylabel("test loss", fontsize=12)
    plt.title(f'a={args.a}')
    plt.savefig('{}/{}/{}/{}_{}.png'.format(plot_dir, args.graph, args.dataset, args.y, args.a), bbox_inches='tight')
    print('{}/{}/{}/{}_{}.png'.format(plot_dir, args.graph, args.dataset, args.y, args.a))
    plt.clf()

if (args.graph == "gridreproduce"):
    epoch = 100
    M = 10
    a = args.a
    plt.figure(figsize=(2.5,2.5))
    accs = []
    epsilon = pow(math.e, -4.5)
    for p in p_grid:
        data = get_data(p, a)
        accs.append(data[args.y][epoch - 1])
    plt.plot(p_grid, accs, 's-r', label='CIFAR10, Res18')
    # plt.plot(p_grid, -a*a*np.log(p_grid) - 2*a*(1-a)*np.log((1-p_grid)/(M-1)+epsilon) - (1-a)*(1-a)*np.log(p_grid)/(M-1) - (1-a)*(1-a)*(M-2)*np.log((1-p)/(M-1) + epsilon)/(M-1), label='beta')
    plt.plot(p_grid, (-a * np.log(p_grid)) - ((1 - a) * np.log((1 - p_grid + epsilon)/(M-1))), '--b', linewidth=4, label=r'$\beta$-type theory')
    plt.legend(loc='upper left')
    plt.xlabel("p", fontsize=12)
    plt.ylabel("test loss", fontsize=12)
    plt.savefig('{}/{}/{}/{}_{}.png'.format(plot_dir, args.graph, args.dataset, args.y, args.a), bbox_inches='tight')
    print('{}/{}/{}/{}_{}.png'.format(plot_dir, args.graph, args.dataset, args.y, args.a))
    plt.clf()

if (args.graph == "heatmap"):
    epoch = 100
    M = 10
    epsilon = 1e-7
    accs = []
    alpha = []
    for a in tqdm(a_grid[::-1]):
        a_accs = []
        # for p in p_grid:
        #     data = get_data(p, a)
        #     a_accs.append(data[args.y][epoch - 1])
        accs.append(a_accs)
        alpha.append((-a * np.log(p_grid)) - ((1 - a) * np.log((1 - p_grid + epsilon)/(M-1))))
    imgplot = plt.imshow(alpha, cmap=cc.cm.bgy, vmin=0.0, vmax=3.5, extent = [0.11, 1.0, 0.11, 1.0])
    # plt.colorbar(imgplot)
    plt.xlabel("p", fontsize=12)
    plt.ylabel("a", fontsize=12)
    plt.savefig('{}/{}/{}/{}.png'.format(plot_dir, args.graph, args.dataset, args.y + "alpha"), bbox_inches='tight')
    print('{}/{}/{}/{}_{}.png'.format(plot_dir, args.graph, args.dataset, args.y, args.a))
    plt.clf()

if (args.graph == "best"):
    epoch = 100
    plt.figure(figsize=(3.0,3.0))
    ps = []
    psb = []
    for a in tqdm(a_grid):
        a_accs = []
        b_accs = []
        for p in p_grid:
            data = get_data(p, a)
            a_accs.append(data['test_loss'][epoch - 1])
            b_accs.append(data['test_acc'][epoch - 1])
        ps.append(p_grid[min(enumerate(a_accs), key=itemgetter(1))[0]])
        psb.append(p_grid[max(enumerate(b_accs), key=itemgetter(1))[0]])
    plt.scatter(a_grid, ps, label='min test loss', s=5)
    plt.scatter(a_grid, psb, label='max test accuracy', s=5)
    plt.legend(loc='upper left')
    plt.xlabel("a", fontsize=12)
    plt.ylabel(r'$p^*$', fontsize=12)
    plt.savefig('{}/{}/{}/{}.png'.format(plot_dir, args.graph, args.dataset, args.y + "alpha"), bbox_inches='tight')
    print('{}/{}/{}/{}_{}.png'.format(plot_dir, args.graph, args.dataset, args.y, args.a))
    plt.clf()

if (args.graph == "big_scatter"):
    epoch = 100
    epsilon = 0
    for a in tqdm(a_grid):
        accs = []
        alphas = []
        for p in p_grid:
            data = get_data(p, a)
            accs.append(data[args.y][epoch - 1])
        plt.plot(accs, (-a * np.log(p_grid)) - ((1 - a) * np.log(1 - p_grid + epsilon)), color=cm.viridis(a))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("empirical", fontsize=12)
    plt.ylabel("alpha", fontsize=12)
    plt.savefig('{}/{}_{}.png'.format(plot_dir, args.graph, args.y), bbox_inches='tight')
    plt.clf()
