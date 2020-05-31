# -*- coding:utf-8 -*-
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
from data.mnist import MNIST
from data.cifar import CIFAR10
from loss import cal_loss
from models import resnet, cnns
import argparse
import sys
import numpy as np
import datetime
import shutil


def train(args,
          model,
          device,
          train_loader,
          optimizer,
          epoch,
          num_classes=10,
          use_method = True,
          text=False):
    model.train()
    loss_a = []

    for batch_idx, batch in enumerate(train_loader):
        if text:
            data = batch.text[0]
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if (data.size()[0] != args.batch_size):
                continue
        else:
            data, target, index = batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        if (use_method):
            loss = cal_loss(output, target, method=args.method, reduction='mean', p=args.smoothing)
        else:
            loss = cal_loss(output, target, method='nll', reduction='mean', p=args.smoothing)

        loss_a.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(loss_a)


def test(args, model, device, test_loader, num_classes, text=False):
    model.eval()
    test_loss = 0
    correct = 0
    acc = []
    with torch.no_grad():
        for batch in test_loader:
            if text:
                data = batch.text[0]
                target = batch.label
                target = torch.autograd.Variable(target).long()
                if (data.size()[0] != args.batch_size):
                    continue
            else:
                data, target, index = batch

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = cal_loss(output, target, reduction='sum')
            test_loss += loss.item()  # sum up batch loss
            pred = output[:, :num_classes].argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), acc))
    return acc, test_loss


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Gambler\'s Loss Runner')

    parser.add_argument(
        '--result_dir',
        type=str,
        help='directory to save result txt files',
        default='results')
    parser.add_argument(
        '--gpu', type=int, help='gpu index', default=0)
    parser.add_argument(
        '--noise_rate',
        type=float,
        help='corruption rate, should be less than 1',
        default=0.5)
    parser.add_argument(
        '--noise_type',
        type=str,
        help='[pairflip, symmetric]',
        default='symmetric')
    parser.add_argument(
        '--dataset', type=str, help='mnist, cifar10, or imdb', default='mnist')
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='how many subprocesses to use for data loading')
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument('--load_model', type=str, default="")

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        metavar='N',
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        metavar='N',
        help='how many batches to wait before logging training status (default: 100)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='learning rate (default: 0.001)')
    parser.add_argument(
        '--start_method', type=int, help='number of epochs before starting method', default=0)

    # label smoothing args
    parser.add_argument(
        '--smoothing',
        type=float,
        default=1.0,
        help='smoothing parameter (default: 1)')
    parser.add_argument(
        '--optimal_smoothing',
        action='store_true',
        default=False)
    parser.add_argument(
        '--scale_lr',
        type=float,
        default=0.0,
        help='exponent to scale learning rate by')
    
    parser.add_argument(
        '--method', type=str, help='[nll, smoothing, fsmoothing]', default="nll")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'mnist':
        input_channel = 1
        num_classes = 10
        train_dataset = MNIST(
            root='./data/',
            download=True,
            train=True,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate)
        test_dataset = MNIST(
            root='./data/',
            download=True,
            train=False,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate)
        print('loading dataset...')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=False)

    if args.dataset == 'cifar10':
        input_channel = 3
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(
            root='./data/',
            download=True,
            train=True,
            transform=transform_train,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate)
        test_dataset = CIFAR10(
            root='./data/',
            download=True,
            train=False,
            transform=transform_test,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate)
        print('loading dataset...')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu) if (use_cuda and args.gpu >= 0) else "cpu")
    print("using {}".format(device))
    
    args.lr *= ((0.9 / (args.smoothing - 0.1)) ** args.scale_lr)
    print("learning rate scaled to {}".format(args.lr))

    print('building model...')
    if args.dataset == 'mnist':
        model = cnns.CNN_basic(num_classes=num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.dataset == 'cifar10':
        model = resnet.ResNet18().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    test_accs = []
    train_losses = []
    test_losses = []
    out = []
    
    if args.optimal_smoothing:
        clean_rate = 1 - args.noise_rate
        args.smoothing = (1 - (2 * clean_rate) + clean_rate * clean_rate * num_classes) / (num_classes - 1)
        print("Using smoothing parameter {:.2f} for clean rate {:.2f}".format(args.smoothing, clean_rate))

    name = "{}_{}_{}_{:.2f}_{:.2f}_{}".format(args.dataset, args.method, args.noise_type, args.smoothing, args.noise_rate, args.seed)

    if not os.path.exists(args.result_dir):
        os.system('mkdir -p %s' % args.result_dir)

    for epoch in range(1, args.n_epoch + 1):
        for param_group in optimizer.param_groups:
            print(epoch, param_group['lr'])
        print(name)
        train_loss = train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            num_classes=num_classes,
            use_method=(epoch >= args.start_method),
            text=(args.dataset == 'imdb'))
        train_losses.append(train_loss)
        test_acc, test_loss = test(args, model, device, test_loader, num_classes, text=(args.dataset == 'imdb'))
        test_accs.append(test_acc)
        test_losses.append(test_loss)

    save_data = {
        "command": " ".join(sys.argv),
        "train_loss" : train_losses,
        "test_loss" : test_losses,
        "test_acc" : test_accs
    }
    save_file = args.result_dir + "/" + name + ".json"
    print(save_file)
    json.dump(save_data, open(save_file, 'w'))

if __name__ == '__main__':
    main()
