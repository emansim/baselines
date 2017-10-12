#from visdom import Visdom
import numpy as np
import glob
import os
import argparse
#from load import load_data, load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 8})
from PIL import Image
import itertools
color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]

import os, sys, shutil, argparse
sys.path.append(os.getcwd())

from baselines.bench.monitor import load_results

def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]

def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0
    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy

def visualize(dir, seeds, name, color, lines):
    episode_rewards = []
    for seed in seeds:
        thedir = "{}-seed{}".format(dir, seed)
        results = load_results(thedir)
        episode_rewards.append(results["episode_rewards"])

    xs, ys = [], []
    for i in range(len(episode_rewards)):
        x, y = smooth_reward_curve(np.arange(0,len(episode_rewards[i])), episode_rewards[i])
        x, y = fix_point(x, y, 10)
        xs.append(x)
        ys.append(y)


    length = min([len(x) for x in xs])
    for j in range(len(xs)):
        xs[j] = xs[j][:length]
        ys[j] = ys[j][:length]

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    x = np.mean(np.array(xs), axis=0)
    y_mean = np.mean(np.array(ys), axis=0)
    y_std = np.std(np.array(ys), axis=0)

    y_upper = y_mean + y_std
    y_lower = y_mean - y_std
    plt.fill_between(
        x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
    )
    line = plt.plot(x, list(y_mean), label=name, color=color)
    lines.append(line[0])
    return lines

if __name__ == "__main__":
    logdir = "/home/mansimov/logdir/ddpg-gpu/Reacher-v1"
    lines = []

    seeds = [1,2]
    color = color_defaults[0]
    name = "ou_0.2"
    visualize(logdir, seeds, name, color, lines)

    seeds = [41,42]
    color = color_defaults[1]
    name = "none"
    visualize(logdir, seeds, name, color, lines)

    seeds = [81,82]
    color = color_defaults[2]
    name = "normal_0.1"
    visualize(logdir, seeds, name, color, lines)

    seeds = [121,122]
    color = color_defaults[3]
    name = "normal_0.1_random"
    visualize(logdir, seeds, name, color, lines)

    plt.ylim(-20,0)

    plt.title("Reacher-ddpg")
    plt.legend(loc=4)
    plt.show()
    plt.draw()
    plt.savefig('reacher-ddpg.png')
