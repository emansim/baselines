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

from baselines.bench.monitor import load_one_result

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

if __name__ == "__main__":
    file_path = "/home/mansimov/logdir/retrieval-simple/SparseJaco150-v1-seed0/"
    result = load_one_result(os.path.join(file_path, 'train.monitor.json'))
    result = result['episode_successes']

    x, y = smooth_reward_curve(np.arange(0,len(result)), result)
    x, y = fix_point(x, y, 100)

    plt.plot(x, y, label="hello", color=color_defaults[0])

    plt.title("3D Jaco Arm Reaching (0 initial demonstrations)")
    plt.ylabel("Success Rate")
    plt.xlabel("Number of episodes")
    plt.legend(loc=4)
    plt.show()
    plt.draw()
    plt.savefig('sparse-jaco.png')
