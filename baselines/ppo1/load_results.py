import os, sys, shutil, argparse
sys.path.append(os.getcwd())

import numpy as np

#from baselines import bench
from baselines.bench.monitor import LoadMonitorResultsError
from baselines.bench.monitor import load_results
dir_path = "/home/mansimov/logdir/ppo-mpi/Humanoid-v1-seed41/"
#results_class = LoadMonitorResultsError()

results = load_results(dir_path)
episode_rewards = results["episode_rewards"]
episode_lengths = results["episode_lengths"]

print (sum(episode_lengths))

print ("Num episodes {}, Mean 100 episode reward {}".format(len(episode_rewards), np.mean(episode_rewards[-100:])))
#print (np.sort(episode_rewards)[-10:])
