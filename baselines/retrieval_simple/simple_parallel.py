#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--num_cpu', help='num procs in parallel', type=int, default=1)
parser.add_argument('--top_k', help='top k retrieved actions', type=int, default=1)
parser.add_argument('--noise_std_start', help='scale of noise at the beginning of episode', type=float, default=1.0)
parser.add_argument('--noise_std_end', help='scale of noise at the end of episode', type=float, default=1.0)

parser.add_argument('--num_iter_train', help='number of episodes total', type=int, default=100) # total number of episodes
parser.add_argument('--eval_after', help='eval after', type=int, default=100) # total number of episodes
parser.add_argument('--num_iter_eval', help='number of episodes to eval', type=int, default=100) # total number of episodes

parser.add_argument('--env', help='environment ID', type=str, default="SparseHalfCheetah-v1")
parser.add_argument('--train', action='store_true')
parser.add_argument('--max_buffer_size', help='max number of demonstrations if train in buffer', type=int, default=1000) # -1 means all of them

args = parser.parse_args()

# SET OPENAI_LOGDIR
folder_name = os.path.join(os.environ["checkpoint_dir"], "retrieval-simple")
try:
    os.mkdir(folder_name)
except:
    pass
log_dir = os.path.join(folder_name, "{}-seed{}".format(args.env, args.seed))
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)
os.environ["OPENAI_LOGDIR"] = log_dir

# Import other random libs
import numpy as np
import pickle
import gym
import logging
from baselines import logger
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import random
random.seed(args.seed)

import scipy
import scipy.misc

from sklearn.neighbors import NearestNeighbors

from collections import deque

def make_env(rank):
    def _thunk():
        env = gym.make(args.env)
        env.seed(args.seed + rank)
        if logger.get_dir():
            env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
        gym.logger.setLevel(logging.WARN)
        return env
    return _thunk

class Runner(object):
    def __init__(self, env, max_timesteps, is_eval=False):
        self.env = env
        self.max_timesteps = max_timesteps
        self.is_eval = is_eval
        self.obs = env.reset()

    def rollout(self, neigh, buffer_paths):
        rollout_paths = [{"obs": [], "rewards": [], "actions": []} for _ in range(args.num_cpu)]
        # add initial obs
        for n in range(args.num_cpu):
            rollout_paths[n]["obs"].append(self.obs[n])

        if neigh == None:
            # do completely random thing !
            actions = np.random.uniform(low=-1*np.ones((self.max_timesteps, args.num_cpu, self.env.action_space.shape[0])),\
                                        high=1*np.ones((self.max_timesteps, args.num_cpu, self.env.action_space.shape[0])))
        else:
            # retrieve closest action
            actions = []
            for n in range(args.num_cpu):
                top_ind = neigh.kneighbors(np.expand_dims(rollout_paths[n]["obs"][0], 0), args.top_k, return_distance=False)
                action = np.zeros((self.max_timesteps, self.env.action_space.shape[0]))
                for ind in top_ind[0]:
                    buffer_path = buffer_paths[ind]
                    action += buffer_path["actions"]
                action /= args.top_k

                # add random noise to actions if requested
                if self.is_eval == False and args.train:
                    noise_start, noise_end = np.asarray([args.noise_std_start]*max_timesteps), np.asarray([args.noise_std_end]*max_timesteps)
                    action_std = (noise_start - noise_end) * (self.max_timesteps - np.arange(self.max_timesteps)) / self.max_timesteps + noise_end
                    action_noise = np.random.normal(loc=np.zeros((self.max_timesteps, self.env.action_space.shape[0])), \
                                                    scale=np.ones((self.max_timesteps, self.env.action_space.shape[0]))*np.expand_dims(action_std,1))
                    action += action_noise

                actions.append(action)
            # done
            actions = np.asarray(actions).swapaxes(1, 0)

        # clip just in case
        actions[actions<max(self.env.action_space.low)] = max(self.env.action_space.low)
        actions[actions>min(self.env.action_space.high)] = min(self.env.action_space.high)

        for i in range(self.max_timesteps):
            action = actions[i]
            obs, rewards, dones, _ = self.env.step(action)
            for n, done in enumerate(dones):
                rollout_paths[n]["actions"].append(action[n])
                rollout_paths[n]["rewards"].append(rewards[n])
                self.obs = obs
                if not done:
                    rollout_paths[n]["obs"].append(obs[n])

        # return rollout out paths
        return rollout_paths

if __name__ == "__main__":
    # Create train env
    train_env = SubprocVecEnv([make_env(i) for i in range(args.num_cpu)])
    # Create eval env
    eval_env = SubprocVecEnv([make_env(i+1000) for i in range(args.num_iter_eval)])

    dummy_env = gym.make(args.env) # used for misc stuff
    max_timesteps = dummy_env.spec.timestep_limit
    episode_rewards = []
    if 'Sparse' in dummy_env.spec.id:
        sparse = True
        episode_successes = []
    else:
        sparse = False

    buffer_paths = []
    neigh = None

    runner = Runner(train_env, max_timesteps, is_eval=False)
    eval_runner = Runner(eval_env, max_timesteps, is_eval=True)

    ##### start rollout
    all_episode_rewards_train, all_episode_successes_train = deque(maxlen=100), deque(maxlen=100)
    all_episode_rewards_eval, all_episode_successes_eval = deque(maxlen=100), deque(maxlen=100)

    for t in range(args.num_iter_train):
        rollout_paths = runner.rollout(neigh, buffer_paths)
        added_to_buffer = False
        for n in range(len(rollout_paths)):
            episode_rewards = rollout_paths[n]["rewards"]
            all_episode_rewards_train.append(sum(episode_rewards))
            if sparse and 1 in episode_rewards:
                all_episode_successes_train.append(1)
                # add to buffer according to conditions
                if args.train and len(buffer_paths) < args.max_buffer_size:
                    buffer_paths.append(rollout_paths[n])
                    added_to_buffer = True
            else:
                all_episode_successes_train.append(0)

        # if something added to buffer then recreate nearest neigh object
        if added_to_buffer:
            print ("added to buffer")
            neigh = NearestNeighbors(n_neighbors=args.top_k, radius=float('inf'))

            neigh.fit(np.concatenate([np.expand_dims(buffer_path["obs"][0], 0) for buffer_path in buffer_paths], axis=0))

        # add evaluation
        if t % args.eval_after == 0:
            print ('evaluating')
            eval_rollout_paths = eval_runner.rollout(neigh, buffer_paths)
            all_episode_rewards_eval = np.sum(np.array([np.sum(eval_rollout_path["rewards"]) for eval_rollout_path in eval_rollout_paths]))
            all_episode_successes_eval = np.sum(np.array([int(np.sum(eval_rollout_path["rewards"])>=1) for eval_rollout_path in eval_rollout_paths]))

            print (all_episode_rewards_eval, all_episode_successes_eval)

    # saving buffer_paths
    print ('saving paths')
    with open(os.path.join(log_dir, "buffer_paths.pkl"), 'wb') as f_buffer_paths:
        pickle.dump(buffer_paths, f_buffer_paths)
    print ('saved')

    sys.exit("done")
