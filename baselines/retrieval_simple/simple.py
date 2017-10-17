#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--top_k', help='top k retrieved actions', type=int, default=1)
parser.add_argument('--num_demon', help='number of demonstrations', type=int, default=-1) # -1 means all of them
parser.add_argument('--noise_std', help='scale of noise', type=float, default=0.05)

parser.add_argument('--num_episodes_train', help='number of episodes total', type=int, default=100) # total number of episodes
parser.add_argument('--eval_after', help='eval after', type=int, default=100) # total number of episodes
parser.add_argument('--num_episodes_eval', help='number of episodes to eval', type=int, default=100) # total number of episodes

parser.add_argument('--env', help='environment ID', type=str, default="SparseReacher-v1")
parser.add_argument('--load_path', help='path to load trained model from', type=str, default="/home/mansimov/logdir/acktr-mujoco/Reacher-v1-seed1")

parser.add_argument('--start_state', action='store_true')
parser.add_argument('--current_state', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--max_demon', help='max number of demonstrations if train', type=int, default=1000) # -1 means all of them

args = parser.parse_args()
assert (args.env == "Reacher-v1" or args.env == "SparseReacher-v1")

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
args.save_path = log_dir

import numpy as np
import pickle
import gym
from baselines import logger
from baselines import bench
import random
#random.seed(args.seed)

import scipy
import scipy.misc

from sklearn.neighbors import NearestNeighbors

from collections import deque

def process_rollouts(max_timesteps):

    rollouts_path = os.path.join(args.load_path, "rollouts-v2.pkl")
    with open(rollouts_path, 'rb') as rollouts_input:
        paths = pickle.load(rollouts_input)

    if args.num_demon != -1:
        # select certain number of episodes
        random.shuffle(paths)
        paths = paths[0:args.num_demon]

    robot_state_start = np.concatenate([np.expand_dims(path["fingertip_com"][0],0) for path in paths])
    goal_state_start = np.concatenate([np.expand_dims(path["target_com"][0],0) for path in paths])
    return robot_state_start, goal_state_start, paths

def get_state(env):
    fingertip_pos = env.env.env.get_body_com("fingertip")
    target_pos = env.env.env.get_body_com("target")
    return fingertip_pos, target_pos

def rollout(env, neigh, paths, is_eval):
    env.reset()

    episode_states, episode_actions, episode_rewards = [], [], []
    robot_state_current, goal_state_current = get_state(env)
    robot_state_current = np.expand_dims(robot_state_current, 0)
    goal_state_current = np.expand_dims(goal_state_current, 0)
    episode_state = np.concatenate([robot_state_current,\
                            goal_state_current], axis=1)
    episode_states.append(episode_state)

    if neigh != None:
        top_ind = neigh.kneighbors(episode_state, args.top_k, return_distance=False)
        actions = np.zeros((max_timesteps, env.action_space.shape[0]))
        for ind in top_ind[0]:
            path = paths[ind]
            actions += path["scaled_action"]
        actions /= args.top_k
    else:
        actions = np.random.uniform(low=env.action_space.low, \
                high=env.action_space.high, size=(max_timesteps, env.action_space.shape[0]))

    # if is_eval is False and args.train add some random noise to action
    if is_eval == False and args.train:
        actions += \
            np.random.normal(np.ones(actions.shape)*0.0, np.ones(actions.shape)*args.noise_std)
    # clip just in case
    actions[actions<max(env.action_space.low)] = max(env.action_space.low)
    actions[actions>min(env.action_space.high)] = min(env.action_space.high)

    # rollout
    for i in range(max_timesteps):
        action = actions[i]
        _, reward, done, _ = env.step(action)
        episode_actions.append(action)
        episode_rewards.append(reward)
        if done:
            break

    episode_actions = np.asarray(episode_actions)
    return episode_rewards, episode_actions, robot_state_current, goal_state_current

if __name__ == "__main__":
    # Create train env and eval_env
    train_env = gym.make(args.env)
    #train_env.seed(args.seed)
    if logger.get_dir():
        train_env = bench.Monitor(train_env, os.path.join(logger.get_dir(), "train.monitor.json"))

    eval_env = gym.make(args.env)
    #eval_env.seed(args.seed+100)
    if logger.get_dir():
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), "eval.monitor.json"))

    max_timesteps = train_env.spec.timestep_limit

    episode_rewards = []
    if 'Sparse' in train_env.spec.id:
        sparse = True
        episode_successes = []
    else:
        sparse = False

    if args.num_demon == 0:
        robot_state_start, goal_state_start, paths = None, None, None
        neigh = None
    else:
        robot_state_start, goal_state_start, paths = process_rollouts(max_timesteps)
        state_start = np.concatenate([robot_state_start, goal_state_start], axis=1)
        # add them to NearestNeighbors library
        neigh = NearestNeighbors(n_neighbors=args.top_k, radius=float('inf'))
        neigh.fit(state_start)

    ##### start rollout
    all_episode_rewards_train, all_episode_successes_train = deque(maxlen=100), deque(maxlen=100)
    all_episode_rewards_eval, all_episode_successes_eval = deque(maxlen=100), deque(maxlen=100)

    for t in range(args.num_episodes_train):
        episode_rewards, episode_actions, robot_state_current, goal_state_current \
            = rollout(train_env, neigh, paths, is_eval=False)

        all_episode_rewards_train.append(sum(episode_rewards))
        if sparse and 1 in episode_rewards:
            all_episode_successes_train.append(1)
            # add to training set
            if args.train and (paths == None or len(paths) < args.max_demon):
                print ("new demon")
                if neigh != None:
                    state_start = np.append(state_start, \
                        np.concatenate([robot_state_current, goal_state_current], axis=1), axis=0)
                    paths.append({"scaled_action": np.asarray(episode_actions)})
                else:
                    state_start = np.concatenate([robot_state_current, goal_state_current], axis=1)
                    paths = [{"scaled_action": np.asarray(episode_actions)}]
                print (state_start.shape, len(paths))
                # recreate nearest neigh object
                neigh = NearestNeighbors(n_neighbors=args.top_k, radius=float('inf'))
                neigh.fit(state_start)

                assert (state_start.shape[0] == len(paths))

        # add evaluation
        if t % args.eval_after == 0:
            for _ in range(args.num_episodes_eval):
                episode_rewards, _, _, _ = rollout(eval_env, neigh, paths, is_eval=True)
                all_episode_rewards_eval.append(sum(episode_rewards))
                if sparse and 1 in episode_rewards:
                    all_episode_successes_eval.append(1)

    print ("Total TRAIN episodes {} ; Average episode reward {} ".format(args.num_episodes_train, np.mean(np.asarray(all_episode_rewards_train))))
    if sparse:
        print ("Total TRAIN episodes {} ; Average success percent {} ".format(args.num_episodes_train, float(sum(all_episode_successes_train)) / float(args.num_episodes_train) * 100))


    print ("Total EVAL episodes {} ; Average episode reward {} ".format(args.num_episodes_eval, np.mean(np.asarray(all_episode_rewards_eval))))
    if sparse:
        print ("Total EVAL episodes {} ; Average success percent {} ".format(args.num_episodes_eval, float(sum(all_episode_successes_eval)) / float(args.num_episodes_eval) * 100))
