#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--top_k', help='top k retrieved actions', type=int, default=1)
parser.add_argument('--noise_std', help='scale of noise', type=float, default=0.05)

parser.add_argument('--num_episodes_train', help='number of episodes total', type=int, default=100) # total number of episodes
parser.add_argument('--eval_after', help='eval after', type=int, default=100) # total number of episodes
parser.add_argument('--num_episodes_eval', help='number of episodes to eval', type=int, default=100) # total number of episodes

parser.add_argument('--env', help='environment ID', type=str, default="SparseReacher-v1")

parser.add_argument('--train', action='store_true')
parser.add_argument('--max_demon', help='max number of demonstrations if train', type=int, default=1000) # -1 means all of them

parser.add_argument('--animate', action='store_true')

parser.add_argument('--noise-type', type=str, default='ou_0.2')  # choices are ou_xx, normal_xx_xx

args = parser.parse_args()

folder_name = os.path.join(os.environ["checkpoint_dir"], "retrieval-simple-explr")
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
random.seed(args.seed)

import scipy
import scipy.misc

from sklearn.neighbors import NearestNeighbors

from collections import deque

from noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

action_noise = None

def get_state(env):
    return np.expand_dims(env.env.env._get_obs(), 0)

def rollout(env, neigh, paths, is_eval, render=False):
    env.reset()
    action_noise.reset()

    episode_states, episode_actions, episode_rewards = [], [], []
    episode_state = get_state(env)

    if neigh != None:
        top_ind = neigh.kneighbors(episode_state, args.top_k, return_distance=False)
        actions = np.zeros((max_timesteps, env.action_space.shape[0]))
        for ind in top_ind[0]:
            path = paths[ind]
            actions += path["scaled_action"]
        actions /= args.top_k

        if is_eval == False and args.train:
            # generate noise
            actions_noise = np.zeros((max_timesteps, env.action_space.shape[0]))
            for tt in range(max_timesteps):
                actions_noise[tt] = action_noise()

            actions += actions_noise
        '''
        if is_eval == False and args.train:
            #(noise_start - noise_end) * (max_timesteps - t) / max_timesteps + noise_end
            noise_start, noise_end = np.asarray([args.noise_std]*max_timesteps), np.asarray([0.0]*max_timesteps)
            actions_std = (noise_start - noise_end) * (max_timesteps - np.arange(max_timesteps)) / max_timesteps + noise_end
            actions_noise = np.random.normal(loc=np.zeros((max_timesteps, env.action_space.shape[0])), \
                                            scale=np.ones((max_timesteps, env.action_space.shape[0]))*np.expand_dims(actions_std,1))
            actions += actions_noise
        '''
    else:
        # hardcoded
        actions = np.random.uniform(low=-1*np.ones((max_timesteps, env.action_space.shape[0])),
                                    high=1*np.ones((max_timesteps, env.action_space.shape[0])),
                                    size=(max_timesteps, env.action_space.shape[0]))
        '''
        noise_start, noise_end = np.asarray([0.33]*max_timesteps), np.asarray([0.0]*max_timesteps)
        actions_std = (noise_start - noise_end) * (max_timesteps - np.arange(max_timesteps)) / max_timesteps + noise_end
        actions = np.random.normal(loc=np.zeros((max_timesteps, env.action_space.shape[0])), \
                                        scale=np.ones((max_timesteps, env.action_space.shape[0]))*np.expand_dims(actions_std,1))
        '''

    # clip just in case
    #### MASSIVE BUG (Lucky doesn't affect much)
    #actions[actions<max(env.action_space.low)] = max(env.action_space.low)
    #actions[actions>min(env.action_space.high)] = min(env.action_space.high)

    #actions[actions>max(env.action_space.low)] = max(env.action_space.low)
    #actions[actions<min(env.action_space.high)] = min(env.action_space.high)

    # rollout
    for i in range(max_timesteps):
        action = actions[i]
        if render:
            env.render()
        _, reward, done, _ = env.step(action)
        episode_actions.append(action)
        episode_rewards.append(reward)
        if done:
            break

    episode_actions = np.asarray(episode_actions)
    return episode_rewards, episode_actions, episode_state

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

    # set noise type
    current_noise_type = args.noise_type.strip()
    nb_actions = train_env.action_space.shape[0]
    if 'normal' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        action_noise.reset()
    if 'ou' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        action_noise.reset()
    else:
        raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    episode_rewards = []
    if 'Sparse' in train_env.spec.id:
        sparse = True
        episode_successes = []
    else:
        sparse = False

    state_start, paths, neigh = None, None, None

    ##### start rollout
    all_episode_rewards_train, all_episode_successes_train = deque(maxlen=100), deque(maxlen=100)
    all_episode_rewards_eval, all_episode_successes_eval = deque(maxlen=100), deque(maxlen=100)

    for t in range(args.num_episodes_train):
        episode_rewards, episode_actions, state_current \
            = rollout(train_env, neigh, paths, is_eval=False, render=False)

        all_episode_rewards_train.append(sum(episode_rewards))
        print ("episode {}".format(t))
        if sparse and 1 in episode_rewards:
            print ("success {}".format(t))
            all_episode_successes_train.append(1)
            # add to training set
            if args.train and (paths == None or len(paths) < args.max_demon):
                print ("new demon")
                if neigh != None:
                    state_start = np.append(state_start, state_current, axis=0)
                    paths.append({"scaled_action": np.asarray(episode_actions)})
                else:
                    state_start = np.copy(state_current)
                    paths = [{"scaled_action": np.asarray(episode_actions)}]
                print (state_start.shape, len(paths))
                # recreate nearest neigh object
                neigh = NearestNeighbors(n_neighbors=args.top_k, radius=float('inf'))
                neigh.fit(state_start)

                assert (state_start.shape[0] == len(paths))
        if sparse and 1 not in episode_rewards:
            all_episode_successes_train.append(0)

        # add evaluation
        if t % args.eval_after == 0:
            for _ in range(args.num_episodes_eval):
                episode_rewards, _, _, = rollout(eval_env, neigh, paths, is_eval=True)
                all_episode_rewards_eval.append(sum(episode_rewards))
                if sparse:
                    if 1 in episode_rewards:
                        all_episode_successes_eval.append(1)
                    else:
                        all_episode_successes_eval.append(0)
            # save the demonstrations
            if paths != None and state_start != None:
                print ('saving paths')
                with open(os.path.join(log_dir, "paths.pkl"), 'wb') as f_paths:
                    pickle.dump(paths, f_paths)
                with open(os.path.join(log_dir, "state_start.pkl"), 'wb') as f_state_start:
                    pickle.dump(state_start, f_state_start)
                print ('saved')

    # animate at the end of training if requested
    if args.animate:
        for _ in range(args.num_episodes_eval):
            rollout(eval_env, neigh, paths, is_eval=True, render=True)


    print ("Total TRAIN episodes {} ; Average episode reward {} ".format(args.num_episodes_train, np.mean(np.asarray(all_episode_rewards_train))))
    if sparse:
        print ("Total TRAIN episodes {} ; Average success percent {} ".format(args.num_episodes_train, np.mean(np.asarray(all_episode_successes_train)) * 100))

    print ("Total EVAL episodes {} ; Average episode reward {} ".format(args.num_episodes_eval, np.mean(np.asarray(all_episode_rewards_eval))))
    if sparse:
        print ("Total EVAL episodes {} ; Average success percent {} ".format(args.num_episodes_eval, np.mean(np.asarray(all_episode_successes_eval)) * 100))
