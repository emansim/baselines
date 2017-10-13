#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--top_k', help='top k', type=int, default=1)
parser.add_argument('--num_demon', help='number of demonstrations', type=int, default=0)
parser.add_argument('--noise_std', help='scale of noise', type=float, default=0.05)

parser.add_argument('--num_episodes_total', help='number of episodes total', type=int, default=100)

parser.add_argument('--env', help='environment ID', type=str, default="Reacher-v1")
parser.add_argument('--load_path', help='path to load trained model from', type=str, default="/home/mansimov/logdir/acktr-mujoco/Reacher-v1-seed1")
parser.add_argument('--animate', help="whether to animate", action='store_true')

parser.add_argument('--start_state', action='store_true')
parser.add_argument('--current_state', action='store_true')
parser.add_argument('--include_timestep', action='store_true')
parser.add_argument('--train', action='store_true')

args = parser.parse_args()

assert (args.env == "Reacher-v1" or args.env == "SparseReacher-v1")

import numpy as np
import pickle
import gym
import random
random.seed(args.seed)

import scipy
import scipy.misc

def process_rollouts(max_timesteps):
    rollouts_path = os.path.join(args.load_path, "rollouts-v2.pkl")
    with open(rollouts_path, 'rb') as rollouts_input:
        paths = pickle.load(rollouts_input)

    if args.num_demon > 0:
        # select certain number of episodes
        random.shuffle(paths)
        paths = paths[0:args.num_demon]

    # do not concatenate paths instead concatenate pos of fingertip and targetpos
    #fingertip_coms = np.concatenate([path["fingertip_com"] for path in paths])
    #target_coms = np.concatenate([path["target_com"] for path in paths])
    if args.start_state:
        fingertip_coms_start = np.concatenate([np.expand_dims(path["fingertip_com"][0],0) for path in paths])
        target_coms_start = np.concatenate([np.expand_dims(path["target_com"][0],0) for path in paths])
        return fingertip_coms_start, target_coms_start, paths
    else:
        fingertip_coms = np.concatenate([path["fingertip_com"] for path in paths])
        target_coms = np.concatenate([path["target_com"] for path in paths])
        scaled_actions = np.concatenate([path["scaled_action"] for path in paths])
        # timesteps
        timesteps_arange = np.arange(max_timesteps)
        timesteps = np.zeros((max_timesteps, max_timesteps))
        timesteps[np.arange(max_timesteps), timesteps_arange] = 1
        timesteps = np.repeat(timesteps, fingertip_coms.shape[0]//max_timesteps, axis=0)
        return fingertip_coms, target_coms, timesteps, scaled_actions

def get_com(env):
    fingertip_com = env.env.get_body_com("fingertip")
    target_com = env.env.get_body_com("target")
    return fingertip_com, target_com

def get_action_current_state(env, t, coms, scaled_actions):
    fingertip_com, target_com = get_com(env)
    com_current = np.concatenate([np.expand_dims(fingertip_com, axis=0), \
                    np.expand_dims(target_com, axis=0)], axis=1)

    if args.include_timestep:
        timestep = np.zeros((1, max_timesteps))
        timestep[0][t] = 1
        com_current = np.concatenate([com_current, timestep], axis=1)

    com_current = np.repeat(com_current, coms.shape[0], axis=0)
    ###### get top ind
    dist = np.sqrt(np.sum((com_current-coms)**2, axis=1))
    argsort_ind = np.argsort(dist)
    top_ind = argsort_ind[:args.top_k]
    ###### get action
    action = np.mean(scaled_actions[top_ind], axis=0)
    return action


if __name__ == "__main__":
    env = gym.make(args.env)
    env.seed(args.seed)
    max_timesteps = env.spec.timestep_limit

    episode_rewards = []
    if 'Sparse' in env.spec.id:
        sparse = True
        episode_successes = []
    else:
        sparse = False

    if args.start_state:
        fingertip_coms_start, target_coms_start, paths = process_rollouts(max_timesteps)
        coms_start = np.concatenate([fingertip_coms_start, target_coms_start], axis=1)

    if args.current_state:
        fingertip_coms, target_coms, timesteps, scaled_actions = process_rollouts(max_timesteps)
        if args.include_timestep:
            coms = np.concatenate([fingertip_coms, target_coms, timesteps], axis=1)
        else:
            coms = np.concatenate([fingertip_coms, target_coms], axis=1)

    ##### start rollout
    for _ in range(args.num_episodes_total):
        # reset the episode
        ob = env.reset()
        episode_com_start = None
        episode_actions = []
        episode_coms = []

        # if only based on the start_state
        if args.start_state:
            fingertip_com, target_com = get_com(env)
            episode_com_start = np.concatenate([np.expand_dims(fingertip_com, 0),\
                                np.expand_dims(target_com, 0)], axis=1)
            episode_coms.append(episode_com_start[0])

            ########################################
            fingertip_com = np.repeat(np.expand_dims(fingertip_com, axis=0), coms_start.shape[0], axis=0)
            target_com = np.repeat(np.expand_dims(target_com, axis=0), coms_start.shape[0], axis=0)
            com_current = np.concatenate([fingertip_com, target_com], axis=1)

            ###### get top ind
            dist = np.sqrt(np.sum((com_current-coms_start)**2, axis=1))
            argsort_ind = np.argsort(dist)
            top_ind = argsort_ind[:args.top_k]
            ###### get actions
            actions = np.zeros((max_timesteps, env.action_space.shape[0]))
            #print (top_ind, len(paths))
            for ind in top_ind:
                path = paths[ind]
                actions += path["scaled_action"]
            actions /= args.top_k

        # retrieve based on the current state
        if args.current_state:
            action = get_action_current_state(env, 0, coms, scaled_actions)

        ######## rollout the episode
        rewards = []

        # just execute the actions
        for i in range(max_timesteps):
            # if based on the current state
            if args.start_state:
                action = actions[i]
            if args.train:
                # add small random noise
                action += np.random.normal([0]*env.action_space.shape[0], [args.noise_std]*env.action_space.shape[0])
            _, rew, done, _ = env.step(action)
            ####append
            episode_actions.append(action)
            fingertip_com, target_com = get_com(env)
            episode_com = np.concatenate([np.expand_dims(fingertip_com, 0),\
                                np.expand_dims(target_com, 0)], axis=1)
            episode_coms.append(episode_com[0])
            rewards.append(rew)
            if done:
                terminated = True
                break
            #### get new action if based on the current state
            if args.current_state:
                action = get_action_current_state(env, i+1, coms, scaled_actions)

        ###### get stats update better way using bench monitor
        episode_rewards.append(np.sum(rewards))

        if sparse and 1 in rewards:
            episode_successes.append(1)

            # if train add as sucessful
            if args.train and args.start_state:
                # max 1000 demonstrations
                if coms_start.shape[0] < 1000:
                    # add to coms_start and add to paths
                    coms_start = np.append(coms_start, episode_com_start, axis=0)
                    # add scaled actions to dictionary
                    paths.append({"scaled_action": np.asarray(episode_actions)})
            if args.train and args.current_state:
                if coms.shape[0] // max_timesteps < 1000:
                    coms = np.append(coms, np.asarray(episode_coms), axis=0)
                    scaled_actions = np.append(scaled_actions, np.asarray(episode_actions), axis=0)


    print ("Total episodes {} ; Average episode reward {} ".format(args.num_episodes_total, np.mean(np.asarray(episode_rewards))))
    if sparse:
        print ("Total episodes {} ; Average success percent {} ".format(args.num_episodes_total, float(sum(episode_successes)) / float(args.num_episodes_total) * 100))
