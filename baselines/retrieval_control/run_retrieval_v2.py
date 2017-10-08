#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--top_k', help='top k', type=int, default=1)
parser.add_argument('--num_demon', help='number of demonstrations', type=int, default=0)

parser.add_argument('--env', help='environment ID', type=str, default="Reacher-v1")
parser.add_argument('--load_path', help='path to load trained model from', type=str, default="/home/mansimov/logdir/acktr-mujoco/Reacher-v1-seed1")
parser.add_argument('--animate', help="whether to animate", action='store_true')

args = parser.parse_args()

from baselines.acktr.filters import ZFilter
import numpy as np
import pickle
import gym
import random

import scipy
import scipy.misc

def process_rollouts():
    # load obfilter ; important !!!
    obfilter_path = os.path.join(args.load_path, "obfilter.pkl")
    with open(obfilter_path, 'rb') as obfilter_input:
        obfilter = pickle.load(obfilter_input)

    # with or without timestep ???
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

    fingertip_coms_start = np.concatenate([np.expand_dims(path["fingertip_com"][0],0) for path in paths])
    target_coms_start = np.concatenate([np.expand_dims(path["target_com"][0],0) for path in paths])

    return fingertip_coms_start, target_coms_start, paths, obfilter


if __name__ == "__main__":
    env = gym.make(args.env)
    max_timesteps, animate = env.spec.timestep_limit, args.animate
    num_episodes_total = 100
    episode_rewards = []
    if 'Sparse' in env.spec.id:
        sparse = True
        episode_successes = []

    fingertip_coms_start, target_coms_start, paths, obfilter = process_rollouts()
    coms_start = np.concatenate([fingertip_coms_start, target_coms_start], axis=1)

    count = 0
    for _ in range(num_episodes_total):

        # do the actual rollout
        ob = env.reset()

        # find closest path based on fingertip_com and target_com
        fingertip_com = env.env.get_body_com("fingertip")
        target_com = env.env.get_body_com("target")
        fingertip_com = np.repeat(np.expand_dims(fingertip_com, axis=0), coms_start.shape[0], axis=0)
        target_com = np.repeat(np.expand_dims(target_com, axis=0), coms_start.shape[0], axis=0)
        com_current = np.concatenate([fingertip_com, target_com], axis=1)

        dist = np.sqrt(np.sum((com_current-coms_start)**2, axis=1))
        argsort_ind = np.argsort(dist)
        top_ind = argsort_ind[:args.top_k]

        actions = np.zeros((max_timesteps, env.action_space.shape[0]))
        for ind in top_ind:
            path = paths[ind]
            actions += path["scaled_action"]

        actions /= args.top_k

        rewards = []
        # just execute the actions
        for i in range(max_timesteps):
            if animate:
                im = env.render('rgb_array')
                scipy.misc.imsave('images/im_{}.jpg'.format(count), im)
                count += 1
            action = actions[i]
            _, rew, done, _ = env.step(action)
            rewards.append(rew)
            if done:
                terminated = True
                break
        episode_rewards.append(np.sum(rewards))
        if sparse and 1 in rewards:
            episode_successes.append(1)
        #print ("Average reward {}".format(np.sum(rewards)))

    print ("Total episodes {} ; Average episode reward {} ".format(num_episodes_total, np.mean(np.asarray(episode_rewards))))
    if sparse:
        print ("Total episodes {} ; Average success percent {} ".format(num_episodes_total, float(sum(episode_successes)) / float(num_episodes_total) * 100))
