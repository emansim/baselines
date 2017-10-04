#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--top_k', help='top k', type=int, default=5)
parser.add_argument('--env', help='environment ID', type=str, default="Jaco-v1")
parser.add_argument('--load_path', help='path to load trained model from', type=str, default="/home/mansimov/logdir/acktr-mujoco/Jaco-v1-seed1")
parser.add_argument('--animate', help="whether to animate", action='store_true')

args = parser.parse_args()

from baselines.acktr.filters import ZFilter
import numpy as np
import pickle
import gym
import random

import scipy
import scipy.misc

def process_rollouts(max_timesteps):
    # load obfilter ; important !!!
    obfilter_path = os.path.join(args.load_path, "obfilter.pkl")
    with open(obfilter_path, 'rb') as obfilter_input:
        obfilter = pickle.load(obfilter_input)

    # with or without timestep ???
    rollouts_path = os.path.join(args.load_path, "rollouts.pkl")
    with open(rollouts_path, 'rb') as rollouts_input:
        paths = pickle.load(rollouts_input)

    # select certain number of episodes
    random.shuffle(paths)
    paths = paths[0:100]

    states = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])

    timesteps_arange = np.arange(max_timesteps)
    timesteps = np.zeros((max_timesteps, max_timesteps))
    timesteps[np.arange(max_timesteps), timesteps_arange] = 1
    timesteps = np.repeat(timesteps, states.shape[0]//max_timesteps, axis=0)

    states = np.concatenate([states, timesteps], -1)

    return states, actions, obfilter


if __name__ == "__main__":
    env = gym.make(args.env)
    max_timesteps, animate = env.spec.timestep_limit, args.animate
    num_total = 20
    episode_rewards = []
    train_states, train_actions, obfilter = process_rollouts(max_timesteps)
    count = 0
    for _ in range(num_total):

        # do the actual rollout
        ob = env.reset()
        prev_ob = np.float32(np.zeros(ob.shape))
        if obfilter: ob = obfilter(ob, update=False)
        rewards = []
        for i in range(max_timesteps):
            if animate:
                im = env.render('rgb_array')
                scipy.misc.imsave('images/im_{}.jpg'.format(count), im)
                count += 1
            timestep = np.zeros([max_timesteps])
            timestep[i] = 1
            state = np.concatenate([ob, prev_ob, timestep], -1)
            state = np.expand_dims(state, 0)
            state_repeated = np.repeat(state, train_states.shape[0], axis=0)
            dist = np.sqrt(np.sum((state_repeated-train_states)**2, axis=1))
            argsort_ind = np.argsort(dist)
            top_ind = argsort_ind[:args.top_k]

            ac = np.mean(train_actions[top_ind])
            scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
            scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)

            prev_ob = np.copy(ob)
            ob, rew, done, _ = env.step(scaled_ac)
            if obfilter: ob = obfilter(ob)
            rewards.append(rew)
            if done:
                terminated = True
                break
        episode_rewards.append(np.sum(rewards))
        #print ("Average reward {}".format(np.sum(rewards)))
    print ("Total episodes {} ; Average episode reward {} ".format(num_total, np.mean(np.asarray(episode_rewards))))
