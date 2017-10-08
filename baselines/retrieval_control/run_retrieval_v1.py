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

    states = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    scaled_actions = np.concatenate([path["scaled_action"] for path in paths])

    timesteps_arange = np.arange(max_timesteps)
    timesteps = np.zeros((max_timesteps, max_timesteps))
    timesteps[np.arange(max_timesteps), timesteps_arange] = 1
    timesteps = np.repeat(timesteps, states.shape[0]//max_timesteps, axis=0)

    states = np.concatenate([states, timesteps], -1)

    return states, actions, scaled_actions, obfilter

if __name__ == "__main__":
    env = gym.make(args.env)
    max_timesteps, animate = env.spec.timestep_limit, args.animate
    num_episodes_total = 100
    episode_rewards = []
    if 'Sparse' in env.spec.id:
        sparse = True
        episode_successes = []

    train_states, train_actions, train_scaled_actions, obfilter = process_rollouts()
    count = 0

    for _ in range(num_episodes_total):

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

            """
            ac = np.mean(train_actions[top_ind])
            scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
            scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
            """

            scaled_ac = np.mean(train_actions[top_ind])

            prev_ob = np.copy(ob)
            ob, rew, done, _ = env.step(scaled_ac)
            if obfilter: ob = obfilter(ob)
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
