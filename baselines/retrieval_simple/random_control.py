#!/usr/bin/env python
import tempfile
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

import numpy as np
import gym

env_id = "Reacher-v1"
num_episodes = 100

noise_end = 0.#0.02
noise_start = 0.33


if __name__ == "__main__":
    env = gym.make(env_id)
    max_timesteps = env.spec.timestep_limit

    for _ in range(num_episodes):
        env.reset()
        episode_rewards = []
        for t in range(max_timesteps):
            env.render()
            #noise = (noise_start - noise_end) * (max_timesteps - t) / max_timesteps + noise_end
            noise = 0.05
            #action = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=(env.action_space.shape[0],))
            #action *= noise
            action = np.random.normal([0]*env.action_space.shape[0], [noise]*env.action_space.shape[0])
            print (noise)
            _, reward, _, _ = env.step(action)
            episode_rewards.append(reward)

    print (episode_rewards)
