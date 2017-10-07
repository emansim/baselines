#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--timesteps_per_batch', help='timesteps_per_batch', type=int, default=2500)
parser.add_argument('--env', help='environment ID', type=str, default="Reacher-v1")
parser.add_argument('--load_path', help='path to load trained model from', type=str, default=None)
parser.add_argument('--save_after', help='save after certain number of iterations', type=int, default=200)
parser.add_argument('--save_rollouts', help="save rollouts", action='store_true')
parser.add_argument('--animate', help="whether to animate", action='store_true')

args = parser.parse_args()

folder_name = os.path.join(os.environ["checkpoint_dir"], "acktr-mujoco")
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

import argparse
import logging
import os
import tensorflow as tf
import gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_cont_internal import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction

def train(env_id, num_timesteps, seed):
    env=gym.make(env_id)
    if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))
    set_global_seeds(seed)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    with tf.Session(config=tf.ConfigProto()) as session:
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=args.timesteps_per_batch,
            desired_kl=0.002,
            num_timesteps=num_timesteps,
            save_path=args.save_path, save_after=args.save_after, load_path=args.load_path,
            save_rollouts=args.save_rollouts, animate=args.animate)

        env.close()

if __name__ == "__main__":
    train(args.env, num_timesteps=30e6, seed=args.seed)
