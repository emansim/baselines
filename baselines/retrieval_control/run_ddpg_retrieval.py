import os, sys, shutil, argparse
import random
sys.path.append(os.getcwd())

from baselines.common.misc_util import boolean_flag

# parser to parse args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--env-id', type=str, default='Reacher-v1')
boolean_flag(parser, 'render-eval', default=False)
boolean_flag(parser, 'layer-norm', default=True)
boolean_flag(parser, 'render', default=False)
boolean_flag(parser, 'normalize-returns', default=False)
boolean_flag(parser, 'normalize-observations', default=True)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
parser.add_argument('--actor-lr', type=float, default=1e-4)
parser.add_argument('--critic-lr', type=float, default=1e-3)
boolean_flag(parser, 'popart', default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--reward-scale', type=float, default=1.)
parser.add_argument('--load-path', type=str, default=None)
parser.add_argument('--clip-norm', type=float, default=None)
parser.add_argument('--nb-epochs', type=int, default=5000)  # with default settings, perform 1M steps total
parser.add_argument('--nb-epoch-cycles', type=int, default=20)
parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
parser.add_argument('--noise-type', type=str, default='ou_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
boolean_flag(parser, 'random-actions', default=False) # randomly sample actions with 20% probability
boolean_flag(parser, 'evaluation', default=False)
parser.add_argument('--top-k', help='top k', type=int, default=1)
parser.add_argument('--num-demon', help='number of demonstrations', type=int, default=0)
parser.add_argument('--demon-path', help='path to load trained model from', type=str, default="/home/mansimov/logdir/acktr-mujoco/Reacher-v1-seed1/")
parser.add_argument('--retrieved-action-scale', type=float, default=0.99)

args = parser.parse_args()
assert (args.env_id == "Reacher-v1" or args.env_id == "SparseReacher-v1")

folder_name = os.path.join(os.environ["checkpoint_dir"], "ddpg-retrieval-gpu")
try:
    os.mkdir(folder_name)
except:
    pass
log_dir = os.path.join(folder_name, "{}-seed{}".format(args.env_id, args.seed))
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)
os.environ["OPENAI_LOGDIR"] = log_dir

import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.retrieval_control.training_retrieval as training
from baselines.retrieval_control.models_retrieval import Actor, Critic
from baselines.retrieval_control.memory_retrieval import Memory
from baselines.retrieval_control.noise import *

import gym
import tensorflow as tf
import pickle

def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):
    rank = 0

    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json"%rank))
    gym.logger.setLevel(logging.WARN)

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'eval.monitor.json'))
        #env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    noise_type = noise_type.strip()
    assert noise_type == 'none' or 'ou' in noise_type or 'normal' in noise_type
    if 'ou' in noise_type:
        _, stddev = noise_type.split('_')
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    elif 'normal' in noise_type:
        _, stddev = noise_type.split('_')
        action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    else:
        pass

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # load demonstrations as a dictionary
    # with or without timestep ???
    demon_path = os.path.join(args.demon_path, "rollouts-v2.pkl")
    with open(demon_path, 'rb') as demon_input:
        demons = pickle.load(demon_input)

    if args.num_demon > 0:
        # select certain number of episodes
        random.shuffle(demons)
        demons = demons[0:args.num_demon]

    fingertip_coms_start = np.concatenate([np.expand_dims(demon["fingertip_com"][0],0) for demon in demons])
    target_coms_start = np.concatenate([np.expand_dims(demon["target_com"][0],0) for demon in demons])

    del kwargs["num_demon"]
    del kwargs["demon_path"]

    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory,
        demons=demons, fingertip_coms_start=fingertip_coms_start,
        target_coms_start=target_coms_start, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    # Run actual script.
    run(**vars(args))
