import os
import time
from collections import deque
import pickle

from baselines.retrieval_control.ddpg_retrieval import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf

from baselines.acktr.filters import ZFilter

from util import *

def get_retrieved_actions(env, demons, coms_start, top_k):
    fingertip_com = env.env.env.get_body_com("fingertip")
    target_com = env.env.env.get_body_com("target")
    fingertip_com = np.repeat(np.expand_dims(fingertip_com, axis=0), coms_start.shape[0], axis=0)
    target_com = np.repeat(np.expand_dims(target_com, axis=0), coms_start.shape[0], axis=0)
    com_current = np.concatenate([fingertip_com, target_com], axis=1)

    dist = np.sqrt(np.sum((com_current-coms_start)**2, axis=1))
    argsort_ind = np.argsort(dist)
    top_ind = argsort_ind[:top_k]

    retrieved_actions = np.zeros((env.spec.timestep_limit, env.action_space.shape[0]))

    for ind in top_ind:
        demon = demons[ind]
        retrieved_actions += demon["scaled_action"]

    retrieved_actions /= top_k
    return retrieved_actions

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    demons, fingertip_coms_start, target_coms_start, top_k=1, retrieved_action_scale=0.99,
    tau=0.01, eval_env=None, load_path=None, param_noise_adaption_interval=50, random_actions=False):
    rank = 0
    coms_start = np.concatenate([fingertip_coms_start, target_coms_start], axis=1)

    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, retrieved_action_scale=retrieved_action_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))
    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # Prepare everything.
        agent.initialize(sess)

        # saver/loader
        if load_path != None:
            saver = tf.train.Saver()
            saver.restore(U.get_session(), os.path.join(load_path, "best_model.ckpt"))
            print ("Loaded Model")
        else:
            # create saver
            saver = tf.train.Saver()

        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        retrieved_actions = get_retrieved_actions(env, demons, coms_start, top_k)

        # normalize obs here
        if eval_env is not None:
            eval_obs = eval_env.reset()
            eval_retrieved_actions = get_retrieved_actions(eval_env, demons, coms_start, top_k)
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        best_epoch_episode_rewards = float("-inf")
        t = 0

        epoch = 0
        start_time = time.time()

        for epoch in range(nb_epochs):
            epoch_episode_rewards = []
            epoch_episode_steps = []
            epoch_episode_eval_rewards = []
            epoch_episode_eval_steps = []
            epoch_start_time = time.time()
            epoch_actions = []
            epoch_qs = []
            epoch_episodes = 0
            t_episode = 0

            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, retrieved_actions[t_episode], apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    #new_obs, r, done, info = env.step(max_action * retrieved_actions[t_episode])

                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    if not done:
                        agent.store_transition(obs, retrieved_actions[t_episode], action, r, new_obs, retrieved_actions[t_episode+1], done)
                    else:
                        agent.store_transition(obs, retrieved_actions[t_episode], action, r, new_obs, retrieved_actions[t_episode], done)

                    obs = new_obs
                    t_episode += 1
                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()
                        # get retrieved actions for new episode
                        retrieved_actions = get_retrieved_actions(env, demons, coms_start, top_k)
                        t_episode = 0

                ################################
                # works up to here
                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    eval_t_episode = 0
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, eval_retrieved_actions[eval_t_episode], apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])

                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r
                        eval_t_episode += 1

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.
                            eval_t_episode = 0

            # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            duration = time.time() - start_time
            stats = agent.get_stats()

            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mean(stats[key])

            # Rollout statistics.
            combined_stats['rollout/return'] = mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_mean'] = mean(epoch_actions)
            combined_stats['rollout/actions_std'] = std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mean(epoch_qs)

            # Train statistics.
            combined_stats['train/loss_actor'] = mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = mean(epoch_adaptive_distances)

            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = mean(np.mean(eval_episode_rewards_history))
                combined_stats['eval/Q'] = mean(eval_qs)
                combined_stats['eval/episodes'] = mean(len(eval_episode_rewards))

            # Total statistics.
            combined_stats['total/duration'] = mean(duration)
            combined_stats['total/steps_per_second'] = mean(float(t) / float(duration))
            combined_stats['total/episodes'] = mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                # save model
                model_path = os.path.join(logdir, "model.ckpt")
                saver.save(U.get_session(), model_path)
                print ("Model saved to {}".format(model_path))

            if rank == 0 and logdir and mean(epoch_episode_rewards) > best_epoch_episode_rewards:
                # save best model
                model_path = os.path.join(logdir, "best_model.ckpt")
                saver.save(U.get_session(), model_path)
                print ("Best Model saved to {}".format(model_path))
                best_epoch_episode_rewards = mean(epoch_episode_rewards)
