from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
import baselines.common.tf_util as U
from baselines.common.running_mean_std import RunningMeanStd
from baselines.retrieval_control.util import reduce_std

from baselines.acktr.filters import ZFilter

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-6)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class DDPG(object):
    def __init__(self, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None, random_actions=False,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        adaptive_param_noise=True, adaptive_param_noise_policy_threshold=.1,
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., retrieved_action_scale=0.99):
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.retrieved_actions0 = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='retrieved_actions0')
        self.retrieved_actions1 = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='retrieved_actions1')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.random_actions = random_actions
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
            self.observation_range[0], self.observation_range[1])

        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = actor(normalized_obs0, self.retrieved_actions0, retrieved_action_scale)
        self.normalized_critic_tf = critic(normalized_obs0, self.actions)
        self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        Q_obs1 = denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1, self.retrieved_actions1, retrieved_action_scale)), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(normalized_obs0)
        self.setup_stats()
        self.setup_target_network_updates()
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        # create optimizer
        actor_grads = U.grad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_grads_flat = U.flatten(self.actor.trainable_vars, actor_grads)
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr,\
                                beta1=0.9, beta2=0.999, epsilon=1e-08)
        grads = list(zip(actor_grads, self.actor.trainable_vars))
        self.actor_update_op = self.actor_optimizer.apply_gradients(grads)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        critic_grads = U.grad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_grads_flat = U.flatten(self.critic.trainable_vars, critic_grads)
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr,\
                                beta1=0.9, beta2=0.999, epsilon=1e-08)
        grads = list(zip(critic_grads, self.critic.trainable_vars))
        self.critic_update_op = self.critic_optimizer.apply_gradients(grads)
        """
        self.critic_optimizer = tf.train.AdamOptimizer(var_list=self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)
        """

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def pi(self, obs, retrieved_action, apply_noise=True, compute_Q=True):
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf

        feed_dict = {self.obs0: [obs], self.retrieved_actions0: [retrieved_action]}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None

        # before flatten it was dimension 1 x 2
        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        # add posibility of doing completely random action
        if np.random.uniform() < 0.2 and self.random_actions:
            action = np.random.uniform(low=-1.0, high=1.0, size=action.shape)

        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q

    def store_transition(self, obs0, retrieved_action0, action, reward, obs1, retrieved_action1, terminal1):
        reward *= self.reward_scale
        self.memory.append(obs0, retrieved_action0, action, reward, obs1, retrieved_action1, terminal1)
        if self.normalize_observations:
            self.obs_rms.update(self.sess, np.array([obs0]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q], feed_dict={
                self.obs1: batch['obs1'],
                self.retrieved_actions1: batch['retrieved_actions1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.ret_rms.update(target_Q.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std : np.array([old_std]),
                self.old_mean : np.array([old_mean]),
            })

            # Run sanity check. Disabled by default since it slows down things considerably.
            # print('running sanity check')
            # target_Q_new, new_mean, new_std = self.sess.run([self.target_Q, self.ret_rms.mean, self.ret_rms.std], feed_dict={
            #     self.obs1: batch['obs1'],
            #     self.rewards: batch['rewards'],
            #     self.terminals1: batch['terminals1'].astype('float32'),
            # })
            # print(target_Q_new, target_Q, new_mean, new_std)
            # assert (np.abs(target_Q - target_Q_new) < 1e-3).all()
        else:
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1: batch['obs1'],
                self.retrieved_actions1: batch['retrieved_actions1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads_flat, self.actor_loss, self.critic_grads_flat, self.critic_loss,
                self.actor_update_op, self.critic_update_op]
        actor_grads, actor_loss, critic_grads, critic_loss, _, _ = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.retrieved_actions0: batch['retrieved_actions0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
        })
        """
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
        })
        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)
        """

        return critic_loss, actor_loss

    def initialize(self, sess):
        # initialize vars
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        #self.actor_optimizer.sync() # we don't need it no mpi
        #self.critic_optimizer.sync() # we don't need it no mpi
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['obs0'],
            self.retrieved_actions0: self.stats_sample['retrieved_actions0'],
            self.actions: self.stats_sample['actions'],
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = mpi_mean(distance)
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })