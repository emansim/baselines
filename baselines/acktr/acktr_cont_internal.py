import os
import pickle
import numpy as np
import tensorflow as tf
from baselines import logger
from baselines import common
from baselines.common import tf_util as U
from baselines.acktr import kfac
from baselines.acktr.filters import ZFilter

def pathlength(path):
    return path["reward"].shape[0]# Loss function that we'll differentiate to get the policy gradient

def rollout(env, policy, max_pathlength, animate=False, obfilter=None, save_rollouts=False):
    """
    Simulate the env and policy for max_pathlength steps
    """
    ob = env.reset()
    env_id = env.spec.id.lower()
    if "reacher" in env_id:
        state_name = "fingertip"
    elif "jaco" in env_id:
        state_name = "jaco_link_hand"
    else:
        print ("something wrong")
        sys.exit()

    if save_rollouts:
        fingertip_com = env.env.env.get_body_com(state_name)
        target_com = env.env.env.get_body_com("target") # com stands for center of mass
    prev_ob = np.float32(np.zeros(ob.shape))
    if obfilter: ob = obfilter(ob, update=not save_rollouts)
    terminated = False

    obs = []
    acs = []
    scaled_acs = []
    ac_dists = []
    logps = []
    rewards = []
    if save_rollouts:
        fingertip_coms = []
        target_coms = []
    for _ in range(max_pathlength):
        if animate:
            env.render()
        state = np.concatenate([ob, prev_ob], -1)
        obs.append(state)
        if save_rollouts:
            fingertip_coms.append(fingertip_com)
            target_coms.append(target_com)
        ac, ac_dist, logp = policy.act(state)
        acs.append(ac)
        ac_dists.append(ac_dist)
        logps.append(logp)
        prev_ob = np.copy(ob)
        scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
        scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
        scaled_acs.append(scaled_ac)
        ob, rew, done, _ = env.step(scaled_ac)
        if obfilter: ob = obfilter(ob)
        if save_rollouts:
            fingertip_com = env.env.env.get_body_com(state_name)
            target_com = env.env.env.get_body_com("target")
        rewards.append(rew)
        if done:
            print ("------------------")
            terminated = True
            break
    if save_rollouts:
        return {"observation" : np.array(obs), "terminated" : terminated,
                "reward" : np.array(rewards), "action" : np.array(acs),
                "scaled_action": np.array(scaled_acs),
                "fingertip_com": np.array(fingertip_coms),
                "target_com": np.array(target_coms),
                "action_dist": np.array(ac_dists), "logp" : np.array(logps)}
    else:
        return {"observation" : np.array(obs), "terminated" : terminated,
                "reward" : np.array(rewards), "action" : np.array(acs),
                "scaled_action": np.array(scaled_acs),
                "action_dist": np.array(ac_dists), "logp" : np.array(logps)}

def save(saver, obfilter, save_path):
    model_path = os.path.join(save_path, "model.ckpt")
    saver.save(U.get_session(), model_path)
    print ("Model saved to {}".format(model_path))
    obfilter_path = os.path.join(save_path, "obfilter.pkl")
    with open(obfilter_path, 'wb') as obfilter_output:
        pickle.dump(obfilter, obfilter_output, pickle.HIGHEST_PROTOCOL)
    print ("Obfilter saved to {}".format(obfilter_path))

def learn(env, policy, vf, gamma, lam, timesteps_per_batch, num_timesteps,
    animate=False, callback=None, desired_kl=0.002,
    save_path="./", save_after=200, load_path=None, save_rollouts=False):

    obfilter = ZFilter(env.observation_space.shape)

    max_pathlength = env.spec.timestep_limit
    stepsize = tf.Variable(initial_value=np.float32(np.array(0.03)), name='stepsize')
    inputs, loss, loss_sampled = policy.update_info
    optim = kfac.KfacOptimizer(learning_rate=stepsize, cold_lr=stepsize*(1-0.9), momentum=0.9, kfac_update=2,\
                                epsilon=1e-2, stats_decay=0.99, async=1, cold_iter=1,
                                weight_decay_dict=policy.wd_dict, max_grad_norm=None)
    pi_var_list = []
    for var in tf.trainable_variables():
        if "pi" in var.name:
            pi_var_list.append(var)

    update_op, q_runner = optim.minimize(loss, loss_sampled, var_list=pi_var_list)
    do_update = U.function(inputs, update_op)
    U.initialize()

    # start queue runners
    enqueue_threads = []
    coord = tf.train.Coordinator()
    for qr in [q_runner, vf.q_runner]:
        assert (qr != None)
        enqueue_threads.extend(qr.create_threads(U.get_session(), coord=coord, start=True))

    if load_path != None:
        saver = tf.train.Saver()
        saver.restore(U.get_session(), os.path.join(load_path, "model.ckpt"))
        obfilter_path = os.path.join(load_path, "obfilter.pkl")
        with open(obfilter_path, 'rb') as obfilter_input:
            obfilter = pickle.load(obfilter_input)
        print ("Loaded Model")
    else:
        # create saver
        saver = tf.train.Saver()

    i = 0
    timesteps_so_far = 0
    while True:
        if timesteps_so_far > num_timesteps:
            break
        logger.log("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            #path = rollout(env, policy, max_pathlength, animate=(len(paths)==0 and (i % 10 == 0) and animate), obfilter=obfilter)
            if "jaco" in env.spec.id.lower():
                path = rollout(env, policy, max_pathlength, animate=animate, obfilter=obfilter, save_rollouts=save_rollouts)
                goal_dist = np.linalg.norm(env.env.env.get_body_com("jaco_link_hand") \
                            - env.env.env.get_body_com("target"))
                if goal_dist <= 0.12:
                    print ("goal_dist {} ; episode added".format(goal_dist))
                    paths.append(path)
            else:
                path = rollout(env, policy, max_pathlength, animate=animate, obfilter=obfilter, save_rollouts=save_rollouts)

            n = pathlength(path)
            timesteps_this_batch += n
            timesteps_so_far += n
            if timesteps_this_batch > timesteps_per_batch:
                break

        if save_rollouts:
            # save the rollouts
            rollouts_path = os.path.join(load_path, "rollouts-v2.pkl")
            with open(rollouts_path, 'wb') as rollouts_output:
                pickle.dump(paths, rollouts_output, pickle.HIGHEST_PROTOCOL)
            sys.exit()

        # Estimate advantage function
        vtargs = []
        advs = []
        for path in paths:
            rew_t = path["reward"]
            return_t = common.discount(rew_t, gamma)
            vtargs.append(return_t)
            vpred_t = vf.predict(path)
            vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])
            delta_t = rew_t + gamma*vpred_t[1:] - vpred_t[:-1]
            adv_t = common.discount(delta_t, gamma * lam)
            advs.append(adv_t)
        # Update value function
        vf.fit(paths, vtargs)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        action_na = np.concatenate([path["action"] for path in paths])
        oldac_dist = np.concatenate([path["action_dist"] for path in paths])
        logp_n = np.concatenate([path["logp"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        # Policy update
        do_update(ob_no, action_na, standardized_adv_n)

        min_stepsize = np.float32(1e-8)
        max_stepsize = np.float32(1e0)
        # Adjust stepsize
        kl = policy.compute_kl(ob_no, oldac_dist)
        if kl > desired_kl * 2:
            logger.log("kl too high")
            U.eval(tf.assign(stepsize, tf.maximum(min_stepsize, stepsize / 1.5)))
        elif kl < desired_kl / 2:
            logger.log("kl too low")
            U.eval(tf.assign(stepsize, tf.minimum(max_stepsize, stepsize * 1.5)))
        else:
            logger.log("kl just right!")

        logger.record_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("EpRewSEM", np.std([path["reward"].sum()/np.sqrt(len(paths)) for path in paths]))
        logger.record_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logger.record_tabular("KL", kl)
        if callback:
            callback()
        logger.dump_tabular()
        # save model if necessary
        if i % save_after == 0:
            save(saver, obfilter, save_path)
        i += 1
