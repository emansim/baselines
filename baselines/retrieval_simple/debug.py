import gym

if __name__ == "__main__":
    env = gym.make("HalfCheetah-v1")
    for _ in range(10):
        ob1 = env.reset()
        ob2 = env.env._get_obs()
        print (ob1==ob2)
'''
from rllab.envs.mujoco.half_cheetah_env_x import HalfCheetahEnvX

if __name__ == "__main__":
    env = HalfCheetahEnvX()
    print (env.frame_skip)
    print (env.spec.timestep_limit)
'''
