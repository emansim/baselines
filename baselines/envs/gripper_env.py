import os, sys
sys.path.append(os.getcwd())
from baselines.envs.mujoco_env import MujocoEnv
import numpy as np

class GripperEnv(MujocoEnv):
    def __init__(self):
        MujocoEnv.__init__(self, "/home/mansimov/projects/baselines-emansim/baselines/envs/xmls/3link_gripper_pusher.xml", 4)

if __name__ == "__main__":
    env = GripperEnv()
    env.reset()
    env._get_viewer()
    env.set_view(0)
    env.get_img()
    print ('got img')
    '''
    while True:
        env.render()
    '''
