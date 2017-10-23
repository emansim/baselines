import numpy as np
import gym
import pickle

if __name__ == "__main__":
    env = gym.make("SparseHalfCheetah-v1")
    env.seed(0)
    path = "/Users/mansimov/logdir/retrieval-simple/SparseHalfCheetah-v1-seed0/success_paths.pkl"
    success_paths = pickle.load(open(path, "rb"))

    print (len(success_paths))

    print (success_paths[0].keys())
    for i in range(10):
        env.reset()
        print (env.env.get_body_com("torso")[0])
    sys.exit()

    for i in range(10):
        print ("trial {}".format(i))
        scaled_actions = success_paths[0]["scaled_action"]
        env.reset()
        for action in scaled_actions:
            env.render()
            env.step(action)
            print ("torso_body_pos")
            print (env.env.get_body_com("torso")[0])
