import sys
import time

import gym
import panda_gym
import custom_env
from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC


argv_len = len(sys.argv)

if argv_len == 1:
    print("Usage: python ./enjoy algo file")
    print("algo: training algorithm, can be TQC or SAC ")
    print("file: path to .zip file of trained model")
    exit(-1)
elif argv_len == 2:
    print("More parameters are required")
    print("Usage: python ./enjoy algo file")
    print("Help: execute python ./enjoy for more information")
    print()
elif argv_len > 3:
    print("Too many parameters")
    print("Usage: python ./enjoy algo file")
    print("Help: execute python ./enjoy for more information")
    exit(-1)

algo = sys.argv[1]

if algo != "TQC" and algo != "SAC":
    print("Wrong training algorithm")
    print("Parameter algo can only be 'TQC' or 'SAC'")
    exit(-1)

path_to_zip = sys.argv[2]

env_id = "PandaPickAndPlaceAndMove-v1"
env = gym.make(env_id, render=True)

model = TQC.load(path_to_zip, env=env)

num_episode = 1000

obs = env.reset()

for i in range(num_episode):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

    time.sleep(0.05)

    if done:
        env.reset()







