import gym
import panda_gym
import time
from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC
import custom_envs

env_id = 'My_PandaPickAndPlace'
env = gym.make(env_id + '-v2', render=True)
model = TQC.load("TQC2.zip", env=env)

num_episode = 1000

obs = env.reset()

for i in range(num_episode):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(rewards)
    env.render()

    if done:
        print("reset")
        env.reset()


