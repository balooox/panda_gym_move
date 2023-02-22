import time
import gym
import custom_env

env_id = 'PandaPickAndPlaceAndMove-v1'
env = gym.make(env_id, render=True)

info = env.reset()
i = 0

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if i % 800 == 0:
        env.reset()

    if done:
        env.reset()

    time.sleep(0.05)
    i += 1
