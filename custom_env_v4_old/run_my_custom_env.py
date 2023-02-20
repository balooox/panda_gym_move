from PandaPickAndPlaceMoveEnv import PandaPickAndPlaceMoveEnv
import time
import gym
import custom_envs

# env = PandaPickAndPlaceMoveEnv(render=True)

env_id = 'My_PandaPickAndPlace'
env = gym.make(env_id + '-v2', render=True)

info = env.reset()
i = 0

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if i % 800 == 0:
        env.reset()

    if done:
        observation, info = env.reset()

    # Simulation verlangsamen
    time.sleep(0.05)
    i += 1
