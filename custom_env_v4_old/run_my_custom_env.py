from PandaPickAndPlaceMoveEnv import PandaPickAndPlaceMoveEnv
import time

env = PandaPickAndPlaceMoveEnv(render=True)
# env = PandaPickAndPlaceEnv(render=True)

info = env.reset()
i = 0

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if i % 400 == 0:
        env.reset()

    if done:
        observation, info = env.reset()

    # Simulation verlangsamen
    time.sleep(0.05)
    i += 1
