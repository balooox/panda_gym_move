from PandaPickAndPlaceMoveEnv import PandaPickAndPlaceMoveEnv
from panda_gym.envs.panda_tasks.panda_pick_and_place import PandaPickAndPlaceEnv

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

    i += 1
