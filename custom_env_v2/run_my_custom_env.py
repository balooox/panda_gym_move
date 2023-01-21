from PickAndPlaceMoveEnv import PickAndPlaceMoveEnv
from panda_gym.envs.panda_tasks import PandaPickAndPlaceEnv

env = PickAndPlaceMoveEnv(render_mode="human")
# env = PandaPickAndPlaceEnv(render_mode="human")

observation, info = env.reset()
i = 0

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if i % 400 == 0:
        env.reset()

    if terminated or truncated:
        observation, info = env.reset()

    i += 1
