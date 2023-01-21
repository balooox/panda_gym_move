from MyPandaStackEnv import MyPandaStackEnv

env = MyPandaStackEnv(render_mode="human")

observation, info = env.reset()
i = 0

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # env.print_cube_position()

    env.set_moving_target_position()

    if terminated or truncated:
        observation, info = env.reset()

    i += 1
