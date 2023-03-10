import os
import sys

sys.path.append("..")

from gym.envs.registration import register

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="My_PandaPickAndPlace{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="custom_env_v3.custom_env.env:PandaPickAndPlaceMoveEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )
