import sys
import time
import gym
import os
import panda_gym
import numpy as np

from stable_baselines3 import TD3, A2C, PPO, SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TRPO, TQC
from PandaPickAndPlaceMoveEnv import PandaPickAndPlaceMoveEnv
from stable_baselines3.common.env_checker import check_env


# Instantiate the env
# env = PandaPickAndPlaceEnv()
env = PandaPickAndPlaceMoveEnv(render=False)
# Define and Train the agent
model = TQC(env=env, policy="MultiInputPolicy", replay_buffer_class=None, learning_rate=0.001)

model.learn(total_timesteps=1000)