import gym
import time
import sys
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer
import custom_env
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback

argv_len = len(sys.argv)

if argv_len == 1:
    print("Usage: python ./train algo amount")
    print("algo: training algorithm, can be TQC or SAC ")
    print("amount: total amount of iterations")
    exit(-1)
elif argv_len == 2:
    print("More parameters are required")
    print("Usage: python ./train algo amount")
    print("Help: execute python ./train for more information")
    print()
elif argv_len > 3:
    print("Too many parameters")
    print("Usage: python ./train algo amount")
    print("Help: execute python ./train for more information")
    exit(-1)

algo = sys.argv[1]

if algo != "TQC" and algo != "SAC":
    print("Wrong training algorithm")
    print("Parameter algo can only be 'TQC' or 'SAC'")
    exit(-1)

total_timesteps = int(sys.argv[2])

if total_timesteps < 0:
    print("Amount must be a positive integer")
    exit(-1)

env_id = 'PandaPickAndPlaceAndMove-v1'
timestamp = time.strftime("%Y%m%d-%H%M")
log_dir = './tensorboard/' + env_id

env = gym.make(env_id)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path='model_checkpoints/' + algo + '/' + env_id + '_' + timestamp,
    name_prefix=env_id)

if algo == 'TQC':
    model = TQC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=1000000, batch_size=2048,
                replay_buffer_class=HerReplayBuffer, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future', ), gamma=0.95, tau=0.05,
                verbose=1,
                tensorboard_log=log_dir)

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save('./trained/' + algo + '/' + env_id + model.__class__.__name__ + timestamp)
elif algo == 'SAC':
    model = SAC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=1000000, batch_size=2048,
                replay_buffer_class=HerReplayBuffer, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future', ), gamma=0.95, tau=0.05,
                verbose=1,
                tensorboard_log=log_dir)

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model.save('./trained/' + algo + '/' + env_id + model.__class__.__name__ + timestamp)
