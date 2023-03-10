import gym
import time
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer
import custom_envs
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback

"""
env_list = ['My_PandaReach', 'Two_PandaReach', 'Three_PandaReach',
            'My_PandaSlide',
            'My_PandaPickAndPlace', 'My_TwoPandaPickAndPlace',
            'Two_PandaPush', 'Three_PandaPush',
            'Two_Obj_PandaPush', 'Three_Obj_PandaPush',
            'My_PandaReachPlate', 'My_TwoPandaReachPlate',
            'My_PandaStack']
env_opts = ['Joints', 'Dense']
"""

env_id = 'My_PandaPickAndPlace'

timestemp = time.strftime("%Y%m%d-%H%M")

env = gym.make(env_id+'-v2')

log_dir = './tensorboard/SAC' + env_id

total_timesteps = 2000000



checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='model_checkpoints/'+env_id + "SAC" + timestemp,
                                         name_prefix=env_id)

model = SAC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=1000000, batch_size=2048,
            replay_buffer_class=HerReplayBuffer, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future', ), gamma=0.95, tau=0.05,
            verbose=1,
            tensorboard_log=log_dir)

model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

model.save('./trained/SAC'+env_id+'/'+env_id+model.__class__.__name__ + "SAC" + timestemp)