import numpy as np
import random


from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from custom_env_v4_old.custom_envs.envs.tasks.PandaPickAndPlaceTask import PandaPickAndPlaceMoveTask
from panda_gym.pybullet import PyBullet
from typing import Any, Dict, Optional, Tuple, Union


class PandaPickAndPlaceMoveEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PandaPickAndPlaceMoveTask(sim, reward_type="dense")
        super().__init__(robot, task)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.task.take_step()
        print("#############################")
        print("ee_position")
        print(self.robot.get_ee_velocity())
        return super(PandaPickAndPlaceMoveEnv, self).step(action)

    def reset(self) -> Dict[str, np.ndarray]:
        self.task.moving_direction = random.randint(0, 1)
        return super(PandaPickAndPlaceMoveEnv, self).reset()
