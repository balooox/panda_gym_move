import numpy as np
from gym import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda

class Panda2(Panda):

    def __init__(
            self,
            sim: PyBullet,
            block_gripper: bool = False,
            base_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
            control_type: str = "ee",
    ) -> None:
        super().__init__(
            sim,
            block_gripper,
            base_position,
            control_type
        )

    def get_gripper_pos(self):
        self.get_link_position(11)




