import numpy as np
import random
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from typing import Any, Dict, Optional, Tuple
from PickAndPlaceMoveTask import PickAndPlaceMoveTask


class PickAndPlaceMoveEnv(RobotTaskEnv):
    def __init__(
            self,
            render_mode,
            reward_type: str = "sparse",
            control_type: str = "ee"
    ):
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, block_gripper=False, base_position=([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlaceMoveTask(sim, reward_type=reward_type)
        super().__init__(robot, task)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.task.take_step()
        return super().step(action)

    def reset(
            self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.task.moving_direction = random.randint(0, 1)
        return super().reset(seed=seed, options=options)
