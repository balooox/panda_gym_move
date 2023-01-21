from panda_gym.envs.tasks.stack import Stack
from panda_gym.pybullet import PyBullet
import numpy as np


class MyStackTask(Stack):
    def __init__(self, sim) -> None:
        super().__init__(sim)

    def _create_scene(self) -> None:
        super()._create_scene()
        self.sim.create_box(
            body_name="moving_platform",
            half_extents=np.array([0.1, 0.1, 0.01]),
            mass=500,
            position=np.array([-0.5, -.1, 0.05]),
            rgba_color=np.array([1, 0.2, 0.5, 1.0]),
        )

        self.sim.create_box(
            body_name="moving_target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=5.0,
            #ghost=True,
            position=np.array([-0.5, -.1, 0.1]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )
