import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

from MyStackTask import MyStackTask


class MyPandaStackEnv(RobotTaskEnv):
    def __init__(
            self,
            render_mode,
    ):
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim)
        task = MyStackTask(sim)
        self.direction = 0
        super().__init__(robot, task)

    def set_moving_target_position(self):
        cur_moving_platform = self.sim.get_base_position("moving_platform")
        cur_moving_target = self.sim.get_base_position("moving_target")
        orientation = np.array([0, 0, 0, 1])
        if self.direction == 1:
            cur_moving_platform[1] += 0.001
            cur_moving_target[1] += 0.001
            self.sim.set_base_pose("moving_platform", cur_moving_platform, orientation)
            #self.sim.set_base_pose("moving_target", cur_moving_target, orientation)
        elif self.direction == 0:
            cur_moving_platform[1] -= 0.001
            cur_moving_target[1] -= 0.001
            self.sim.set_base_pose("moving_platform", cur_moving_platform, orientation)
            #self.sim.set_base_pose("moving_target", cur_moving_target, orientation)

        if cur_moving_platform[1] > 0.25:
            self.direction = 0
        elif cur_moving_platform[1] < -0.25:
            self.direction = 1
