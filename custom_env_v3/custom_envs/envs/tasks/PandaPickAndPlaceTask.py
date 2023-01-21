from typing import Any, Dict, Union

import numpy as np
import random

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class PandaPickAndPlaceMoveTask(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([0, obj_xy_range / 2, 0])
        self.moving_direction = 1
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="moving_platform",
            half_extents=np.array([0.1, 0.1, 0.01]),
            mass=500,
            position=np.array([.15, 0, 0.05]),
            rgba_color=np.array([1, 0.2, 0.5, 1.0]),
        )

    def take_step(self):
        cur_moving_platform = self.sim.get_base_position("moving_platform")
        cur_moving_target = self.sim.get_base_position("target")
        orientation = np.array([0, 0, 0, 1])
        if self.moving_direction == 1:
            cur_moving_platform[1] += 0.001
            cur_moving_target[1] += 0.001
            self.sim.set_base_pose("moving_platform", cur_moving_platform, orientation)
            self.sim.set_base_pose("target", cur_moving_target, orientation)
        elif self.moving_direction == 0:
            cur_moving_platform[1] -= 0.001
            cur_moving_target[1] -= 0.001
            self.sim.set_base_pose("moving_platform", cur_moving_platform, orientation)
            self.sim.set_base_pose("target", cur_moving_target, orientation)

        if cur_moving_platform[1] > 0.25:
            self.moving_direction = 0
        elif cur_moving_platform[1] < -0.25:
            self.moving_direction = 1

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("moving_platform", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        """
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal
        """
        goal_position = np.array([.15, 0, 0.05])
        y_noise = random.randrange(-25, 25) / 100
        # z_noise = randrange(0, 25) / 100
        goal_position[1] += y_noise
        # goal_position[2] += z_noise

        return goal_position

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "dense":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
