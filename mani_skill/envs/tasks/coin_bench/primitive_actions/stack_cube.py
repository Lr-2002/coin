from typing import Any, Dict, Union
import numpy as np
import torch
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.coin_bench.primitive_actions.pick_place import PickPlaceEnv
import sapien
from transforms3d.euler import euler2quat


@register_env("Tabletop-Stack-Cubes-v1", max_episode_steps=5000)
class StackCubeEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    The goal is to pick up a red cube and stack it on top of a green cube and let go of the cube without it falling

    **Randomizations:**
    - both cubes have their z-axis rotation randomized
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the red cube is on top of the green cube (to within half of the cube size)
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)
    """

    description = "stack all the cube"

    def __init__(self, *args, **kwargs):
        self.object_num = 2

        super().__init__(*args, **kwargs)

    def _load_scene(self, options):
        super()._load_scene(options)

        self.object_list = []
        for i in range(self.object_num):
            self.object_list.append(
                self._create_default_object(name="cube" + str(i + 1))
            )

    def get_random_pose(self, env_idx, xy_range=0.2):
        object_xy = torch.rand(self.num_envs, 2) * xy_range * 2 - xy_range
        object_z = 0.0  # Place slightly above table

        # Randomize object orientation (only around z-axis)
        # object_ori = torch.zeros(self.num_envs, 3)
        # object_ori[:, 0] = 0.5 * np.pi
        # Set object pose
        object_pose = sapien.Pose(
            p=[object_xy[env_idx, 0].item(), object_xy[env_idx, 1].item(), object_z],
        )
        return object_pose

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        table_height = self.table_scene.table_height

        # Randomize object position on table
        xy_range = 0.0
        for i, obj in enumerate(self.object_list):
            obj.set_pose(sapien.Pose(p=[0.1 * i, 0.2 * i, 0]))

        # Calculate target height based on cube size
        # Use a fixed cube size since we can't access collision shapes directly
        cube_size = (
            0.04  # Standard cube size (matches the default in _create_default_object)
        )
        # Target height is the sum of heights of all cubes
        self.target_height = cube_size * self.object_num

        # Additional initialization specific to this task

    def get_max_height(self):
        max_height = 0
        for obj in self.object_list:
            minn, maxx = self.get_aabb(obj)
            max_height = max(maxx[2], max_height)
        return max_height

    def is_stable(self):
        is_st = True
        for obj in self.object_list:
            if not self.is_static(obj):
                is_st = False
        return is_st

    def is_grasp(self):
        is_gs = False
        for obj in self.object_list:
            if self.agent.is_grasping(obj)[0]:
                is_gs = True
        return is_gs

    def _get_success(self, env_idx=None):
        suc = self.target_height - self.get_max_height() < 0.03
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if suc and self.is_stable() and not self.is_grasp():
            success = torch.ones_like(success)
        # for obj in self.object_list:
        #     self.show_two_objects(obj)
        return {
            "success": success,
        }
