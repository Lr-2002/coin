from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench import UniversalTabletopEnv
import numpy as np


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Put-Fork-OnPlate-v1", max_episode_steps=5000)
class PutForkOnPlate(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to pick up an object and place it at a target location.

    **Randomizations:**
    - The object's position is randomized on the table
    - The object's orientation is randomized around the z-axis
    - The target location is randomized on the table

    **Success Conditions:**
    - The object is placed at the target location (within a threshold)
    - The robot is static (velocity < 0.2)
    """

    description = "put the fork on the plate"

    def __init__(
        self,
        *args,
        object_path=None,  # Path to the object GLB/USD file
        object_scale=0.01,  # Scale of the object
        object_mass=0.5,  # Mass of the object in kg
        object_friction=1.0,  # Friction coefficient of the object
        placement_threshold=0.05,  # Distance threshold for successful placement
        # orientation=[0, 0, 0],
        object_config="configs/plate_new.json",
        **kwargs,
    ):
        config = {}
        self.object_config = object_config
        # Initialize target marker properties
        self.target_size = 0.01
        self.target_position = None

        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Load the object to be manipulated
        # if self.object_path and os.path.exists(self.object_path):
        #     self.object = self.load_asset(
        #         asset_path=self.object_path,
        #         scale=self.object_scale,
        #         mass=self.object_mass,
        #         friction=self.object_friction,
        #     )
        # else:
        #     # Create a default cube if no object is specified
        #     self.object = self._create_default_object()
        self.plate = self.load_from_config(
            self.object_config, "plate", convex=False, body_type="static"
        )

        self.fork = self.load_from_config(
            object_config="configs/fork.json", name="fork", convex=True
        )
        # Create target marker

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized object and target positions"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        table_height = self.table_scene.table_height

        # Randomize object position on table
        xy_range = 0.1
        object_xy = torch.rand(self.num_envs, 2) * 0 * 2 - xy_range
        object_z = 0.02  # Place slightly above table

        # Randomize object orientation (only around z-axis)
        # object_ori = torch.zeros(self.num_envs, 3)
        # object_ori[:, 0] = 0.5 * np.pi
        object_ori = (
            torch.tensor([-np.pi / 2, 0, 0]).unsqueeze(0).expand(self.num_envs, -1)
        )
        # Set object pose
        # breakpoint()
        object_pose = sapien.Pose(
            p=[object_xy[env_idx, 0].item(), object_xy[env_idx, 1].item(), object_z],
            q=euler2quat(
                object_ori[env_idx, 0].item(),
                object_ori[env_idx, 1].item(),
                object_ori[env_idx, 2].item(),
            ),
        )
        self.plate.set_pose(object_pose)
        self.fork.set_pose(sapien.Pose([0.12, -0.05, 0.01], [1, 0, 0, 0]))

    def _get_success(self, env_idx=None):
        """Evaluate task success"""
        success = super()._get_success()
        if self.calculate_object_distance(self.fork, self.plate) < 0.10:
            success = torch.ones_like(success)
        return success
