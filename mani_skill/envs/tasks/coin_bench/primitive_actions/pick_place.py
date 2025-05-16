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


@register_env("Tabletop-Pick-Apple-v1", max_episode_steps=5000)
class PickPlaceEnv(UniversalTabletopEnv):
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

    description = "pick the apple to the marker"

    def __init__(
        self,
        *args,
        object_path=None,  # Path to the object GLB/USD file
        object_scale=0.01,  # Scale of the object
        object_mass=0.5,  # Mass of the object in kg
        object_friction=1.0,  # Friction coefficient of the object
        placement_threshold=0.05,  # Distance threshold for successful placement
        orientation=[0, 0, 0],
        object_config="configs/apple.json",
        **kwargs,
    ):
        config = {}
        self.object_config = object_config
        # self.object_path = config.get("usd-path", object_path)
        # self.object_scale = config.get("scale", object_scale)
        # self.object_mass = config.get("mass", object_mass)
        # self.object_friction = config.get("friction", object_friction)
        # orientation = config.get("orientation", orientation)
        self.object_orientation = create_orientation_from_degree(*orientation)
        self.placement_threshold = placement_threshold
        # Initialize target marker properties
        self.target_size = 0.01
        self.target_position = None
        super().__init__(*args, **kwargs)
        self.object_dict = {"apple": self.object}

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
        # self.object = self.load_asset(
        #     asset_path=self.object_path,
        #     scale=self.object_scale,
        #     mass=self.object_mass,
        #     friction=self.object_friction,
        # )
        #
        self.object = self.load_from_config(self.object_config, "apple", convex=True)
        # Create target marker
        # self.target_marker = self._create_target_marker()
        self.target_marker = self._create_goal_area()
        # self.goal_region = self.create_goal_region()

    def _create_target_marker(self):
        """Create a visual marker for the target placement location"""
        # Create actor builder
        builder = self.scene.create_actor_builder()

        # Add visual shape (green transparent cylinder)
        builder.add_cylinder_visual(
            radius=self.target_size,
            half_length=0.04,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.5]),
        )
        # builder.add_box_collision(sapien.Pose(), half_size=[0.03, 0.03, 0.02])

        # Build the actor (kinematic - no physics)
        target = builder.build_kinematic(name="target_marker")

        return target

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized object and target positions"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        table_height = self.table_scene.table_height

        # Randomize object position on table
        xy_range = 0.01
        object_z = 0.03  # Place slightly above table

        # Randomize object orientation (only around z-axis)
        # object_ori = torch.zeros(self.num_envs, 3)
        # object_ori[:, 0] = 0.5 * np.pi
        object_ori = (
            torch.tensor(self.object_orientation).unsqueeze(0).expand(self.num_envs, -1)
        )
        # Set object pose
        object_pose = sapien.Pose(
            p=[0.07, 0.07, object_z],
            q=euler2quat(
                object_ori[env_idx, 0].item(),
                object_ori[env_idx, 1].item(),
                object_ori[env_idx, 2].item(),
            ),
        )
        self.object.set_pose(object_pose)

        # Randomize target position (different from object position)
        target_z = 0.001  # Place just above table

        # Set target marker pose

    def _get_success(self, env_idx=None):
        """Evaluate task success"""
        # Check if object is grasped
        success = super()._get_success()
        is_grasped = self.agent.is_grasping(self.object)

        # input()
        # Check if robot is static
        is_robot_static = self.agent.is_static(0.2).numpy()[0]
        is_at_target = (
            self.calculate_object_distance(self.object, self.target_marker)
            < self.placement_threshold
        )
        # Success if object is at target and robot is static
        # breakpoint()
        suc = is_at_target & is_robot_static
        if suc:
            success = torch.ones_like(success)

        return success

    def display(self, obs):
        display_camera_views(obs)
