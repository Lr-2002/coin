"""
Tabletop-Pick-Bottle-v1 Environment

This environment implements a tabletop task where the robot needs to pick a bottle.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import articulations
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
    load_articulation_from_json,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import table
from mani_skill.utils.structs import Articulation, Link, Pose
import os


@register_env("Tabletop-Pick-Bottle-v1", max_episode_steps=5000)
class PickBottleEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A tabletop environment where the robot needs to pick a bottle.

    **Randomizations:**
    - The bottle's position on the table

    **Success Conditions:**
    - The bottle has been successfully picked
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda

    description = "pick up the bottle and put it on the marker"
    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Add any task-specific parameters here
        super().__init__(
            *args,
            **kwargs,
        )

    def _load_scene(self, options=None):
        """Load the task-specific objects into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)
        self.bottle = self.load_from_config(
            "configs/water_bottle.json", "bottle", convex=True
        )
        self.goal_area = self._create_goal_area()

    def _initialize_episode(self, env_idx, options=None):
        """Initialize the episode by setting up objects and robot"""
        # Call the parent method to initialize the base scene
        super()._initialize_episode(env_idx, options)

        # TODO: Initialize task-specific objects
        # Example:
        self.bottle.set_pose(sapien.Pose(p=[0.1, 0.05, 0], q=euler2quat(np.pi, 0, 0)))

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)

        # TODO: Implement task-specific success criteria
        # Example:
        # distance = torch.norm(self.object.pose.p - self.target_position)
        # success = distance < 0.05
        distance = self.calculate_object_distance(
            self.bottle, self.goal_area, axis=[0, 1]
        )
        if distance <= 0.05 and self.is_static(self.bottle):
            success = torch.ones_like(success)
        return {
            "success": success,
        }
