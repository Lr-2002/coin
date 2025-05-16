from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat
import random

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
)

"""
Task: Find and pick up an apple that is inside a cabinet.

The robot needs to open the cabinet to reach and pick up the apple.
"""


@register_env("Tabletop-Seek-Objects-WithObstacle-v1", max_episode_steps=5000)
class SeekObjectsWithObstacleEnv(UniversalTabletopEnv):
    description = "find the cube in the cabinet and pick it up"
    workflow = ["open the door", "pick the cube", "put it on the marker"]

    def __init__(self, *args, **kwargs):
        # Set default friction for the apple
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.OBSTACLE],
            "rob": [Robot.MORPH, Robot.ACT_NAV, Robot.PERSPECTIVE],
            "iter": [Inter.FAIL_ADAPT, Inter.PLAN],
        }
        self.object_friction = random.uniform(0.1, 1)
        super().__init__(*args, **kwargs)
        self.query_query = "Where is the cube?"
        self.query_selection = {"A": "In the drawer", "B": "In the cabinet"}
        self.query_answer = "B"
        # Did it open the drawer?
        # Did it open the cabinet?
        # Did it pick up the cube?
        # Did it place the cube to the marker?

    def _load_scene(self, options: dict):
        """Load the scene with table, cabinet, and apple"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Load the cabinet using articulation loader
        self.cabinet = self._load_cabinet("1054")

        self.goal = self._create_goal_area()
        # Load the apple from config file
        self.apple = self._load_apple()

    def _load_apple(self):
        """Load an apple from config file"""
        apple_config = "configs/cc.json"
        apple = self.load_from_config(
            apple_config, name="cube", body_type="dynamic", convex=True
        )

        return apple

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with cabinet and apple"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Position the cabinet on the table
        # cabinet_position = torch.tensor([0.3, 0.0, 0.01], device=self.device)
        self.cabinet.set_pose(
            sapien.Pose(
                p=[-0.1, -0.5, 0.35],
                q=euler2quat(0, 0, -np.pi / 2),  # No rotation
            )
        )
        self.set_articulation_joint(self.cabinet, "joint_0", 0)
        cabinet_position = self.cabinet.pose.p[0]
        # Position the apple inside the cabinet
        # Offset from the cabinet position to place it inside
        offset = torch.tensor([0.0, 0.1, -0.15], device=self.device)  # TODO
        # offset = torch.tensor([0.0, 0.0, 0.25], device=self.device)
        apple_position = cabinet_position + offset
        print(apple_position)

        # self.apple.set_pose(
        #     sapien.Pose(
        #         # q=euler2quat(0, 0, 0),  # No rotation
        #         p=apple_position,
        #     )
        # )
        self.set_pos(self.apple, apple_position)

        # Set target position for the task (where to place the apple)kk
        # self.target_position = torch.tensor([0.5, 0.0, 0.05], device=self.device)
        # self.set_articulation_joint(self.cabinet, 'joint_0', 1, pd_version='hard')
        self.set_articulation_joint(self.cabinet, "joint_1", 0.1)

    def _get_success(self, env_idx=None):
        """Evaluate task success"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self.calculate_object_distance(self.apple, self.goal) <= 0.04:
            success = torch.ones_like(success)
        #
        return success
