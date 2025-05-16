from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import random

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv


@register_env("Tabletop-Find-Cube-RedDown-v1", max_episode_steps=5000)
class FindSealEnv(UniversalTabletopEnv):
    """
    Find Seal Environment

    Task: Place seal1 on the red and white target area

    Features:
    1. Three seals are placed on the ground
    2. A white and red target area is drawn using the ManiSkill API
    3. The goal is to place seal1 on the red and white target area
    """

    description = "find the cube which have red face downward, and put it on the marker with red face upward"
    workflow = [
        "see all the face of farest cube ",
        "see all the face of left cube ",
        "see all the face of nearst cube ",
        "find the cube have no red face over the ground and put it to the marker",
    ]

    def __init__(self, *args, **kwargs):
        self.seal_friction = 1.0
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.ORIENT],
            "rob": [Robot.PERSPECTIVE, Robot.ACT_NAV],
            "iter": [Inter.HISTORY, Inter.PLAN]
        }
        self.seal_mass = 0.5
        self.target_radius = 0.05
        self.success_distance_threshold = (
            0.03  # Distance threshold for successful placement
        )
        super().__init__(*args, **kwargs)

        self.query_query = "which seal has red face downward"
        self.query_selection = {"A": "A", "B": "B", "C": "C"}
        self.query_answer = "B"

    def _load_scene(self, options: dict):
        """Load the scene with three seals and a target area"""
        super()._load_scene(options)

        # Create the three seals
        # self.seal1 = self.load_from_config('configs/seal_with_circle.json', 'swc')
        # self.seal1 = self.load_from_config('configs/seal_with_chinese_number.json', 'swcn')
        # self.seal1 = self.load_from_config('configs/seal_with_chinese_word.json', 'swcw')
        self.seal1 = self._create_seal(
            "configs/rubik_cube.json", "rubik_cube_1", [0.2, -0.1, 0.08]
        )
        self.seal2 = self._create_seal(
            "configs/rubik_cube.json", "rubik_cube_2", [0.4, -0.0, 0.08]
        )
        self.seal3 = self._create_seal(
            "configs/rubik_cube.json", "rubik_cube_3", [-0.8, 0.1, 0.08]
        )

        # Create the red and white target area
        self.target_area = actors.build_red_white_target(
            self.scene,
            radius=self.target_radius,
            thickness=1e-5,
            name="target_area",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[0.3, 0.3, 0.001], q=euler2quat(0, np.pi / 2, 0).tolist()
            ),
        )

    def _create_seal(self, config_path, name, position, ori=[0, 0, 0]):
        """Create a seal object"""
        # Create a seal-like object (using a default object for simplicity)
        seal = self.load_from_config(
            config_path, name, convex=True
        )  # Set name using actor.name property seal.name = name seal.set_mass(self.seal_mass)
        seal.set_pose(sapien.Pose(p=position, q=euler2quat(*np.deg2rad(ori)).tolist()))
        return seal

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode"""
        super()._initialize_episode(env_idx, options)

        # Randomize positions of seals
        # positions = [
        #     [0.0 + random.uniform(-0.05, 0.05), 0.1 + random.uniform(-0.05, 0.05), 0.03],
        #     [0.0 + random.uniform(-0.05, 0.05), -0.1 + random.uniform(-0.05, 0.05), 0.03],
        #     [-0.0 + random.uniform(-0.05, 0.05), 0.2 + random.uniform(-0.05, 0.05), 0.03]
        # ]
        #
        total_z = 0.03

        self.seal1.set_pose(
            sapien.Pose(p=[0, 0.1, total_z], q=euler2quat(0, np.pi / 2, 0).tolist())
        )
        self.seal2.set_pose(
            sapien.Pose(
                p=[0.2, -0.1, total_z], q=euler2quat(0, np.pi / 2 * 2, 0).tolist()
            )
        )
        self.seal3.set_pose(
            sapien.Pose(p=[-0.2, 0.0, total_z], q=euler2quat(0, 0, 0).tolist())
        )
        #
        # Randomize target position (but keep it on the table)
        target_pos = [
            0.0 + random.uniform(-0.1, 0.1),
            0.0 + random.uniform(-0.1, 0.1),
            0.001,
        ]
        self.target_area.set_pose(
            sapien.Pose(p=target_pos, q=euler2quat(0, np.pi / 2, 0).tolist())
        )

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        # print (self.calculate_object_distance(self.seal1, self.target_area) <= 0.05 , self.is_static(self.seal1) , self.compare_angle(self.seal1,[0, -90, 0] , specific_axis='pitch'))
        if (
            self.calculate_object_distance(self.seal1, self.target_area) <= 0.05
            and self.is_static(self.seal1)
            and self.compare_angle(self.seal1, [0, -90, 0], specific_axis="pitch")
        ):
            success = torch.ones_like(success)

        # Check if seal1 is on the target area
        # seal1_pos = self.seal1.pose.p
        # target_pos = self.target_area.pose.p
        #
        # # Calculate horizontal distance (ignoring height)
        # distance = torch.norm(
        #     torch.tensor([seal1_pos[0] - target_pos[0], seal1_pos[1] - target_pos[1]], device=self.device)
        # )
        #
        # # Task is successful if seal1 is close enough to the target
        # success = distance < self.success_distance_threshold
        #
        return success
