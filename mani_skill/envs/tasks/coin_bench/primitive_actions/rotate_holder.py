from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat, quat2euler
import random

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv


@register_env("Tabletop-Rotate-Holder-v1", max_episode_steps=5000)
class RotateHolderEnv(UniversalTabletopEnv):
    """
    Rotate Twice Environment

    Task: Rotate a Rubik's cube 180 degrees from downward to upward orientation

    Features:
    1. A Rubik's cube is placed on the table
    2. The initial orientation is downward
    3. The goal is to rotate the cube 180 degrees to an upward orientation
    """

    description = "rotate the holder till the hole upward"

    def __init__(self, *args, **kwargs):
        self.success_angle_threshold = 5.0  # Degrees threshold for successful rotation
        self.cube = None
        self.initial_orientation = np.array([0, 0, 0])
        self.target_orientation = np.array([90, 0, 0])

        self.initial_position = [0.05, 0.0, 0.04]
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        """Load the scene with a Rubik's cube"""
        super()._load_scene(options)

        # Create the Rubik's cube
        self.cube = self.load_from_config(
            "configs/pen_holder.json", "holder", body_type="dynamic", convex=True
        )

        # Set initial position and orientation (downward)
        if self.cube is not None:
            # Set initial orientation to downward (180 degrees around x-axis)

            self.cube.set_pose(
                sapien.Pose(
                    p=self.initial_position,
                    q=euler2quat(*np.deg2rad(self.initial_orientation)).tolist(),
                )
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode"""
        super()._initialize_episode(env_idx, options)

        if self.cube is not None:
            self.cube.set_pose(
                sapien.Pose(
                    p=self.initial_position,
                    q=euler2quat(*np.deg2rad(self.initial_orientation)).tolist(),
                )
            )

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)

        # Get current orientation in degrees for debugging
        obj_quat = self.cube.pose.q.cpu().numpy()[0]
        obj_euler_deg = np.rad2deg(quat2euler(obj_quat))
        # print(f"Current orientation (degrees): {obj_euler_deg}")
        # print(f"Target orientation (degrees): {self.target_orientation}")

        # Use the refactored compare_angle function with the success_angle_threshold
        if self.compare_angle(
            self.cube,
            self.target_orientation,
            threshold=self.success_angle_threshold,
            specific_axis="roll",
        ) and self.is_static(self.cube):
            success = torch.ones_like(success)

        return {"success": success}
