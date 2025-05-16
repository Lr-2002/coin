from logging import debug
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


@register_env("Tabletop-Rotate-Cube-Twice-v1", max_episode_steps=5000)
class RotateTwiceEnv(UniversalTabletopEnv):
    """
    Rotate Twice Environment

    Task: Rotate a Rubik's cube 180 degrees from downward to upward orientation

    Features:
    1. A Rubik's cube is placed on the table
    2. The initial orientation is downward
    3. The goal is to rotate the cube 180 degrees to an upward orientation
    """

    description = "rotate the cube till the green face upward"
    workflow = ["rotate the cube for 90", "rotate the cube for 90"]

    def __init__(self, *args, **kwargs):
        self.cube_friction = 1.0
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.ORIENT, Obj.MESH],
            "rob": [Robot.ACT_NAV],
            "iter": [Inter.PLAN,  Inter.FAIL_ADAPT],
        }
        self.cube_mass = 0.5
        self.success_angle_threshold = 20  # Degrees threshold for successful rotation
        self.cube = None
        self.initial_orientation = np.array([180, 0, 0])
        self.target_orientation = np.array([0, 0, 0])

        super().__init__(*args, **kwargs)
        self.query_query = "Where is the green face?"
        self.query_selection = {
            "A": "On the top of the cube",
            "B": "On the side of the cube",
            "C": "On the bottom of the cube",
        }
        self.query_answer = "C"
        # Did it rotate the cube once?
        # Did it rotate the cube twice?

    def _load_scene(self, options: dict):
        """Load the scene with a Rubik's cube"""
        super()._load_scene(options)

        # Create the Rubik's cube
        self.cube = self.load_from_config(
            "configs/rubik_cube.json", "rubik_cube", body_type="dynamic", convex=True
        )

        # Set initial position and orientation (downward)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode"""
        super()._initialize_episode(env_idx, options)

        # Randomize position slightly but keep orientation downward
        # position = [
        #     0.2 + random.uniform(-0.05, 0.05),
        #     0.0 + random.uniform(-0.05, 0.05),
        #     -0.05,
        # ]
        #
        # # Downward orientation (180 degrees around x-axis)
        if self.cube is not None:
            initial_position = [0.1, 0.0, 0]
            # Set initial orientation to downward (180 degrees around x-axis)
            initial_orientation = self.initial_orientation

            self.cube.set_pose(
                sapien.Pose(
                    p=initial_position,
                    q=euler2quat(*np.deg2rad(initial_orientation)).tolist(),
                )
            )

    def _get_success(self, env_idx=None):
        """Compute dense reward"""
        success = super()._get_success()
        print(
            self.compare_angle(
                self.cube,
                self.target_orientation,
                specific_axis="roll",
                threshold=self.success_angle_threshold,
                debug=True,
            )
        )
        if self.compare_angle(
            self.cube,
            self.target_orientation,
            specific_axis="roll",
            threshold=self.success_angle_threshold,
        ):
            success = torch.ones_like(success)
        return success
