from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench import UniversalTabletopEnv
from mani_skill.utils.building.actors.common import _build_by_type


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Finish-Hanobi-v1", max_episode_steps=5000)
class FinishHanoiEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to arrange the Tower of Hanoi disks on the base.
    The base has three sticks, and the disks need to be arranged on one stick with
    the biggest at the bottom, medium in the middle, and smallest at the top.

    **Randomizations:**
    - The initial positions of the disks are randomized on the table

    **Success Conditions:**
    - All disks are stacked on one of the sticks in the correct order
    - The robot is static (velocity < 0.2)
    """

    description = "Place all the hanobi in big to small from bottom to up"
    workflow = [
        "move the red one to the right",
        "move the yellow one to the right",
        "move the blue one to the top of right",
    ]

    def __init__(
        self,
        *args,
        success_threshold=0.05,  # Distance threshold for successful completion
        **kwargs,
    ):
        # Set success threshold
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.SEQ_NAV, Obj.MESH],
            "rob": [Robot.ACT_NAV],
            "iter": [Inter.PLAN, Inter.HISTORY]
        }
        self.success_threshold = success_threshold

        # Set task description
        # Initialize objects
        self.hanoi_base = None
        self.hanoi_biggest = None
        self.hanoi_mid = None
        self.hanoi_small = None

        # Store stick positions for checking success
        self.stick_positions = []

        # Config paths
        self.base_config = "configs/hanoi_base.json"
        self.biggest_config = "configs/hanoi_biggest.json"
        self.mid_config = "configs/hanoi_mid.json"
        self.small_config = "configs/hanoi_small.json"

        super().__init__(*args, **kwargs)
        self.query_query = "What is the size order of the yellow, red, and blue hanoi?"
        self.query_selection = {
            "A": "Yellow > Blue > Red",
            "B": "Red > Yellow > Blue",
            "C": "Blue > Yellow > Red",
        }
        self.query_answer = "B"

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Load the base (static)
        self.hanoi_base = self.load_from_config(
            self.base_config,
            "hanoi_base",
            body_type="static",
            scale_override=[0.08, 0.08, 0.08],
        )
        # Load the biggest disk (dynamic)
        self.hanoi_biggest = self.load_from_config(
            self.biggest_config, "hanoi_biggest", body_type="dynamic"
        )
        self.box = self.load_from_config("configs/box.json", "box", "static")
        # Load the medium disk (dynamic)
        self.hanoi_mid = self.load_from_config(
            self.mid_config, "hanoi_mid", body_type="dynamic"
        )

        # Load the small disk (dynamic)
        self.hanoi_small = self.load_from_config(
            self.small_config, "hanoi_small", body_type="dynamic"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode with positions for objects"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        table_height = self.table_height

        init_x = -0.25
        # Place the base at the center of the table
        if self.hanoi_base is not None:
            base_pos_x = init_x
            base_pos_y = 0.0
            base_pos_z = 0.11  # Slightly above the table to avoid z-fighting

            self.hanoi_base.set_pose(
                sapien.Pose(
                    p=[base_pos_x, base_pos_y, base_pos_z],
                    q=euler2quat(*np.deg2rad([90, 0, 90])),
                )
            )

            # Analyze the base to find stick positions
            # This is an approximation - in a real scenario, we would need to analyze the mesh
            # Here we assume the base has three sticks at fixed positions
            self.stick_positions = [
                [base_pos_x - 0.05, base_pos_y, base_pos_z + 0.05],  # Left stick
                [base_pos_x, base_pos_y, base_pos_z + 0.05],  # Middle stick
                [base_pos_x + 0.05, base_pos_y, base_pos_z + 0.05],  # Right stick
            ]

        # Randomize initial positions of the disks
        # For testing, we'll place them on the table at different positions
        if self.hanoi_biggest is not None:
            biggest_pos_x = init_x
            biggest_pos_y = 0
            biggest_pos_z = 0.15

            self.hanoi_biggest.set_pose(
                sapien.Pose(
                    p=[biggest_pos_x, biggest_pos_y, biggest_pos_z],
                    q=[0.707, 0.707, 0, 0],
                )
            )

        if self.hanoi_mid is not None:
            mid_pos_x = init_x
            mid_pos_y = 0
            mid_pos_z = 0.15

            self.hanoi_mid.set_pose(
                sapien.Pose(
                    p=[mid_pos_x, mid_pos_y, mid_pos_z], q=[0.707, 0.7070, 0, 0]
                )
            )

        if self.hanoi_small is not None:
            small_pos_x = init_x
            small_pos_y = 0.20
            small_pos_z = 0.11

            self.hanoi_small.set_pose(
                sapien.Pose(
                    p=[small_pos_x, small_pos_y, small_pos_z], q=[0.707, 0.707, 0, 0]
                )
            )

    def check_success(self):
        print(
            self.calculate_object_distance(self.hanoi_small, self.hanoi_mid),
            self.calculate_object_distance(self.hanoi_biggest, self.hanoi_mid),
        )
        if (
            self.calculate_object_distance(self.hanoi_small, self.hanoi_mid) <= 0.09
            and self.is_stable(self.hanoi_small)
            and self.calculate_object_distance(self.hanoi_biggest, self.hanoi_mid)
            <= 0.09
            and self.is_stable(self.hanoi_mid)
            # and self.is_static(self.hanoi_biggest)
        ):
            return True
            # print(
            #     self.hanoi_biggest.pose.p[:, 2],
            #     self.hanoi_mid.pose.p[:, 2],
            #     self.hanoi_small.pose.p[:, 2] + 0.02,
            # )
            # if (
            #     self.hanoi_biggest.pose.p[:, 2] <= self.hanoi_mid.pose.p[:, 2] + 0.02
            #     and self.hanoi_mid.pose.p[:, 2] <= self.hanoi_small.pose.p[:, 2] + 0.03
            # ):
            #     return True

        return False

    def _get_success(self, env_idx=None):
        """Check if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        # Initialize success tensor
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self.check_success():
            success = torch.ones_like(success)
        # # Check if the robot is static
        # robot_static = self.agent.is_static(0.2)
        #
        # # Set success if robot is static
        # if robot_static:
        #     success = torch.ones(len(env_idx), dtype=torch.bool, device=self.device)
        #
        return success
