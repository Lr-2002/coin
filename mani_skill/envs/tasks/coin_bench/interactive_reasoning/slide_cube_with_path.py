from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import table
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench import UniversalTabletopEnv


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Slide-Cube-WithPath-v1", max_episode_steps=5000)
class SlideCubeWithPathEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to slide a cube down a slope into a container.
    The slope is positioned on the table, with a container at the bottom to catch the cube.

    **Randomizations:**
    - The initial position of the cube on the slope

    **Success Conditions:**
    - The cube is inside the container
    - The robot is static (velocity < 0.2)
    """

    description = "Slide a cube down a slope to the marker "
    workflow = ["pull the cube target at the marker", "push the cube down the slope"]

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
            "obj": [Obj.FRICTION, Obj.ORIENT, Obj.GEOMETRY],
            "rob": [Robot.ACT_NAV],
            "iter": [Inter.PLAN]
        }
        self.success_threshold = success_threshold

        # Set task description

        # Initialize objects
        self.slope = None
        self.container = None
        self.cube = None

        # Config paths
        self.slope_config = "configs/slope.json"
        self.container_config = "configs/dustpan.json"
        self.cube_config = "configs/cube.json"

        self.plat_config = "configs/box.json"
        # Container position to check if cube is inside
        self.container_position = None
        self.container_size = None

        super().__init__(*args, **kwargs)
        self.query_query = (
            "Where should the block be placed to slide to target position?"
        )
        self.query_selection = {"A": "A", "B": "B"}
        self.query_answer = "B"
        # Did it pick up or push the cube?
        # Did the cube slide to the target position?

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Load the slope (static)
        self.slope = self.load_from_config(
            self.slope_config,
            "slope",
            body_type="static",
            friction_override=0,
            scale_override=0.3,
        )

        # Load the container (static)
        self.container = self.load_from_config(
            self.container_config, "container", body_type="static"
        )
        self.target_area = self._create_goal_area(
            sapien.Pose(p=[0.23, -0.07, 0], q=euler2quat(0, np.pi / 2, 0))
        )
        # Load the cube (dynamic)
        # self.cube = self.load_from_config(self.cube_config, "cube", body_type="dynamic")

        self.cube = self._create_default_object(
            "configxx.", name="cube", body_type="dynamic"
        )
        self.plat = self.load_from_config(
            self.plat_config, "plat", body_type="static", scale_override=[0.5, 0.4, 0.4]
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode with positions for objects"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        # table_height = self.table_scene.table_height
        table_height = self.table_height
        # Place the slope on the table
        if self.slope is not None:
            # Position the slope at an angle on the table
            slope_pos_x = 0.0
            slope_pos_y = -0.0
            slope_pos_z = table_height + 0.01  # Slightly above the table

            # Create rotation for the slope (angled)
            # Convert to list to avoid type error with numpy array
            slope_rotation = list(
                euler2quat(np.pi / 2, degree2rad(0), np.pi * 0)
            )  # 15-degree incline

            self.slope.set_pose(
                sapien.Pose(p=[slope_pos_x, slope_pos_y, slope_pos_z], q=slope_rotation)
            )

        # Place the container at the bottom of the slope
        if self.container is not None:
            container_pos_x = 0.35  # In front of the slope
            container_pos_y = 0.1
            container_pos_z = table_height + 0.01

            self.container.set_pose(
                sapien.Pose(
                    p=[container_pos_x, container_pos_y, container_pos_z],
                    q=euler2quat(np.pi / 2, 0, -np.pi / 2),
                )
            )

            # Store container position and approximate size for success checking
            self.container_position = np.array(
                [container_pos_x, container_pos_y, container_pos_z]
            )
            self.container_size = np.array(
                [0.1, 0.1, 0.05]
            )  # Approximate container dimensions

        # Place the cube on the slope
        if self.cube is not None:
            # Position the cube at the top of the slope
            cube_pos_x = -0.15  # At the top of the slope
            cube_pos_y = 0  # Slight randomization
            cube_pos_z = table_height + 0.15  # Above the slope

            self.cube.set_pose(
                sapien.Pose(p=[cube_pos_x, cube_pos_y, cube_pos_z], q=[1, 0, 0, 0])
            )

        self.plat.set_pose(sapien.Pose(p=(-0.2, 0.0, 0.02), q=(0.707, 0, 0, 0.707)))

    # def _is_cube_in_container(self):
    #     """Check if the cube is inside the container"""
    #     roi = max(self.calculate_obj_roi(self.container, self.cube))
    #     print(roi)
    #     if roi >= 0.64:
    #         is_inside = True
    #     else:
    #         is_inside = False
    #     return is_inside

    def _get_success(self, env_idx=None):
        """Check if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        # Initialize success tensor
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)

        # Check if the cube is in the container
        # cube_in_container = self._is_cube_in_container()

        # Check if the robot is static
        robot_static = self.agent.is_static(0.2)
        # Set success if both conditions are met
        if (
            self.calculate_object_distance(self.cube, self.target_area) <= 0.07
            and robot_static
        ):
            success = torch.ones(len(env_idx), dtype=torch.bool, device=self.device)

        return success
