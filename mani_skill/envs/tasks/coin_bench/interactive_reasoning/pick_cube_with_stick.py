from typing import Any, Dict, List, Optional, Tuple, Union
import os
from git.cmd import slots_to_dict
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


@register_env("Tabletop-Pick-Cube-WithStick-v1", max_episode_steps=5000)
class PickCubeWithStickEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to use a stick to move a small cube along a T-shaped path
    made of fixed cubes.

    **Randomizations:**
    - The small cube's position is randomized at the start of one path
    - The stick's position is randomized on the table

    **Success Conditions:**
    - The small cube is moved to the end of the T-shaped path
    - The robot is static (velocity < 0.2)
    """

    description = "Use the stick to move the small cube along the T-shaped path to the target position "
    workflow = [
        "pick the stick and put it on the entry ",
        "push the cube with the stick to the T point",
        "pick up the stick and put it to the space between the wall and the cube",
        "rotate the stick and push the cube to the marker",
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
            "obj": [Obj.GEOMETRY, Obj.ORIENT],
            "rob": [Robot.ACT_NAV],
            "iter": [Inter.PLAN, Inter.TOOL]
        }
        self.success_threshold = success_threshold

        # Set task description

        # Define T-path parameters
        self.path_cube_size = 0.04  # Size of each cube in the path
        self.path_cube_spacing = 0.001  # Small gap between path cubes
        self.path_length_horizontal = 7  # Number of cubes in horizontal path
        self.path_length_vertical = 4  # Number of cubes in vertical path

        # Initialize objects
        self.path_cubes = {}
        self.small_cube = None
        self.stick = None

        # Target position (end of T path)
        self.target_position = None

        super().__init__(*args, **kwargs)
        self.query_query = "Why we need a stick to move the cube?"
        self.query_selection = {
            "A": "Because the cube is too small and the path is too narrow",
            "B": "Because the cube is too heavy to move",
            "C": "Because the cube is too slippery",
        }
        self.query_answer = "A"

    def _create_cuboid(self, name, half_size, mass, color, body_type="dynamic"):
        """Create a cuboid object (path cube, small cube, or stick)"""
        # Create a builder for the cuboid
        builder = self.scene.create_actor_builder()

        # Add collision component
        builder.add_box_collision(half_size=half_size)

        # Add visual component with color
        try:
            # Try with material parameter (newer SAPIEN versions)
            material = sapien.render.RenderMaterial()
            material.set_base_color(color)
            builder.add_box_visual(half_size=half_size, material=material)
        except Exception as e:
            print(f"Warning: Could not set material: {e}")
            # Fallback with no material
            builder.add_box_visual(half_size=half_size)

        # Build the actor
        # cuboid = builder.build(name=name)
        cuboid = _build_by_type(builder, name, body_type=body_type)
        # Set mass and density
        # cuboid.set_mass(mass)

        return cuboid

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Create T-shaped path with fixed cubes
        self._create_t_path()

        # Create a small cube to be moved
        self.small_cube = self._create_cuboid(
            name="small_cube",
            half_size=[0.02, 0.02, 0.02],  # Smaller than path cubes
            mass=0.1,
            color=[0.2, 0.8, 0.2, 1.0],  # Green
        )

        # Create a stick to move the cube
        self.stick = self._create_cuboid(
            name="stick",
            half_size=[0.01, 0.15, 0.02],  # Long in Y direction
            mass=0.02,
            color=[0.8, 0.4, 0.0, 1.0],  # Orange
            # body_type='static'
        )

        self.target_area = self._create_goal_area(position=[0, 0.15, 0])

    def _create_t_path(self):
        """Create a T-shaped slot path with walls"""
        # Clear any existing path cubes
        self.path_cubes = {}

        # Define the dimensions of the T-slot
        slot_width = 0.06  # Width of the slot (slightly larger than small cube)
        wall_height = 0.08  # Height of the walls
        wall_thickness = 0.02  # Thickness of the walls

        # Calculate the total length of horizontal and vertical paths
        horizontal_length = self.path_length_horizontal * self.path_cube_size
        vertical_length = self.path_length_vertical * self.path_cube_size

        # Create the floor of the T-path (a single flat piece)
        # floor = self._create_cuboid(
        #     name="t_path_floor",
        #     half_size=[horizontal_length/2, vertical_length/2, 0.005],  # Thin floor
        #     mass=5.0,  # Heavy to stay in place
        #     color=[0.7, 0.7, 0.7, 1.0],  # Light gray
        #     body_type='static'  # Fixed in place
        # )
        # self.path_cubes['t_path_floor'] = floor

        # Create the left wall of the horizontal path
        left_h_wall = self._create_cuboid(
            name="left_h_wall",
            half_size=[horizontal_length / 2, wall_thickness / 2, wall_height / 2],
            mass=2.0,
            color=[0.5, 0.5, 0.5, 1.0],  # Gray
            body_type="static",
        )
        self.path_cubes["left_h_wall"] = left_h_wall

        # Create the right wall of the horizontal path
        right_h_wall_1 = self._create_cuboid(
            name="right_h_wall_1",
            half_size=[horizontal_length / 5, wall_thickness / 2, wall_height / 2],
            mass=2.0,
            color=[0.5, 0.5, 0.5, 1.0],  # Gray
            body_type="static",
        )
        self.path_cubes["right_h_wall_1"] = right_h_wall_1

        right_h_wall_2 = self._create_cuboid(
            name="right_h_wall_2",
            half_size=[horizontal_length / 5, wall_thickness / 2, wall_height / 2],
            mass=2.0,
            color=[0.5, 0.5, 0.5, 1.0],  # Gray
            body_type="static",
        )
        self.path_cubes["right_h_wall_2"] = right_h_wall_2

        # # Create the left wall of the vertical path
        left_v_wall = self._create_cuboid(
            name="left_v_wall",
            half_size=[wall_thickness / 2, vertical_length / 2, wall_height / 2],
            mass=2.0,
            color=[0.5, 0.5, 0.5, 1.0],  # Gray
            body_type="static",
        )
        self.path_cubes["left_v_wall"] = left_v_wall

        # # Create the right wall of the vertical path
        right_v_wall = self._create_cuboid(
            name="right_v_wall",
            half_size=[wall_thickness / 2, vertical_length / 2, wall_height / 2],
            mass=2.0,
            color=[0.5, 0.5, 0.5, 1.0],  # Gray
            body_type="static",
        )
        self.path_cubes["right_v_wall"] = right_v_wall
        # right_v_wall_1 = self._create_cuboid(
        #     name="right_v_wall_1",
        #     half_size=[horizontal_length/6, wall_thickness/2, wall_height/2],
        #     mass=2.0,
        #     color=[0.5, 0.5, 0.5, 1.0],  # Gray
        #     body_type='static'
        # )
        # self.path_cubes['right_v_wall_1'] = right_v_wall_1
        #
        # right_v_wall_2 = self._create_cuboid(
        #     name="right_v_wall_2",
        #     half_size=[horizontal_length/6, wall_thickness/2, wall_height/2],
        #     mass=2.0,
        #     color=[0.5, 0.5, 0.5, 1.0],  # Gray
        #     body_type='static'
        # )
        # self.path_cubes['right_v_wall_2'] = right_v_wall_2
        #

        # Create end caps for the horizontal path
        # left_end_cap = self._create_cuboid(
        #     name="left_end_cap",
        #     half_size=[wall_thickness/2, slot_width/2 + wall_thickness, wall_height/2],
        #     mass=1.0,
        #     color=[0.5, 0.5, 0.5, 1.0],  # Gray
        #     body_type='static'
        # )
        # self.path_cubes['left_end_cap'] = left_end_cap
        #
        right_end_cap = self._create_cuboid(
            name="right_end_cap",
            half_size=[
                wall_thickness / 2,
                slot_width / 2 + wall_thickness,
                wall_height / 2,
            ],
            mass=1.0,
            color=[0.5, 0.5, 0.5, 1.0],  # Gray
            body_type="static",
        )
        self.path_cubes["right_end_cap"] = right_end_cap
        #
        # # Create end cap for the vertical path (top)
        # top_end_cap = self._create_cuboid(
        #     name="top_end_cap",
        #     half_size=[slot_width/2 + wall_thickness, wall_thickness/2, wall_height/2],
        #     mass=1.0,
        #     color=[0.5, 0.5, 0.5, 1.0],  # Gray
        #     body_type='static'
        # )
        # self.path_cubes['top_end_cap'] = top_end_cap

        # Create a visual marker at the target position (end of vertical path)
        # target_marker = self._create_cuboid(
        #     name="target_marker",
        #     half_size=[slot_width/4, slot_width/4, 0.002],  # Very thin marker
        #     mass=0.1,
        #     color=[1.0, 0.0, 0.0, 1.0],  # Red
        #     body_type='static'
        # )
        # self.path_cubes['target_marker'] = target_marker

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode with positions for objects"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        table_height = self.table_scene.table_height

        # Define the dimensions of the T-slot
        slot_width = 0.04  # Width of the slot (slightly larger than small cube)
        wall_height = 0.08  # Height of the walls
        wall_thickness = 0.05  # Thickness of the walls

        # Calculate the total length of horizontal and vertical paths
        horizontal_length = self.path_length_horizontal * self.path_cube_size
        vertical_length = self.path_length_vertical * self.path_cube_size

        # Base position for the T-path (center of horizontal path)
        base_pos_x = 0.0
        base_pos_y = 0.0
        base_pos_z = 0.00  # Place directly on the table

        # Position the floor
        if "t_path_floor" in self.path_cubes:
            self.path_cubes["t_path_floor"].set_pose(
                sapien.Pose(
                    p=[base_pos_x, base_pos_y + vertical_length / 4, base_pos_z],
                    q=[1, 0, 0, 0],
                )
            )

        # Position the left wall of horizontal path
        if "left_h_wall" in self.path_cubes:
            self.path_cubes["left_h_wall"].set_pose(
                sapien.Pose(
                    p=[
                        base_pos_x,
                        base_pos_y - slot_width / 2 - wall_thickness / 2,
                        base_pos_z + wall_height / 2,
                    ],
                    q=[1, 0, 0, 0],
                )
            )

        # Position the right wall of horizontal path
        if "right_h_wall_1" in self.path_cubes:
            self.path_cubes["right_h_wall_1"].set_pose(
                sapien.Pose(
                    p=[
                        base_pos_x + slot_width * 2.4,
                        base_pos_y + slot_width / 2 + wall_thickness / 2,
                        base_pos_z + wall_height / 2,
                    ],
                    q=[1, 0, 0, 0],
                )
            )
        if "right_h_wall_2" in self.path_cubes:
            self.path_cubes["right_h_wall_2"].set_pose(
                sapien.Pose(
                    p=[
                        base_pos_x - slot_width * 2.4,
                        base_pos_y + slot_width / 2 + wall_thickness / 2,
                        base_pos_z + wall_height / 2,
                    ],
                    q=[1, 0, 0, 0],
                )
            )
        # Position the left wall of vertical path
        if "left_v_wall" in self.path_cubes:
            self.path_cubes["left_v_wall"].set_pose(
                sapien.Pose(
                    p=[
                        base_pos_x - slot_width / 2 - wall_thickness / 2,
                        base_pos_y + vertical_length / 1.5,
                        base_pos_z + wall_height / 2,
                    ],
                    q=[1, 0, 0, 0],
                )
            )

        # Position the right wall of vertical path
        if "right_v_wall" in self.path_cubes:
            self.path_cubes["right_v_wall"].set_pose(
                sapien.Pose(
                    p=[
                        base_pos_x + slot_width / 2 + wall_thickness / 2,
                        base_pos_y + vertical_length / 1.5,
                        base_pos_z + wall_height / 2,
                    ],
                    q=[1, 0, 0, 0],
                )
            )
        # if 'right_v_wall_2' in self.path_cubes:
        #     self.path_cubes['right_v_wall_2'].set_pose(sapien.Pose(
        #         p=[base_pos_x + slot_width/2 + wall_thickness/2, base_pos_y + vertical_length/2, base_pos_z + wall_height/2],
        #         q=[1, 0, 0, 0]
        #     ))
        #
        # Position the left end cap
        if "left_end_cap" in self.path_cubes:
            self.path_cubes["left_end_cap"].set_pose(
                sapien.Pose(
                    p=[
                        base_pos_x - horizontal_length / 2 - wall_thickness / 2,
                        base_pos_y,
                        base_pos_z + wall_height / 2,
                    ],
                    q=[1, 0, 0, 0],
                )
            )

        # Position the right end cap
        if "right_end_cap" in self.path_cubes:
            self.path_cubes["right_end_cap"].set_pose(
                sapien.Pose(
                    p=[
                        base_pos_x + horizontal_length / 2 + wall_thickness / 2,
                        base_pos_y,
                        base_pos_z + wall_height / 2,
                    ],
                    q=[1, 0, 0, 0],
                )
            )

        # Position the top end cap
        if "top_end_cap" in self.path_cubes:
            self.path_cubes["top_end_cap"].set_pose(
                sapien.Pose(
                    p=[
                        base_pos_x,
                        base_pos_y + vertical_length + wall_thickness / 2,
                        base_pos_z + wall_height / 2,
                    ],
                    q=[1, 0, 0, 0],
                )
            )

        # Position the target marker
        if "target_marker" in self.path_cubes:
            self.path_cubes["target_marker"].set_pose(
                sapien.Pose(
                    p=[
                        base_pos_x,
                        base_pos_y + vertical_length - slot_width / 2,
                        base_pos_z + 0.00,
                    ],
                    q=[1, 0, 0, 0],
                )
            )

        # Set the target position (end of vertical path)
        self.target_position = np.array(
            [
                base_pos_x,
                base_pos_y + vertical_length - slot_width / 2,
                base_pos_z + 0.01,  # Slightly above the floor
            ]
        )

        # Place the small cube at the start of the horizontal path (left end)
        if self.small_cube is not None:
            start_pos_x = base_pos_x - horizontal_length / 2 + slot_width / 2
            start_pos_y = base_pos_y
            start_pos_z = base_pos_z + 0.0  # Place on the floor of the path

            self.small_cube.set_pose(
                sapien.Pose(p=[start_pos_x, start_pos_y, start_pos_z], q=[1, 0, 0, 0])
            )

        # Place the stick on the side of the table
        if self.stick is not None:
            stick_pos_x = -0.2
            stick_pos_y = -0.1
            stick_pos_z = 0.02  # Place directly on the table

            # Random orientation for the stick
            stick_quat = euler2quat(0, 0, np.pi / 2).tolist()

            self.stick.set_pose(
                sapien.Pose(p=[stick_pos_x, stick_pos_y, stick_pos_z], q=stick_quat)
            )

    def _is_cube_on_path(self):
        """Check if the small cube is on the T-path"""
        if self.small_cube is None or not self.path_cubes:
            return False

        # Get small cube position
        small_cube_pos = np.array(self.small_cube.pose.p)

        # Check distance to each path cube
        for path_cube_name, path_cube in self.path_cubes.items():
            path_cube_pos = np.array(path_cube.pose.p)

            # Check horizontal distance (xy plane)
            horizontal_dist = np.linalg.norm(small_cube_pos[:2] - path_cube_pos[:2])

            # Check if small cube is above the path cube
            height_condition = (
                small_cube_pos[2] > path_cube_pos[2]
                and small_cube_pos[2] < path_cube_pos[2] + self.path_cube_size + 0.04
            )

            # If the cube is above any path cube, it's on the path
            if horizontal_dist < self.path_cube_size / 2 and height_condition:
                return True

        return False

    def _get_success(self, env_idx=None):
        """Check if the task is successful"""
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.calculate_object_distance(
            self.target_area, self.small_cube
        ) <= 0.05 and self.is_stable(self.small_cube):
            success = torch.ones_like(success)
        return success
