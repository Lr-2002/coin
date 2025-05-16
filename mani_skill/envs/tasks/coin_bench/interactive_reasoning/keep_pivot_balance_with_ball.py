from typing import Any, Dict, List, Optional, Tuple, Union
import random
import os
import numpy as np
import sapien
import torch
from torch.serialization import SourceChangeWarning
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench import UniversalTabletopEnv


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Balance-Pivot-WithBalls-v1", max_episode_steps=5000)
class BalancePivotWithBallsEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to balance a long board on a triangular prism (pivot)
    and then place two cubes on the board to maintain balance.

    **Randomizations:**
    - The triangular prism's position is randomized on the table
    - The long board's position is randomized on the table
    - The two cubes' positions are randomized on the table

    **Success Conditions:**
    - The long board is balanced on the triangular prism
    - The two cubes are placed on the board in a way that maintains balance
    - The system remains stable for a certain period of time
    """

    description = "Put the balls in to the holder to balance the long board on the triangular prism"
    workflow = [
        "pick and put one ball in the holder and see whether balance",
        "pick and put one ball in the holder and see whether balance",
        "pick and put one ball in the holder and see whether balance",
        "pick and put one ball in the holder and see whether balance",
        "pick and put one ball in the holder and see whether balance",
    ]

    def __init__(
        self,
        *args,
        balance_time_threshold=30,  # Number of steps the balance must be maintained
        balance_angle_threshold=0.1,  # Maximum angle deviation for the board to be considered balanced
        **kwargs,
    ):
        # Set balance thresholds
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.MASS, Obj.SCALE],
            "rob": [],
            "iter": [Inter.TOOL,  Inter.FAIL_ADAPT, Inter.PLAN]
        }
        self.balance_time_threshold = balance_time_threshold
        self.balance_angle_threshold = balance_angle_threshold
        self.balance_counter = 0
        # Set task description
        self.ball_radius = 0.03
        self.ball_mass = 0.08
        self.ball_num = 6
        self.ball_friction = 0.7
        super().__init__(*args, **kwargs)

        self.query_query = "How many balls can make the pivot balance?"
        self.query_selection = {"A": "4", "B": "5", "C": "6"}
        self.query_answer = "B"

    def _create_ball(self):
        """Create a small ball with the specified properties"""
        builder = self.scene.create_actor_builder()

        # Add collision component (sphere)
        builder.add_sphere_collision(radius=self.ball_radius)

        # Add visual component (colored sphere)
        try:
            # Try with material parameter (newer SAPIEN versions)
            material = sapien.render.RenderMaterial()
            # Random color for each ball
            color = [random.random(), random.random(), random.random(), 1.0]
            material.set_base_color(color)
            builder.add_sphere_visual(radius=self.ball_radius, material=material)
        except TypeError:
            # Fallback for older SAPIEN versions
            try:
                # Try with color parameter
                color = [random.random(), random.random(), random.random(), 1.0]
                builder.add_sphere_visual(radius=self.ball_radius, color=color)
            except TypeError:
                # Fallback with no color
                builder.add_sphere_visual(radius=self.ball_radius)

        # Create the actor
        ball = builder.build(name=f"ball_{random.randint(0, 9999)}")

        # Set physical properties
        try:
            ball.set_mass(self.ball_mass)
            ball.set_damping(linear=0.5, angular=0.5)
        except Exception as e:
            print(f"Warning: Could not set some physical properties: {e}")

        # Set friction
        try:
            for collision_shape in ball.get_collision_shapes():
                collision_shape.set_friction(self.ball_friction)
        except Exception as e:
            print(f"Warning: Could not set friction: {e}")

        return ball

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Create a triangular prism (pivot)
        # self.triangular_prism = self._create_triangular_prism(
        #     name="triangular_prism",
        #     size=[0.05, 0.15, 0.05],  # [width, length, height]
        #     mass=0.5,
        #     color=[0.8, 0.2, 0.2, 1.0]  # Red
        # )
        self.triangular_prism = self.load_from_config(
            "configs/plugin_triangle.json", "pivot_base", "static"
        )
        # Create a long board
        self.long_board = self._create_cuboid(
            name="long_board",
            half_size=[0.06, 0.3, 0.01],  # [half_width, half_length, half_height]
            mass=0.3,
            color=[0.2, 0.6, 0.8, 1.0],  # Blue
        )

        # Create two cubes with different colors
        self.cube1 = self._create_cuboid(
            name="cube1",
            half_size=[0.03, 0.03, 0.03],  # [half_width, half_length, half_height]
            mass=0.4,
            color=[0.2, 0.8, 0.2, 1.0],  # Green
        )
        self.holder = self.load_from_config(
            "configs/pen_holder.json", "holder", "dynamic"
        )
        self.balls = [self._create_ball() for i in range(self.ball_num)]

    def _create_cuboid(self, name, half_size, mass, color, is_static=False):
        """Create a cuboid object (board or cube)"""
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
        if not is_static:
            cuboid = builder.build(name=name)
        else:
            cuboid = builder.build_kinematic(name=name)

        # Set mass
        cuboid.set_mass(mass)

        return cuboid

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode with random positions for objects"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Reset balance counter
        self.balance_counter = 0

        # Get table height
        table_height = 0

        # Random orientation for the prism (but keep it upright)
        prism_quat = euler2quat(0, 0, self.np_random.uniform(0, 2 * np.pi)).tolist()

        self.triangular_prism.set_pose(
            sapien.Pose(p=(0.155, 0.0, -0.06), q=euler2quat(*np.deg2rad([0, 180, 90])))
        )
        # Place the long board nearby
        board_pos_x = -0.0
        board_pos_y = 0
        board_pos_z = table_height + 0.05
        board_pos = np.array([board_pos_x, board_pos_y, board_pos_z])

        # Random orientation for the board
        board_quat = euler2quat(0, 0, 0).tolist()

        if self.long_board is not None:
            self.long_board.set_pose(sapien.Pose(p=board_pos, q=board_quat))

        # Place cube1 on the left side of the table
        cube1_pos_x = 0
        cube1_pos_y = -0.2
        cube1_pos_z = 0.05
        cube1_pos = np.array([cube1_pos_x, cube1_pos_y, cube1_pos_z])

        # Random orientation for cube1
        cube1_quat = euler2quat(0, 0, self.np_random.uniform(0, 2 * np.pi)).tolist()

        if self.cube1 is not None:
            self.cube1.set_pose(sapien.Pose(p=cube1_pos, q=cube1_quat))

        self.holder.set_pose(sapien.Pose(p=[0.0, 0.2, 0.05], q=(0.707, 0.7070, 0, 0)))
        for i, ball in enumerate(self.balls):
            ball.set_pose(sapien.Pose(p=[-0.2, -0.1 + i * 0.05, 0]))

    def update_balance(self):
        if self.compare_angle(
            self.long_board, [0, 0, 0], specific_axis="roll"
        ) and self.is_stable(self.long_board):
            self.balance_counter += 1
        else:
            self.balance_counter += 0
        return self.balance_counter >= self.balance_time_threshold
        self.balance_counter

    def _get_success(self, env_idx=None):
        """Check if the task is successful"""
        success = super()._get_success(env_idx)
        bal = self.update_balance()
        if bal:
            success = torch.ones_like(success)
        return success
