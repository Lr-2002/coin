from typing import Any, Dict, List, Optional, Tuple, Union
import os
from cv2 import borderInterpolate
import numpy as np
import sapien
import torch
import json
from transforms3d import euler
from transforms3d.euler import euler2quat
import random
from mani_skill.envs import sapien_env
import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
import numpy as np
from mani_skill.envs.tasks.coin_bench.primitive_actions.pick_place import PickPlaceEnv


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Move-Balls-WithDustpan-v1", max_episode_steps=5000)
class MoveBallsWithDustpanEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to place as many small balls as possible into a container.

    **Randomizations:**
    - The container position is randomized on the table
    - The balls are randomly placed on the table

    **Success Conditions:**
    - Success is measured by how many balls are placed in the container
    - The robot is static (velocity < 0.2)
    """

    description = "move all the balls into the holder with dustpan"
    workflow = [
        "pick the dustpan",
        "push all the ball into the dustpan with the fixed holder",
        "move all the ball into the container with the dustpan",
    ]

    def __init__(
        self,
        *args,
        num_balls=5,  # Number of balls to place
        ball_radius=0.01,  # Radius of each ball
        ball_mass=0.001,  # Mass of each ball
        ball_friction=0.7,  # Friction of the balls
        **kwargs,
    ):
        self.num_balls = num_balls
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.MASS, Obj.SCALE, Obj.GEOMETRY],
            "rob": [],
            "iter": [Inter.TOOL, Inter.PLAN, Inter.FAIL_ADAPT],
        }
        self.ball_radius = ball_radius
        self.ball_mass = ball_mass
        self.ball_friction = ball_friction
        self.balls = []  # List to store ball actors
        self.balls_in_container = 0  # Counter for balls in container
        self.dustpan_config = "configs/dustpan.json"
        self.plat_config = "configs/cube__hole_12_12_12__r9.json"
        # Initialize the container as the main object
        #
        super().__init__(*args, **kwargs)
        self.query_query = (
            "What strategy should be used to collect multiple balls efficiently?"
        )
        self.query_selection = {
            "A": "Collect all balls at once with a single sweeping motion",
            "B": "Move the balls one by one",
        }
        self.query_answer = "A"

    def _load_scene(self, options: dict):
        """Load the scene with table, container, and balls"""
        # Load the basic scene with table and container
        super()._load_scene(options)

        # Rename the main object to container for clarity
        self.container = self.load_from_config(
            self.dustpan_config, "dustpan", body_type="dynamic"
        )

        # Create the balls
        self.balls = [self._create_ball() for _ in range(self.num_balls)]
        self.plat = self.load_from_config(
            self.plat_config, "plat", body_type="static", scale_override=0.002
        )

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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized container and ball positions"""
        # Initialize the table scene, robot, and container
        super()._initialize_episode(env_idx, options)

        # Get table height

        self.container.set_pose(
            sapien.Pose(p=[0, 0.1, 0.0], q=euler2quat(*np.deg2rad([90, 0, 0])))
        )
        self.plat.set_pose(
            sapien.Pose(p=[0.0, -0.2, 0], q=euler2quat(-np.pi / 2, 0, 0))
        )
        # Get container position and dimensions
        # container_pose = self.container.pose
        # container_aabb = self.container.get_aabb()
        # container_center = container_pose.p
        # container_size = container_aabb.max - container_aabb.min
        #
        # # Randomize ball positions on table (avoiding container area)
        table_range = 0.08  # Range for ball placement on table
        y_offset = 0.05
        min_distance = 0.04  # Minimum distance between balls

        # Place balls randomly on the table
        ball_positions = []
        for i, ball in enumerate(self.balls):
            # Try to find a valid position for the ball
            max_attempts = 20
            for _ in range(max_attempts):
                # Random position on table
                pos_x = random.uniform(-table_range, table_range)
                pos_y = random.uniform(-table_range, table_range) + y_offset
                pos_z = (
                    self.table_height + self.ball_radius + 0.01
                )  # Slightly above table

                # # Check if too close to container
                # dist_to_container = np.sqrt(
                #     (pos_x - container_center[0])**2 +
                #     (pos_y - container_center[1])**2
                # )
                # if dist_to_container < max(container_size[0], container_size[1]) + self.ball_radius:
                #     continue
                #
                # Check if too close to other balls
                too_close = False
                for other_pos in ball_positions:
                    dist = np.sqrt(
                        (pos_x - other_pos[0]) ** 2 + (pos_y - other_pos[1]) ** 2
                    )
                    if dist < min_distance:
                        too_close = True
                        break

                if not too_close:
                    # Valid position found
                    ball_positions.append([pos_x, pos_y, pos_z])
                    break

            # Set ball pose
            ball_pose = sapien.Pose(
                p=[pos_x, pos_y, pos_z],
                q=[1, 0, 0, 0],  # Identity quaternion
            )
            ball.set_pose(ball_pose)

            # Reset velocity
            # ball.set_velocity([0, 0, 0])
            # ball.set_angular_velocity([0, 0, 0])

        # Reset counter for balls in container
        self.balls_in_container = 0

    def _count_balls_in_container(self):
        """Count how many balls are inside the container"""
        count = 0
        for ball in self.balls:
            print(max(self.calculate_obj_roi(self.plat, ball)))
            if max(self.calculate_obj_roi(self.plat, ball)) >= 0.7 and self.is_stable(
                ball
            ):
                count += 1
        if count / len(self.balls) >= 0.8:
            return True
        else:
            return False

    def _get_success(self, env_idx=None):
        """Evaluate task success"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self._count_balls_in_container():
            success = torch.ones_like(success)
        return success
