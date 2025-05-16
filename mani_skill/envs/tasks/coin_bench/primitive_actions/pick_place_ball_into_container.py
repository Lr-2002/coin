from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat
import random
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


@register_env("Tabletop-Put-Ball-IntoContainer-v1", max_episode_steps=5000)
class PickPlaceBallIntoContainerEnv(UniversalTabletopEnv):
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

    description = "put the ball into the container "

    def __init__(
        self,
        *args,
        num_balls=1,  # Number of balls to place
        ball_radius=0.03,  # Radius of each ball
        ball_mass=0.1,  # Mass of each ball
        ball_friction=0.7,  # Friction of the balls
        container_config="configs/pen_holder.json",  # Container config
        **kwargs,
    ):
        self.num_balls = num_balls
        self.ball_radius = ball_radius
        self.ball_mass = ball_mass
        self.ball_friction = ball_friction
        self.balls = []  # List to store ball actors
        self.balls_in_container = 0  # Counter for balls in container
        self.container_path = container_config
        # Initialize the container as the main object
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        """Load the scene with table, container, and balls"""
        # Load the basic scene with table and container
        super()._load_scene(options)

        # Rename the main object to container for clarity
        self.container = self.load_from_config(
            self.container_path, "container", body_type="static"
        )
        print(dir(self.container))
        print(self.get_actor_size(self.container))
        # print(self.container.get_collision_meshes())
        # Create the balls

        self.balls = [self._create_ball() for _ in range(self.num_balls)]

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
            color = [1, 0, 0, 1.0]
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
        if self.num_balls == 1:
            ball_id = 1
        else: 
            ball_id = random.randint(0, 9999)
        ball = builder.build(name=f"ball_{ball_id}")
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
        table_height = self.table_scene.table_height

        # Get container position and dimensions
        container_pose = self.container.pose

        # Define container dimensions based on its scale and type
        # These values are approximations and might need adjustment based on the actual container
        container_half_size = np.array(
            [0.1, 0.1, 0.15]
        )  # Half-size of the container in x, y, z
        container_center = container_pose.p

        # Calculate container bounds
        container_min = container_center - container_half_size
        container_max = container_center + container_half_size
        container_size = container_max - container_min
        container_position = [0.1, -0.15, 0.0]
        # pen_holder_rotation = [
        #     1,
        #     0,
        #     0.0,
        #     0.0,
        # ]
        container_rotation = [
            0.707,
            0.707,
            0.0,
            0.0,
        ]  # w, x, y, z quaternion format
        self.container.set_pose(sapien.Pose(p=container_position, q=container_rotation))
        # Randomize ball positions on table (avoiding container area)
        table_range = 0.3  # Range for ball placement on table
        min_distance = 0.05  # Minimum distance between balls

        # Place balls randomly on the table
        ball_positions = []

        # Reset counter for balls in container
        self.balls_in_container = 0

    def _get_obs_extra(self, info: Dict):
        """Get task-specific observations"""
        obs = super()._get_obs_extra(info)

        # Add ball positions and velocities to observations
        ball_positions = []
        ball_velocities = []

        for ball in self.balls:
            ball_pose = ball.pose
            # ball_vel = ball.get_velocity()

            ball_positions.append(ball_pose.p)
            # ball_velocities.append(ball_vel)

        # Flatten and add to observations
        obs["ball_positions"] = np.array(ball_positions, dtype=np.float32).flatten()
        # obs["ball_velocities"] = np.array(ball_velocities, dtype=np.float31).flatten()

        # Add number of balls in container
        obs["balls_in_container"] = np.array(
            [self.balls_in_container], dtype=np.float32
        )

        return obs

    # def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
    #     """Compute dense reward for the task"""
    #     # Count balls in container
    #     self.balls_in_container = self._count_balls_in_container()
    #
    #     # Base reward is the number of balls in container
    #     reward = self.balls_in_container / self.num_balls
    #
    #     # Add small penalty for large actions to encourage smoother motion
    #     action_penalty = 0.01 * torch.norm(action).item()
    #
    #     return reward - action_penalty

    # def (self):
    #     """Evaluate task success"""
    #     # Count balls in container
    #     self.balls_in_container = self._count_balls_in_container()
    #
    #     # Success is based on number of balls in container (partial success possible)
    #     success_rate = self.balls_in_container / self.num_balls
    #
    #     # Check if robot is static
    #     # robot_velocity = torch.norm(self.robot.get_velocity()).item()
    #     # is_static = robot_velocity < 0.2
    #
    #     # Only consider success if robot is static
    #     # if not is_static:
    #     #     success_rate = 0.0
    #
    #     return {
    #         "success": torch.tensor(success_rate == 1),  # Success if at least half the balls are in
    #         # "success_rate": success_rate,
    #         "balls_in_container": self.balls_in_container,
    #         "total_balls": self.num_balls

    #     }
    #
    def _get_success(self, env_idx=None):
        success = super()._get_success(env_idx)
        sr = 0
        # if self.debug:
        #     self.show_two_objects(self.balls[0], self.container)
        for ball in self.balls:
            if max(self.calculate_obj_roi(ball, self.container)) > 0.8:
                sr += 1
        sr /= len(self.balls)
        if sr >= 0.7:
            success = torch.ones_like(success)
        return success
