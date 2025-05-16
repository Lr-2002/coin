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


@register_env("Tabletop-Put-Balls-IntoContainer-v1", max_episode_steps=5000)
class PickPlaceBallsIntoContainerEnv(UniversalTabletopEnv):
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

    description = "move all the balls into the dustpan as fast as you can "
    workflow = [
        "pick the dustpan",
        "push all the ball in to the dustpan with the fixed body",
        "move the dustpand upward",
        "put all balls into the container",
    ]

    def __init__(
        self,
        *args,
        num_balls=5,  # Number of balls to place
        ball_radius=0.02,  # Radius of each ball
        ball_mass=0.01,  # Mass of each ball
        ball_friction=0.7,  # Friction of the balls
        **kwargs,
    ):
        self.num_balls = num_balls
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.MOVEABLE],
            "rob": [Robot.ACT_NAV],
            "iter": [Inter.TOOL, Inter.PLAN, Inter.FAIL_ADAPT],
        }
        self.ball_radius = ball_radius
        self.ball_mass = ball_mass
        self.ball_friction = ball_friction
        self.balls = []  # List to store ball actors
        self.balls_in_container = 0  # Counter for balls in container
        self.dustpan_config = "configs/dustpan.json"
        # Initialize the container as the main object
        #
        super().__init__(*args, **kwargs)
        self.query_query = (
            "What is the most efficient way to get all the balls into the dustpan?"
        )
        self.query_selection = {
            "A": "Pick up and place each ball individually into the dustpan",
            "B": "Push multiple balls at once with arm",
        }
        self.query_answer = "B"

    def _load_scene(self, options: dict):
        """Load the scene with table, container, and balls"""
        # Load the basic scene with table and container
        super()._load_scene(options)

        # Rename the main object to container for clarity
        self.container = self.load_from_config(
            self.dustpan_config, "dustpan", body_type="static"
        )

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
        table_height = self.table_scene.table_height

        self.container.set_pose(
            sapien.Pose(p=[0, 0.2, 0.01], q=euler2quat(*np.deg2rad([87, 0, 0])))
        )
        # Get container position and dimensions
        # container_pose = self.container.pose
        # container_aabb = self.container.get_aabb()
        # container_center = container_pose.p
        # container_size = container_aabb.max - container_aabb.min
        #
        # # Randomize ball positions on table (avoiding container area)
        table_range = 0.1  # Range for ball placement on table
        min_distance = 0.05  # Minimum distance between balls

        # Place balls randomly on the table
        ball_positions = []
        for i, ball in enumerate(self.balls):
            # Try to find a valid position for the ball
            max_attempts = 20
            for _ in range(max_attempts):
                # Random position on table
                pos_x = random.uniform(-table_range, table_range)
                pos_y = random.uniform(-table_range, table_range)
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

    def _get_obs_extra(self, info: Dict):
        """Get task-specific observations"""
        obs = super()._get_obs_extra(info)

        # # Add ball positions and velocities to observations
        # ball_positions = []
        # ball_velocities = []
        #
        # ball_pose = ball.get_pose()
        # for ball in self.balls:
        #     ball_vel = ball.get_velocity()
        #
        #     ball_positions.append(ball_pose.p)
        #     ball_velocities.append(ball_vel)
        #
        # # Flatten and add to observations
        # obs["ball_positions"] = np.array(ball_positions, dtype=np.float32).flatten()
        # obs["ball_velocities"] = np.array(ball_velocities, dtype=np.float32).flatten()
        #
        # # Add number of balls in container
        # obs["balls_in_container"] = np.array([self.balls_in_container], dtype=np.float32)
        #
        return obs

    def _count_balls_in_container(self):
        """Count how many balls are inside the container"""
        # container_pose = self.container.get_pose()
        # container_aabb = self.container.get_aabb()

        # Get container dimensions and position
        container_center = self.container.pose.p
        container_min = min(self.get_actor_size(self.container))
        container_max = max(self.get_actor_size(self.container))
        # print(container_min, container_max)
        # Count balls inside container
        count = 0
        # print(self.container, self.balls)
        for ball in self.balls:
            # ball_pose = ball.pose
            # ball_pos = ball_pose.p
            if max(self.calculate_obj_roi(ball, self.container)) > 0.7:
                count += 1
            # Check if ball is inside container bounds
            # if (ball_pos[0] >= container_min[0] and ball_pos[0] <= container_max[0] and
            #     ball_pos[1] >= container_min[1] and ball_pos[1] <= container_max[1] and
            #     ball_pos[2] >= container_min[2]):
            #     count += 1
            #
        return count

    def _get_success(self, env_idx=None):
        """Evaluate task success"""
        # Count balls in container
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self._count_balls_in_container() / len(self.balls) > 0.5:
            success = torch.ones_like(success)
        return {"success": success}
        #
        # self.balls_in_container = self._count_balls_in_container()
        #
        # # Success is based on number of balls in container (partial success possible)
        # success_rate = self.balls_in_container / self.num_balls
        #
        # Check if robot is static
        # robot_velocity = torch.norm(self.robot.get_velocity()).item()
        # is_static = robot_velocity < 0.2
        #
        # # Only consider success if robot is static
        # if not is_static:
        #     success_rate = 0.0
        #
        return {
            "success": torch.tensor(
                success_rate >= 0.5
            ),  # Success if at least half the balls are in
            # "success_rate": success_rate,
            # "balls_in_container": self.balls_in_container,
            # "total_balls": self.num_balls
        }
