from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat

from mani_skill.envs import sapien_env
import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench import UniversalTabletopEnv
import numpy as np


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Pick-Pen-v1", max_episode_steps=5000)
class PickPenEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to pick up an object and place it at a target location.

    **Randomizations:**
    - The object's position is randomized on the table
    - The object's orientation is randomized around the z-axis
    - The target location is randomized on the table

    **Success Conditions:**
    - The object is placed at the target location (within a threshold)
    - The robot is static (velocity < 0.2)
    """
    description = "pick up the pen and put it to the marker"
    def __init__(
        self,
        *args,
        object_path=None,  # Path to the object GLB/USD file
        object_scale=0.01,  # Scale of the object
        object_mass=0.5,  # Mass of the object in kg
        object_friction=1.0,  # Friction coefficient of the object
        placement_threshold=0.05,  # Distance threshold for successful placement
        orientation=[0, 0, 0],
        object_config="configs/pen.json",
        **kwargs,
    ):
        config = {}

        self.placement_threshold = placement_threshold
        self.object_config = object_config
        # Initialize target marker properties
        self.target_size = 0.01
        self.target_position = None

        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Load the object to be manipulated
        # if self.object_path and os.path.exists(self.object_path):
        #     self.object = self.load_asset(
        #         asset_path=self.object_path,
        #         scale=self.object_scale,
        #         mass=self.object_mass,
        #         friction=self.object_friction,
        #     )
        # else:
        #     # Create a default cube if no object is specified
        #     self.object = self._create_default_object()
        self.pen = self.load_from_config(self.object_config, "pen", convex=True)

        # Create target marker
        self.goal_region = self._create_goal_area()

    def _create_default_object(self):
        """Create a default object (red cube) if no object path is provided."""
        builder = self.scene.create_actor_builder()

        # Add collision component
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])

        # Add visual component (red cube)
        try:
            # Try with material parameter (newer SAPIEN versions)
            material = sapien.render.RenderMaterial()
            material.set_base_color([1, 0, 0, 1])  # Red color with alpha=1
            builder.add_box_visual(half_size=[0.02, 0.02, 0.02], material=material)
        except TypeError:
            # Fallback for older SAPIEN versions
            try:
                # Try with color parameter
                builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1, 0, 0, 1])
            except TypeError:
                # Fallback with no color
                builder.add_box_visual(half_size=[0.02, 0.02, 0.02])

        # Create the actor
        actor = builder.build(name="default_object")

        # Set physical properties
        try:
            actor.set_damping(linear=0.5, angular=0.5)
        except Exception as e:
            print(f"Warning: Could not set some physical properties: {e}")

        # Set friction
        try:
            for collision_shape in actor.get_collision_shapes():
                collision_shape.set_friction(self.object_friction)
        except Exception as e:
            print(f"Warning: Could not set friction: {e}")

        return actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized object and target positions"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        table_height = self.table_scene.table_height

        # Randomize object position on table
        xy_range = 0.1
        object_xy = torch.rand(self.num_envs, 2) * xy_range * 2 - xy_range
        object_z = 0.005  # Place slightly above table

        # Randomize object orientation (only around z-axis)
        # object_ori = torch.zeros(self.num_envs, 3)
        # object_ori[:, 0] = 0.5 * np.pi
        # object_ori = (
        #     torch.tensor(self.object_orientation).unsqueeze(0).expand(self.num_envs, -1)
        # )
        # # Set object pose
        # object_pose = sapien.Pose(
        #     p=[object_xy[env_idx, 0].item(), object_xy[env_idx, 1].item(), object_z],
        #     q=euler2quat(
        #         object_ori[env_idx, 0].item(),
        #         object_ori[env_idx, 1].item(),
        #         object_ori[env_idx, 2].item(),
        #     ),
        # )
        self.pen.set_pose(sapien.Pose(p=[0.1, 0.1, 0]))
        #
        # Randomize target position (different from object position)
        while True:
            target_xy = torch.rand(self.num_envs, 2) * xy_range * 2 - xy_range
            # Ensure target is not too close to object
            if torch.norm(target_xy[env_idx] - object_xy[env_idx]) > 0.1:
                break

        target_z = 0.000  # Place just above table

        # Set target marker pose
        target_pose = sapien.Pose(
            p=[target_xy[env_idx, 0].item(), target_xy[env_idx, 1].item(), target_z],
            q=[1, 0, 0, 0],
        )

    # def _get_success(self, env_idx=None):
    #     """Evaluate task success"""
    #     # Check if object is grasped
    #     is_grasped = self.agent.is_grasping(self.object)
    #
    #     # Check if robot is static
    #     is_robot_static = self.agent.is_static(0.2)
    #
    #     # Check if object is at target location
    #     object_xy = self.object.pose.p[:, :2]
    #     target_xy = self.target_position[:2]
    #
    #     # Calculate distance to target
    #     distance_to_target = torch.norm(object_xy - target_xy, dim=1)
    #
    #     # Object is at target if distance is below threshold
    #     is_at_target = distance_to_target < self.placement_threshold
    #
    #     # Success if object is at target and robot is static
    #     success = is_at_target & is_robot_static
    #
    #     return {
    #         "success": success,
    #         "is_at_target": is_at_target,
    #         "is_robot_static": is_robot_static,
    #         "is_grasped": is_grasped,
    #         "distance_to_target": distance_to_target,
    #     }

    # def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
    #     """Compute dense reward for the task"""
    #     # Distance from gripper to object
    #     tcp_to_obj_dist = torch.linalg.norm(
    #         self.object.pose.p - self.agent.tcp.pose.p, axis=1
    #     )
    #
    #     # Reward for reaching the object
    #     reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
    #     reward = reaching_reward
    #
    #     # Reward for grasping the object
    #     is_grasped = info["is_grasped"]
    #     reward += is_grasped
    #
    #     # Distance from object to target
    #     obj_to_target_dist = torch.linalg.norm(
    #         self.object.pose.p[:, :2] - self.target_position[:2], axis=1
    #     )
    #
    #     # Reward for moving object toward target (only when grasped)
    #     placement_reward = 1 - torch.tanh(5 * obj_to_target_dist)
    #     reward += placement_reward * is_grasped
    #
    #     # Reward for being static when at target
    #     static_reward = 1 - torch.tanh(
    #         5 * torch.linalg.norm(self.agent.robot.get_qvel(), axis=1)
    #     )
    #     reward += static_reward * info["is_at_target"]
    #
    #     # Bonus reward for success
    #     reward[info["success"]] = 10
    #
    #     return reward
    #
    def _get_success(self, env_idx=None):
        success = super()._get_success(env_idx)
        if self.calculate_object_distance(
            self.goal_region, self.pen
        ) <= 0.05 and self.is_static(self.pen):
            success = torch.ones_like(success)
        return success
