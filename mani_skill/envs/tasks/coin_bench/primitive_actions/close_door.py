from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import articulations
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
    load_articulation_from_json,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import table
from mani_skill.utils.structs import Articulation, Link, Pose
import os


@register_env("Tabletop-Close-Door-v1", max_episode_steps=5000)
class CloseDoorEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A tabletop environment with a door that needs to be opened.

    **Randomizations:**
    - The door's position on the table is fixed at (0.3, 0.0, 0.0)

    **Success Conditions:**
    - The door has been opened (joint position > 90% of its limit)
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda

    description = "close the door"

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        door_scale=0.3,  # Scale factor for the door
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.door_scale = door_scale
        # self.door_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "configs", "door_8867.json")
        self.door_config_path = "configs/door_8867.json"
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    def _load_scene(self, options=None):
        """Load the door model into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)

        # Load the door model
        self.door = self._load_door()

    def _load_door(self):
        """Load the door model from URDF using a JSON configuration file"""
        # Load from JSON configuration
        door = load_articulation_from_json(
            scene=self.scene,
            json_path=self.door_config_path,
            json_type="urdf",
            scale_override=self.door_scale,
            position_override=[0.25, 0.0, 0.28],
            prefix_function=self.update_prefix,
        )

        # Set the door position
        # position = torch.tensor([0.3, 0.0, 0.0], device=self.device)
        # door.set_root_pose(Pose(position))

        return door

    def _initialize_episode(self, env_idx, options=None):
        """Initialize the episode by setting up the door and robot"""
        # Call the parent method to initialize the base scene
        super()._initialize_episode(env_idx, options)

        # Initialize all door joints to position 0
        for joint in self.door.get_active_joints():
            joint.set_drive_target(1)

    # def compute_dense_reward(self, obs, action, info):
    #     """Compute the reward for the task"""
    #     # Get the current joint positions
    #     rewards = torch.zeros(self.num_envs, device=self.device)
    #
    #     # Get the active joints
    #     active_joints = self.door.get_active_joints()
    #     if len(active_joints) == 0:
    #         return rewards
    #
    #     # Calculate rewards for all joints
    #     joint_rewards = []
    #     for joint in active_joints:
    #         # Skip joint_2 as in the evaluate function
    #         if joint.name == "joint_2":
    #             continue
    #
    #         # Get the actual current position
    #         current_pos = joint.qpos.item()
    #
    #         # Get the joint limits
    #         limits = joint.get_limits()
    #         if limits is None or limits.shape[0] == 0:
    #             continue
    #
    #         # Calculate the normalized position (0 = closed, 1 = open)
    #         min_pos, max_pos = limits[0][0], limits[0][1]
    #         range_pos = max_pos - min_pos
    #         if range_pos < 1e-6:  # Avoid division by zero
    #             continue
    #
    #         normalized_pos = (current_pos - min_pos) / range_pos
    #
    #         # Reward is higher when the door is more closed (normalized_pos closer to 0)
    #         # Using an inverse reward function where smaller values are better
    #         joint_reward = (
    #             1.0 - normalized_pos
    #         )  # Linear reward based on how closed the door is
    #
    #         # Add a bonus if the door is almost closed
    #         if normalized_pos < 0.3:  # Within 30% of closed position
    #             joint_reward += 0.5
    #
    #         # Add an even bigger bonus if the door is fully closed
    #         if (
    #             normalized_pos < 0.08
    #         ):  # Within 8% of fully closed position (matching success condition)
    #             joint_reward += 1.0
    #
    #         joint_rewards.append(joint_reward)
    #
    #     # Average the rewards across all joints
    #     if joint_rewards:
    #         rewards[0] = sum(joint_rewards) / len(joint_rewards)
    #
    #         # Print debugging info when in human render mode
    #         if self.render_mode == "human":
    #             print(
    #                 f"Joint positions: {[joint.qpos.item() for joint in active_joints if joint.name != 'joint_2']}"
    #             )
    #             print(f"Reward: {rewards[0].item()}")
    #
    #     return rewards
    #
    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        success = super()._get_success(env_idx)
        # Get the active joints
        active_joints = self.door.get_active_joints()
        if len(active_joints) == 0:
            return {"success": success}

        # Check each joint
        for joint in active_joints:
            # Get the actual current position
            if joint.name == "joint_2":
                continue
            current_pos = joint.qpos.item()
            # print(joint.name)
            # Get the joint limits
            limits = joint.get_limits()
            if limits is None or limits.shape[0] == 0:
                continue

            # Calculate the normalized position (0 = closed, 1 = open)
            min_pos, max_pos = limits[0][0], limits[0][1]
            range_pos = max_pos - min_pos
            if range_pos < 1e-6:  # Avoid division by zero
                continue

            normalized_pos = (current_pos - min_pos) / range_pos
            # Success if the door is open (position > 90% of its limit)

            # print("normalized_pos", normalized_pos)
            if normalized_pos < 0.08:
                success[0] = True
                break
        # print("success", success)
        return {
            "success": success,
        }
