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
    load_articulation_from_urdf,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import table
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.io_utils import load_json
import os


@register_env(
    "Tabletop-Close-Cabinet-v1",
    max_episode_steps=5000,
    asset_download_ids=["partnet_mobility_cabinet"],
)
class CloseCabinetEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A tabletop environment with an open cabinet from PartNet Mobility placed on the table.
    The task is to close the cabinet door or drawer.

    **Randomizations:**
    - The cabinet model is randomly sampled from all available PartNet Mobility cabinet models
    - The cabinet's position on the table is randomized

    **Success Conditions:**
    - The cabinet door/drawer is closed (joint position is near its minimum limit)
    """

    description = "close the cabinet door"
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        cabinet_scale=0.4,  # Scale factor for the cabinet (cabinets can be large)
        cabinet_config_path="configs/drawer_cabinet.json",  # Path to a JSON configuration file for the cabinet
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cabinet_scale = cabinet_scale
        self.cabinet_config_path = cabinet_config_path

        # Load cabinet model IDs
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    def _load_scene(self, options=None):
        """Load the cabinet model into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)

        # Sample a random cabinet model for each environment
        b = self.num_envs
        model_indices = torch.randint(0, len(self.all_model_ids), (b,))
        self.cabinet_ids = [self.all_model_ids[i.item()] for i in model_indices]

        # Load the cabinet models
        self.cabinets = []
        for i in range(b):
            cabinet = self._load_cabinet(self.cabinet_ids[i])
            self.cabinets.append(cabinet)

    def _load_cabinet(self, cabinet_id):
        """Load a cabinet model from PartNet Mobility dataset"""
        # If a cabinet configuration file is provided, use it
        if self.cabinet_config_path:
            # Load from JSON configuration with scale override
            return load_articulation_from_json(
                scene=self.scene,
                json_path=self.cabinet_config_path,
                scale_override=self.cabinet_scale,
                json_type="urdf",
                position_override=[-0.15, -0.6, 0.35],
                orientation_override=[0, 0, np.pi * -0.5],
                prefix_function=self.update_prefix,
            )
            # Otherwise, use the high-level articulation loader
        table_height = 0.3
        offset = 0.05  # Small offset to prevent intersection with the table

        # Set the cabinet position
        position = np.array([-0.15, -0.6, table_height + offset])
        orientation = np.array([0, 0, np.pi * -0.5])  # No rotation

        # Load the articulation
        cabinet = load_articulation(
            scene=self.scene,
            position=position,
            orientation=orientation,
            scale=self.cabinet_scale,
            data_source="partnet-mobility",
            class_name="cabinet",
            class_id=cabinet_id,
            fix_root_link=True,
            name=f"cabinet-{cabinet_id}",
        )

        return cabinet

    def _initialize_episode(self, env_idx, options=None):
        """Initialize the episode by setting up the cabinet and robot"""
        # Call the parent method to initialize the base scene
        super()._initialize_episode(env_idx, options)

        # Initialize the cabinet with doors/drawers open
        for i, cabinet in enumerate(self.cabinets):
            if i >= len(env_idx):
                continue

            # Store the target joint positions and indices for reward calculation
            self.target_joint_positions = []
            self.target_joint_indices = []

            # Set the cabinet's joint states to open position
            # For closing task, we start with the cabinet fully open
            # active_joints = cabinet.get_active_joints()
            # active_joints = active_joints[:1]
            # qpos_list = []
            self.target_joint_name = "joint_0"
            self.set_articulation_joint(
                cabinet,
                self.target_joint_name,
                0.8,
            )
            # for j, joint in enumerate(active_joints):
            #     # Get the joint limits
            #
            #     limits = joint.get_limits()
            #     print("joint", joint, j)
            #     if limits is not None and limits.shape[0] > 0:
            #         # Store target (closed) position for reward calculation
            #         self.target_joint_positions.append(limits[0][0])
            #         self.target_joint_indices.append(j)
            #
            #         # Set joint to maximum position (fully open)
            #         pos = (
            #             limits[0][1] * 0.5
            #         )  # Use 70% of maximum limit to ensure doors are clearly open
            #         qpos_list.append(pos)
            #
            #         # Also set drive properties to maintain the position until robot interacts
            #         joint.set_drive_target(pos)
            #         joint.set_drive_properties(stiffness=5.0, damping=1.0)
            #
            #         # Set moderate friction to make it realistic but not too hard to close
            #         joint.set_friction(0.5)
            #     else:
            #         self.target_joint_positions.append(0.0)
            #         self.target_joint_indices.append(j)
            #
            # # Set all joint positions at once
            # cabinet.set_qpos(qpos_list)
            #
            # # Choose one joint to focus on (for simplicity)
            # if len(active_joints) > 0:
            #     # Choose the first joint with non-zero range
            #     for j, joint in enumerate(active_joints):
            #         limits = joint.get_limits()
            #         if (
            #             limits is not None
            #             and limits.shape[0] > 0
            #             and limits[0][1] - limits[0][0] > 0.01
            #         ):
            #             self.target_joint_idx = j
            #             break
            #     else:
            #         # If no suitable joint found, use the first one
            #         self.target_joint_idx = 0
            #

    # def compute_dense_reward(self, obs, action, info):
    #     """Compute the reward for the task"""
    #     # Get the current joint positions
    #     rewards = torch.zeros(self.num_envs, device=self.device)
    #
    #     for i, cabinet in enumerate(self.cabinets):
    #         if i >= self.num_envs:
    #             continue
    #
    #         # Get the active joints
    #         active_joints = cabinet.get_active_joints()
    #         if len(active_joints) == 0:
    #             continue
    #
    #         # Calculate rewards for all joints
    #         joint_rewards = []
    #         for joint in active_joints:
    #             # Get the ACTUAL current position using the qpos property
    #             current_pos = joint.qpos.item()
    #
    #             # Get the joint limits
    #             limits = joint.get_limits()
    #             if limits is None or limits.shape[0] == 0:
    #                 continue
    #
    #             # Calculate the normalized position (0 = closed, 1 = open)
    #             min_pos, max_pos = limits[0][0], limits[0][1]
    #             range_pos = max_pos - min_pos
    #             if range_pos < 1e-6:  # Avoid division by zero
    #                 continue
    #
    #             normalized_pos = (current_pos - min_pos) / range_pos
    #
    #             # Reward is higher when the cabinet is more closed (normalized_pos closer to 0)
    #             # Using a negative exponential to create a smooth reward gradient
    #             joint_reward = torch.exp(
    #                 -2.0
    #                 * torch.as_tensor(
    #                     normalized_pos, dtype=torch.float32, device=self.device
    #                 )
    #             )
    #
    #             # Add a bonus if the cabinet is almost closed
    #             if normalized_pos < 0.1:  # Within 10% of closed position
    #                 joint_reward += 0.5
    #
    #             # Add an even bigger bonus if the cabinet is fully closed
    #             if normalized_pos < 0.05:  # Within 5% of closed position
    #                 joint_reward += 1.0
    #
    #             joint_rewards.append(joint_reward)
    #
    #         # Average the rewards across all joints
    #         if joint_rewards:
    #             rewards[i] = sum(joint_rewards) / len(joint_rewards)
    #
    #             # Print debugging info when in human render mode
    #             if self.render_mode == "human" and i == 0:
    #                 positions = [joint.qpos.item() for joint in active_joints]
    #                 norm_positions = [
    #                     (joint.qpos.item() - joint.get_limits()[0][0])
    #                     / (joint.get_limits()[0][1] - joint.get_limits()[0][0])
    #                     for joint in active_joints
    #                 ]
    #                 print(f"Joint positions: {positions}")
    #                 print(f"Normalized positions: {norm_positions}")
    #                 print(f"Reward: {rewards[i].item()}")
    #
    #     return rewards
    #
    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        success = super()._get_success(env_idx)
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        for i, idx in enumerate(env_idx):
            if idx >= len(self.cabinets):
                continue

            cabinet = self.cabinets[idx]
            active_joints = cabinet.get_active_joints()
            if len(active_joints) == 0:
                continue

            # Get the target joint
            target_joint = active_joints[self.target_joint_idx]
            current_pos = target_joint.qpos

            # Get the joint limits
            limits = target_joint.get_limits()
            if limits is None or limits.shape[0] == 0:
                continue

            # Calculate the normalized position (0 = closed, 1 = open)
            min_pos, max_pos = limits[0][0], limits[0][1]
            range_pos = max_pos - min_pos
            if range_pos < 1e-6:  # Avoid division by zero
                continue

            # normalized_pos = (current_pos - min_pos) / range_pos
            normalized_pos = self.get_articulation_joint_info(
                cabinet, self.target_joint_name
            )
            # Success if the cabinet is almost closed (within 10% of closed position)
            if normalized_pos < 0.1:
                success[i] = True

        return {
            "success": success,
        }

    @property
    def cabinet(self):
        """Return the first cabinet for easy access in the test script"""
        if hasattr(self, "cabinets") and len(self.cabinets) > 0:
            return self.cabinets[0]
        return None
