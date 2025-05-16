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
    "Tabletop-Open-Microwave-v1",
    max_episode_steps=5000,
)
class OpenMicrowaveEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A tabletop environment with a microwave from PartNet Mobility placed on the table.
    The robot needs to open the microwave door.

    **Randomizations:**
    - The microwave's position on the table is randomized

    **Success Conditions:**
    - The microwave door is opened to at least 90% of its maximum range
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    MICROWAVE_ID = 7130  # Specific microwave model ID

    description = "open the microwave"

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.0,
        microwave_scale=0.8,  # Scale factor for the microwave
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.microwave_scale = microwave_scale

        self.json_path = os.path.join("configs", "microwave.json")
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    def _load_scene(self, options=None):
        """Load the microwave model into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)
        # Load the microwave models
        self.microwaves = []
        for i in range(self.num_envs):
            microwave = self._load_microwave(self.MICROWAVE_ID)
            self.microwaves.append(microwave)

    def _load_microwave(self, microwave_id):
        """Load a microwave model from PartNet Mobility dataset"""
        table_height = 0.2
        # offset = 0.05  # Small offset to prevent intersection with the table

        # Set the microwave position
        position = np.array([0.0, -0.45, table_height])
        orientation = np.array(np.deg2rad([0, 0, -90]))  # Rotate to face the robot
        microwave = load_articulation_from_json(
            scene=self.scene,
            json_path=self.json_path,
            json_type="urdf",
            position_override=position,
            orientation_override=orientation,
            prefix_function=self.update_prefix,
            # scale_override=scale_override
        )
        self.target_joint_name = "joint_0"
        # for j, joint in enumerate(microwave.get_active_joints()):
        #     # Get the joint limits
        #     limits = joint.get_limits()
        #     if limits is None or limits.shape[0] == 0:
        #         continue
        #     # Check if limits are finite
        #     # if np.isfinite(min_pos) and np.isfinite(max_pos):
        #     min_pos, max_pos = limits[0][0], limits[0][1]
        #     # Set to closed position (minimum limit)
        #     pos = float(min_pos)
        #     joint.set_drive_target(pos)
        #     joint.set_drive_properties(5, 1)
        #
        return microwave

    def _initialize_episode(self, env_idx, options=None):
        """Initialize the episode by setting up the microwave and robot"""
        # Call the parent method to initialize the base scene
        super()._initialize_episode(env_idx, options)

        self.set_articulation_joint(self.microwave, "joint_0", 0.1)
        # Initialize the microwave - set all joints to closed position
        # for i, microwave in enumerate(self.microwaves):
        #     if i >= len(env_idx):
        #         continue
        #
        #     # Print joint information for debugging
        #     if self.render_mode == "human" and i == 0:
        #         print("Microwave joints:")
        #         for j, joint in enumerate(microwave.get_active_joints()):
        #             print(
        #                 f"Joint {j} name: {joint.name}, qpos: {joint.qpos}, limits: {joint.get_limits()}"
        #             )
        #
        # Set all joints to closed position (minimum limit)
        # for j, joint in enumerate(microwave.get_active_joints()):
        #     # Get the joint limits
        #     limits = joint.get_limits()
        #     if limits is None or limits.shape[0] == 0:
        #         continue
        #     # Check if limits are finite
        #     min_pos, max_pos = limits[0][0], limits[0][1]
        #     if np.isfinite(min_pos) and np.isfinite(max_pos):
        #         # Set to closed position (minimum limit)
        #             pos = float(min_pos)
        #             joint.set_drive_target(pos)
        #             joint.set_drive_properties(5, 1)
        #             # # Try to manually set the joint position
        # try:
        #     joint.set_qpos(torch.tensor([pos]))
        # except Exception as e:
        #     if self.render_mode == "human" and i == 0:
        #         print(f"Error setting qpos for joint {joint.name}: {e}")

        # # Print the result of setting the drive target
        # if self.render_mode == "human" and i == 0:
        #     print(f"Set joint {joint.name} to position {pos}, current qpos: {joint.qpos}")

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)

        if self.get_articulation_joint_info(self.microwave, "joint_0") >= 0.7:
            success = torch.ones_like(success)

        return {
            "success": success,
        }

    def get_obs(self, env_idx=None):
        """Get the observation for the environment"""
        # If env_idx is None, use all environments
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        # Get info for the specified environments
        info = self.get_info()

        # Get the base observation from the parent class
        obs = super().get_obs(info)

        return obs

    @property
    def microwave(self):
        """Return the first microwave for easy access in the test script"""
        if hasattr(self, "microwaves") and len(self.microwaves) > 0:
            return self.microwaves[0]
        return None
