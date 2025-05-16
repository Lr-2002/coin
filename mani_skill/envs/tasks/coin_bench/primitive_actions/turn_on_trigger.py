from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
import os
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import articulations
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation_from_json,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import table


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Open-Trigger-v1", max_episode_steps=5000)
class TurnOnTriggerEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A tabletop environment with a trigger/switch. The goal is to turn on the trigger.

    **Randomizations:**
    - The trigger's position on the table is fixed
    - The initial state of the trigger is randomized

    **Success Conditions:**
    - The trigger is turned on
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda

    description = "turn on the trigger"

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_wristcam",
        robot_init_qpos_noise: float = 0.00,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # Initialize trigger state variables
        self.trigger_is_on = False
        self.trigger_threshold = -0.7  # Threshold to determine if trigger is on/off
        self.last_trigger_state = (
            False  # Track previous trigger state for change detection
        )
        self.trigger_changed = (
            False  # Track if trigger state has changed during episode
        )
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    def _load_scene(self, options: Optional[Dict] = None):
        """Load the trigger into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)

        # Load the trigger
        self.trigger = self._load_trigger()

    def _load_trigger(self):
        """Load a trigger model from a JSON configuration file"""
        # Path to the JSON configuration file
        json_path = os.path.join("configs", "switch.json")
        position_override = np.array([0.0, -0.1, 0.03])
        orientation_override = np.array(create_orientation_from_degree(0, 90, 0))
        # scale_override = 0.05

        trigger = load_articulation_from_json(
            scene=self.scene,
            json_path=json_path,
            json_type="urdf",
            position_override=position_override,
            orientation_override=orientation_override,
            # scale_override=scale_override,
            prefix_function=self.update_prefix,
        )

        return trigger

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: Optional[Dict] = None
    ):
        """Initialize the episode by setting up the trigger and robot"""
        # Call the parent method to initialize the base scene
        super()._initialize_episode(env_idx, options)

        # Initialize the trigger with a random state
        self.set_articulation_joint(
            self.trigger, "joint_0", 1, easy_pd=(1, 0.2)
        )  # 1 for off , 0 for on

    def _get_success(self, env_idx=None) -> Dict:
        """
        Evaluate if the task is successful.
        Success is defined as turning on the trigger.

        Returns:
            Dict: Dictionary containing success information
        """
        # Check if the trigger is currently on
        success = super()._get_success(env_idx)
        # if self.get_articulation_joint_info(self.trigger) <
        if self.get_articulation_joint_info(self.trigger, "joint_0") < 0.4:
            self.get_joint(self.trigger, "joint_0").set_drive_properties(0, 0)
        if self.get_articulation_joint_info(self.trigger, "joint_0") < 0.2:
            success = torch.ones_like(success)

        # self.show_two_objects(self.trigger)
        #
        # is_on = self.get_trigger_state()
        #
        # # Create success dictionary as required by the parent class
        # success = {
        #     "success": torch.tensor(
        #         [is_on for _ in range(self.num_envs)],
        #         dtype=torch.bool,
        #         device=self.device,
        #     )
        # }

        return success
