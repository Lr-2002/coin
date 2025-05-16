from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
from torch.serialization import SourceChangeWarning
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
    "Tabletop-Open-Cabinet-WithSwitch-v1",
    max_episode_steps=5000,
    asset_download_ids=["partnet_mobility_cabinet"],
)
class OpenCabinetWithSwitchEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A tabletop environment with a cabinet from PartNet Mobility placed on the table.

    **Randomizations:**
    - The cabinet model is randomly sampled from all available PartNet Mobility cabinet models
    - The cabinet's position on the table is randomized

    **Success Conditions:**
    - This is an exploratory environment without specific success conditions
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )

    description = "open the door, notice the switch will control the state of the door "
    workflow = ["turn on the trigger", "open the cabinet door"]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        cabinet_scale=0.5,  # Scale factor for the cabinet (cabinets can be large)
        cabinet_config_path="configs/drawer_cabinet.json",  # Path to a JSON configuration file for the cabinet
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cabinet_scale = cabinet_scale
        self.cabinet_config_path = cabinet_config_path

        # Load cabinet model IDs
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.LOCK],
            "rob": [],
            "iter": [Inter.PLAN, Inter.FAIL_ADAPT],
        }
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )
        
        self.query_query = "What is the efficient way to open the cabinet"
        self.query_selection = {
            "A": "Open the cabinet directly",
            "B": "Push the switch to open the cabinet",
        }
        self.query_answer = "B"

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

        self.switch = self._load_switch()

    def _load_switch(self):
        """Load a switch model from a JSON configuration file"""
        # Path to the JSON configuration file
        json_path = os.path.join("configs", "switch.json")

        # Load the switch using the load_articulation_from_json function with json_type="urdf"
        switch = load_articulation_from_json(
            scene=self.scene,
            json_path=json_path,
            json_type="urdf",
            prefix_function=self.update_prefix,
        )

        return switch

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
        position = np.array([-0.15, -0.45, table_height + offset])
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

        # Additional initialization for the cabinet if needed
        # For example, you could randomize the joint states of the cabinet here
        self.switch.set_pose(
            sapien.Pose(
                p=np.array([0.1, 0.03, 0]), q=euler2quat(*np.deg2rad((0, 90, 0)))
            )
        )
        self.set_articulation_joint(self.switch, "joint_0", 1)  # 1 for off , 0 for on
        for joint in self.cabinet.get_active_joints():
            self.set_articulation_joint(
                self.cabinet, joint.name, 0, pd_version="hard", hard_pd=(500, 30)
            )

    def _update_cabinet_from_switch(self):
        if self.get_articulation_joint_info(self.switch, "joint_0") < 0.4:
            self.get_joint(self.switch, "joint_0").set_drive_properties(0, 0)
        if self.get_articulation_joint_info(self.switch, "joint_0") < 0.2:
            self.get_joint(self.cabinet, "joint_0").set_drive_properties(0, 0)
            self.get_joint(self.cabinet, "joint_1").set_drive_properties(0, 0)

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        # This is an exploratory environment, so there's no specific success condition
        # We'll just return False for all environments
        success = super()._get_success(env_idx)
        self._update_cabinet_from_switch()
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        print(
            "joint info is ",
            self.get_articulation_joint_info(self.cabinet, "joint_0"),
            self.get_articulation_joint_info(self.switch, "joint_0"),
        )
        if self.get_articulation_joint_info(self.cabinet, "joint_0") > 0.7:
            success = torch.ones_like(success)
        return success

    @property
    def cabinet(self):
        """Return the first cabinet for easy access in the test script"""
        if hasattr(self, "cabinets") and len(self.cabinets) > 0:
            return self.cabinets[0]
        return None
