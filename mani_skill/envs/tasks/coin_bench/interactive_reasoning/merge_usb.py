from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
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


@register_env("Tabletop-Merge-USB-v1", max_episode_steps=5000)
class MergeUSBEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to pick up a USB body and insert it into a USB hub.

    **Randomizations:**
    - The USB body's position is randomized on the table
    - The USB hub's position is randomized on the table

    **Success Conditions:**
    - The USB body is inserted into the USB hub (close enough in position and orientation)
    - The robot is static (velocity < 0.2)
    """

    description = "Pick up the USB body and insert it into the USB hub"
    workflow = [
        "rotate the usb body",
        "align the body with the hub",
        "insert it to the usb hub",
    ]

    def __init__(
        self,
        *args,
        usb_body_config="configs/usb_body.json",
        usb_hub_config="configs/usb_hub.json",
        insertion_threshold=0.02,  # Distance threshold for successful insertion
        **kwargs,
    ):
        # Load USB body configuration
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY],
            "rob": [],
            "iter": [Inter.PLAN]
        }
        self.usb_body_config = usb_body_config
        self.usb_body_params = self._load_config(usb_body_config)

        # Load USB hub configuration
        self.usb_hub_config = usb_hub_config
        self.usb_hub_params = self._load_config(usb_hub_config)

        # Set insertion threshold for success
        self.insertion_threshold = insertion_threshold

        # Set task description

        super().__init__(*args, **kwargs)
        self.query_query = "What is the correct orientation for inserting the USB?"
        self.query_selection = {
            "A": "Align the USB connector with the port and insert",
            "B": "Rotate the USB 180 degrees and insert with the port facing down",
        }
        self.query_answer = "B"

    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"Loaded object configuration from {config_path}")
            return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Load the USB body
        # self.usb_body = self.load_asset(
        #     asset_path=self.usb_body_params.get("usd-path", "assets_glb/usb_body.glb"),
        #     scale=self.usb_body_params.get("scale", 0.01),
        #     mass=self.usb_body_params.get("mass", 0.5),
        #     friction=self.usb_body_params.get("friction", 1.0),
        #     name="usb_body",
        #     body_type='dynamic',
        #     density=1000.0
        # )
        self.usb_body = self.load_from_config(
            self.usb_body_config, "usb_body", convex=True, body_type="dynamic"
        )
        self.usb_hub = self.load_from_config(
            self.usb_hub_config, "usb_hub", convex=False, body_type="static"
        )
        # Load the USB hub
        # self.usb_hub = self.load_asset(
        #     asset_path=self.usb_hub_params.get("usd-path", "assets_glb/usb_hub.glb"),
        #     scale=self.usb_hub_params.get("scale", 0.01),
        #     mass=self.usb_hub_params.get("mass", 0.5),
        #     friction=self.usb_hub_params.get("friction", 1.0),
        #     name="usb_hub",
        #     body_type='static',
        #     density=1000.0
        # )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode with random positions for objects"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        table_height = 0

        # Randomize the position of the USB body
        usb_body_pos_x = self.np_random.uniform(0.0, 0.2)
        usb_body_pos_y = self.np_random.uniform(-0.2, 0.0)
        usb_body_pos_z = table_height + 0.02
        usb_body_pos = np.array([usb_body_pos_x, usb_body_pos_y, usb_body_pos_z])
        usb_body_quat = euler2quat(0, 0, self.np_random.uniform(0, 2 * np.pi))

        if self.usb_body is not None:
            self.usb_body.set_pose(
                sapien.Pose(p=usb_body_pos, q=usb_body_quat.tolist())
            )

        # Place the USB hub on the right side of the table
        usb_hub_pos_x = self.np_random.uniform(0.0, 0.1)
        usb_hub_pos_y = self.np_random.uniform(0.0, 0.1)
        usb_hub_pos_z = table_height + 0.02
        usb_hub_pos = np.array([usb_hub_pos_x, usb_hub_pos_y, usb_hub_pos_z])
        usb_hub_quat = euler2quat(np.pi, 0, 0)

        if self.usb_hub is not None:
            self.usb_hub.set_pose(sapien.Pose(p=usb_hub_pos, q=usb_hub_quat.tolist()))

    def _get_success(self, env_idx=None):
        """Check if the task is successful"""
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            self.calculate_object_distance(self.usb_body, self.usb_hub)
            <= self.insertion_threshold
            and self.is_static(self.usb_body)
            and self.is_stable(self.usb_body)
            and self.usb_body.pose.p[0][2] >= self.usb_hub.pose.p[0][2]
        ):
            success = torch.ones_like(success)

        return success
