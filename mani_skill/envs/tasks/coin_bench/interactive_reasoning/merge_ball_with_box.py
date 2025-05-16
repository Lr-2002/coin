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


@register_env("Tabletop-Merge-Box-v1", max_episode_steps=5000)
class MergeBoxEnv(UniversalTabletopEnv):
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

    description = "Merge ball and boxs up "
    workflow = [
        "pick the ball and put it on the hole of the box",
        "pick the cube which have plat on the top and put it on the ball",
    ]

    def __init__(
        self,
        *args,
        ball_config="configs/ball__r9.json",
        box_config="configs/cube__hole_12_12_12__r9.json",
        insertion_threshold=0.09,  # Distance threshold for successful insertion
        **kwargs,
    ):
        # Set insertion threshold for success
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.ORIENT],
            "rob": [],
            "iter": [Inter.PLAN]
        }
        self.ball_config = ball_config
        self.box_config = box_config
        self.insertion_threshold = insertion_threshold

        # Set task description

        super().__init__(*args, **kwargs)
        self.query_query = "which box is to be inserted"
        self.query_selection = {"A": "Box A", "B": "Box B"}
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
        self.ball = self.load_from_config(
            self.ball_config, "ball", convex=True, body_type="dynamic"
        )
        self.box1 = self.load_from_config(
            self.box_config, "box1", convex=False, body_type="dynamic"
        )
        self.box2 = self.load_from_config(
            self.box_config, "upper", convex=False, body_type="dynamic"
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
        self.ball.set_pose(sapien.Pose(p=[0.1, 0.1, table_height + 0.05]))
        self.box2.set_pose(
            sapien.Pose(
                p=[0.0, 0.0, table_height + 0.10], q=euler2quat(np.pi / 2, 0, 0)
            )
        )
        self.box1.set_pose(
            sapien.Pose(
                p=[0.0, -0.1, table_height + 0.02], q=euler2quat(-np.pi / 2, 0, 0)
            )
        )

    def _get_success(self, env_idx=None):
        """Check if the task is successful"""
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            self.calculate_object_distance(self.ball, self.box1)
            <= self.insertion_threshold
            and self.calculate_object_distance(self.ball, self.box2)
            <= self.insertion_threshold
            and self.is_stable(self.box1)
            and self.is_stable(self.box2)
            and (
                min(self.box1.pose.p[0][2], self.box2.pose.p[0][2])
                <= self.ball.pose.p[0][2]
                and max(self.box1.pose.p[0][2], self.box2.pose.p[0][2])
                >= self.ball.pose.p[0][2]
                # and self.box1.pose.p[0][2] >= 0.3
            )
        ):
            success = torch.ones_like(success)
        return success
