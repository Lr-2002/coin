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


@register_env("Tabletop-Rotate-USB-v1", max_episode_steps=5000)
class RotateUSBEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to rotate a USB body.

    **Randomizations:**
    - The USB body's position is randomized on the table

    **Success Conditions:**
    - The USB body is inserted into the USB hub (close enough in position and orientation)
    - The robot is static (velocity < 0.2)
    """

    description = "Rotate the USB body for 90 degree with plug face left "

    def __init__(
        self,
        *args,
        usb_body_config="configs/usb_body.json",
        **kwargs,
    ):
        # Load USB body configuration
        self.usb_body_config = usb_body_config
        self.success_angle_threshold = 5.1
        self.initial_orientation = np.array([0, 0, 0])
        self.target_orientation = np.array([-90, 0, 0])

        # Set task description

        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Load the USB body
        self.usb_body = self.load_from_config(
            self.usb_body_config, "usb_body", body_type="dynamic", convex=True
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode with random positions for objects"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height

        # Randomize the position of the USB body
        usb_body_pos_x = self.np_random.uniform(0.0, 0.2)
        usb_body_pos_y = self.np_random.uniform(-0.2, 0.0)
        usb_body_pos_z = 0.02
        usb_body_pos = np.array([usb_body_pos_x, usb_body_pos_y, usb_body_pos_z])
        usb_body_quat = euler2quat(*np.deg2rad(self.initial_orientation)).tolist()
        if self.usb_body is not None:
            self.usb_body.set_pose(sapien.Pose(p=usb_body_pos, q=usb_body_quat))

    def _get_success(self, env_idx=None):
        """Check if the task is successful"""
        success = super()._get_success(env_idx)
        if self.compare_angle(
            self.usb_body,
            self.target_orientation,
            threshold=self.success_angle_threshold,
            specific_axis="roll",
        ) and self.is_static(self.usb_body):
            success = torch.ones_like(success)
        return success
