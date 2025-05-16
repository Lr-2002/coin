from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.coin_bench.primitive_actions.pick_place import PickPlaceEnv
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import table
import os


@register_env("Tabletop-Pick-Cube-ToHolder-v1", max_episode_steps=5000)
class PickCubeToHolderEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A tabletop environment with a pen and a pen holder. The goal is to pick up the pen.

    **Randomizations:**
    - The pen's position on the table is fixed
    - The pen holder's position on the table is fixed

    **Success Conditions:**
    - The pen is grasped by the robot
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    description = "pick up the cube, put it in the holder"

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        pen_scale=1.0,
        **kwargs,
    ):
        # Set paths for pen asset
        self.pen_path = "configs/nonoe.json"
        # Set path for pen holder
        self.pen_holder_path = "configs/pen_holder.json"
        super().__init__(*args, **kwargs)

    def _load_scene(self, options=None):
        """Load the pen and pen holder into the scene"""
        # Call the parent method to set up the base scene (table, etc.) and the pen
        super()._load_scene(options)

        # Load the pen holder as an additional object
        self.pen_holder = self.load_from_config(
            self.pen_holder_path, "holder", body_type="static"
        )
        self.pen = self.load_from_config(self.pen_path, "cube")

    def _initialize_episode(self, env_idx, options=None):
        """Initialize the episode by setting up the pen, pen holder, and robot"""
        # Call the parent method to initialize the base scene
        super()._initialize_episode(env_idx, options)

        # Set the initial position of the pen holder
        if self.pen_holder is not None:
            pen_holder_position = [0.1, 0.0, 0.0]
            # pen_holder_rotation = [
            #     1,
            #     0,
            #     0.0,
            #     0.0,
            # ]
            pen_holder_rotation = [
                0.707,
                0.707,
                0.0,
                0.0,
            ]  # w, x, y, z quaternion format
            self.pen_holder.set_pose(
                sapien.Pose(p=pen_holder_position, q=pen_holder_rotation)
            )
        self.pen.set_pose(
            sapien.Pose(
                p=(0.1, 0.2, 0.2),
            )
        )

    def compute_dense_reward(self, obs, action, info):
        """Compute the reward for the task"""
        # Initialize rewards with zeros
        rewards = torch.zeros(self.num_envs, device=self.device)

        # Check if pen and pen_holder are initialized
        if (
            not hasattr(self, "pen")
            or self.pen is None
            or not hasattr(self, "pen_holder")
            or self.pen_holder is None
        ):
            return rewards

        # Initialize variables to avoid unbound errors
        pen_pos = None
        holder_pos = None

        try:
            # Get positions of the pen and pen holder
            pen_pos = self.pen.pose.p
            holder_pos = self.pen_holder.pose.p

            # Ensure we have valid position data
            if not isinstance(pen_pos, np.ndarray) or not isinstance(
                holder_pos, np.ndarray
            ):
                return rewards

            # Make sure positions are properly shaped
            if len(pen_pos.shape) == 1:
                # Single position vector [x, y, z]
                if pen_pos.shape[0] >= 3 and holder_pos.shape[0] >= 3:
                    # Calculate the horizontal distance between pen and holder
                    horizontal_dist = np.sqrt(
                        (pen_pos[0] - holder_pos[0]) ** 2
                        + (pen_pos[1] - holder_pos[1]) ** 2
                    )

                    # Check if the pen is above the holder (vertically aligned)
                    is_above_holder = (
                        horizontal_dist < 0.05
                    )  # Threshold for horizontal alignment

                    # Check if the pen is at the right height to be considered "in" the holder
                    pen_height = pen_pos[2]
                    holder_height = holder_pos[2]
                    is_in_holder_height = (
                        pen_height > holder_height and pen_height < holder_height + 0.1
                    )  # Adjust threshold as needed

                    # Reward is 1 if the pen is in the holder, 0 otherwise
                    if is_above_holder and is_in_holder_height:
                        rewards = torch.ones(self.num_envs, device=self.device)
            else:
                # Handle batched positions if needed
                print(
                    f"Warning: Unexpected position shape: pen_pos={pen_pos.shape}, holder_pos={holder_pos.shape}"
                )
        except (IndexError, AttributeError) as e:
            # Log the error for debugging
            print(f"Error in compute_dense_reward: {e}")
            print(f"pen_pos shape: {pen_pos.shape if pen_pos is not None else 'None'}")
            print(
                f"holder_pos shape: {holder_pos.shape if holder_pos is not None else 'None'}"
            )

        return rewards

    def _get_success(self, env_idx=None):
        self.show_two_objects(self.pen_holder, self.pen)
        ra1, ra2 = self.calculate_obj_roi(self.pen_holder, self.pen)
        if ra1 > 0.8 or ra2 > 0.8:
            success = super()._get_success(env_idx)
            success = torch.ones_like(success) * 1
            return success
        return super()._get_success(env_idx)
