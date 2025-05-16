from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat
import random

import mani_skill.envs.utils.randomization as randomization
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.utils.building import articulations
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
    load_articulation_from_json,
)
from mani_skill.utils.io_utils import load_json
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.envs.tasks.coin_bench.primitive_actions.pick_place import PickPlaceEnv

"""
Task: Get an object from inside a cabinet and place it on the upper surface of the cabinet.

The robot needs to:
1. Open the cabinet door/drawer
2. Pick up the object from inside the cabinet
3. Place the object on the top surface of the cabinet
"""


@register_env(
    "Tabletop-Pick-Object-FromCabinet-v1",
    max_episode_steps=5000,
    asset_download_ids=["partnet_mobility_cabinet"],
)
class PickObjectFromCabinetEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to get an object from inside a cabinet and place it on the upper surface of the cabinet.

    **Randomizations:**
    - The cabinet model is randomly sampled from PartNet Mobility cabinet models
    - The object's position is randomized inside the cabinet
    - The cabinet's position on the table is randomized

    **Success Conditions:**
    - The object is placed on the top surface of the cabinet (within a threshold)
    - The robot is static (velocity < 0.2)
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )

    description = "pick up the object from the cabinet "
    workflow = [
        "open the cabinet door ",
        "pick the object in the cabinet ",
        "put it on the marker",
    ]

    def __init__(
        self,
        *args,
        object_config="configs/book",
        cabinet_scale=0.5,  # Scale factor for the cabinet
        cabinet_config_path="configs/drawer_cabinet.json",  # Path to a JSON configuration file for the cabinet
        **kwargs,
    ):
        self.cabinet_scale = cabinet_scale
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.OBSTACLE, Obj.GEOMETRY, Obj.SPATIALRELATE],
            "rob": [Robot.PERSPECTIVE, Robot.MORPH, Robot.ACT_NAV],
            "iter": [Inter.PLAN, Inter.FAIL_ADAPT],
        }
        self.cabinet_config_path = cabinet_config_path

        # Load cabinet model IDs
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        # Initialize with the book object
        super().__init__(*args, **kwargs)
        self.query_query = "What's the object in the cabinet?"
        self.query_selection = {"A": "A book", "B": "A cube", "C": "A pen"}
        self.query_answer = "A"
        # Did it move toward the cabinet?
        # Did it see the object in the cabinet?
        # Did it get the object from the cabinet?
        # Did it move the object to the target?

    def _load_scene(self, options: dict):
        """Load the scene with table, cabinet, and object"""
        # Load the basic scene with table
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

        self.object = self.load_from_config("configs/apple.json", "apple")
        self._create_goal_area()

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
        orientation = np.array([0, 0, -0.5 * np.pi])  # No rotation

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
        # self.set_articulation_joint(
        #     cabinet,
        #     "joint_0",
        #     0.4,
        # )
        return cabinet

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized object and cabinet"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Randomize the cabinet's joint states
        # This will open/close drawers or doors randomly
        for i, cabinet in enumerate(self.cabinets):
            if i >= len(env_idx):
                continue

            # Get all joints of the cabinet
            joints = cabinet.get_active_joints()

            # Randomly open one of the doors/drawers
            if len(joints) > 0:
                # Select a random joint to open
                joint_idx = np.random.randint(0, len(joints))
                joint = joints[joint_idx]

                # Get the joint limits
                limits = joint.get_limits()
                if limits is not None and limits.shape[0] > 0:
                    # Open the door/drawer (set to maximum position)
                    joint.set_drive_target(
                        limits[0][1] * 0.8
                    )  # 80% of max to avoid collision issues

                    # Set other joints to closed position
                    for j, other_joint in enumerate(joints):
                        if j != joint_idx:
                            other_limits = other_joint.get_limits()
                            if other_limits is not None and other_limits.shape[0] > 0:
                                other_joint.set_drive_target(
                                    other_limits[0][0]
                                )  # Minimum position (closed)

        # Place the object inside the opened cabinet
        self._place_object_in_cabinet(env_idx)

        # Set target position on top of the cabinet
        self._set_target_on_cabinet_top(env_idx)

    def _place_object_in_cabinet(self, env_idx):
        """Place the object inside the opened cabinet"""
        cabinet = self.cabinets[0]  # Use the first cabinet for simplicity
        # Find the cabinet's dimensions and position
        cabinet_links = cabinet.get_links()
        cabinet_aabb_min, cabinet_aabb_max = None, None

        # Find the cabinet's bounding box
        # for link in cabinet_links:
        #     link_aabb_min, link_aabb_max = link.get_aabb()
        #     if cabinet_aabb_min is None:
        #         cabinet_aabb_min, cabinet_aabb_max = link_aabb_min, link_aabb_max
        #     else:
        #         cabinet_aabb_min = np.minimum(cabinet_aabb_min, link_aabb_min)
        #         cabinet_aabb_max = np.maximum(cabinet_aabb_max, link_aabb_max)
        #
        # Calculate a position inside the cabinet
        # Use the center of the cabinet with a slight offset from the bottom
        cabinet_aabb_min, cabinet_aabb_max = self.get_aabb(cabinet)
        cabinet_center = (cabinet_aabb_min + cabinet_aabb_max) / 2

        # Place the object inside the cabinet
        object_z = 0 + 0.3  # Slightly above the bottom of the cabinet
        object_pose = sapien.Pose(
            p=[cabinet_center[0], cabinet_center[1], object_z],
            # q=euler2quat(
            #     self.object_orientation[0],
            #     self.object_orientation[1],
            #     self.object_orientation[2],
            # ),
        )
        self.object.set_pose(object_pose)

    def _set_target_on_cabinet_top(self, env_idx):
        """Set the target position on top of the cabinet"""
        cabinet = self.cabinets[0]  # Use the first cabinet for simplicity
        return
        # Find the cabinet's dimensions and position
        cabinet_links = cabinet.get_links()
        cabinet_aabb_min, cabinet_aabb_max = None, None

        # Find the cabinet's bounding box
        for link in cabinet_links:
            link_aabb_min, link_aabb_max = link.get_aabb()
            if cabinet_aabb_min is None:
                cabinet_aabb_min, cabinet_aabb_max = link_aabb_min, link_aabb_max
            else:
                cabinet_aabb_min = np.minimum(cabinet_aabb_min, link_aabb_min)
                cabinet_aabb_max = np.maximum(cabinet_aabb_max, link_aabb_max)

        # Calculate a position on top of the cabinet
        target_z = cabinet_aabb_max[2] + 0.001  # Slightly above the top of the cabinet

        # Randomize the target position on the top surface
        top_surface_min = np.array([cabinet_aabb_min[0], cabinet_aabb_min[1]])
        top_surface_max = np.array([cabinet_aabb_max[0], cabinet_aabb_max[1]])

        # Calculate a random position on the top surface
        target_xy = top_surface_min + np.random.rand(2) * (
            top_surface_max - top_surface_min
        )

        # Set target marker pose
        target_pose = sapien.Pose(
            p=[target_xy[0], target_xy[1], target_z],
            q=[1, 0, 0, 0],
        )
        self.target_marker.set_pose(target_pose)

        # Store target position for reward calculation
        self.target_position = torch.tensor(
            [target_xy[0], target_xy[1], target_z],
            device=self.device,
        )

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

    def _get_success(self, env_idx=None):
        success = super()._get_success(env_idx)
        if self.calculate_object_distance(self.goal_region, self.object) < 0.05:
            success = torch.ones_like(success)
        return success

    @property
    def cabinet(self):
        """Return the first cabinet for easy access in the test script"""
        if hasattr(self, "cabinets") and len(self.cabinets) > 0:
            return self.cabinets[0]
        return None
