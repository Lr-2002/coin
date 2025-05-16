from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.building import actors
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


@register_env("Tabletop-Open-Door-WithObstacle-v1", max_episode_steps=5000)
class OpenDoorWithObstacleEnv(UniversalTabletopEnv):
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

    description = "open the door"
    workflow = ["move the blue board", "open the door"]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        door_scale=0.3,  # Scale factor for the door
        **kwargs,
    ):
        # self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.OBSTACLE],
            "rob": [],
            "iter": [Inter.PLAN, Inter.FAIL_ADAPT]
        }
        self.door_scale = door_scale
        # self.door_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "configs", "door_8867.json")
        self.door_config_path = "configs/door_8867.json"
        super().__init__(
            *args,
            robot_uids=robot_uids,
            # robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

        self.query_query = "What is the main challenge in opening this door?"
        self.query_selection = {
            "A": "The door is unreachable",
            "B": "The door is too heavy to be opened directly",
            "C": "There is an obstacle blocking the door's path",
        }
        self.query_answer = "C"

    def _load_scene(self, options=None):
        """Load the door model into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)

        # Load the door model
        self.door = self._load_door()

        self.fix_cube = actors.build_cylinder(
            self.scene,
            radius=0.05,
            half_length=0.02,
            body_type="static",
            color=np.array([1, 0, 0, 1]),
            name="obstacle",
        )
        # self.pivot = actors.build_twocolor_peg(
        #     self.scene,
        #     length=0.15,
        #     width=0.02,
        #     color_1=np.array([12, 42, 160, 255]) / 255,
        #     color_2=np.array([12, 42, 160, 255]) / 255,
        #     name="pivot",
        #     body_type="dynamic",
        #     initial_pose=sapien.Pose(p=[0.0, -0.1, 0]),
        # )
        self.pivot = actors.build_box(
            self.scene,
            [
                0.15,
                0.03,
                0.01,
            ],
            name="pivot",
            body_type="dynamic",
            color=np.array([12, 42, 160, 255]) / 255,
        )

    def _load_door(self):
        """Load the door model from URDF using a JSON configuration file"""
        # Load from JSON configuration
        # door = load_articulation_from_json(
        #     scene=self.scene,
        #     json_path=self.door_config_path,
        #     json_type="urdf",
        #     scale_override=self.door_scale,
        #     position_override=[0.15, 0.0, 0.4]
        # )
        door = self.load_articulation_from_json(
            self.door_config_path,
            scale_override=self.door_scale,
            position_override=[0.15, 0.0, 0.28],
        )

        # door.set_pose(sapien.Pose(p=[0.15, 0.0, 0.4]))  # Set the door position
        # position = torch.tensor([0.3, 0.0, 0.0], device=self.device)
        # door.set_root_pose(Pose(position))

        return door

    def _initialize_episode(self, env_idx, options=None):
        """Initialize the episode by setting up the door and robot"""
        # Call the parent method to initialize the base scene
        super()._initialize_episode(env_idx, options)

        # Initialize all door joints to position 0
        for joint in self.door.get_active_joints():
            self.set_articulation_joint(
                self.door, joint.name, 0.2, easy_pd=(1, 0.1), friction=0
            )
        self.fix_cube.set_pose(
            sapien.Pose(p=[-0.28, -0.0, 0.0], q=[0.707, 0, 0.707, 0])
        )
        self.pivot.set_pose(sapien.Pose(p=[-0.1, 0, 0.0]))

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        # Check if any joint is open (position > 90% of its limit)
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)

        # Get the active joints
        active_joints = self.door.get_active_joints()
        if len(active_joints) == 0:
            return {"success": success}

        if self.get_articulation_joint_info(self.door, "joint_1") >= 0.9:
            success = torch.ones_like(success)
        return {
            "success": success,
        }
