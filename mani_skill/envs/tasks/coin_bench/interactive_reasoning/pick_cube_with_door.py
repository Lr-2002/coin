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


@register_env("Tabletop-Pick-Cube-WithDoor-v1", max_episode_steps=5000)
class PickCubeWithDoorEnv(UniversalTabletopEnv):
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

    description = "put the cube to the marker"
    workflow = ["open the door", "push the cube to the marker"]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        door_scale=0.4,  # Scale factor for the door
        **kwargs,
    ):
        # self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "rob": [Robot.JOINT_AWARE, Robot.ACT_NAV],
            "obj": [Obj.OBSTACLE],
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
        self.query_query = "What is the correct sequence to retrieve the cube?"
        self.query_selection = {
            "A": "Reach for the cube without opening the door",
            "B": "Open the door first, then pick the cube",
        }
        self.query_answer = "B"

    def _load_scene(self, options=None):
        """Load the door model into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)

        # Load the door model
        self.door = self._load_door()

        self.cube = (
            self._create_default_object()
        )  # self.pivot = actors.build_twocolor_peg(
        #     self.scene,
        #     length=0.15,
        #     width=0.02,
        #     color_1=np.array([12, 42, 160, 255]) / 255,
        #     color_2=np.array([12, 42, 160, 255]) / 255,
        #     name="pivot",
        #     body_type="dynamic",
        #     initial_pose=sapien.Pose(p=[0.0, -0.1, 0]),
        # )
        # self.pivot = actors.build_box(
        #     self.scene,
        #     [
        #         0.15,
        #         0.03,
        #         0.01,
        #     ],
        #     name="pivot",
        #     body_type="dynamic",
        #     color=np.array([12, 42, 160, 255]) / 255,
        # )
        self.target_area = self._create_goal_area()

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
            position_override=[0.15, 0.0, 0.38],
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
                self.door, joint.name, 0.2, easy_pd=(1, 0.1), friction=0.0
            )
        self.cube.set_pose(sapien.Pose(p=[0.2, 0.0, 0]))

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        # Check if any joint is open (position > 90% of its limit)
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self.get_articulation_joint_info(self.door, "joint_1") >= 0.8:
            self.set_articulation_joint(self.door, "joint_1", 1.0)
        # Get the active joints
        active_joints = self.door.get_active_joints()
        if len(active_joints) == 0:
            return {"success": success}

        if self.calculate_object_distance(
            self.cube, self.target_area
        ) <= 0.07 and self.is_stable(self.cube):
            success = torch.ones_like(success)
        return {
            "success": success,
        }
