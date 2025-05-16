from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors, articulations
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
    "Tabletop-Put-Cube-IntoCabinetWithObstacle-v1",
    max_episode_steps=5000,
)
class PutCubeIntoCabinetWithObstacleEnv(UniversalTabletopEnv):
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

    description = "put the object into the cabinet"
    workflow = [
        "open the cabinet door",
        "pull/push the obstacle",
        "pick the cube and put it in the cabinet",
    ]

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
            "obj": [Obj.OBSTACLE, Obj.GEOMETRY],
            "rob": [Robot.PERSPECTIVE, Robot.MORPH, Robot.ACT_NAV],
            "iter": [Inter.PLAN, Inter.FAIL_ADAPT],
        }
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )
        self.query_query = "What is the best approach to put the cube into the cabinet"
        self.query_selection = {
            "A": "Move the obstacle first, then place the cube into the cabinet",
            "B": "Try to navigate around the obstacle without moving it",
        }
        self.query_answer = "A"
        # Did it see the obstacle?
        # Did it move the obstacle away?
        # Did it pick up the cube?
        # Did it place the cube in the cabinet?

    def _load_scene(self, options=None):
        """Load the cabinet model into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)

        # Sample a random cabinet model for each environment
        b = self.num_envs
        model_indices = torch.randint(0, len(self.all_model_ids), (b,))
        self.cabinet_ids = [self.all_model_ids[i.item()] for i in model_indices]
        print(self.cabinet_ids)
        # input()
        # Load the cabinet models
        self.cabinets = []
        for i in range(b):
            cabinet = self._load_cabinet(self.cabinet_ids[i])
            self.cabinets.append(cabinet)

        self.block = actors.build_box(
            self.scene, [0.15, 0.02, 0.2], np.array([1, 1, 1, 1]), name="obstacle"
        )
        self.cube = self._create_default_object()

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
        orientation = np.array([0, 0, np.pi * -0.5])  # No rotation
        # print("cabinet_id", cabinet_id)
        # breakpoint()
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
        self.target_joint_name = "joint_0"
        self.set_articulation_joint(
            self.cabinet,
            self.target_joint_name,
            0.0,
        )

        self.block.set_pose(sapien.Pose(p=[-0.15, -0.45, 0.04]))

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        # This is an exploratory environment, so there's no specific success condition
        # We'll just return False for all environments
        success = super()._get_success(env_idx)

        if (
            self.cube.pose.p[0][2] > 0
            and self.is_stable(self.cube)
            and max(
                self.calculate_obj_aabb_roi(
                    self.cube, ((-0.15, -0.48, 0.01), (0.15, -0.35, 0.2))
                )
            )
            >= 0.7
        ):
            success = torch.ones_like(success)
        return success

    @property
    def cabinet(self):
        """Return the first cabinet for easy access in the test script"""
        if hasattr(self, "cabinets") and len(self.cabinets) > 0:
            return self.cabinets[0]
        return None
