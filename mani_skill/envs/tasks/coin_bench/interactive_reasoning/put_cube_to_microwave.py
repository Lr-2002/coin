from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
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
    "Tabletop-Put-Cube-IntoMicrowave-v1",
    max_episode_steps=5000,
)
class PutCubeToMicrowaveEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A tabletop environment with a microwave from PartNet Mobility placed on the table.
    The robot needs to open the microwave door.

    **Randomizations:**
    - The microwave's position on the table is randomized

    **Success Conditions:**
    - The microwave door is opened to at least 90% of its maximum range
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    MICROWAVE_ID = 7130  # Specific microwave model ID

    description = "put the cube into the microwave "
    workflow = [
        "open the microwave",
        "pick the cube and put it into the microwave",
        "close the door",
    ]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.0,
        microwave_scale=0.8,  # Scale factor for the microwave
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.OBSTACLE, Obj.GEOMETRY],
            "rob": [Robot.ACT_NAV, Robot.MORPH],
            "iter": [Inter.PLAN, Inter.FAIL_ADAPT]
        }
        self.microwave_scale = microwave_scale
        self.json_path = os.path.join("configs", "microwave.json")
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )
        self.query_query = (
            "What is the correct order of actions to put the cube into the microwave?"
        )
        self.query_selection = {
            "A": "Pick up the cube first, then open the microwave door and place the cube inside",
            "B": "Open the microwave door first, then pick and place the cube inside",
            "C": "Directly pick the cube into microwave",
        }
        self.query_answer = "B"
        # Did it open the microwave firstly?
        # Did it open the microwave?
        # Did it pick up the cube?
        # Did it put the cube in to the mircrowave?

    def _load_scene(self, options=None):
        """Load the microwave model into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)

        self.cube = self._create_default_object()
        # Load the microwave models
        self.microwaves = []
        for i in range(self.num_envs):
            microwave = self._load_microwave(self.MICROWAVE_ID)
            self.microwaves.append(microwave)

    def _load_microwave(self, microwave_id):
        """Load a microwave model from PartNet Mobility dataset"""
        table_height = 0.2
        # offset = 0.05  # Small offset to prevent intersection with the table
        # Set the microwave position
        position = np.array([0.0, -0.5, table_height])
        orientation = np.array(np.deg2rad([0, 0, -90]))  # Rotate to face the robot
        microwave = load_articulation_from_json(
            scene=self.scene,
            json_path=self.json_path,
            json_type="urdf",
            position_override=position,
            orientation_override=orientation,
            prefix_function=self.update_prefix,
            # scale_override=scale_override
        )

        return microwave

    def _initialize_episode(self, env_idx, options=None):
        """Initialize the episode by setting up the microwave and robot"""
        # Call the parent method to initialize the base scene
        super()._initialize_episode(env_idx, options)
        self.cube.set_pose(sapien.Pose(p=[0.0, 0.1, 0.2]))
        # Initialize the microwave - set all joints to closed position
        for i, microwave in enumerate(self.microwaves):
            if i >= len(env_idx):
                continue

            # Print joint information for debugging
            if self.render_mode == "human" and i == 0:
                print("Microwave joints:")
                for j, joint in enumerate(microwave.get_active_joints()):
                    print(
                        f"Joint {j} name: {joint.name}, qpos: {joint.qpos}, limits: {joint.get_limits()}"
                    )

            # Set all joints to closed position (minimum limit)
            self.set_articulation_joint(self.microwave, "joint_1", 0, easy_pd=(1, 0.1))

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        # aabb_microwave = self.get_aabb(self.microwave)
        # aabb_cube = self.get_aabb(self.cube)
        # print('------', aabb_microwave, aabb_cube)
        if max(self.calculate_obj_roi(self.cube, self.microwave)) > 0.5:
            success = torch.ones_like(success)
        return {
            "success": success,
        }

    @property
    def microwave(self):
        """Return the first microwave for easy access in the test script"""
        if hasattr(self, "microwaves") and len(self.microwaves) > 0:
            return self.microwaves[0]
        return None
