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

from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
    load_articulation_from_json,
    load_articulation_from_urdf,
)


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Pick-Eraser-FromHolder-v1", max_episode_steps=5000)
class PickEraserAndPenEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to pick up an eraser and a pen and place them in a pen holder.

    **Randomizations:**
    - The eraser's position is randomized on the table
    - The pen's position is randomized on the table
    - The pen holder's position is randomized on the table

    **Success Conditions:**
    - Both the eraser and pen are placed in the pen holder
    - The robot is static (velocity < 0.2)
    """

    description = "Pick up the eraser in the holder and place it to the marker"
    workflow = [
        "rotate the holder",
        "rotate the hodler",
        "pick the holder",
        "pick the earser and put it on the marker",
    ]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Load eraser configuration
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.ORIENT, Obj.SPATIALRELATE],
            "iter": [Inter.PLAN, Inter.HISTORY],
            "rob": [Robot.MORPH],
        }
        eraser_config = "configs/eraser.json"
        pen_config = "configs/pen.json"
        holder_config = "configs/pen_holder.json"
        placement_threshold = 0.05  # Distance threshold for successful placement

        self.eraser_config = eraser_config

        # Load pen configuration
        self.pen_config = pen_config

        # Load pen holder configuration
        self.holder_config = holder_config

        # Set placement threshold for success
        self.placement_threshold = placement_threshold

        # Set task description

        super().__init__(*args, **kwargs)
        self.query_query = "How can we pick the eraser from the holder?"
        self.query_selection = {
            "A": "Pick the eraser directly",
            "B": "Pour out the rubber from the holder",
        }
        self.query_answer = "B"

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        self.eraser = self.load_from_config(
            self.eraser_config, name="eraser", body_type="dynamic", convex=True
        )
        self.pen = self.load_from_config(
            self.pen_config, name="pen", body_type="dynamic"
        )
        self.goal = self._create_goal_area(position=[0, 0.2, 0])
        # # Load the pen holder
        # self.holder = self.load_asset(
        #     asset_path=self.holder_params.get(
        #         "usd-path", "assets_glb/simple_container.glb"
        #     ),
        #     scale=self.holder_params.get("scale", 0.6),
        #     mass=self.holder_params.get("mass", 0.1),
        #     friction=self.holder_params.get("friction", 2.0),
        #     name="holder",
        #     convex=False,
        #     body_type="dynamic",  # Holder should be static
        # )
        # self.holder = self.load_from_config(
        #     "configs/simple_container.json",
        #     name="container",
        #     convex=False,
        #     body_type="static",
        # )
        self.holder = self._load_holder()
        # self.holder = self.load_
        #
        # self.holder = self.load_articulation_from_urdf()
        #

    #
    def _load_holder(self):
        """Load a switch model from a JSON configuration file"""
        # Path to the JSON configuration file
        json_path = os.path.join("configs", "pen_holder.json")

        # Load the switch using the load_articulation_from_json function with json_type="urdf"
        holder = load_articulation_from_json(
            scene=self.scene,
            json_path=json_path,
            json_type="urdf",
            prefix_function=self.update_prefix,
        )

        return holder

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode with random positions for objects"""
        # Randomize the position of the eraser

        super()._initialize_episode(env_idx, options)
        # Place the holder at the center of the table
        holder_pos_x = 0.0
        holder_pos_y = -0.0
        holder_pos_z = self.table_height + 0.01
        holder_pos = np.array([holder_pos_x, holder_pos_y, holder_pos_z])
        holder_quat = euler2quat(90 / 180 * np.pi, 0, 0)

        if self.holder is not None:
            self.holder.set_pose(sapien.Pose(p=holder_pos, q=holder_quat.tolist()))

        eraser_pos_x = holder_pos_x
        eraser_pos_y = holder_pos_y
        eraser_pos_z = self.table_height + 0.1
        eraser_pos = np.array([eraser_pos_x, eraser_pos_y, eraser_pos_z])
        eraser_quat = euler2quat(0.0, 0, self.np_random.uniform(0, 2 * np.pi))

        if self.eraser is not None:
            self.eraser.set_pose(sapien.Pose(p=eraser_pos, q=eraser_quat.tolist()))

        # Randomize the position of the pen
        pen_pos_x = holder_pos_x + 0.1
        pen_pos_y = holder_pos_y
        pen_pos_z = self.table_height + 0.1
        pen_pos = np.array([pen_pos_x, pen_pos_y, pen_pos_z])
        pen_quat = euler2quat(0, 0.5 * np.pi, self.np_random.uniform(0, 0 * np.pi))

        if self.pen is not None:
            self.pen.set_pose(sapien.Pose(p=pen_pos, q=pen_quat.tolist()))

    def _get_success(self, env_idx=None):
        success = super()._get_success(env_idx)
        if self.calculate_object_distance(
            self.goal_region, self.eraser
        ) <= 0.02 and self.is_static(self.eraser):
            success = torch.ones_like(success)
        return success
