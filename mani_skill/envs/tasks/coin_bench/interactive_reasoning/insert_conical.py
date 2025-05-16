from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat
import random
from mani_skill.envs import sapien_env
import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
import numpy as np
from mani_skill.utils import sapien_utils


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Insert-Conical-v1", max_episode_steps=5000)
class InsertConicalEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to insert three different shaped objects (cube, cylinder, triangle)
    into a container with matching holes.

    **Randomizations:**
    - The objects' positions are randomized on the table
    - The objects' orientations are randomized around the z-axis
    - The container position is randomized on the table

    **Success Conditions:**
    - All three objects are inserted into their corresponding holes in the container
    - The robot is static (velocity < 0.2)
    """

    description = "insert the conical to the container"
    workflow = [
        "pick the bottle up ",
        "align the bottle with the holder",
        "place the bottle down",
    ]

    def __init__(
        self,
        *args,
        object_mass=0.5,  # Mass of the objects in kg
        object_friction=1.0,  # Friction coefficient of the objects
        insertion_threshold=0.02,  # Distance threshold for successful insertion
        container_scale=0.5,  # Scale of the container
        **kwargs,
    ):
        print("===== args ", args, kwargs)
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.ORIENT],
            "rob": [],
            "iter": []
        }
        object_mass = random.uniform(0.1, 10)

        # Object properties
        self.object_mass = object_mass
        self.object_friction = object_friction
        self.insertion_threshold = insertion_threshold
        self.container_scale = container_scale

        # Object paths
        self.container_path = "configs/pen_holder.json"
        self.conical_config = "configs/water_bottle.json"
        # Check if files exist

        # Store objects and their target positions
        self.objects = {}
        self.object_initial_poses = {}
        self.target_positions = {}

        super().__init__(*args, **kwargs)

        self.query_query = "What to prevent while insert the water bottle"
        self.query_selection = {
            "A": "Tilting the bottle too much, which may cause misalignment",
            "B": "Moving too slowly, which wastes time and energy",
        }
        self.query_answer = "A"

    def _load_scene(self, options):
        super()._load_scene(options)
        self.container = self.load_from_config(
            self.container_path, "container", body_type="static", scale_override=0.6
        )

        self.conical = self.load_from_config(
            self.conical_config, "conical", convex=True
        )
        self.conical.set_mass(0.1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode by placing objects and container"""
        # Create the container
        super()._initialize_episode(env_idx, options)
        self.container.set_pose(sapien.Pose(p=[0.0, 0.1, 0], q=[0.707, 0.707, 0, 0]))
        self.conical.set_pose(sapien.Pose(p=[0.0, -0.1, 0], q=[0, 1, 0, 0]))

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        # Check if all objects are in their target positions
        all_inserted = True
        inserted_objects = []

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        print(max(self.calculate_obj_roi(self.container, self.conical)))
        if max(self.calculate_obj_roi(self.container, self.conical)) > 0.2:
            success = torch.ones_like(success)
        return success
