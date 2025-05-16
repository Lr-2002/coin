from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat
import random
import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
import numpy as np
from mani_skill.envs.tasks.coin_bench.primitive_actions.pick_place import PickPlaceEnv
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
    load_articulation_from_json,
    load_articulation_from_urdf,
)

"""
this task need to be finished with some target pose over the table 

"""


@register_env("Tabletop-Lift-Book-v1", max_episode_steps=5000)
class LiftBookEnv(UniversalTabletopEnv):
    description = "lift the book up to the higher platform"
    workflow = [
        "move the book to the side of the book",
        "pick the book from the short side",
        "lift the book and put it on the high box",
    ]

    def __init__(self, *args, **kwargs):
        # print("===== args ", args, kwargs)
        # object_friction = random.uniform(0.1, 1)
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.ORIENT, Obj.SCALE],
            "rob": [Robot.MORPH],
            "iter": [ Inter.PLAN, Inter.TOOL]
        }
        self.book_config = "configs/book9.json"
        self.box_config = "configs/box.json"
        self.plat_config = "configs/box.json"
        super().__init__(*args, **kwargs)
        self.query_query = "How to move the book to the higher platform?"
        self.query_selection = {
            "A": "Directly grasp the book and lift it up to the higher platform",
            "B": "Push the book to create an overhang at the table edge, then grasp and lift",
            "C": "Attempt to slide fingers underneath the book without creating an overhang first",
        }
        self.query_answer = "B"

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.book = self.load_from_config(self.book_config, "book", convex=True)
        position = np.array([0.0, -0.0, self.table_height])
        orientation = np.array(np.deg2rad([0, 0, 0]))  # Rotate to face the robot
        self.box = self.load_from_config(self.box_config, "box", body_type="static")

        self.plat = self.load_from_config(
            self.plat_config, "plat", body_type="static", scale_override=[0.7, 0.7, 0.6]
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.box.set_pose(sapien.Pose(p=(0, 0, 0.03), q=(0.7070, 0, 0, 0.707)))
        self.plat.set_pose(sapien.Pose(p=(0.0, 0.2, 0.02), q=(1, 0, 0, 0)))
        self.book.set_pose(
            sapien.Pose(p=(-0.05, -0.0, 0.13), q=(0.0, 0.7070, 0, 0.707))
        )

    # raise NotImplementedError
    def _get_success(self, env_idx=None):
        succ = super()._get_success(env_idx)
        if (
            self.book.pose.p[0][2] > 0.17
            and self.is_static(self.book)
            and self.book.pose.p[0][1] >= 0.1
            and self.is_stable(self.book)
        ):
            succ = torch.ones_like(succ)
        return succ
        # if

    # self.object_mass = randomization.uniform(0.1, 100)
