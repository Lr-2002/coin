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


@register_env("Tabletop-Move-Cube-DynamicFriction-v1", max_episode_steps=5000)
class PickPlaceDynamicFrictionCubeEnv(UniversalTabletopEnv):
    description = "move the cube to the marker"
    workflow = ["pick the cube and put it on the marker"]

    def __init__(self, *args, **kwargs):
        print("===== args ", args, kwargs)

        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.FRICTION, Obj.MASS],
            "iter": [Inter.PLAN,   Inter.FAIL_ADAPT, Inter.HISTORY],
            "rob": [Robot.ACT_NAV],
        }
        self.object_friction = 0
        self.config = "xx.config"
        super().__init__(*args, **kwargs)
        self.query_query = (
            "What's the best approach to move a cube with unpredictable friction?"
        )
        self.query_selection = {
            "A": "Lift the cube and directly place it to the target position",
            "B": "Slide the cube while maintaining contact with the surface",
            "C": "Pick up blocks, re-pick them up if it falls",
        }
        self.query_answer = "B"
        # Did it pick up or push the cube?
        # Did it make the cube to the target?

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)
        self.cube = self.load_from_config(
            self.config, "cube", friction_override=self.object_friction
        )
        # self.cube.set_friction(object_friction=self.object_friction)
        self.goal = self._create_goal_area()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.cube.set_pose(sapien.Pose(p=[0.1, 0.1, 0.1], q=euler2quat(0, 0, 0)))

    def _get_success(self, env_idx=None):
        succ = super()._get_success()
        if self.calculate_object_distance(
            self.cube, self.goal
        ) <= 0.04 and self.is_stable(self.cube):
            succ = torch.ones_like(succ)
        return succ

    # self.object_mass = randomization.uniform(0.1, 100)
