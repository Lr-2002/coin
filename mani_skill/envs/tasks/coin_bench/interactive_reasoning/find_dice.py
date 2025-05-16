from typing import Any, Dict, List, Optional, Tuple, Union
import random
import os
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import random

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv


@register_env("Tabletop-Find-Dice-v1", max_episode_steps=5000)
class FindDiceEnv(UniversalTabletopEnv):
    """
    Find Seal Environment

    Task: Place seal1 on the red and white target area

    Features:
    1. Three seals are placed on the ground
    2. A white and red target area is drawn using the ManiSkill API
    3. The goal is to place seal1 on the red and white target area
    """

    description = "find the dice which have 2 and 4 point in the corresponding face and put it on the marker"

    workflow = [
        "rotate left dice",
        "rotate right dice",
        "find the dice have 2 and 4 in the corresponding face and put it on the marker",
    ]

    def __init__(self, *args, **kwargs):
        self.random_number = random.randint(0, 1)
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.ORIENT, Obj.MESH],
            "rob": [Robot.PERSPECTIVE],
            "iter": [Inter.HISTORY, Inter.PLAN]
        }
        self.random_number = 0
        super().__init__(*args, **kwargs)

        self.query_query = "which dice has 2 and 4 point in the corresponding face"
        self.query_selection = {"A": "A", "B": "B"}
        self.query_answer = "A"  # A is dice 1

    def _load_scene(self, options):
        super()._load_scene(options)
        self.dice1 = self.load_from_config("configs/dice1.json", "dice1", convex=True)

        self.dice2 = self.load_from_config("configs/dice2.json", "dice2", convex=True)

        self.marker = self._create_goal_area()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.dice1.set_pose(sapien.Pose([0.1, -0.1, 0.00], euler2quat(0, np.pi / 2, 0)))
        self.dice2.set_pose(sapien.Pose([0.1, 0.1, 0.00], euler2quat(0, np.pi / 2, 0)))

    def _get_success(self, env_idx=None):
        success = super()._get_success(env_idx)

        if self.random_number == 0:
            if self.calculate_object_distance(
                self.dice2, self.marker
            ) <= 0.05 and self.is_static(self.dice2):
                success = torch.ones_like(success)

        else:
            if self.calculate_object_distance(
                self.dice1, self.marker
            ) <= 0.05 and self.is_static(self.dice1):
                success = torch.ones_like(success)
        return success
